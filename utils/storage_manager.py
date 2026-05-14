import sqlite3
import h5py
import numpy as np
import collections
import json

class StorageManager:
    def __init__(self, db_path, h5_path, flush_interval=100, buffer_limit=50):
        self.db_path = db_path
        self.h5_path = h5_path
        self.conn = self._init_sqlite()
        self.h5_file = h5py.File(h5_path, 'a')
        self.flush_interval = flush_interval
        self._write_count = 0

        # Write buffers: collect appends in memory, flush to HDF5 in bulk
        self._tensor_buffers = {}       # dataset_path -> list of numpy arrays
        self._tensor_metadata = {}      # dataset_path -> metadata dict (stored until first flush)
        self._vlen_buffers = {}         # dataset_path -> list of vlen arrays
        self._vlen_metadata = {}
        self._buffer_limit = buffer_limit

    def close(self):
        self.flush_all_buffers()
        if self.h5_file:
            self.h5_file.flush()
            self.h5_file.close()
        if self.conn:
            self.conn.close()

    # ==========================================
    #  SQLite Setup
    # ==========================================
    def _init_sqlite(self):
        # conn = sqlite3.connect(self.db_path)
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.execute('PRAGMA journal_mode=WAL')
        cursor = conn.cursor()
        
        cursor.execute('CREATE TABLE IF NOT EXISTS Models (id INTEGER PRIMARY KEY, model_size TEXT, step INTEGER, UNIQUE(model_size, step))')
        cursor.execute('CREATE TABLE IF NOT EXISTS Sentences (id INTEGER PRIMARY KEY, content TEXT UNIQUE)')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS Rankings (
            id INTEGER PRIMARY KEY, model_id INTEGER, layer_index INTEGER,
            component_name TEXT, head_index INTEGER, rank_type INTEGER, 
            sentence_id INTEGER, token_index INTEGER, score REAL,
            text_snippet TEXT, prediction_data TEXT, 
            FOREIGN KEY(model_id) REFERENCES Models(id),
            FOREIGN KEY(sentence_id) REFERENCES Sentences(id))''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rankings_lookup ON Rankings(model_id, layer_index, component_name, head_index, rank_type)')
        conn.commit()
        return conn

    # ==========================================
    #  Transaction Control
    # ==========================================
    def begin_transaction(self):
        self.conn.execute('BEGIN')

    def commit(self):
        self.conn.commit()

    def rollback_sqlite(self):
        self.conn.rollback()

    # ==========================================
    #  Basic Operations
    # ==========================================
    def get_model_id(self, model_size, step):
        c = self.conn.cursor()
        c.execute('INSERT OR IGNORE INTO Models (model_size, step) VALUES (?, ?)', (model_size, step))
        c.execute('SELECT id FROM Models WHERE model_size = ? AND step = ?', (model_size, step))
        return c.fetchone()[0]

    def register_sentences(self, sentences):
        c = self.conn.cursor()

        # Bulk insert (one round-trip)
        c.executemany(
            'INSERT OR IGNORE INTO Sentences (content) VALUES (?)',
            [(s,) for s in sentences]
        )

        # Bulk select (one round-trip)
        placeholders = ','.join('?' * len(sentences))
        c.execute(
            f'SELECT id, content FROM Sentences WHERE content IN ({placeholders})',
            sentences
        )

        # Map back to original order
        content_to_id = {row[1]: row[0] for row in c.fetchall()}
        return [content_to_id[s] for s in sentences]

    # ==========================================
    #  Ranking Operations
    # ==========================================
    def fetch_rankings(self, model_id):
        """Loads existing rankings for the tracker."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, layer_index, component_name, head_index, rank_type, 
                   sentence_id, token_index, score, text_snippet, prediction_data
            FROM Rankings WHERE model_id = ?
        ''', (model_id,))
        
        rankings = collections.defaultdict(list)
        rows = cursor.fetchall()
        existing_ids = set(r[0] for r in rows)
        
        for row in rows:
            row_id, layer, comp, head, r_type, s_id, t_idx, score, snippet, pred_data = row
            rankings[(layer, comp, head)].append(
                (score, s_id, t_idx, row_id, snippet, pred_data)
            )
            
        return rankings, existing_ids

    def sync_rankings(self, model_id, rows_to_add, ids_to_delete):
        """Deletes old rankings and inserts new ones.
        
        Returns:
            list of row IDs for the newly inserted rows (in same order as rows_to_add).
        """
        cursor = self.conn.cursor()
        
        if ids_to_delete:
            cursor.executemany('DELETE FROM Rankings WHERE id = ?', [(rid,) for rid in ids_to_delete])
        
        new_ids = []
        if rows_to_add:
            for r in rows_to_add:
                cursor.execute('''
                    INSERT INTO Rankings (model_id, layer_index, component_name, head_index, rank_type, 
                                          sentence_id, token_index, score, text_snippet, prediction_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (model_id, *r))
                new_ids.append(cursor.lastrowid)
        
        # No commit here — runner controls the transaction
        return new_ids

    # ==========================================
    #  HDF5 Write Buffering
    # ==========================================
    def flush_all_buffers(self):
        """Drain all in-memory buffers to HDF5."""
        for path in list(self._tensor_buffers.keys()):
            self._flush_tensor_buffer(path)
        for path in list(self._vlen_buffers.keys()):
            self._flush_vlen_buffer(path)
        self.h5_file.flush()
        self._write_count = 0

    def _flush_tensor_buffer(self, dataset_path):
        buf = self._tensor_buffers.get(dataset_path)
        if not buf:
            return

        combined = np.concatenate(buf, axis=0)
        metadata = self._tensor_metadata.pop(dataset_path, None)

        if dataset_path not in self.h5_file:
            shape = (0,) + combined.shape[1:]
            maxshape = (None,) + combined.shape[1:]
            dset = self.h5_file.create_dataset(
                dataset_path, shape=shape, maxshape=maxshape,
                dtype=combined.dtype, chunks=True, compression="gzip"
            )
            if metadata:
                for k, v in metadata.items():
                    dset.attrs[k] = v

        dset = self.h5_file[dataset_path]
        curr = dset.shape[0]
        dset.resize(curr + combined.shape[0], axis=0)
        dset[curr:] = combined

        self._tensor_buffers[dataset_path] = []

    def _flush_vlen_buffer(self, dataset_path):
        buf = self._vlen_buffers.get(dataset_path)
        if not buf:
            return

        # Flatten list of lists into one list
        combined = []
        for chunk in buf:
            combined.extend(chunk)

        metadata = self._vlen_metadata.pop(dataset_path, None)

        if dataset_path not in self.h5_file:
            dt = h5py.vlen_dtype(np.dtype('float32'))
            dset = self.h5_file.create_dataset(
                dataset_path, shape=(0,), maxshape=(None,),
                dtype=dt, chunks=True, compression="gzip"
            )
            if metadata:
                for k, v in metadata.items():
                    dset.attrs[k] = v

        dset = self.h5_file[dataset_path]
        curr = dset.shape[0]
        dset.resize(curr + len(combined), axis=0)
        dset[curr:] = combined

        self._vlen_buffers[dataset_path] = []

    # ==========================================
    #  HDF5 Checkpoint / Rollback
    # ==========================================
    def checkpoint_h5(self):
        """Flush buffers, then snapshot current dataset sizes."""
        self.flush_all_buffers()
        sizes = {}
        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                sizes[name] = obj.shape[0]
        self.h5_file.visititems(_visit)
        self.h5_file.attrs['_checkpoint_sizes'] = json.dumps(sizes)

    def rollback_to_checkpoint(self):
        """Discard buffers and truncate datasets back to last checkpoint."""
        # Discard anything not yet written
        self._tensor_buffers.clear()
        self._tensor_metadata.clear()
        self._vlen_buffers.clear()
        self._vlen_metadata.clear()

        raw = self.h5_file.attrs.get('_checkpoint_sizes', None)
        if raw is None:
            return
        sizes = json.loads(raw)
        for name, expected_rows in sizes.items():
            if name in self.h5_file:
                dset = self.h5_file[name]
                if dset.shape[0] > expected_rows:
                    print(f"Rolling back {name}: {dset.shape[0]} -> {expected_rows}")
                    dset.resize(expected_rows, axis=0)
        self.h5_file.flush()

    # ==========================================
    #  HDF5 Operations (buffered)
    # ==========================================
    def log_tensor(self, dataset_path, tensor_data, metadata=None):
        if tensor_data.shape[0] == 0:
            return

        if dataset_path not in self._tensor_buffers:
            self._tensor_buffers[dataset_path] = []
            if metadata:
                self._tensor_metadata[dataset_path] = metadata

        self._tensor_buffers[dataset_path].append(tensor_data)

        if len(self._tensor_buffers[dataset_path]) >= self._buffer_limit:
            self._flush_tensor_buffer(dataset_path)
    
    def log_variable_length_tensor(self, dataset_path, tensor_data_list, metadata=None):
        """
        Store variable-length arrays (e.g., components with different counts per layer).
        
        Args:
            dataset_path: HDF5 path for dataset
            tensor_data_list: list of 1D numpy arrays with potentially different lengths
            metadata: optional dict of attributes
        """
        if len(tensor_data_list) == 0:
            return

        if dataset_path not in self._vlen_buffers:
            self._vlen_buffers[dataset_path] = []
            if metadata:
                self._vlen_metadata[dataset_path] = metadata

        self._vlen_buffers[dataset_path].append(tensor_data_list)

        if len(self._vlen_buffers[dataset_path]) >= self._buffer_limit:
            self._flush_vlen_buffer(dataset_path)