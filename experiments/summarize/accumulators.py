"""Vectorized streaming accumulators for summary statistics.

GlobalAccumulator: sum-based (not Welford) for overall mean/std.
PerSentenceAccumulator: streaming flush to HDF5, handles reappearing sentences.
"""

import numpy as np
import h5py


class GlobalAccumulator:
    """Accumulates (sum, sum_sq, count) over chunks for mean/std."""

    def __init__(self, shape, valid_mask):
        self._sum = np.zeros(shape, dtype=np.float64)
        self._sum_sq = np.zeros(shape, dtype=np.float64)
        self._count = 0
        self._valid = valid_mask

    def update_chunk(self, data):
        masked = data * self._valid[np.newaxis, ...]
        self._sum += masked.sum(axis=0)
        self._sum_sq += (masked ** 2).sum(axis=0)
        self._count += data.shape[0]

    def finalize(self):
        safe_c = max(self._count, 1)
        mean = self._sum / safe_c
        variance = (self._sum_sq / safe_c) - (mean ** 2)
        std = np.sqrt(np.maximum(variance, 0.0))
        mean = np.where(self._valid, mean, 0.0).astype(np.float32)
        std = np.where(self._valid, std, 0.0).astype(np.float32)
        count = np.full(mean.shape, self._count, dtype=np.int64)
        return mean, std, count


class PerSentenceAccumulator:
    """Streaming per-sentence stats with flush-to-HDF5.

    Tokens from the same sentence are *mostly* contiguous in the HDF5,
    but may reappear after being flushed. Handles this by reloading
    previously-flushed data and merging.
    """

    def __init__(self, edge_shape, valid_mask, out_h5_group, prefix_attn=True):
        self.edge_shape = edge_shape
        self.valid_mask = valid_mask
        self._out_grp = out_h5_group
        self._mean_key = "attn_edge_mean" if prefix_attn else "mlp_edge_mean"
        self._std_key = "attn_edge_std" if prefix_attn else "mlp_edge_std"
        self._data = {}          # sid -> (sum, sum_sq, count)
        self._active_sids = set()
        self._prev_sids = set()
        self.flushed = 0

    def _ensure_sid(self, sid):
        if sid not in self._data:
            # Check if previously flushed — reload and merge
            sid_grp = self._out_grp.get(str(sid))
            if sid_grp and self._mean_key in sid_grp:
                old_mean = np.array(sid_grp[self._mean_key], dtype=np.float64)
                old_std = np.array(sid_grp[self._std_key], dtype=np.float64)
                old_count = int(sid_grp.attrs.get(self._mean_key + "_count", 0))
                if old_count > 0:
                    old_sum = old_mean * old_count
                    old_sum_sq = (old_std ** 2 + old_mean ** 2) * old_count
                    del sid_grp[self._mean_key]
                    del sid_grp[self._std_key]
                    self._data[sid] = (old_sum, old_sum_sq, old_count)
                    self.flushed -= 1
                    return
            self._data[sid] = (
                np.zeros(self.edge_shape, dtype=np.float64),
                np.zeros(self.edge_shape, dtype=np.float64),
                0,
            )

    def update_chunk(self, data, sids):
        self._active_sids = set()
        unique_sids = np.unique(sids)
        for sid in unique_sids:
            sid = int(sid)
            self._active_sids.add(sid)
            self._ensure_sid(sid)
            mask = sids == sid
            subset = data[mask]
            s, sq, c = self._data[sid]
            masked = subset * self.valid_mask[np.newaxis, ...]
            s += masked.sum(axis=0)
            sq += (masked ** 2).sum(axis=0)
            self._data[sid] = (s, sq, c + int(mask.sum()))

    def flush_stale(self):
        stale = self._prev_sids - self._active_sids
        for sid in stale:
            if sid in self._data:
                self._write_and_delete(sid)
        self._prev_sids = set(self._active_sids)

    def flush_all(self):
        for sid in list(self._data.keys()):
            self._write_and_delete(sid)

    def _write_and_delete(self, sid):
        s, sq, c = self._data[sid]
        safe_c = max(c, 1)
        mean = s / safe_c
        variance = (sq / safe_c) - (mean ** 2)
        std = np.sqrt(np.maximum(variance, 0.0))
        mean = np.where(self.valid_mask, mean, 0.0).astype(np.float32)
        std = np.where(self.valid_mask, std, 0.0).astype(np.float32)

        sid_grp = self._out_grp.require_group(str(sid))
        if self._mean_key in sid_grp:
            del sid_grp[self._mean_key]
        if self._std_key in sid_grp:
            del sid_grp[self._std_key]
        sid_grp.create_dataset(self._mean_key, data=mean, compression="gzip")
        sid_grp.create_dataset(self._std_key, data=std, compression="gzip")
        sid_grp.attrs[self._mean_key + "_count"] = c

        del self._data[sid]
        self.flushed += 1
