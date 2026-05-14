# utils/state.py
import os

def get_last_processed_index(state_file):
    """Reads the last processed index from a simple text file."""
    if not os.path.exists(state_file):
        return -1
    try:
        with open(state_file, 'r') as f:
            content = f.read().strip()
            return int(content) if content else -1
    except ValueError:
        return -1

def update_last_processed_index(state_file, index):
    """Writes the current index to a simple text file."""
    with open(state_file, 'w') as f:
        f.write(str(index))