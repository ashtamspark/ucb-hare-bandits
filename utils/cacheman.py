import os
import numpy as np

CACHE_DIR = "cached_data"
os.makedirs(CACHE_DIR, exist_ok=True)


def load_or_run(filename, func, force_rerun=False):
    """
    Loads a cached NumPy array from disk, or executes the provided function to generate it.
    """
    filepath = os.path.join(CACHE_DIR, filename)
    
    if not force_rerun and os.path.exists(filepath):
        print(f"[CACHE HIT] Loading {filename}...")
        return np.load(filepath, allow_pickle=True)
    
    print(f"[CACHE MISS] Simulating {filename}...")
    result = func()
    
    # Removed dtype=object so the array retains its native int64 typing
    result_array = np.asarray(result) 
    np.save(filepath, result_array)
    
    return result_array