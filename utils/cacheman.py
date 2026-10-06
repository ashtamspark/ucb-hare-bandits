from pathlib import Path
import numpy as np

CACHE_DIR = Path(__file__).resolve().parents[1] / "cached_data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_or_run(filename, func, force_rerun=False):
    """
    Loads a cached NumPy array from disk, or executes the provided function to generate it.
    """
    filepath = CACHE_DIR / filename
    
    if not force_rerun and filepath.exists():
        print(f"[CACHE HIT] Loading {filename}...")
        return np.load(filepath, allow_pickle=True)
    
    print(f"[CACHE MISS] Simulating {filename}...")
    result = func()
    
    # Removed dtype=object so the array retains its native int64 typing
    result_array = np.asarray(result) 
    np.save(filepath, result_array)
    
    return result_array
