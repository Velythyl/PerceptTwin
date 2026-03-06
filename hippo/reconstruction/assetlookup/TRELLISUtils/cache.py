import os
import shutil
from pathlib import Path
home = Path.home()

TRELLIS_GEN_DIR = f"{home}/.TRELLIS"

def path_in_cache(path):
    path = path.replace(".", "/")
    path = f"{TRELLIS_GEN_DIR}/{path}"
    while "//" in path:
        path = path.replace("//", "/")
    return path

def path_in_cache_for_raw(path):
    return f"{path_in_cache(path)}/raw"

def path_in_cache_for_convert(path):
    return f"{path_in_cache(path)}/convert"

def path_in_cache_for_metadata(path):
    return f"{path_in_cache(path)}/metadata"

def path_in_cache_for_trellisfailure(path):
    return f"{path_in_cache(path)}/failure"

def is_in_cache(path):
    return os.path.exists(path_in_cache_for_raw(path)) and os.path.exists(path_in_cache_for_convert(path)) and os.path.exists(path_in_cache_for_metadata(path))

def cache_says_trellis_fails(path):
    return os.path.exists(path_in_cache_for_trellisfailure(path))

def clear_cache(path):
    shutil.rmtree(path_in_cache(path), ignore_errors=True)