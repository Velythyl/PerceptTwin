import os
import uuid


def get_temp_filename(prefix="hippo",suffix="",ext=".tmp"):
    ret = f"{prefix}{uuid.uuid4()}{suffix}"
    if ext == "":
        return ret
    if ext.startswith("."):
        ext = ext[1:]
    return f"{ret}.{ext}"

def get_tmp_file(folder="/tmp", prefix="hippo", suffix="", ext=".tmp"):
    return f"{folder}/{get_temp_filename(prefix,suffix,ext)}"

def get_tmp_folder():
    folder_name = get_temp_filename(ext="")
    folder_path = f"/tmp/{folder_name}"
    os.makedirs(folder_path)
    return folder_path

def get_next_file_counter(folder, subpath):
    os.makedirs(folder, exist_ok=True)
    counter = 0
    def p(c, s):
        return f"{c}{'_'+s if (s or len(s) >= 1) else ''}"
    for file in os.listdir(folder):
        if file.endswith(subpath):
            try:
                int(file.split("_")[0])
                counter += 1
            except:
                pass
    return f"{folder}/{p(counter, subpath)}"
