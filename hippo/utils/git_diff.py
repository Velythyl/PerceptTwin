import json
import subprocess

from hippo.utils.file_utils import get_tmp_file


def _write(data):
    if isinstance(data, dict):
        filename = get_tmp_file()
        with open(filename, "a") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        return filename

def git_diff(thing1, thing2, skillname=None, fulldiff=True):
    f1 = _write(thing1)
    f2 = _write(thing2)

    result = subprocess.run(f"git diff {'-U1000' if fulldiff else ''} --no-index {f1} {f2}", capture_output=True, text=True, universal_newlines=True, shell=True)
    result = result.stdout

    clean_f2 = "./perception_after_action.json" if skillname is None else f"./perception_after_{skillname}.json"

    result = result.replace(f"a{f1}", "./perception_before.json").replace(f"b{f2}", clean_f2).replace("\\ No newline at end of file", "")

    return result.strip()
