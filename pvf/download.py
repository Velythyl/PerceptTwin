import os
import json
import pandas as pd
import wandb
from wandb.apis.public import Api
from concurrent.futures import ThreadPoolExecutor, as_completed

wandb.login()
api = Api()
entity = "velythyl"
project = "pvf_sep7_7pm"

runs = api.runs(f"{entity}/{project}")

base_root = "downloaded_runs"
os.makedirs(base_root, exist_ok=True)

def download_run(run):
    run_id = run.id
    run_dir = os.path.join(base_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    try:
        # Save config
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Download all files
        for file in run.files():
            try:
                file.download(root=run_dir, replace=True)
            except Exception as e:
                print(f"[{run_id}] Failed to download {file.name}: {e}")

        # Save history
        try:
            history_df = run.history(pandas=True)
            history_df.to_csv(os.path.join(run_dir, "history.csv"), index=False)
        except Exception as e:
            print(f"[{run_id}] Failed to download history: {e}")

        return f"[{run_id}] ✅ Finished"

    except Exception as e:
        return f"[{run_id}] ❌ Error: {e}"


# Run downloads in parallel
max_workers = 8  # adjust depending on your system and WANDB rate limits
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(download_run, run): run.id for run in runs}
    for future in as_completed(futures):
        print(future.result())
