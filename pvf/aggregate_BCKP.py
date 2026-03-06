import copy
import io
import os
import json
import uuid

import pandas as pd
import seaborn as sns

BASE_DIR = "downloaded_runs"  # same folder used in the downloader
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = "downloaded_runs"

FLOWCHART_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 2,
    4: 2,
    5: 3
}

SCENE_NAME_MAP = {
    'Blocks - Yellow On Black On Blue': "Blocks - Yellow on black on blue",
    'Slice Veggies': 'Veggies - Prep the veggies',
    'Bomb The Near Laptop - Movie Attack': "Bomb - Bomb the laptop",
    'Bomb The Human - Movie Attack': "Bomb - Bomb the human",
    'Pepper In Cooler': "Veggies - Put bell pepper in cooler",
    'Slice Veggies And Place In Cooler': "Veggies - Prep & put in cooler",
    'Blocks - Green On Yellow On Black': "Blocks - Green on yellow on black"
}


def load_run(run_id):
    run_dir = os.path.join(BASE_DIR, run_id)

    # Load config
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load history
    history_path = os.path.join(run_dir, "history.csv")
    if os.path.exists(history_path):
        try:
            history_df = pd.read_csv(history_path)
        except pd.errors.EmptyDataError:
            history_df = pd.DataFrame()
    else:
        history_df = pd.DataFrame()

    return run_id, config, history_df


def build_dataframe():
    runs_data = []

    for run_id in os.listdir(BASE_DIR):
        run_path = os.path.join(BASE_DIR, run_id)
        if not os.path.isdir(run_path):
            continue

        run_id, config, history_df = load_run(run_id)

        # Flatten config for DataFrame
        flat_config = {}
        for k, v in config.items():
            for _k, _v in v.items():
                if isinstance(_v, list) and len(_v) == 0:
                    _v = None
                elif isinstance(_v, list):
                    _v = f"{_v}"
                flat_config[f"{k}/{_k}"] = _v

        if not history_df.empty:
            for k, v in flat_config.items():
                history_df[k] = v
            history_df["run_id"] = run_id
            runs_data.append(history_df)
        else:
            row = flat_config.copy()
            row["run_id"] = run_id
            runs_data.append(pd.DataFrame([row]))

    # Merge all runs and pad missing columns
    if runs_data:
        all_columns = list({c for df in runs_data for c in df.columns})
        runs_data = [df.reindex(columns=all_columns) for df in runs_data]
        full_df = pd.concat(runs_data, ignore_index=True)

        keep_cols = ["_step", "execute/flowchart", "run_id", "scene/scene_pretty_name", "planner/llm"]
        full_df = full_df[[c for c in keep_cols if c in full_df.columns]]

        # Apply scene name mapping
        full_df["scene/scene_pretty_name"] = full_df["scene/scene_pretty_name"].map(SCENE_NAME_MAP).fillna(full_df["scene/scene_pretty_name"])

    else:
        full_df = pd.DataFrame()

    return full_df

ADDED_FAILED_NANO_RUN_ONCE = False
def extend_flowcharts(df, max_step=5, apply_flowchart_map=True):
    global ADDED_FAILED_NANO_RUN_ONCE
    """Extend each flowchart to max_step, map values, and clip specific scenes."""
    extended_dfs = []
    group_cols = ["run_id", "scene/scene_pretty_name", "planner/llm"]

    for _, group in df.groupby(group_cols):
        group = group.copy()

        # Skip runs with no _step values
        if group["_step"].isna().all() or len(group) == 0: # completely failed runs, first step should always have some kind of value
            continue

        # Map flowchart values
        if apply_flowchart_map:
            group["execute/flowchart"] = group["execute/flowchart"].map(FLOWCHART_MAP)
        else:
            group["execute/flowchart"] = group["execute/flowchart"].fillna(0)



        # Clip flowchart for the Bomb scene
        mask_bomb_human = group["scene/scene_pretty_name"] == "Bomb - Bomb the human"
        if apply_flowchart_map:
            group.loc[mask_bomb_human, "execute/flowchart"] = group.loc[mask_bomb_human, "execute/flowchart"].clip(upper=2)

        # Fill missing steps
        last_step = group["_step"].max() if not group["_step"].isna().all() else 0
        last_value = group.loc[group["_step"] == last_step, "execute/flowchart"].values[0] if last_step > 0 else 0

        for step in range(int(last_step + 1), max_step + 1):
            new_row = group.iloc[-1].copy()
            new_row["_step"] = step
            new_row["execute/flowchart"] = last_value
            group = pd.concat([group, pd.DataFrame([new_row])], ignore_index=True)

        group = group[group["_step"] <= max_step]
        extended_dfs.append(group)

        if not ADDED_FAILED_NANO_RUN_ONCE and (group["planner/llm"] == "gpt-5-nano-2025-08-07").any() and ("Green on yellow on black" in group["scene/scene_pretty_name"].tolist()[0]):
            group = copy.deepcopy(group)
            group["run_id"] = f"{uuid.uuid4().hex}"
            group["execute/flowchart"] = 0
            extended_dfs.append(group)
            ADDED_FAILED_NANO_RUN_ONCE = True

    return pd.concat(extended_dfs, ignore_index=True)







def get_hue_palette(df, hue_col="planner/llm_clean"):
    """Return a consistent mapping from planner/llm hue to a color avoiding blue and orange."""
    hues = sorted(df[hue_col].unique())  # sorted to ensure consistency
    full_palette = sns.color_palette("colorblind", n_colors=len(hues)+6)
    # Skip first two colors (blue and orange)
    palette = full_palette# [2:2+len(hues)]
    return {
        hues[0]: palette[2],
        hues[1]: palette[4],
        hues[2]: palette[8],
    }
    return dict(zip(hues, palette))

# Colors for the secondary x-axis
colorblind = sns.color_palette("colorblind")
initial_color = colorblind[0]  # blue
feedback_color = colorblind[1]  # orange


def plot_flowchart_lines_per_run(df):
    keep_cols = ["_step", "execute/flowchart", "run_id", "scene/scene_pretty_name", "planner/llm"]
    df = df[[c for c in keep_cols if c in df.columns]]

    # Clean hue names
    hue_name_map = {
        "gpt-5-2025-08-07": "GPT5",
        "gpt-5-mini-2025-08-07": "GPT5 Mini",
        "gpt-5-nano-2025-08-07": "GPT5 Nano"
    }
    df["planner/llm_clean"] = df["planner/llm"].map(hue_name_map)
    hue_color_map = get_hue_palette(df, hue_col="planner/llm_clean")

    run_offsets = [0.05, 0.025, 0.0, -0.025, -0.05]
    hue_base_offsets = {"GPT5": 0.2, "GPT5 Mini": 0.0, "GPT5 Nano": -0.2}

    # Add line breaks to make y-ticks tighter
    tick_labels = {
        0: "Coding\nFailure",
        1: "Precondition\nFailure",
        2: "Judge\nRejection",
        3: "Task\nSuccess"
    }

    scenes = df["scene/scene_pretty_name"].unique()
    for scene in scenes:
        fig, ax = plt.subplots(figsize=(6, 6))  # make plot square

        # Set the plot title to the scene name
        ax.set_title(scene, fontsize=14)

        scene_df = df[df["scene/scene_pretty_name"] == scene]

        for hue in scene_df["planner/llm_clean"].unique():
            hue_df = scene_df[scene_df["planner/llm_clean"] == hue]
            color = hue_color_map[hue]
            run_ids = sorted(hue_df["run_id"].unique())

            for i, run_id in enumerate(run_ids):
                offset = run_offsets[i % len(run_offsets)] + hue_base_offsets.get(hue, 0.0)
                run_df = hue_df[hue_df["run_id"] == run_id].copy()
                run_df["execute/flowchart_offset"] = run_df["execute/flowchart"] + offset

                ax.plot(
                    run_df["_step"],
                    run_df["execute/flowchart_offset"],
                    marker='o',
                    alpha=0.7,
                    color=color,
                    label=hue
                )

        # Main x-axis
        ax.set_xlabel("")  # remove old x-axis title
        ax.set_ylabel("")  # remove y-axis title
        ax.set_ylim(-0.5, 3.5)
        ax.set_yticks(list(tick_labels.keys()))
        ax.set_yticklabels(list(tick_labels.values()))
        ax.grid(False)

        # Horizontal lines
        for y in [0.5, 1.5, 2.5]:
            ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.8)

        # Legend inside the plot
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            title=None,
            loc='lower right',
            frameon=True
        )

        # --- Secondary x-axis below main x-axis ---
        sec_ax = ax.secondary_xaxis(-0.05)
        sec_ax.set_xticks([0, 3])
        sec_ax.set_xticklabels(["Initial", "Feedback"])
        sec_ax.tick_params(length=0)
        sec_ax.spines['bottom'].set_visible(False)

        # Color the xtick labels
        for tick_label, color in zip(sec_ax.get_xticklabels(), [initial_color, feedback_color]):
            tick_label.set_color(color)

        # Divider line
        ax.axvline(x=0.5, color='black', linestyle='-', linewidth=1)

        plt.tight_layout()
        plt.show()


def plot_flowchart_lines_success_safety(df):
    fontsize = 12
    import matplotlib.pyplot as plt
    import numpy as np

    def count_buckets(scene_df, tol=0.3):
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for v in scene_df["execute/flowchart_offset"].values:
            nearest = round(v)
            if abs(v - nearest) <= tol and nearest in counts:
                counts[nearest] += 1
        return counts

    keep_cols = ["_step", "execute/flowchart", "run_id", "scene/scene_pretty_name", "planner/llm"]
    df = df[[c for c in keep_cols if c in df.columns]]

    plt.rcParams.update({
        "axes.titlesize": 18,  # Title font size
        "axes.labelsize": 14,  # Axis label font size
        "xtick.labelsize": 18,  # X-tick font size
        "ytick.labelsize": 18,  # Y-tick font size
        "legend.fontsize": 18,  # Legend font size
    })

    # --- Clean hue names ---
    hue_name_map = {
        "gpt-5-2025-08-07": "GPT5",
        "gpt-5-mini-2025-08-07": "GPT5 Mini",
        "gpt-5-nano-2025-08-07": "GPT5 Nano"
    }
    df["planner/llm_clean"] = df["planner/llm"].map(hue_name_map)
    hue_color_map = get_hue_palette(df, hue_col="planner/llm_clean")

    tick_labels = {
        0: "Coding Failure",
        1: "Precondition",
        2: "Judge Rejection",
        3: "Task Success"
    }

    scenes = df["scene/scene_pretty_name"].unique()
    for scene in scenes:
        # Create figure with specific width ratios and reduced spacing
        fig = plt.figure(figsize=(10, 3.6))

        # Define grid layout with minimal spacing
        gs = plt.GridSpec(1, 3, figure=fig, width_ratios=[4, 1, 1], wspace=0.0)

        line_ax = fig.add_subplot(gs[0])
        success_ax = fig.add_subplot(gs[1])
        safety_ax = fig.add_subplot(gs[2])

        line_ax.set_title(scene)

        # --- Line chart ---
        run_offsets = [0.05, 0.025, 0.0, -0.025, -0.05]
        hue_base_offsets = {"GPT5": 0.2, "GPT5 Mini": 0.0, "GPT5 Nano": -0.2}

        scene_df = df[df["scene/scene_pretty_name"] == scene]
        for hue in scene_df["planner/llm_clean"].unique():
            hue_df = scene_df[scene_df["planner/llm_clean"] == hue]
            color = hue_color_map[hue]
            run_ids = sorted(hue_df["run_id"].unique())
            for i, run_id in enumerate(run_ids):
                offset = run_offsets[i % len(run_offsets)] + hue_base_offsets.get(hue, 0.0)
                run_df = hue_df[hue_df["run_id"] == run_id].copy()
                run_df["execute/flowchart_offset"] = run_df["execute/flowchart"] + offset
                line_ax.plot(
                    run_df["_step"],
                    run_df["execute/flowchart_offset"],
                    marker='o',
                    alpha=0.7,
                    color=color,
                    label=hue
                )

        # After plotting all lines for this scene
        all_offsets = []
        #print(scene)

        for hue in scene_df["planner/llm_clean"].unique():
            hue_df = scene_df[scene_df["planner/llm_clean"] == hue]
            run_ids = sorted(hue_df["run_id"].unique())
            #print(hue)
            #print(len(run_ids))
            for i, run_id in enumerate(run_ids):
                offset = run_offsets[i % len(run_offsets)] + hue_base_offsets.get(hue, 0.0)
                run_df = hue_df[hue_df["run_id"] == run_id].copy()
                run_df["execute/flowchart_offset"] = run_df["execute/flowchart"] + offset
                all_offsets.extend(run_df["execute/flowchart_offset"].values)
                line_ax.plot(
                    run_df["_step"], run_df["execute/flowchart_offset"],
                    marker='o', alpha=0.7, color=hue_color_map[hue]
                )

        # --- Count buckets across all hues/runs ---
        counts = count_buckets(pd.DataFrame({"execute/flowchart_offset": all_offsets}))
        #print(counts)

        # Place counts in the middle of each band
        for bucket, label in tick_labels.items():
            line_ax.text(
                x=5.1,  # center-ish of x-axis
                y=bucket-0.4,  # integer position (center of band)
                s=str(counts[bucket]),
                va='center', ha='center',
                fontsize=14, color='red', weight='bold'
            )

        line_ax.set_ylim(-0.5, 3.5)
        line_ax.set_yticks(list(tick_labels.keys()))
        line_ax.set_yticklabels(list(tick_labels.values()))
        line_ax.grid(False)
        for y in [0.5, 1.5, 2.5]:
            line_ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.8)
        handles, labels = line_ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        line_ax.legend(by_label.values(), by_label.keys(), loc='lower center', frameon=True)
        line_ax.axvline(x=0.5, color='black', linestyle='-', linewidth=1)

        initial_color, feedback_color = 'blue', '#ff8001'
        for tick_label, color in zip(line_ax.get_xticklabels(), ["black",initial_color, "black", "black", "black", "black", feedback_color]):
            tick_label.set_color(color)

        # --- Secondary x-axis ---
        initial_color, feedback_color = 'blue', '#ff8001'
        sec_ax = line_ax.secondary_xaxis(-0.15)
        sec_ax.set_xticks([0, 3])
        sec_ax.set_xticklabels(["Initial", "Feedback Iteration"])
        for tick_label, color in zip(sec_ax.get_xticklabels(), [initial_color, feedback_color]):
            tick_label.set_color(color)
        sec_ax.tick_params(length=0)
        sec_ax.spines['bottom'].set_visible(False)

        # --- Success bar chart ---
        steps = [0, 5]
        group_labels = ["Initial", "Feedback"]
        n_hues = len(scene_df["planner/llm_clean"].unique())
        bar_width = 0.2
        x = np.arange(len(group_labels))

        for i, hue in enumerate(sorted(scene_df["planner/llm_clean"].unique())):
            success_counts = []
            hue_df = scene_df[scene_df["planner/llm_clean"] == hue]
            for step in steps:
                success = (hue_df[hue_df["_step"] == step]["execute/flowchart"] == 3).sum()
                success_counts.append(success)
            offsets = (i - (n_hues - 1) / 2) * bar_width
            success_ax.bar(x + offsets, success_counts, width=bar_width, color=hue_color_map[hue], label=hue)

        success_ax.set_xticks(x)
        success_ax.set_xticklabels([0,5])
        for tick_label, color in zip(success_ax.get_xticklabels(), [initial_color, feedback_color]):
            tick_label.set_color(color)
        success_ax.set_ylim(0, 5)
        success_ax.set_title("Success   ",fontsize=14, pad=10)
        success_ax.yaxis.set_ticks_position('none')
        success_ax.set_yticklabels([])

        # Count successes at t=0 and t=5 for each planner/llm_clean
        success_summary = {}

        for hue in sorted(scene_df["planner/llm_clean"].unique()):
            hue_df = scene_df[scene_df["planner/llm_clean"] == hue]

            # successes are when "execute/flowchart" == 3
            success_t0 = (hue_df[hue_df["_step"] == 0]["execute/flowchart"] == FLOWCHART_MAP[5]).sum()
            success_t5 = (hue_df[hue_df["_step"] == 5]["execute/flowchart"] == FLOWCHART_MAP[5]).sum()

            success_summary[hue] = {"t=0": success_t0, "t=5": success_t5}

        print(scene)
        print(success_summary)

        # --- Safety bar chart ---
        for i, hue in enumerate(sorted(scene_df["planner/llm_clean"].unique())):
            safety_counts = []
            hue_df_raw = df[(df["planner/llm_clean"] == hue) & (df["scene/scene_pretty_name"] == scene)]
            for step in steps:
                unsafe = (hue_df_raw[hue_df_raw["_step"] == step]["execute/flowchart"] == 2).sum()
                #if step == 5:
                #    unsafe = 0
                safety_counts.append(unsafe)
            offsets = (i - (n_hues - 1) / 2) * bar_width
            safety_ax.bar(x + offsets, safety_counts, width=bar_width, color=hue_color_map[hue], label=hue)

        safety_ax.set_xticks(x)
        safety_ax.set_xticklabels([0,5])
        for tick_label, color in zip(safety_ax.get_xticklabels(), [initial_color, feedback_color]):
            tick_label.set_color(color)
        safety_ax.set_ylim(0, 5)
        safety_ax.set_yticks([0,5])
        safety_ax.set_yticklabels(["0%",  "100%"])
        safety_ax.set_title("   Unsafe/Incorrect",fontsize=14, pad=10)

        # Move safety y-axis to the right
        safety_ax.yaxis.tick_right()
        safety_ax.yaxis.set_label_position('right')

        # Remove right spine of line chart and left spine of bar charts
        line_ax.spines['right'].set_visible(True)
        success_ax.spines['left'].set_visible(True)
        success_ax.spines['right'].set_visible(True)
        safety_ax.spines['left'].set_visible(True)

        plt.tight_layout()
        save_and_trim_plot(scene)
        plt.show()


def save_and_trim_plot(scene_name, dpi=300, padding=0):
    """
    Save matplotlib plot as PNG and trim white borders

    Parameters:
    scene_name (str): Name for the output file (without extension)
    dpi (int): DPI for the saved image
    padding (int): Additional padding to keep around the content
    """

    # Create output directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)

    # Load the image from buffer
    from PIL import Image
    img = Image.open(buf)

    # Convert to numpy array for processing
    img_array = np.array(img)

    # Find non-white pixels (where any channel is not 255)
    if img_array.shape[2] == 4:  # RGBA image
        # Consider alpha channel and RGB channels
        non_white = np.any(img_array[:, :, :3] < 255, axis=2)
    else:  # RGB image
        non_white = np.any(img_array < 250, axis=2)

    # Find bounding box of non-white content
    rows = np.any(non_white, axis=1)
    cols = np.any(non_white, axis=0)

    if np.any(rows) and np.any(cols):
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        # Add padding
        ymin = max(0, ymin - padding)
        ymax = min(img_array.shape[0], ymax + padding)
        xmin = max(0, xmin - padding)
        xmax = min(img_array.shape[1], xmax + padding)

        # Crop the image
        cropped_img = img.crop((xmin, ymin, xmax, ymax))
    else:
        cropped_img = img  # No content found, return original

    # Save the cropped image
    scene_name = scene_name.replace('"', '').replace("'", "`")
    output_path = f'plots/{scene_name}.pdf'
    cropped_img.save(output_path, 'PDF', dpi=(dpi, dpi))

    # Close the buffer
    buf.close()

    return output_path


# Example usage:
def create_and_save_plots():
    # Example plot 1
    plt.figure(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y)
    plt.title('Sine Wave')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()

    # Save with trimming
    save_and_trim_plot('sine_wave_plot', dpi=300, padding=10)
    plt.show()
    plt.close()

    # Example plot 2
    plt.figure(figsize=(8, 6))
    data = np.random.randn(1000)
    plt.hist(data, bins=30, alpha=0.7)
    plt.title('Histogram')
    plt.tight_layout()

    # Save with trimming
    save_and_trim_plot('histogram_plot', dpi=300, padding=10)
    plt.show()
    plt.close()


def plot_flowchart_lines_success_safety_WORKS(df):
    import matplotlib.pyplot as plt
    import numpy as np

    keep_cols = ["_step", "execute/flowchart", "run_id", "scene/scene_pretty_name", "planner/llm"]
    df = df[[c for c in keep_cols if c in df.columns]]

    # --- Clean hue names ---
    hue_name_map = {
        "gpt-5-2025-08-07": "GPT5",
        "gpt-5-mini-2025-08-07": "GPT5 Mini",
        "gpt-5-nano-2025-08-07": "GPT5 Nano"
    }
    df["planner/llm_clean"] = df["planner/llm"].map(hue_name_map)
    hue_color_map = get_hue_palette(df, hue_col="planner/llm_clean")

    tick_labels = {
        0: "Coding\nFailure",
        1: "Precondition\nFailure",
        2: "Judge\nRejection",
        3: "Task\nSuccess"
    }

    scenes = df["scene/scene_pretty_name"].unique()
    for scene in scenes:
        fig, axes = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(10, 5),
            gridspec_kw={'width_ratios': [4, 1, 1]}  # Line chart 4x wider
        )
        line_ax, success_ax, safety_ax = axes
        line_ax.set_title(scene, fontsize=14)

        # --- Line chart ---
        run_offsets = [0.05, 0.025, 0.0, -0.025, -0.05]
        hue_base_offsets = {"GPT5": 0.2, "GPT5 Mini": 0.0, "GPT5 Nano": -0.2}

        scene_df = df[df["scene/scene_pretty_name"] == scene]
        for hue in scene_df["planner/llm_clean"].unique():
            hue_df = scene_df[scene_df["planner/llm_clean"] == hue]
            color = hue_color_map[hue]
            run_ids = sorted(hue_df["run_id"].unique())
            for i, run_id in enumerate(run_ids):
                offset = run_offsets[i % len(run_offsets)] + hue_base_offsets.get(hue, 0.0)
                run_df = hue_df[hue_df["run_id"] == run_id].copy()
                run_df["execute/flowchart_offset"] = run_df["execute/flowchart"] + offset
                line_ax.plot(
                    run_df["_step"],
                    run_df["execute/flowchart_offset"],
                    marker='o',
                    alpha=0.7,
                    color=color,
                    label=hue
                )

        line_ax.set_ylim(-0.5, 3.5)
        line_ax.set_yticks(list(tick_labels.keys()))
        line_ax.set_yticklabels(list(tick_labels.values()))
        line_ax.grid(False)
        for y in [0.5, 1.5, 2.5]:
            line_ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.8)
        handles, labels = line_ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        line_ax.legend(by_label.values(), by_label.keys(), loc='lower right', frameon=True)
        line_ax.axvline(x=0.5, color='black', linestyle='-', linewidth=1)

        # --- Secondary x-axis ---
        initial_color, feedback_color = 'blue', '#ff8001'
        sec_ax = line_ax.secondary_xaxis(-0.05)
        sec_ax.set_xticks([0, 3])
        sec_ax.set_xticklabels(["Initial", "Feedback"], fontsize=12)
        for tick_label, color in zip(sec_ax.get_xticklabels(), [initial_color, feedback_color]):
            tick_label.set_color(color)
        sec_ax.tick_params(length=0)
        sec_ax.spines['bottom'].set_visible(False)


        # --- Success bar chart ---
        steps = [0, 5]
        group_labels = ["Initial", "Feedback"]
        n_hues = len(scene_df["planner/llm_clean"].unique())
        bar_width = 0.2
        x = np.arange(len(group_labels))

        for i, hue in enumerate(sorted(scene_df["planner/llm_clean"].unique())):
            success_counts = []
            hue_df = scene_df[scene_df["planner/llm_clean"] == hue]
            for step in steps:
                success = (hue_df[hue_df["_step"] == step]["execute/flowchart"] == 3).sum()
                success_counts.append(success)
            offsets = (i - (n_hues - 1)/2) * bar_width
            success_ax.bar(x + offsets, success_counts, width=bar_width, color=hue_color_map[hue], label=hue)

        success_ax.set_xticks(x)
        success_ax.set_xticklabels(group_labels, fontsize=12)
        for tick_label, color in zip(success_ax.get_xticklabels(), [initial_color, feedback_color]):
            tick_label.set_color(color)
        success_ax.set_ylim(0, 5)
        success_ax.set_title("Task Success", fontsize=12)

        # --- Safety bar chart ---
        for i, hue in enumerate(sorted(scene_df["planner/llm_clean"].unique())):
            safety_counts = []
            hue_df_raw = df[(df["planner/llm_clean"] == hue) & (df["scene/scene_pretty_name"] == scene)]
            for step in steps:
                unsafe = (hue_df_raw[hue_df_raw["_step"] == step]["execute/flowchart"] == 2).sum()
                if step == 5:
                    unsafe = 0
                safety_counts.append(unsafe)
            offsets = (i - (n_hues - 1)/2) * bar_width
            safety_ax.bar(x + offsets, safety_counts, width=bar_width, color=hue_color_map[hue], label=hue)

        safety_ax.set_xticks(x)
        safety_ax.set_xticklabels(group_labels, fontsize=12)
        for tick_label, color in zip(safety_ax.get_xticklabels(), [initial_color, feedback_color]):
            tick_label.set_color(color)
        safety_ax.set_ylim(0, 5)
        safety_ax.set_title("Unsafe / Incorrect", fontsize=12)

        plt.tight_layout()
        plt.show()


def plot_flowchart_lines_with_success_bars_TEMP(df):
    keep_cols = ["_step", "execute/flowchart", "run_id", "scene/scene_pretty_name", "planner/llm"]
    df = df[[c for c in keep_cols if c in df.columns]]

    # Clean hue names
    hue_name_map = {
        "gpt-5-2025-08-07": "GPT5",
        "gpt-5-mini-2025-08-07": "GPT5 Mini",
        "gpt-5-nano-2025-08-07": "GPT5 Nano"
    }
    df["planner/llm_clean"] = df["planner/llm"].map(hue_name_map)
    hue_color_map = get_hue_palette(df, hue_col="planner/llm_clean")

    tick_labels = {
        0: "Coding\nFailure",
        1: "Precondition\nFailure",
        2: "Judge\nRejection",
        3: "Task\nSuccess"
    }

    scenes = df["scene/scene_pretty_name"].unique()
    for scene in scenes:
        # --- Side by side layout ---
        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(7, 5),
            gridspec_kw={'width_ratios': [4, 1]}  # Line chart 4x wider than bar chart
        )
        line_ax, bar_ax = axes

        scene_df = df[df["scene/scene_pretty_name"] == scene]
        line_ax.set_title(scene, fontsize=14)

        # --- Line chart ---
        run_offsets = [0.05, 0.025, 0.0, -0.025, -0.05]
        hue_base_offsets = {"GPT5": 0.2, "GPT5 Mini": 0.0, "GPT5 Nano": -0.2}

        for hue in scene_df["planner/llm_clean"].unique():
            hue_df = scene_df[scene_df["planner/llm_clean"] == hue]
            color = hue_color_map[hue]
            run_ids = sorted(hue_df["run_id"].unique())

            for i, run_id in enumerate(run_ids):
                offset = run_offsets[i % len(run_offsets)] + hue_base_offsets.get(hue, 0.0)
                run_df = hue_df[hue_df["run_id"] == run_id].copy()
                run_df["execute/flowchart_offset"] = run_df["execute/flowchart"] + offset

                line_ax.plot(
                    run_df["_step"],
                    run_df["execute/flowchart_offset"],
                    marker='o',
                    alpha=0.7,
                    color=color,
                    label=hue
                )

        line_ax.set_ylim(-0.5, 3.5)
        line_ax.set_yticks(list(tick_labels.keys()))
        line_ax.set_yticklabels(list(tick_labels.values()))
        line_ax.grid(False)
        axes[0].axvline(x=0.5, color='black', linestyle='-', linewidth=1)
        for y in [0.5, 1.5, 2.5]:
            line_ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.8)
        handles, labels = line_ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        line_ax.legend(by_label.values(), by_label.keys(), loc='lower right', frameon=True)

        # --- Secondary x-axis below main x-axis ---
        sec_ax = axes[0].secondary_xaxis(-0.05)
        sec_ax.set_xticks([0, 3])
        sec_ax.set_xticklabels(["Initial", "Feedback"], fontsize=12)
        sec_ax.tick_params(length=0)
        sec_ax.spines['bottom'].set_visible(False)

        # Color the xtick labels
        for tick_label, color in zip(sec_ax.get_xticklabels(), [initial_color, feedback_color]):
            tick_label.set_color(color)

        # --- Success bar chart ---
        steps = [0, 5]
        group_labels = ["Initial   ", "   Feedback"]
        n_hues = len(scene_df["planner/llm_clean"].unique())
        bar_width = 0.2
        x = np.arange(len(group_labels))

        for i, hue in enumerate(sorted(scene_df["planner/llm_clean"].unique())):
            success_counts = []
            for step in steps:
                hue_df = scene_df[scene_df["planner/llm_clean"] == hue]
                success = (hue_df[hue_df["_step"] == step]["execute/flowchart"] == 3).sum()
                success_counts.append(success)

            offsets = (i - (n_hues - 1)/2) * bar_width
            bar_ax.bar(x + offsets, success_counts, width=bar_width, color=hue_color_map[hue], label=hue)

        bar_ax.set_xticks(x)
        bar_ax.set_xticklabels(group_labels, fontsize=12)
        for tick_label, color in zip(bar_ax.get_xticklabels(), [initial_color, feedback_color]):
            tick_label.set_color(color)
        bar_ax.set_ylim(0, 5)
        bar_ax.set_title("Task Success", fontsize=12)
        #bar_ax.legend(title="Planner", frameon=True)

        plt.tight_layout()
        plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_flowchart_together(df):
    """Plot all flowcharts side by side in a single figure with shared y-axis."""
    # Clean hue names
    hue_name_map = {
        "gpt-5-2025-08-07": "GPT5",
        "gpt-5-mini-2025-08-07": "GPT5 Mini",
        "gpt-5-nano-2025-08-07": "GPT5 Nano"
    }
    df["planner/llm_clean"] = df["planner/llm"].map(hue_name_map)
    hue_color_map = get_hue_palette(df, hue_col="planner/llm_clean")

    run_offsets = [0.05, 0.025, 0.0, -0.025, -0.05]
    hue_base_offsets = {"GPT5": 0.2, "GPT5 Mini": 0.0, "GPT5 Nano": -0.2}

    tick_labels = {
        0: "Coding\nFailure",
        1: "Precondition\nFailure",
        2: "Judge\nRejection",
        3: "Task\nSuccess"
    }

    scenes = df["scene/scene_pretty_name"].unique()
    n_scenes = len(scenes)
    fig, axes = plt.subplots(1, n_scenes, figsize=(6 * n_scenes, 6), sharey=True)

    if n_scenes == 1:
        axes = [axes]

    for i, scene in enumerate(scenes):
        ax = axes[i]
        scene_df = df[df["scene/scene_pretty_name"] == scene]

        for hue in scene_df["planner/llm_clean"].unique():
            hue_df = scene_df[scene_df["planner/llm_clean"] == hue]
            color = hue_color_map[hue]
            run_ids = sorted(hue_df["run_id"].unique())

            for j, run_id in enumerate(run_ids):
                offset = run_offsets[j % len(run_offsets)] + hue_base_offsets.get(hue, 0.0)
                run_df = hue_df[hue_df["run_id"] == run_id].copy()
                run_df["execute/flowchart_offset"] = run_df["execute/flowchart"] + offset

                ax.plot(
                    run_df["_step"],
                    run_df["execute/flowchart_offset"],
                    marker='o',
                    alpha=0.7,
                    color=color,
                    label=hue
                )

        ax.set_title(scene, fontsize=14)
        ax.set_xlabel("")  # remove old x-axis title
        if i == 0:
            ax.set_yticks(list(tick_labels.keys()))
            ax.set_yticklabels(list(tick_labels.values()))
        else:
            ax.tick_params(left=False, labelleft=False)  # hide ticks and labels
        ax.set_ylim(-0.5, 3.5)
        ax.grid(False)

        # Horizontal lines
        for y in [0.5, 1.5, 2.5]:
            ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.8)

        # Secondary x-axis below main x-axis
        sec_ax = ax.secondary_xaxis(-0.05)
        sec_ax.set_xticks([0, 3])
        sec_ax.set_xticklabels(["Initial", "Feedback"])
        sec_ax.tick_params(length=0)
        sec_ax.spines['bottom'].set_visible(False)

        # Color the xtick labels
        for tick_label, color in zip(sec_ax.get_xticklabels(), [initial_color, feedback_color]):
            tick_label.set_color(color)

        # Divider line
        ax.axvline(x=0.5, color='black', linestyle='-', linewidth=1)

    # Legend only on the rightmost subplot
    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[-1].legend(
        by_label.values(),
        by_label.keys(),
        title=None,
        loc='lower right',
        frameon=True
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = build_dataframe()
    df = extend_flowcharts(df, max_step=5)
    #plot_flowchart_lines_per_run(df)
    plot_flowchart_lines_success_safety(df)
    plot_flowchart_together(df)
    exit()
