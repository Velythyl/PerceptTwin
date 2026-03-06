import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ==== Config ====
directory = '/Users/charlie/Projects/Holodeck/llmannotationexperiment/exp4'
use_std = True  # Toggle between using needle_std or needle_stderr
FONT_SIZE = 16   # Controls all font sizes
# ================

# Seaborn style
sns.set(style="white", palette="colorblind")
plt.rcParams.update({'font.size': FONT_SIZE})

# Match filenames like: gpt-4.1-mini-2025-04-14.json
filename_regex = re.compile(r'gpt-(\d+(?:\.\d+)?)(?:-([a-z]+))?')

entries = []

# Load files and extract data
for filename in os.listdir(directory):
    if not filename.endswith('.json'):
        continue

    match = filename_regex.search(filename)
    if not match:
        continue

    size, tag = match.groups()
    tag = tag if tag else None
    tag = tag.replace("Turbo", "").replace("turbo", "") if tag else None

    with open(os.path.join(directory, filename), 'r') as f:
        data = json.load(f)

    acc = data['needle_acc']
    err = data['needle_std'] if use_std else data['needle_stderr']

    lower = max(0, acc - err)
    upper = min(1, acc + err)

    entries.append({
        'size': size,
        'tag': tag,
        'name': f"gpt-{size}" + (f"-{tag}" if tag and tag.lower() != "turbo" else ""),
        'pretty_name': f"{size}" + (f" {tag.capitalize()}" if tag else ""),
        'acc': acc,
        'lower': acc - lower,
        'upper': upper - acc
    })

# Custom sort: 3.5, 4.1-nano, 4.1-mini, 4, 4.1
def custom_sort_key(e):
    size = e['size']
    tag = e['tag']
    if size == '3.5':
        return (0,)
    elif size == '4.1' and tag == 'nano':
        return (1,)
    elif size == '4.1' and tag == 'mini':
        return (2,)
    elif size == '4' and tag is None:
        return (3,)
    elif size == '4.1' and tag is None:
        return (4,)
    else:
        return (99,)  # fallback

entries.sort(key=custom_sort_key)

# Build DataFrame for plotting
df = pd.DataFrame(entries)
df['label'] = df['pretty_name']
df['yerr'] = list(zip(df['lower'], df['upper']))

# Plot
plt.figure(figsize=(7, 6))
ax = sns.barplot(data=df, x='label', y='acc', palette='colorblind')

# Add asymmetric error bars manually
for i, (acc, (low, up)) in enumerate(zip(df['acc'], df['yerr'])):
    ax.errorbar(i, acc, yerr=[[low], [up]], fmt='none', c='black', capsize=5)

plt.ylim(0, 1)
plt.title("Accuracy (±{})".format("Std Dev" if use_std else "Std Err"), fontsize=FONT_SIZE)
plt.xlabel("LLM (GPT Model)", fontsize=FONT_SIZE)
plt.ylabel("Accuracy", fontsize=FONT_SIZE)
plt.xticks(fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.tight_layout()

# Save plot
output_path = os.path.join(directory, "acc_gpt.pdf")
plt.savefig(output_path)
plt.show()

import pandas as pd

# Build dict: {ModelName: "mean ± std"}
row = {}
for e in entries:
    row[e["pretty_name"]] = f"{e['acc']:.3f} ± {e['upper']:.3f}"

# Turn into 1-row dataframe
df_table = pd.DataFrame([row])

print(df_table.to_latex(index=False))

import pandas as pd

# Only keep mainline models
wanted = {"3.5", "4", "4.1", "5"}

row = {}
for e in entries:
    if e["size"] in wanted and e["tag"] is None or e["tag"] == "":  # exclude mini/nano
        row[e["size"]] = f"{e['acc']:.3f} ± {e['upper']:.3f}"

# Sort columns in order: 3.5, 4, 4.1, 5
order = ["3.5", "4", "4.1", "5"]
row_sorted = {k: row[k] for k in order if k in row}

# Build 1-row dataframe
df_table = pd.DataFrame([row_sorted])

print(df_table.to_latex(index=False))


