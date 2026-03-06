
from pathlib import Path
from llmqueries.llm import set_api_key

set_api_key(Path("/home/velythyl/Desktop/Holodeck/hippo/secrets/openai_api_key.txt").read_text())

from tqdm import tqdm, trange
import ast
import json
import re
import os
import sys
import numpy as np
import random

#@cache.memoize()
def get_experiments(seed, NUM_EXPERIMENTS=10, NUM_ACTIONS_PER_EXPERIMENT=10, NUM_OBJECTS_PER_ACTION=1):
    def reset_seeds(seed):
        np.random.seed(seed)
        random.seed(seed)
    reset_seeds(seed)


    from datasets import load_dataset
    # Load the HICO-DET dataset
    dataset = load_dataset("zhimeng/hico_det", cache_dir="/tmp/llmannotationexperiment")
    splits = dataset.keys()
    print("Available splits:", list(splits))





    def loadit(actlist):
        return ast.literal_eval(actlist)

    def extract_actions_for_sample(sample):
        positive_captions_str = loadit(sample["positive_captions"])
        #posobj = [obj for obj, _ in positive_captions_str]
        #pos = [act for _, act in positive_captions_str]

        negative_captions_str = loadit(sample["negative_captions"])
        #negobj = [obj for obj, _ in neg]
        #neg = [act for _, act in negative_captions_str]

        amb_captions_str = loadit(sample["ambiguous_captions"])
        #ambobj = [obj for obj, _ in amb_captions_str]
        #amb = [act for _, act in amb_captions_str]
        return positive_captions_str + negative_captions_str + amb_captions_str#, pos + neg + amb


    def extract_actions_for_sample_idx(dataset_split, idx):
        sample = dataset_split[idx]
        return extract_actions_for_sample(sample)

    

    #@cache.memoize()
    def process_dataset(dataset_split, desc, TO_REMOVE, max_workers=4):
        obj2i = {}
        i2obj = {}
        total_samples = len(dataset_split)


        for i in trange(total_samples):
            try:
                objs = extract_actions_for_sample_idx(dataset_split, i)
                for o in objs:
                    if o[0] in ["person", "human", "robot", "LoCoBot", "robot arm", "robot hand"]:
                        continue
                    if o[1] in TO_REMOVE:
                        continue

                    if o[0] not in obj2i:
                        obj2i[o[0]] = set()
                    obj2i[o[0]].add(o[1])

                    if o[1] not in i2obj:
                        i2obj[o[1]] = set()
                    i2obj[o[1]].add(o[0])
            except Exception as e:
                print(f"Error processing sample: {str(e)}")
        return obj2i, i2obj

    # Process datasets
    obj2i, i2obj = process_dataset(dataset["train"], "Processing train set", TO_REMOVE = ["no_interaction", "move", "inspect", "look_at", "drag", "pull", "push"])
    
    for k in obj2i.keys():
        obj2i[k] = list(sorted(list(obj2i[k])))
    for k in i2obj.keys():
        i2obj[k] = list(sorted(list(i2obj[k])))

    ALL_HOI_CLASSES = list(i2obj.keys())

    # now, we prepare the experiments

    def add_experiment(i):
        selected_actions = random.sample(ALL_HOI_CLASSES, NUM_ACTIONS_PER_EXPERIMENT)
        selected_objects = []
        for a in selected_actions:
            obj = random.sample(list(i2obj[a]), NUM_OBJECTS_PER_ACTION)
            selected_objects.extend(obj)
        return {
            "experiment_id": i,
            "selected_actions": selected_actions,
            "selected_objects": selected_objects
        }
    experiments = [add_experiment(i) for i in range(NUM_EXPERIMENTS)]
    return experiments, obj2i, i2obj

EXPERIMENTS, OBJ_TO_ACTION, ACTION_TO_OBJ = get_experiments(3, NUM_EXPERIMENTS=20, NUM_ACTIONS_PER_EXPERIMENT=10, NUM_OBJECTS_PER_ACTION=1)


RESULTS = {}
def init_llm_in_results(llm_name):
    RESULTS[llm_name] = {"interactions": [], "needle_acc": 0, "needle_std": 0, "needle_stderr": 0}

def add_interaction_to_results(llm_name, interaction):
    if llm_name not in RESULTS:
        init_llm_in_results(llm_name)
    RESULTS[llm_name]["interactions"].append(interaction)

def set_interaction(llm_name, experiment_id, object_name, predicted_labels, available_labels, found_needle, gt_labels):
    if llm_name not in RESULTS:
        init_llm_in_results(llm_name)
    add_interaction_to_results(llm_name, {"experiment_id": experiment_id, "object_name": object_name, "predicted_labels": predicted_labels, "available_labels": available_labels, "found_needle": found_needle, "gt_labels": gt_labels})



def human_labeler(object_name, actions):
    """Prompt a human user to label an object with available actions."""
    # Create the actions table with aligned numbers
    num_actions = len(actions)
    col_width = max(len(action) for action in actions) + 2  # Add padding for spacing

    print(f"\nLabel: {object_name}")

    # Print numbers centered over each action
    number_row = " | ".join(str(i + 1).center(col_width) for i in range(num_actions))
    action_row = " | ".join(action.ljust(col_width) for action in actions)

    print(number_row)
    print(action_row)

    HUMAN_SELECTED = None
    while True:
        user_input = input("Your answer: ").strip()

        # Parse the input (accept comma or any whitespace-separated values)
        if ',' in user_input:
            selected = [s.strip() for s in user_input.split(',')]
        else:
            selected = re.split(r'\s+', user_input)

        # Validate the input
        try:
            selected_indices = [int(s) for s in selected]
            if all(1 <= idx <= num_actions for idx in selected_indices):
                HUMAN_SELECTED = [actions[idx - 1] for idx in selected_indices]
                break
            else:
                print(f"Please enter numbers between 1 and {num_actions}")
        except ValueError:
            print("Please enter numbers only, separated by commas or whitespace")
    print(f"Human selected: {HUMAN_SELECTED}")
    return HUMAN_SELECTED



def llm_annotate(gpt_model, sample_name, target_actions):
    from llmannotationexperiment.llm_annotation import LLM_annotate

    if gpt_model == "human":
        return human_labeler(sample_name, target_actions)
    # Call the real LLM-based annotator
    return LLM_annotate(gpt_model, sample_name, target_actions)


for llm in ["gpt-4"]:# "gpt-3.5-turbo",  "gpt-4.1-nano-2025-04-14", "gpt-4.1-mini-2025-04-14", "gpt-4.1-2025-04-14"]:# ,"gpt-4"]:
    NEEDLES_ACC = []
    for experiment in EXPERIMENTS:
        NEEDLES_FOR_THIS_EXPERIMENT = 0
        NEEDLES_FOUND_FOR_THIS_EXPERIMENT = 0
        for object_name in tqdm(experiment["selected_objects"]):
            # Simulate the LLM annotation process
            predicted_labels = llm_annotate(llm, object_name, experiment["selected_actions"])
            found_needle = False
            for pred_label in predicted_labels:
                if pred_label in experiment["selected_actions"] and pred_label in OBJ_TO_ACTION[object_name]:
                    found_needle = True
                    break
            set_interaction(llm_name=llm, experiment_id=experiment["experiment_id"], object_name=object_name, predicted_labels=predicted_labels, available_labels=experiment["selected_actions"], found_needle=found_needle, gt_labels=OBJ_TO_ACTION[object_name])
            NEEDLES_FOUND_FOR_THIS_EXPERIMENT += int(found_needle)
            NEEDLES_FOR_THIS_EXPERIMENT += 1
        NEEDLES_ACC.append(NEEDLES_FOUND_FOR_THIS_EXPERIMENT / NEEDLES_FOR_THIS_EXPERIMENT)
            
    RESULTS[llm]["needle_acc"] = np.mean(NEEDLES_ACC)
    #RESULTS[llm]["num_needles"] = len(NEEDLES_ACC)
    RESULTS[llm]["needle_std"] = np.std(NEEDLES_ACC)
    RESULTS[llm]["needle_stderr"] = np.std(NEEDLES_ACC) / np.sqrt(len(NEEDLES_ACC))
    with open(f"main2_results_for_{llm}.json", "w") as f:
        json.dump(RESULTS[llm], f, indent=4)
    print("\n\n~~~")
    print(f"Results for {llm} saved to main2_results_for_{llm}.json")
    print("~~~\n\n")

with open(f"main2_results_for_ALL.json", "w") as f:
    json.dump(RESULTS, f, indent=4)
print(f"Results for {llm} saved to main2_results_for_ALL.json")



