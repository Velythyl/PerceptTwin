import argparse
import dataclasses
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import openai

from cleanplanner.parse_scene import SceneTask, PlanLog, get_objects_list
from hippo.ai2thor_hippo_controller import get_list_of_objects
from hippo.utils.file_utils import get_tmp_folder
from llmqueries.llm import LLM, approx_num_tokens

sys.path.append(".")

import resources.actions as actions
import resources.robots as robots


def set_api_key(openai_api_key):
    openai.api_key = Path(openai_api_key + '.txt').read_text()


# Function returns object list with name and properties.
def convert_to_dict_objprop(objs, obj_mass):
    objs_dict = []
    for i, obj in enumerate(objs):
        obj_dict = {'name': obj, 'mass': obj_mass[i]}
        # obj_dict = {'name': obj , 'mass' : 1.0}
        objs_dict.append(obj_dict)
    return objs_dict


TARGET_TMP_DIR = get_tmp_folder()





def parse_ai2thor_plan(scene, task_path):
    # read the tasks
    test_tasks = []
    robots_test_tasks = []
    gt_test_tasks = []
    trans_cnt_tasks = []
    max_trans_cnt_tasks = []
    with open(task_path, "r") as f:
        for line in f.readlines():
            if line.startswith("//"):
                continue
            test_tasks.append(list(json.loads(line).values())[0])
            robots_test_tasks.append(list(json.loads(line).values())[1])
            gt_test_tasks.append(list(json.loads(line).values())[2])
            trans_cnt_tasks.append(list(json.loads(line).values())[3])
            max_trans_cnt_tasks.append(list(json.loads(line).values())[4])

    available_robots = []
    for robots_list in robots_test_tasks:
        task_robots = []
        for i, r_id in enumerate(robots_list):
            rob = robots.robots[r_id - 1]
            # rename the robot
            rob['name'] = 'robot' + str(i + 1)
            task_robots.append(rob)
        available_robots.append(task_robots)

    return SceneTask(test_tasks=test_tasks, robots_test_tasks=robots_test_tasks, gt_test_tasks=gt_test_tasks,
                     trans_cnt_tasks=trans_cnt_tasks, max_trans_cnt_tasks=max_trans_cnt_tasks, scene=scene,
                     available_robots=available_robots)

SKILL_ALIASES = {
    "TurnOnObject": "SwitchOn",
    "TurnOffObject": "SwitchOff",
    "ActivateObject": "SwitchOn",
    "DeactivateObject": "SwitchOff",
}

AI2THOR_ACTIONS = actions.ai2thor_actions_list
for ALIAS, TARGET in SKILL_ALIASES.items():

    FOUND = None
    for act in AI2THOR_ACTIONS:
        if act.startswith(TARGET):
            FOUND = act
    assert FOUND is not None

    AI2THOR_ACTIONS.append(f"{ALIAS} {' '.join(FOUND.split(' ')[1:])}")
AI2THOR_ACTIONS = ", ".join(AI2THOR_ACTIONS)

PROMPT = f"""

You are PLANR, an excellent LLM planner for open-vocabulary scenes. The PLANR system is able to plan using these skills:
{AI2THOR_ACTIONS}

The SwitchOn/SwitchOff skills are used to toggle things on/off, not just "switch" objects.

INPUT: a list of objects found in the scene, as well as a task description. 
OUTPUT: PLANR first REASONS about the task. It then OUTPUTs a sequence of skill to complete the task.

Here is an example:

```
# INPUT:
# Put Tomato in Fridge
objects = ['Apple-9213#jk', 'Bowl|aekgj|o', 'CounterTop|2|0', 'Tomato-#211a-be',
           'Fridge|2|1': {{"hasInsideOf": ["Apple-3981#0"]}}', 'GarbageBin-2199-ae98']

# OUTPUT: 
# REASONING: We must pickup the tomato and place it inside the fridge. We'll need GoToObject, PickupObject, OpenObject, CloseObject, and PutObject skills.

# CODE
def put_tomato_in_fridge():
    # 0: Task : Put Tomato in Fridge
    # 1: Go to the Tomato.
    GoToObject('Tomato-#211a-be')
    # 2: Pick up the Tomato.
    PickupObject('Tomato-#211a-be')
    # 3: Go to the Fridge.
    GoToObject('Fridge|2|1')
    # 4: Open the Fridge.
    OpenObject('Fridge|2|1')
    # 5: Put the Tomato in the Fridge.
    PutObject('Tomato-#211a-be', 'Fridge|2|1')
    # 6: Close the Fridge.
    CloseObject('Fridge|2|1')
slice_potato()
```

PLANR is very smart, and when a task is not feasible, it aborts the plan.

```
# INPUT:
# Slice the Lettuce
objects = ['Bowl|aekgj|o', 'CounterTop|2|0', 'Tomato-#211a-be',
           'Fridge|2|1', 'Lettuce-2199-ae98', "Dolphin-2919|I@1", "Jeremy|219|9"]

# OUTPUT:
# REASONING:
# We must slice the lettuce. But! There's no knife! We cannot accomplish this plan

AbortPlan("We must use SliceObject, but there is no knife in the scene.")
```

PLANR can also receive FEEDBACK. This FEEDBACK will contain a previous plan from PLANR as well as the reason why the plan failed.
The FEEDBACK will be included as part of the INPUT. When there is FEEDBACK, PLANR must talk about it in the REASONING section.

And don't forget, all reasoning must be inside python comments!

Now, apply PLANR to this input:

"""

def gen_plan(cfg, scenetask: SceneTask, output_dir, feedback=""):
    assert os.path.exists(output_dir)

    ######## Train Task Decomposition ########

    # prepare train decompostion demonstration for ai2thor samples
    #prompt = f"from skills import " + actions.ai2thor_actions
    #prompt += f"\nimport time"
    #prompt += f"\nimport threading"

    # read input train prompts
    #decompose_prompt = Path(cfg.paths.curdir + "/datasmartllm/pythonic_plans/" + cfg.planner.prompt + ".py").read_text()

    #prompt += "\n\n" + decompose_prompt

    print("Generating Decomposed Plans...")

    prompt = f"{PROMPT}\n\n```# INPUT: \n# TASK: {scenetask.tasks[0]}\nobjects = {get_list_of_objects(scenetask.scene)}\n```"
    if feedback is not None and len(feedback) > 0:
        prompt = f"{prompt}\n# START FEEDBACK\n{feedback}\n# END FEEDBACK"
    prompt = f"{prompt}\n# OUTPUT:"

    #curr_prompt = f"{prompt}\n\n# TASK: {scenetask.tasks[0]}"
    #curr_prompt = f"{curr_prompt}\n# OBJECTS IN SCENE:\nobjects = {get_list_of_objects(scenetask.scene)}"
    #curr_prompt = f"{curr_prompt}\n# generate here..."

    NUM_INPUT_TOKENS = approx_num_tokens(cfg.planner.llm, prompt)
    _, decomposed_plan = LLM(prompt, cfg.planner.llm, max_tokens=1300, frequency_penalty=0.0, ignore_cache=cfg.planner.ignore_cache)
    decomposed_plan = decomposed_plan.replace('("robot0",', "(").replace('("robot1",', "(").replace("('robot0',", "(").replace("('robot1',", "(") # common llm mistake
    NUM_OUTPUT_TOKENS = approx_num_tokens(cfg.planner.llm, decomposed_plan)
    for alias, truename in SKILL_ALIASES.items():
        decomposed_plan = decomposed_plan.replace(f"{alias}(", f"{truename}(")
    #decomposed_plan = decomposed_plan.replace("ActivateObject(", "SwitchOn(").replace("TurnOnObject(", "SwitchOn(")
    decomposed_plan = decomposed_plan.split("def ")[1:]
    if len(decomposed_plan) == 1:
        decomposed_plan = [""] + decomposed_plan
    decomposed_plan = "def ".join(decomposed_plan)
    decomposed_plan = decomposed_plan.replace("```", "\n")


    print("Plan obtained! Saving...")
    """
    ######## Train Task Allocation - SOLUTION ########
    prompt = f"from skills import " + actions.ai2thor_actions
    prompt += f"\nimport time"
    prompt += f"\nimport threading"

    allocated_prompt = Path(cfg.paths.curdir + "/datasmartllm/pythonic_plans/" + cfg.planner.prompt_allocation_set + "_solution.py").read_text()

    prompt += "\n\n" + allocated_prompt + "\n\n"


    no_robot = len(scenetask.robots)
    curr_prompt = prompt + decomposed_plan
    curr_prompt += f"\n# TASK ALLOCATION"
    curr_prompt += f"\n# Scenario: There are {no_robot} robots available, The task should be performed using the minimum number of robots necessary. Robots should be assigned to subtasks that match its skills and mass capacity. Using your reasoning come up with a solution to satisfy all contraints."
    curr_prompt += f"\n\nrobots = {scenetask.robots}"
    curr_prompt += f"\n{objects_ai}"
    curr_prompt += f"\n\n# IMPORTANT: The AI should ensure that the robots assigned to the tasks have all the necessary skills to perform the tasks. IMPORTANT: Determine whether the subtasks must be performed sequentially or in parallel, or a combination of both and allocate robots based on availablitiy. "
    curr_prompt += f"\n# SOLUTION  \n"

    if "gpt-3.5" in cfg.planner.llm:
        _, allocated_plan = LLM(curr_prompt, cfg.planner.llm, max_tokens=1500, frequency_penalty=0.35)

    else:
        # gpt 4.0
        messages = [{"role": "system",
                     "content": "You are a Robot Task Allocation Expert. Determine whether the subtasks must be performed sequentially or in parallel, or a combination of both based on your reasoning. In the case of Task Allocation based on Robot Skills alone - First check if robot teams are required. Then Ensure that robot skills or robot team skills match the required skills for the subtask when allocating. Make sure that condition is met. In the case of Task Allocation based on Mass alone - First check if robot teams are required. Then Ensure that robot mass capacity or robot team combined mass capacity is greater than or equal to the mass for the object when allocating. Make sure that condition is met. In both the Task Task Allocation based on Mass alone and Task Allocation based on Skill alone, if there are multiple options for allocation, pick the best available option by reasoning to the best of your ability."},
                    #{"role": "system", "content": "You are a Robot Task Allocation Expert"},
                    {"role": "user", "content": curr_prompt}]
        _, allocated_plan = LLM(messages, cfg.planner.llm, max_tokens=400, frequency_penalty=0.69)

    print("Generating Allocated Code...")

    ######## Train Task Allocation - CODE Solution ########

    prompt = f"from skills import " + actions.ai2thor_actions
    prompt += f"\nimport time"
    prompt += f"\nimport threading"
    prompt += objects_ai


    code_prompt = Path(cfg.paths.curdir + "/datasmartllm/pythonic_plans/" + cfg.planner.prompt_allocation_set + "_code.py").read_text()

    prompt += "\n\n" + code_prompt + "\n\n"

    curr_prompt = prompt + decomposed_plan
    curr_prompt += f"\n# TASK ALLOCATION"
    curr_prompt += f"\n\nrobots = {scenetask.robots}"
    curr_prompt += allocated_plan
    curr_prompt += f"\n# CODE Solution  \n"

    messages = [{"role": "system", "content": "You are a Robot Task Allocation Expert"},
                {"role": "user", "content": curr_prompt}]
    _, code_plan = LLM(messages, cfg.planner.llm, max_tokens=1400, frequency_penalty=0.4)

    """

    # save generated plan
    date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")


    plan_log = PlanLog(
        scenetask = scenetask,
        llm= cfg.planner.llm,
        num_input_tokens=NUM_INPUT_TOKENS,
        num_output_tokens=NUM_OUTPUT_TOKENS,
        llm_outputs={"code_plan": decomposed_plan},
        code_plan=decomposed_plan,
    )

    with open(f"{output_dir}/plan_log.json", "w") as f:
        f.write(json.dumps(plan_log.asdict(), indent=2, sort_keys=True))

    def zero_shot_plan_verif():
        #
        PROMPT = f"""
You are an LLM being used as part of a robotic planning pipeline. 
Your job is to verify the validity of the plan.
Part of the experiment is to show that smaller LLMs can verify plans of costlier LLMs, so do your best to prove us right!

You need to verify aspects of the plan that require semantic/commonsense thinking that we can't otherwise do with an hardcoded simulator.
For example, the robot shouldn't use knives near humans, it shouldn't pour water on electronics (unless it is a necessary part of completing the high-level task), etc.
Do not worry about held items seeming too far from the robot, this is an implementation detail, and does not indicate a fault state.

Please follow the RESPONSE FORMAT exactly.
EXAMPLE RESPONSE:

**Plan reasoning**: <reasoning about the plan. justification for the actions. etc.>

**Answer reasoning**: It's an <okay> plan because <reason>. 

**Final answer**: ```<
CorrectPlan("<reason>")
# OR
UnsafePlan"<reason>")
# OR
IncorrectPlan("<reason>")
>
```

{"The high-level task is " + scenetask.tasks[0]}

INITIAL STATE:
{get_list_of_objects(scenetask.scene)}

PLAN:
{plan_log.code_plan}
    """
        _, response = LLM(PROMPT, cfg.feedback.judge_llm)

        if "CorrectPlan" in response:
            return True
        if "UnsafePlan" in response or "IncorrectPlan" in response:
            return False
        return True

    return plan_log, zero_shot_plan_verif()

    with open(f"{output_dir}/available_robots.json", "w") as f:
        f.write(json.dumps(scenetask.robots, indent=2))

    with open(f"{output_dir}/scene_name.txt", "w") as f:
        f.write(scenetask.scene_id)

    with open(f"{output_dir}/abstract_task_prompt.txt", "w") as f:
        f.write(scenetask.tasks)

    with open(f"{output_dir}/log.txt", 'w') as f:
        f.write(task)
        f.write(f"\n\nGPT Version: {args.gpt_version}")
        f.write(f"\n\nFloor Plan: {args.floor_plan}")
        f.write(f"\n{objects_ai}")
        f.write(f"\nrobots = {scenetask.available_robots[idx]}")
        f.write(
            f"\nground_truth = {scenetask.gt_test_tasks[idx] if scenetask.gt_test_tasks is not None else None}")
        f.write(
            f"\ntrans = {scenetask.trans_cnt_tasks[idx] if scenetask.trans_cnt_tasks is not None else None}")
        f.write(
            f"\nmax_trans = {scenetask.max_trans_cnt_tasks[idx] if scenetask.max_trans_cnt_tasks is not None else None}")

    #with open(f"./logs/{folder_name}/decomposed_plan.py", 'w') as d:
    #    d.write(decomposed_plan[idx])

    #with open(f"./logs/{folder_name}/allocated_plan.py", 'w') as a:
    #    a.write(allocated_plan[idx])

    #with open(f"./logs/{folder_name}/code_plan.py", 'w') as x:
    #    x.write(code_plan[idx])
