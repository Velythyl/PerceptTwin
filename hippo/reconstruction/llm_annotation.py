import ast
import json


from hippo.reconstruction.scenedata import HippoObject
from hippo.simulation.skillsandconditions.skill_names import get_enabled_2_llm_name
from llmqueries.llm import LLM


def get_prompt(ho: HippoObject):

    PROMPT = f"""
You are helping out with robotic experiments.
The robot is a single-arm robot with a gripper (similar to a LoCobot).
It is slightly shorter than a human.
It cannot interact with objects that are taller than itself.
It can open doors and drawers, slice items, etc.
All skills in the list below can be performed by the robot, but not on all objects.
Note: the turn on/off skills are also used for toggling electronics on/off, or activating/deactivating electronics or other such devices.

Please assign labels to the object <{ho.object_name}> of description <{ho.object_description}>. 

Think logically before assigning. Detail your reasoning, including object weight and size. 
Furthermore, do not be creative with your labels, use common sense.
Pick ONLY one label for each category, otherwise the robot will explode.
    
LABELS: {json.dumps(get_enabled_2_llm_name(), indent=2)}

We need to parse your response using python, so make sure to respect the following format. 
```[<label1>, <label2>, ...]```. Write this response inside a ``` block. Do not use ``` blocks for anything else. Do not write a dict, write a simple list.

    """.strip()

    return PROMPT

def parse_response(response):
    if "```" in response:
        splitted = response.split("```")[-2].strip()

        if splitted.count("[") != splitted.count("]"):
            return None


        if splitted.count("[") == 1:
            try:
                return json.loads(splitted)
            except:
                return ast.literal_eval(splitted)


        splitted = splitted.split("\n")
        ret = []
        for s in splitted:
            if "," in s:
                return None
            try:
                s = json.loads(s)
            except:
                s = ast.literal_eval(s)
            if len(s) > 1:
                return None
            ret.append(s[0])
        return ret

    return None


def LLM_query(cfg, ho: HippoObject):
    prompt = get_prompt(ho)

    _, response = LLM(prompt, cfg.skillprediction.method, max_tokens=1000, temperature=0, stop=None, logprobs=1, frequency_penalty=0)

    parsed = parse_response(response)
    if parsed is not None:
        return parsed

    prompt = f"""
---

In a previous chat session, you answered:

{response}

But we could not parse your response correctly. Please try again.

---
""".strip()

    _, response = LLM(prompt, cfg.skillprediction.method, max_tokens=500, temperature=0, stop=None, logprobs=1, frequency_penalty=0)

    parsed = parse_response(response)
    if parsed is not None:
        return parsed
    
    return []

    raise AssertionError(f"Could not parse LLM response: {response}")

def LLM_annotate(cfg, ho: HippoObject) -> HippoObject:
    parsed = LLM_query(cfg, ho)

    from hippo.simulation.skillsandconditions.skill_names import assert_names_are_valid
    if len(assert_names_are_valid(parsed)) > 0:
        raise AssertionError("LLM annotation failed because of duplicate skill names, maybe need to code feedback loop?")

    return ho.set_skill_metadata(tuple(parsed))
