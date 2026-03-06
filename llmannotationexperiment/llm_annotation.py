import ast
import json


from hippo.reconstruction.scenedata import HippoObject
from hippo.simulation.skillsandconditions.skill_names import get_enabled_2_llm_name
from llmqueries.llm import LLM


def get_prompt(object_name, possible_actions):

    PROMPT = f"""
    
You are helping out with robotic experiments. The robot is a LoCoBot, a low-cost mobile manipulator suitable for both navigation and manipulation.

Please assign action labels to the object <{object_name}>. 
These labels will be used to determine if the robot can apply the action to the object.

Think logically before assigning. Detail your reasoning. 
Furthermore, do not be creative with your labels, use common sense, otherwise the robot will explode.
    
LABELS: {possible_actions}

We need to parse your response using python, so make sure to respect the following format. 
```[<label1>, <label2>, ...]```. Write this response inside a ``` block. Do not use ``` blocks for anything else. Do not write a dict, write a simple list.

    """.strip()

    return PROMPT

def parse_response(response):
    try:
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
    except:
        return None

    return None


def LLM_query(LLM_model, object_name, possible_actions):
    prompt = get_prompt(object_name, possible_actions)

    _, response = LLM(prompt, LLM_model, max_tokens=1000, temperature=0, stop=None, logprobs=1, frequency_penalty=0)

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

    _, response = LLM(prompt, LLM_model, max_tokens=500, temperature=0, stop=None, logprobs=1, frequency_penalty=0)

    parsed = parse_response(response)
    if parsed is not None:
        return parsed

    return []

def LLM_annotate(LLM_model, object_name, possible_actions) -> []:
    return LLM_query(LLM_model, object_name, possible_actions)


