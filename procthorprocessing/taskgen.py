import json


def get_prompt(scene):
    prompt = f"""
You are an LLM being used to generate abstract, high-level tasks to evaluate robotic planners. For example, in the presence of cheese and a knife, you might output "make me a snack". The tasks you output must be completable using a specific list of skills:

```
actions_list = [
  "GoToObject",
  "OpenObject",
  "CloseObject",
  "BreakObject",
  "SliceObject",  # requires a knife to be present in the scene!
  "SwitchOn",
  "SwitchOff",
  "CleanObject",
  "PickupObject",
  "PutObject"]
```

The task be be doable using just the skills outlined above, but it has to be abstracted, and high-level.
Do not use the verbs from the skill list, reformulate in natural language.
Please generate 10 tasks for each scene. Be succinct in your response, only output the tasks according to the RESPONSE FORMAT.

### RESPONSE FORMAT:

```
["<task 1>", "<task 2>", "<task 3>", ...]
```

---

### SCENE OBJECTS:

```
{json.dumps(scene['scene_objects'], indent=2)}
```

### YOUR RESPONSE:
    
    """.strip()
    return prompt


def parse_response(response):
    try:
        # Remove any leading/trailing whitespace or code block markers
        cleaned_response = response.strip()

        if '```' in cleaned_response:
            cleaned_response = cleaned_response.split("```")[1].strip()
        cleaned_response = cleaned_response.replace("'", '"')

        # Parse the JSON list
        tasks = json.loads(cleaned_response)

        # Validate it's a list of strings
        if isinstance(tasks, list) and all(isinstance(task, str) for task in tasks):
            return tasks
        else:
            raise Exception("Response is not a list of strings")
    except Exception:
        return None

def gen_tasks_for_scene(scene, llm_id):
    from llmqueries.llm import LLM

    prompt = get_prompt(scene)
    _, response = LLM(prompt, llm_id, max_tokens=250, temperature=0, stop=None, logprobs=1, frequency_penalty=0)

    return parse_response(response)
