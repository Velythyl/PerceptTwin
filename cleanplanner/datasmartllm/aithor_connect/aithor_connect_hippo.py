no_robot = len(robots)
print(floor_no)

from SMARTLLM.smartllm.utils.get_controller import get_sim
from llmqueries import set_api_key

set_api_key("../api_key")
simulator = get_sim(floor_no)
simulator.set_task_description(abstract_task_prompt)
