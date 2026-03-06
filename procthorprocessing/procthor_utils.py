import hashlib
import json

from tqdm import tqdm, trange
import prior
from prior.utils.types import LazyJsonDataset

from llmqueries.llm import set_api_key
from procthorprocessing.taskgen import gen_tasks_for_scene


def get_procthor10k():
    d = prior.load_dataset("procthor-10k")#, revision="ab3cacd0fc17754d4c080a3fd50b18395fae8647")
    d = LazyJsonDataset(d.train.data+d.test.data+d.val.data, "traintestval", "train")

    #def get_sha_for_index(i):
    #    return bytes_to_unique_str(d.data[i])

    #d.get_sha_for_index = get_sha_for_index
    return d

def get_uuid(id_with_vbar):
    return id_with_vbar.split("|")[0].lower().strip()
    import hashlib
    import uuid

    m = hashlib.md5()
    m.update(id_with_vbar.encode('utf-8'))
    new_uuid = uuid.UUID(m.hexdigest(), version=4)
    new_uuid = new_uuid.hex.replace("-", "")
    return id_with_vbar.split("|")[0].lower() + f"-{new_uuid[:4]}-{new_uuid[4:8]}"

def get_objects_list_single_p10k(scene):
    objs = []
    for obj in scene["objects"]:
        o = {
            "item": get_uuid(obj["id"]),
        }

        if "children" in obj:
            children = obj["children"]
            children = [get_uuid(x["id"]) for x in children]

            o["children items"] = children


        objs.append(o)
    return objs

#def bytes_to_unique_str(data: bytes) -> str:
#    # Compute SHA-256 hash of the bytes
#    hash_obj = hashlib.sha256(data)
#    return hash_obj.hexdigest()  # Returns a hex string

objlist_path = "objlist.json"

def get_objects_list_p10k(d, num_todo=100):
    objlist = {}
    for i in trange(len(d)):
        objlist[i] = {"scene_objects" :get_objects_list_single_p10k(d[i])}

    with open(objlist_path, "w") as f:
        f.write(json.dumps(objlist, indent=2))

def gen_tasks(llm_id, start_index=-1, num_todo=10, num_tasks=10):
    with open(objlist_path, "r") as f:
        objlist = json.load(f)

    def get_scene(scene_id):
        return objlist[f"{scene_id}"]

    num_scenes = len(objlist)

    DONE_UP_UNTIL = 0
    for scene_id in trange(num_scenes, desc="Finding start point..."):
        if "tasks" not in get_scene(scene_id) or get_scene(scene_id)["tasks"] is None:
            DONE_UP_UNTIL = scene_id
            break

    if start_index == -1:
        start_index = DONE_UP_UNTIL

    if num_todo == -1:
        num_todo = num_scenes - start_index

    todos = []
    for scene_id in trange(start_index, num_scenes, desc="Collecting TODO scenes..."):
        if "tasks" not in get_scene(scene_id) or get_scene(scene_id)["tasks"] is None:
            todos.append(scene_id)
        if len(todos) >= num_todo:
            break

    for scene_id in tqdm(todos, desc="Using LLM to generate tasks..."):
        scene = get_scene(scene_id)

        tasks = gen_tasks_for_scene(scene, llm_id)
        scene["tasks"] = tasks
        scene["llm_id"] = llm_id

    temp_path = objlist_path.replace(".json", "_tasked.json")
    with open(temp_path, "w") as f:
        f.write(json.dumps(objlist, indent=2))


if __name__ == "__main__":
    set_api_key("./api_key")

    d = get_procthor10k()
    get_objects_list_p10k(d)

    gen_tasks("gpt-3.5-turbo")
    exit()


    exit()

    objlist = []
    for i in tqdm(d):
        get_objects_list_single_p10k(i)
    exit()
    objlists = list()
    print(get_objects_list_single_p10k(d[0]))

    i=0