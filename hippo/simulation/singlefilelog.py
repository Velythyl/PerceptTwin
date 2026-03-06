import dataclasses
import json
import uuid
from typing import Union, List

import jax.numpy as jnp
import numpy as np


def todict(obj, classkey=None):
    if isinstance(obj, dict):
        data = {}
        for (k, v) in obj.items():
            data[k] = todict(v, classkey)
        return data
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, jnp.ndarray):
        return np.array(obj).tolist()
    elif hasattr(obj, "_ast"):
        return todict(obj._ast())
    elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        return [todict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict([(key, todict(value, classkey))
            for key, value in obj.__dict__.items()
            if not callable(value) and not key.startswith('_')])
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj

@dataclasses.dataclass
class FeedbackMixin:
    @property
    def feedback_type(self):
        return self.__class__.__name__

    @property
    def feedback_necessary(self) -> bool:
        raise NotImplementedError()

    def error_message(self) -> Union[List[str], str]:
        raise NotImplementedError()

    def feedback_dict(self):
        try:
            asdict = self.asdict()
        except Exception as e:
            asdict = {"ERROR": "SERIALIZATION ERROR"}

        error_message = self.error_message()
        if isinstance(error_message, list):
            error_message = "\n".join(error_message)
        assert isinstance(error_message, str)

        #feedback_necessary = error_message is not None
        #if hasattr(self, "feedback_necessary"):
        feedback_necessary = self.feedback_necessary

        return {
            "Type": self.feedback_type,
            "Error message": error_message,
            "Full class": asdict,
            "Feedback necessary": feedback_necessary,
        }

filepath = f"/tmp/{uuid.uuid4().hex}.txt"
print(filepath)
def set_filepath(_filepath: str):
    global filepath
    filepath = _filepath

def log_feedback_to_file(feedback):
    with open(filepath, "a") as f:
        f.write(json.dumps(feedback.feedback_dict()) + "\n")

def log_scenedict_to_file(id, scenedict):
    with open(filepath, "a") as f:
        dico = {"IS_SCENEDICT": True, "scenedict_index": id, "scenedict": scenedict}
        f.write(json.dumps(dico) + "\n")

def iterator(_filepath = None, reverse=False):
    if _filepath is not None:
        do_filepath = _filepath
    else:
        do_filepath = filepath

    with open(do_filepath, "r") as f:
        lines = f.readlines()

    if reverse:
        lines = list(reversed(lines))

    for line in lines:
        yield json.loads(line)

def is_obj_FeedbackMixin(obj):
    return "Type" in obj and "Feedback necessary" in obj

def is_obj_scenedict(obj):
    return "IS_SCENEDICT" in obj and "scenedict_index" in obj

def is_plan_success(according_to_big_llm=False, _filepath = None):
    TRIPPED_ONCE = not according_to_big_llm # if we want according to big, then tripped once is False, so we need to ignore the small llm's answer
    for i, obj in enumerate(iterator(_filepath, reverse=True)):
        if is_obj_FeedbackMixin(obj) and (obj["Type"] in ["CorrectFinalState", "IncorrectFinalState", "UnsafeFinalState"]):
            if TRIPPED_ONCE:
                return obj["Type"] == "CorrectFinalState"
            else:
                TRIPPED_ONCE = True
    return False

def get_necessary_plan_feedback(_filepath = None):
    for obj in iterator(_filepath, reverse=True):
        if is_obj_FeedbackMixin(obj) and obj["Feedback necessary"] == True:
            return obj

def get_last_plan_feedback(_filepath = None):
    for obj in iterator(_filepath, reverse=True):
        if is_obj_FeedbackMixin(obj):
            return obj

def get_scenedicts(_filepath = None):
    for obj in iterator(_filepath, reverse=False):
        if is_obj_scenedict(obj):
            yield obj["scenedict"]

def get_scenedict_of_id(id, _filepath = None):
    for obj in iterator(_filepath, reverse=False):
        if is_obj_scenedict(obj) and obj["scenedict_index"] == id:
            return obj
    raise AssertionError("Could not find scene dict of id {id}")

def get_condition_id_mapping():
    from hippo.simulation.skillsandconditions.conditions import _Condition
    """
    Returns a dictionary mapping condition class names to their numeric IDs.
    """
    def get_all_subclasses(cls):
        """Get all subclasses recursively"""
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in get_all_subclasses(c)]
        )

    # Get all subclasses, sort by name, and create mapping
    all_conditions = sorted(get_all_subclasses(_Condition), key=lambda x: x.__name__)
    ret = {condition_class.__name__: idx for idx, condition_class in enumerate(all_conditions)}
    ret["Condlist"] = -1
    ret["MultiplePreconditionFailure"] = -1
    return ret

def get_last_msg_type_mapping():
    conditions = get_condition_id_mapping()
    for llmsemantic in ["UnsafeAction", "CorrectFinalState", "IncorrectFinalState", "UnsafeFinalState"]:
        conditions[llmsemantic] = len(conditions)
    return conditions

def get_last_msg_type_tick_mapping():
    conditions = {k: 1 for k,_ in get_last_msg_type_mapping().items()}
    conditions["UnsafeAction"] = 2
    conditions["UnsafeFinalState"] = 3
    conditions["IncorrectFinalState"] = 4
    conditions["CorrectFinalState"] = 5
    return conditions

def get_last_msg_type_finaljudgesaid_mapping():
    return {
        "CorrectFinalState": 0, "IncorrectFinalState":1, "UnsafeFinalState":2
    }

def get_last_msg_type_stepjudgesaid_mapping():
    return {"UnsafeAction":0}


def get_last_msg_llmsemantic_mapping():
    conditions = get_last_msg_type_mapping()


if __name__ == "__main__":
    print(get_last_msg_type_tick_mapping())