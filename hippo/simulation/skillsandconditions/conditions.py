from dataclasses import dataclass
from typing import Callable, List, Tuple, Any

from typing_extensions import Self

from hippo.simulation.ai2thor_metadata_reader import get_robot_inventory, get_object_list_from_controller, \
    get_object_from_controller
from hippo.simulation.semanticverifllm.llm_semantic_verification import _LLMSemanticVerification, UnsafeAction, \
    UnsafeFinalState, IncorrectFinalState
from hippo.simulation.singlefilelog import log_feedback_to_file, FeedbackMixin
from hippo.simulation.skillsandconditions.sas import SimulationActionState
from hippo.utils.selfdataclass import SelfDataclass

class ConditionIDMeta(type):
    _counter = 0
    def __new__(mcls, name, bases, attrs):
        cls = super().__new__(mcls, name, bases, attrs)
        cls._condition_id = ConditionIDMeta._counter
        ConditionIDMeta._counter += 1
        return cls
    # todo



@dataclass
class _Condition(Callable, SelfDataclass, FeedbackMixin):
    name: str = ""
    state: bool = None
    sas: SimulationActionState = None

    prev: Self = None
    success: bool = None

    @property
    def feedback_necessary(self) -> bool:
        return isinstance(self.state, bool) and self.state == False

    def __bool__(self):
        return self.success

    def error_message(self) -> List[str]:
        if self.state is None:
            raise AssertionError("You tried to get the Condition's error message, but it has not been evaluated yet.")

        acc = [] + self.prev.error_message()

        if self.state is False:
            acc = acc + [self._error_message()]

        acc = list(filter(lambda x: x is not None, acc))

        return acc

    def _error_message(self):
        raise NotImplementedError()

    def __call__(self, sas: SimulationActionState) -> Self:
        return self


@dataclass
class TrivialCondition(_Condition):
    name: str = ""
    state: bool = True
    sas: SimulationActionState = None

    prev: Callable = None
    success: bool = True


    def __call__(self, sas: SimulationActionState) -> Self:
        return self.replace(sas=sas)

    def error_message(self) -> List[str]:
        return []

    def _error_message(self):
        return None


@dataclass
class Condition(_Condition):
    name: str
    state: bool = None
    sas: SimulationActionState = None

    prev: _Condition = TrivialCondition()
    success: bool = None

    def __call__(self, sas: SimulationActionState) -> Self:
        self = self.replace(sas=sas)
        if self.prev is not None:
            prev = self.prev(sas)
            self = self.replace(prev=prev)

        state = self.call(sas)
        self = self.replace(state=state)
        self = self.replace(success=self.state and self.prev.success)
        return self

    def call(self, sas: SimulationActionState) -> bool:
        # sets self.state to True/False
        raise NotImplementedError()

@dataclass
class CONDITION_ObjectExists(Condition):
    name: str = "ObjectExists"

    def _error_message(self):
        return f"Object {self.sas.target_object_id} does not exist."

    def call(self, sas: SimulationActionState) -> bool:
        if get_object_from_controller(sas.controller, sas.target_object_id) is None:
            return False
        return True

@dataclass
class CONDITION_NotUnderAnotherObject(Condition):
    name: str = "NotUnderAnotherObject"

    def _error_message(self):
        return f"Object {self.sas.target_object_id} is under another object."

    def call(self, sas: SimulationActionState) -> bool:
        return sas.pre_container.is_obj_surface_free(sas.target_object_id)

@dataclass
class CONDITION_IsInteractable(Condition):
    name: str = "IsInteractable"
    prev: _Condition = CONDITION_ObjectExists()

    def _error_message(self):
        return f"Object {self.sas.target_object_id} is not interactable. It might be too far, be inside a closed object, or not visible by the robot."

    def call(self, sas: SimulationActionState) -> bool:
        object = get_object_from_controller(sas.controller, sas.target_object_id)

        is_obj_interactable = object["isInteractable"]

        if is_obj_interactable:
            return True

        obj_insideness = sas.pre_container.get_obj2id_that_obj1id_is_inside_of(sas.target_object_id)
        if obj_insideness is not None:
            inside_of = get_object_from_controller(sas.controller, obj_insideness)
            if inside_of["openable"]:
                if inside_of["isInteractable"] and inside_of["isOpen"]:
                    return True


        obj_ontopness = sas.pre_container.get_obj2id_that_obj1id_is_ontop_of(sas.target_object_id)
        if obj_ontopness is not None:
            ontop_of = get_object_from_controller(sas.controller, obj_ontopness)
            if ontop_of["isInteractable"]:
                return True
        return False

@dataclass
class CONDITION_AuxiliaryObjectIsInInventory(Condition):
    name: str = "IsInInventory"

    def _error_message(self):
        if self.sas.pre_container.get_held_object() is None:
            return f"Specified object is not in robot's inventory or a wrong ID was used."

        return f"Object {self.sas.auxiliary_object_id} is not in the robot's inventory."

    def call(self, sas: SimulationActionState) -> bool:
        if sas.pre_container.get_held_object() is None:
            return False

        return sas.auxiliary_object.heldBy == f"robot{sas.robot+1}"

@dataclass
class CONDITION_IsInInventory(Condition):
    name: str = "IsInInventory"

    def _error_message(self):
        return f"Object {self.sas.target_object_id} is not in the robot's inventory."

    def call(self, sas: SimulationActionState) -> bool:
        return sas.target_object.heldBy == f"robot{sas.robot+1}"

@dataclass
class CONDITION_InventoryIsEmpty(Condition):
    name: str = "InventoryIsEmpty"

    def _error_message(self):
        return f"Robot hand is not free because robot is holding {self.sas.pre_container.get_held_object().id}."

    def call(self, sas: SimulationActionState) -> bool:
        return sas.pre_container.get_held_object() is None

@dataclass
class CONDITION_AttributeEnabled(Condition):
    name: str = "AttributeEnabled"

    def _error_message(self):
        return f"The object {self.sas.target_object_id} does not support attribute {self.sas.skill_object.enabled_name}."

    def call(self, sas: SimulationActionState) -> bool:
        return bool(sas.skill_object.is_enabled)

@dataclass
class __COND_AttributeEnabledInAi2Thor(Condition):
    name: str = "AI2ThorAttributeEnabled"

    def _error_message(self):
        return f"The object {self.sas.target_object_id} does not support attribute {self.sas.skill_object.enabled_name}. This is likely due to a bug, and not a planning problem. Please report this bug."

    def call(self, sas: SimulationActionState) -> bool:
        obj = get_object_from_controller(sas.controller, sas.target_object_id)
        return obj[sas.skill_object.enabled_name]

@dataclass
class CONDITION_SkillEnabled(Condition):
    name: str = "SkillEnabled"
    prev: _Condition = CONDITION_AttributeEnabled()

    def _error_message(self):
        return f"The object {self.sas.target_object_id} does not support the skill {self.sas.skill_name}."

    def call(self, sas: SimulationActionState) -> bool:
        return sas.skill_object.has_skill_of_name(sas) and bool(sas.skill_object.is_enabled)

def get_slicing_implement_from_inventory(sas: SimulationActionState | Tuple[Any,int, Any]):
    if isinstance(sas, SimulationActionState):
        controller = sas.controller
        robot = sas.robot
        container = sas.pre_container
    else:
        controller, robot, container = sas

    inventory = get_robot_inventory(controller, robot)
    for item in inventory:
        item = container.get_object_by_id(item)
        toolskill = item.skill_portfolio.find_skill("SlicingTool")
        if toolskill.is_enabled:
            return item
    return None


@dataclass
class CONDITION_SlicingImplementInInventory(Condition):
    name: str = "SlicingImplementInInventory"
    prev: _Condition = CONDITION_AttributeEnabled()

    def _error_message(self):
        return f"The robot does not have a knife/slicing tool in hand."

    def call(self, sas: SimulationActionState) -> bool:
        tool = get_slicing_implement_from_inventory(sas)
        if tool is None:
            return False
        return True


@dataclass
class Condlist(SelfDataclass, FeedbackMixin):
    condlist: List[Condition]

    @property
    def feedback_type(self):
        if len(self.badconds) == 0:
            return self.__class__.__name__
        elif len(self.badconds) == 1:
            return self.badconds[0].__class__.__name__
        else:
            return "MultiplePreconditionFailure"

    @property
    def badconds(self) -> List[Condition]:
        badconds = []
        for cond in self.condlist:
            if cond.success is False:
                badconds.append(cond)
        return badconds

    @property
    def feedback_necessary(self):
        return len(self.badconds) >= 1

    def error_message(self):
        errors = []
        for cond in self.badconds:
            if isinstance(cond, _LLMSemanticVerification):
                return [cond.error_message()]
            msg = " AND ".join(cond.error_message())
            errors.append(msg)
        return errors

    def __str__(self):
        return "\n".join(self.error_message())

class ConditionFailure(Exception):
    pass

class PreconditionFailure(ConditionFailure):
    pass


class _PostconditionFailure(ConditionFailure):
    pass

class LLMVerificationFailure(_PostconditionFailure):
    pass


def eval_conditions(sas):
    preconditions = [c(sas) for c in sas.skill_object.pre_conditions]

    return maybe_raise_condition_exception(preconditions)


def maybe_raise_condition_exception(condlist):
    condlist = Condlist(condlist)
    log_feedback_to_file(condlist)

    if all(condlist.condlist):
        return condlist

    raise PreconditionFailure(condlist.badconds)




def maybe_raise_llmcondition_exception(llmreturn: _LLMSemanticVerification):
    log_feedback_to_file(llmreturn)
    if isinstance(llmreturn, (UnsafeAction, UnsafeFinalState, IncorrectFinalState)):
        raise LLMVerificationFailure(llmreturn)


if __name__ == "__main__":
    c = CONDITION_SkillEnabled()
    i=0