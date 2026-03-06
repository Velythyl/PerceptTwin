from dataclasses import dataclass

from hippo.simulation.skillsandconditions.conditions import CONDITION_IsInteractable, \
    CONDITION_ObjectExists, CONDITION_SkillEnabled, CONDITION_AuxiliaryObjectIsInInventory, \
    CONDITION_SlicingImplementInInventory, \
    get_slicing_implement_from_inventory, CONDITION_NotUnderAnotherObject, CONDITION_InventoryIsEmpty
from hippo.simulation.skillsandconditions.sas import SimulationActionState
from hippo.simulation.skillsandconditions.skills_abstract import _Skill



@dataclass
class _ToggleObject(_Skill):
    enabled_name: str = "toggleable"
    state_name: str = "isToggled"
    anti_llm_name: str = "cannot be turned on or off"

    @property
    def pre_conditions(self):
        return [CONDITION_ObjectExists(), CONDITION_IsInteractable(), CONDITION_SkillEnabled()]



@dataclass
class ToggleObjectOnAndOff(_ToggleObject):
    llm_name: str = "can be turned on/off"

    def ToggleObjectOn(self, sas: SimulationActionState):
        ret = self.replace(state_value=True)
        sas.action_callback(sas)
        return ret

    def ToggleObjectOff(self, sas: SimulationActionState):
        ret = self.replace(state_value=False)
        sas.action_callback(sas)
        return ret

@dataclass
class ToggleObjectOn(_ToggleObject):
    llm_name: str = "can only be turned on"

    def ToggleObjectOn(self, sas: SimulationActionState):
        ret = self.replace(state_value=True)
        sas.action_callback(sas)
        return ret

@dataclass
class ToggleObjectOff(_ToggleObject):
    llm_name: str = "can only be turned off"

    def ToggleObjectOff_(self, sas: SimulationActionState):
        ret = self.replace(state_value=True)
        sas.action_callback(sas)
        return ret

@dataclass
class _Ai2ThorSkill(_Skill):
    # movement skill, always supported
    state_value: bool = None    # must be inferred from the controller, so runtimeobject handles it...
    state_name: str = None

@dataclass
class GoToObject(_Ai2ThorSkill):
    # movement skill, always supported
    enabled_name: str = "navigable"
    is_enabled: bool = True # always enabled

    llm_name: str = None    # LLM cant turn on/off

    def GoToObject(self, sas: SimulationActionState):
        sas.action_callback(sas)
        return None

    @property
    def pre_conditions(self):
        return [CONDITION_ObjectExists()]



@dataclass
class PickupObject(_Ai2ThorSkill):
    # movement skill, always supported
    enabled_name: str = "pickupable"  # LLM CAN turn on/off
    llm_name: str = "can be picked up"
    anti_llm_name: str = "cannot be picked up"

    @property
    def pre_conditions(self):
        return [CONDITION_ObjectExists(), CONDITION_IsInteractable(), CONDITION_SkillEnabled(), CONDITION_NotUnderAnotherObject(), CONDITION_InventoryIsEmpty()]

    def PickupObject(self, sas: SimulationActionState):
        sas.action_callback(sas)
        return None

    def specific_post_conditions(self):
        return []# fixme make condition that checks that (1) item has isPickedUp and also that (2) robot has it in the inventory

@dataclass
class ThrowObject(_Ai2ThorSkill):
    enabled_name: str = "throwable" # used when mapping llm names to skills, if "can be picked up" is not specified, we'll have to set ThrowObject to disabled
    llm_name: str = "can be picked up"
    anti_llm_name: str = "cannot be picked up"

    @property
    def pre_conditions(self):
        return [CONDITION_ObjectExists(),
                CONDITION_SkillEnabled()]

    def ThrowObject(self, sas: SimulationActionState):
        sas.action_callback(sas)
        return None

@dataclass
class PutObject(_Ai2ThorSkill):
    enabled_name: str = "receptacle"  # LLM CAN turn on/off
    llm_name: str = "objects can be put down on this"
    anti_llm_name: str = "objects can't or shouldn't be put down on this"

    def PutObject(self, sas: SimulationActionState):
        ret = sas.action_callback(sas)
        if ret is None:
            return None
        return ret

    @property
    def pre_conditions(self):
        return [CONDITION_ObjectExists(), CONDITION_IsInteractable(), CONDITION_SkillEnabled(), CONDITION_AuxiliaryObjectIsInInventory()]
    # fixme checks for the post conditions of PickupObject. So, the robot must have an object in its inventory.  Note: the target_object is the target surface, not the object that will be put down!

    @property
    def post_conditions(self):
        return [] # fixme  checks that the runtime container says the object that was in the inventory is now placed on the target object "isOnTop" on whatever



@dataclass
class _OpenableObject(_Skill):
    enabled_name: str = "openable"
    state_name: str = "isOpen"

    anti_llm_name: str = "cannot be opened or closed"

    @property
    def pre_conditions(self):
        return [CONDITION_ObjectExists(), CONDITION_IsInteractable(), CONDITION_SkillEnabled(), CONDITION_InventoryIsEmpty()]


@dataclass
class OpenAndCloseObject(_OpenableObject):
    llm_name: str = "can be opened and closed"

    def OpenObject(self, sas: SimulationActionState):
        ret = self.replace(state_value=True)
        sas.action_callback(sas)
        return ret

    def CloseObject(self, sas: SimulationActionState):
        ret = self.replace(state_value=False)
        sas.action_callback(sas)
        return ret

@dataclass
class OpenObject(_OpenableObject):
    llm_name: str = "can only be opened"

    def OpenObject(self, sas: SimulationActionState):
        ret = self.replace(state_value=True)
        sas.action_callback(sas)
        return ret

@dataclass
class CloseObject(_OpenableObject):
    llm_name: str = "can only be closed"

    def CloseObject(self, sas: SimulationActionState):
        ret = self.replace(state_value=False)
        sas.action_callback(sas)
        return ret

@dataclass
class SliceObject(_Skill):
    enabled_name: str = "sliceable"
    state_name: str = "isSliced"

    tool_llm_name:str = "is a slicing implement"
    anti_tool_llm_name:str = "cannot use this object to slice things with"

    llm_name: str = "can be sliced"
    anti_llm_name: str = "cannot be sliced in most situations"

    def SliceObject(self, sas: SimulationActionState):
        #slicing_tool = get_slicing_implement_from_inventory(sas)

        #slicing_portfolio = slicing_tool.skill_portfolio
        #skill = slicing_portfolio.find_skill("CleanObject").replace(state_value=True)
        #slicing_portfolio.effectuate_skill(skill)
        #slic
        #toolskill = slicing_tool.skill_portfolio.find_skill("SlicingTool")
        #toolskill

        ret = self.replace(state_value=True)
        sas.action_callback(sas)
        return ret

    @property
    def pre_conditions(self):
        return [CONDITION_ObjectExists(), CONDITION_IsInteractable(),
                CONDITION_SkillEnabled(), CONDITION_SlicingImplementInInventory()]

@dataclass
class SlicingTool(_Skill):
    enabled_name: str = "slicing implement"
    #state_name: str = "isDirty"

    llm_name: str = "is a slicing implement"
    anti_llm_name: str = "cannot use this object to slice things with"

    def SlicingTool(self, sas: SimulationActionState):
        #return self.replace(state_value=True)
        raise AssertionError("This is a tool-skill, and should never be called. Please report this bug.")

    @property
    def pre_conditions(self):
        return [CONDITION_ObjectExists(), CONDITION_IsInteractable(),
                CONDITION_SkillEnabled()]


@dataclass
class BreakObject(_Skill):
    enabled_name: str = "breakable"
    state_name: str = "isBroken"

    llm_name: str = "can be broken"
    anti_llm_name: str = "cannot be broken in most situations"

    @property
    def pre_conditions(self):
        return [CONDITION_ObjectExists(), CONDITION_IsInteractable(),
                CONDITION_SkillEnabled(), CONDITION_InventoryIsEmpty()]

    def BreakObject(self, sas: SimulationActionState):
        ret = self.replace(state_value=True)
        sas.action_callback(sas)
        return ret

@dataclass
class CleanObject(_Skill):
    enabled_name: str = "dirtyable"
    state_name: str = "isDirty"
    llm_name: str = "can be considered dirty or clean"
    anti_llm_name: str = "most people wouldn't use 'dirty' or 'clean' to describe this object"

    @property
    def pre_conditions(self):
        return [CONDITION_ObjectExists(), CONDITION_IsInteractable(),
                CONDITION_SkillEnabled()]

    def CleanObject(self, sas: SimulationActionState):
        return self.replace(state_value=False)

    def DirtyObject(self, sas: SimulationActionState):
        return self.replace(state_value=True)
