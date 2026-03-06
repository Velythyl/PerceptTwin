import copy
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict

from hippo.simulation.ai2thor_metadata_reader import get_object_list_from_controller
from hippo.simulation.skillsandconditions.sas import SimulationActionState

from hippo.utils.selfdataclass import SelfDataclass


@dataclass
class _Skill(SelfDataclass):
    enabled_name: str = None
    state_name: str = None
    llm_name: str = None
    anti_llm_name: str = None

    is_enabled: bool = None
    state_value: Any = None

    @property
    def pre_conditions(self):
        return []

    @property
    def post_conditions(self):
        return []

    #def verify_preconditions(self, sas):
    #    return verify_all_conditions(sas, self.pre_conditions)

    #def verify_postconditions(self, sas):
    #    return verify_all_conditions(sas, self.post_conditions)

    def has_skill_of_name(self, skill_name: str | SimulationActionState):
        if isinstance(skill_name, SimulationActionState):
            skill_name = skill_name.skill_name

        method = getattr(self, skill_name, None)
        if callable(method):
            return True

    def get_skill_of_name(self, skill_name: str | SimulationActionState):
        if isinstance(skill_name, SimulationActionState):
            skill_name = skill_name.skill_name

        assert self.has_skill_of_name(skill_name)
        return getattr(self, skill_name)

    def output_dict(self):
        if self.state_name is not None and self.is_enabled:
            return {self.state_name: self.state_value, self.enabled_name: self.is_enabled}
        return {}




@dataclass
class SkillPortfolio(SelfDataclass):
    skills: Dict[str, _Skill] = dataclasses.field(default_factory=dict)

    @classmethod
    def create(cls, skill_metadata):
        from hippo.simulation.skillsandconditions.skill_names import get_llm_name_2_skill, get_anti_llm_name_2_skill
        llmname_2_skillobj = get_llm_name_2_skill()
        anti_llmname_2_skillobj = get_anti_llm_name_2_skill()

        ret = []
        for llm_name in skill_metadata:
            if llm_name not in llmname_2_skillobj:
                assert llm_name in anti_llmname_2_skillobj
                continue
            applicable_skills = llmname_2_skillobj[llm_name]
            applicable_skills = [x(is_enabled=True, state_value=False) for x in applicable_skills]
            ret += applicable_skills #.append(llmname_2_skillobj[llm_name](is_enabled=True, state_value=False))    # fixme let the llm pick the initial state value?

        found_enablednames = [ret.enabled_name for ret in ret]

        from hippo.simulation.skillsandconditions.skill_names import get_enabledname_2_skills
        enabledname_2_skills = get_enabledname_2_skills()
        for enabledname in enabledname_2_skills:
            if enabledname not in found_enablednames:
                ret.append(enabledname_2_skills[enabledname][0]())

        ret = {r.enabled_name: r for r in ret}

        return cls(ret)

    def find_skill(self, sas: SimulationActionState | str):
        import hippo.simulation.skillsandconditions.skills  # noqa
        from hippo.simulation.skillsandconditions.skill_names import find_skillname_2_enabledname
        attribute_enabled_name = find_skillname_2_enabledname(skill_name=sas if isinstance(sas, str) else sas.skill_name)
        assert attribute_enabled_name is not None
        return self.skills[attribute_enabled_name]

    def apply(self, sas: SimulationActionState) -> SimulationActionState:
        raise AssertionError()
        skill = self.find_skill(sas)

        sas = sas.replace(skill_object=skill)

        preconditions = skill.verify_preconditions(sas)

        # todo the stuff below probably belongs in simulator...
        if all(preconditions):
            skill_method = skill.get_skill_of_name(sas)

        result = skill_method(sas)
        if isinstance(result, _Skill):
            new_portfolio = self.effectuate_skill(result)
            target_object_instance = sas.target_object.replace(skill_portfolio=new_portfolio)
            new_object_container = sas.pre_container.update_object(target_object_instance)
        elif result is None:
            new_object_container = sas.pre_container.update_from_ai2thor(get_object_list_from_controller(sas.controller))
        else:
            raise AssertionError("Could not recognize the result of the skill method. Make sure that the skill method returns either a skill or None.")

        sas = sas.replace(post_container=new_object_container)
        skill.verify_postconditions(sas)

        return sas

    def effectuate_skill(self, skill):
        skills = copy.deepcopy(self.skills)
        skills[skill.enabled_name] = skill
        return self.replace(skills=skills)

    def output_dict(self):
        ret = {}
        for k, v in self.skills.items():
            ret.update(v.output_dict())
        return ret

