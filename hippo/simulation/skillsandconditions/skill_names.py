import json
from functools import lru_cache

from hippo.simulation.skillsandconditions.skills_abstract import _Skill
from hippo.simulation.skillsandconditions.skills import * # noqa

def _get_final_subclasses(cls):
    subclasses = cls.__subclasses__()
    if not subclasses:  # No further subclasses
        return [cls]
    final_nodes = []
    for subclass in subclasses:
        final_nodes.extend(_get_final_subclasses(subclass))  # Recursive call
    return final_nodes


def get_all_possible_skills():
    return _get_final_subclasses(_Skill)


def get_enabledname_2_skills():
    all_possible_skills = get_all_possible_skills()

    ret = {}
    for skill in all_possible_skills:
        if skill.enabled_name not in ret:
            ret[skill.enabled_name] = []
        ret[skill.enabled_name].append(skill)

    return ret


def get_enabled_2_llm_name():
    temp = get_enabledname_2_skills()
    ret = {}
    for k, v in temp.items():
        if k not in ret:
            ret[k] = []
        for _v in v:
            ret[k].append(_v.anti_llm_name)
            if _v.llm_name is None:
                continue
            ret[k].append(_v.llm_name)
        if len(ret[k]) == 0:
            del ret[k]
    ret = {k: list(filter(lambda x : x is not None, v)) for k,v in ret.items()}
    ret = {k: list(set(v)) for k,v in ret.items() if len(v) > 0}

    # edge case: throwable
    del ret["throwable"]

    def sortedDeep(d):
        if isinstance(d, list):
            return sorted(sortedDeep(v) for v in d)
        if isinstance(d, dict):
            return {k: sortedDeep(d[k]) for k in sorted(d)}
        return d

    return sortedDeep(ret)


def get_llm_name_2_skill():
    all_possible_skills = get_all_possible_skills()

    ret = {}
    for skill in all_possible_skills:
        if skill.llm_name is not None:
            #assert skill.llm_name not in ret, f"Duplicate LLM name {skill.llm_name}"
            if skill.llm_name not in ret:
                ret[skill.llm_name] = []
            ret[skill.llm_name].append(skill)

    return ret

def get_anti_llm_name_2_skill():
    all_possible_skills = get_all_possible_skills()

    ret = {}
    for skill in all_possible_skills:
        if skill.anti_llm_name is not None:
            #assert skill.llm_name not in ret, f"Duplicate LLM name {skill.llm_name}"
            if skill.anti_llm_name not in ret:
                ret[skill.anti_llm_name] = []
            ret[skill.anti_llm_name].append(skill)

    return ret

#def get_all_llm_names():
#    temp = get_llm_name_2_skill()
#    return list(temp.keys())


#def get_exclusive_llm_names():
#    temp = get_enabled_2_llm_name()
#    return list(temp.values())

@lru_cache(maxsize=128)
def canonical_enabledname_2_llmname():
    # if ai2thor obj has enabled set to true, it corresponds to this llmname
    ret = {"receptacle": PutObject, "toggleable": ToggleObjectOnAndOff, "breakable": BreakObject, "dirtyable": CleanObject, "sliceable": SliceObject, "openable": OpenAndCloseObject, "pickupable": PickupObject}
    return {k: v().llm_name for k, v in ret.items()}


@lru_cache(maxsize=128)
def find_skillname_2_enabledname(skill_name):
    attribute_enabled_name = None
    for skill in get_all_possible_skills():
        if skill().has_skill_of_name(skill_name):
            attribute_enabled_name = skill.enabled_name
            break
    return attribute_enabled_name

def assert_names_are_valid(list_of_skill_metadata):

    dico = get_enabled_2_llm_name()
    for k, counter in dico.items():
        counter.append(0)

    for skill_llm_name in list_of_skill_metadata:
        for k, counter in dico.items():
            if skill_llm_name in counter:
                counter.append(counter.pop() + 1)

    errors = []
    for k, counter in dico.items():
        if counter[-1] > 1:
            errors.append(f"Duplicates for {k}: {counter[:-1]}")
    return errors



if __name__ == "__main__":
    print(get_enabled_2_llm_name())#, indent=2))
    print(len(get_enabledname_2_skills()))