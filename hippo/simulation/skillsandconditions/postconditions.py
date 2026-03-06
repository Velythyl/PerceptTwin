from dataclasses import dataclass
from typing import Callable, List

from typing_extensions import Self

from hippo.simulation.ai2thor_metadata_reader import get_robot_inventory, get_object_list_from_controller
from hippo.simulation.skillsandconditions.sas import SimulationActionState
from hippo.utils.selfdataclass import SelfDataclass


@dataclass
class _PostCondition(Callable, SelfDataclass):
    name: str = ""

    def __call__(self, sas: SimulationActionState) -> SimulationActionState:
        return sas

@dataclass
class POSTCOND_MakeSlicingImplementDirty(_PostCondition):
    def __call__(self, sas: SimulationActionState) -> Self:
        # todo
        pass