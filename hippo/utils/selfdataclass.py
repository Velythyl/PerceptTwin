import dataclasses
from typing import Any

import numpy as np
from typing_extensions import Self

from hippo.utils.dict_utils import recursive_map


def try_marshall(field_value):
    field = field_value.arg1
    value = field_value.arg2
    field_type = field.type
    try:
        if isinstance(value, field_type):
            return value
    except:
        pass

    try:
        if field_type == np.ndarray:
            return np.array(value)
    except:
        pass

    try:
        return field_type(value)
    except TypeError:
        pass

    #print(f"Could not marshall field of type {field_type} for value {value}")
    return value

@dataclasses.dataclass
class SelfDataclass:

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    def asdict(self):
        return dataclasses.asdict(self)

    def astuple(self):
        return dataclasses.astuple(self)

    @classmethod
    def fromdict(cls, kwargs: dict) -> Self:
        fields = dataclasses.fields(cls)

        fields = {f.name: f for f in fields}

        final_kwargs = {}
        for k, v in fields.items():
            if k in kwargs:
                final_kwargs[k] = kwargs[k]
            else:
                if fields[k].default is not dataclasses.MISSING:
                    final_kwargs[k] = fields[k].default
                elif fields[k].default_factory is not dataclasses.MISSING:
                    final_kwargs[k] = fields[k].default_factory()
                else:
                    final_kwargs[k] = None

        @dataclasses.dataclass
        class Arg:
            arg1: Any
            arg2: Any
        final_kwargs = {k: Arg(fields[k], v) for k,v in final_kwargs.items()}

        final_kwargs = recursive_map(final_kwargs, try_marshall)

        ret = cls(**final_kwargs)
        return ret



