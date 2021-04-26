#  Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
#  with the License. A copy of the License is located at
#
#  http://aws.amazon.com/apache2.0/
#
#  or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
#  OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
#  and limitations under the License.
try:
    from ._version import __version__  # noqa
except ImportError:
    # For raw code without installing, use a dummy version
    __version__ = "0.0.0"

import dataclasses as dc
import copy


_class_mapping_ = {}


class MetaClass(type):
    @staticmethod
    def class_fullname(cls):
        return f"{cls.__module__}###{cls.__qualname__}"

    def __new__(cls, name, bases, attr):
        cls = super().__new__(cls, name, bases, attr)
        _class_mapping_[MetaClass.class_fullname(cls)] = cls
        return cls


class BaseClass(metaclass=MetaClass):
    @classmethod
    def class_fullname(cls):
        return MetaClass.class_fullname(cls)

    @classmethod
    def append_meta(cls, d: dict = None):
        meta = {"__meta__": {"class_fullname": cls.class_fullname()}}
        if d is not None:
            meta.update(d)
        return meta


@dc.dataclass
class BaseParams(BaseClass):
    @classmethod
    def from_dict(cls, param=None, recursive=False):
        def get_param(x, type_hint=None):
            if isinstance(x, BaseParams):
                return copy.deepcopy(x)
            elif isinstance(x, dict):
                meta = x.get("__meta__", None)
                if meta:
                    cls = _class_mapping_[meta["class_fullname"]]
                    x = cls.from_dict(x)
                elif type_hint and issubclass(type_hint, BaseParams):
                    x = type_hint.from_dict(x)
            elif isinstance(x, (list, tuple)):
                x = [get_param(y, type_hint=type_hint) for y in x]
            else:
                x = copy.deepcopy(x)
            return x

        if param is None:
            return cls()
        elif isinstance(param, cls):
            return copy.deepcopy(param)
        elif isinstance(param, dict):
            d = {}
            for f in dc.fields(cls):
                if f.name not in param:
                    if recursive and issubclass(f.type, BaseParams):
                        d[f.name] = f.type.from_dict(param, recursive=recursive)
                    continue
                d[f.name] = get_param(param[f.name], type_hint=f.type)
            return cls(**d)
        raise ValueError(f"{param} is not a valid parameter dictionary for {cls.name}")

    def to_dict(self, with_meta=True):
        d = {}
        for f in dc.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, BaseParams):
                d[f.name] = value.to_dict(with_meta)
            elif isinstance(value, (tuple, list)):
                d[f.name] = [
                    x.to_dict(with_meta) if isinstance(x, BaseParams) else x for x in value
                ]
            elif isinstance(value, dict):
                d[f.name] = {
                    k: v.to_dict(with_meta) if isinstance(v, BaseParams) else v
                    for k, v in value.items()
                }
            else:
                d[f.name] = value
        return self.append_meta(d) if with_meta else d
