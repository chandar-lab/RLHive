import argparse
from copy import deepcopy
import inspect
from typing import List, Mapping, Sequence, _GenericAlias
import yaml
from functools import partial, update_wrapper


class Registrable:
    @classmethod
    def type_name(cls):
        raise ValueError


class CallableType(Registrable):
    def __init__(self, fn):
        self._fn = fn
        update_wrapper(self, self._fn)

    def __call__(self, *args, **kwargs):
        return partial(self._fn, *args, **kwargs)

    @classmethod
    def type_name(cls):
        return "callable"


class Registry:
    def __init__(self) -> None:
        self._registry = {}

    def register(self, name, constructor, type):
        if not issubclass(type, Registrable):
            raise ValueError(f"{type} is not Registrable")
        if type.type_name() not in self._registry:
            self._registry[type.type_name()] = {}

            def getter(self, object_or_config, prefix=None):
                if object_or_config is None:
                    return None, {}
                elif isinstance(object_or_config, type):
                    return object_or_config, {}
                name = object_or_config["name"]
                kwargs = object_or_config.get("kwargs", {})
                expanded_config = deepcopy(object_or_config)
                if name in self._registry[type.type_name()]:
                    object_class = self._registry[type.type_name()][name]
                    parsed_args = get_callable_parsed_args(object_class, prefix=prefix)
                    kwargs.update(parsed_args)
                    kwargs, kwargs_config = construct_objects(
                        object_class, kwargs, prefix
                    )
                    expanded_config["kwargs"] = kwargs_config
                    return object_class(**kwargs), expanded_config
                else:
                    raise ValueError(f"{name} class not found")

            setattr(self.__class__, f"get_{type.type_name()}", getter)
        self._registry[type.type_name()][name] = constructor

    def register_all(self, base_class, class_dict):
        for cls in class_dict:
            self.register(cls, class_dict[cls], base_class)


def construct_objects(object_constructor, config, prefix):
    signature = inspect.signature(object_constructor)
    prefix = "" if prefix is None else f"{prefix}."
    expanded_config = deepcopy(config)
    for argument in signature.parameters:
        if argument not in config:
            continue
        expected_type = signature.parameters[argument].annotation

        if isinstance(expected_type, type) and issubclass(expected_type, Registrable):
            config[argument], expanded_config[argument] = registry.__getattribute__(
                f"get_{expected_type.type_name()}"
            )(config[argument], f"{prefix}{argument}")
        if isinstance(expected_type, _GenericAlias):
            origin = expected_type.__origin__
            args = expected_type.__args__
            if (
                issubclass(origin, Sequence)
                and len(args) == 1
                and issubclass(args[0], Registrable)
                and isinstance(config[argument], Sequence)
            ):
                objs = []
                expanded_config[argument] = []
                for idx, item in enumerate(config[argument]):
                    obj, obj_config = registry.__getattribute__(
                        f"get_{args[0].type_name()}"
                    )(item, f"{prefix}{argument}.{idx}")
                    objs.append(obj)
                    expanded_config[argument].append(obj_config)
                config[argument] = objs
            elif (
                issubclass(origin, Mapping)
                and len(args) == 2
                and issubclass(args[1], Registrable)
                and isinstance(config[argument], Mapping)
            ):
                objs = {}
                expanded_config[argument] = {}
                for key, val in config[argument].items():
                    obj, obj_config = registry.__getattribute__(
                        f"get_{args[1].type_name()}"
                    )(val, f"{prefix}{argument}.{key}")
                    objs[key] = obj
                    expanded_config[argument][key] = obj_config
                config[argument] = objs

    return config, expanded_config


def get_callable_parsed_args(callable, prefix=None):
    signature = inspect.signature(callable)
    arguments = {
        argument: signature.parameters[argument]
        for argument in signature.parameters
        if argument != "self"
    }
    return get_parsed_args(arguments, prefix)


def get_parsed_args(arguments, prefix=None):
    prefix = "" if prefix is None else f"{prefix}."
    parser = argparse.ArgumentParser()
    for argument in arguments:
        parser.add_argument(f"--{prefix}{argument}")
    parsed_args, _ = parser.parse_known_args()
    parsed_args = vars(parsed_args)

    # Strip the prefix from the parsed arguments and remove arguments not present
    parsed_args = {
        (key[len(prefix) :] if key.startswith(prefix) else key): parsed_args[key]
        for key in parsed_args
        if parsed_args[key] is not None
    }

    for argument in parsed_args:
        expected_type = arguments[argument]
        if expected_type in [int, bool, str, float]:
            parsed_args[argument] = expected_type(parsed_args[argument])
        else:
            parsed_args[argument] = yaml.safe_load(parsed_args[argument])

    return parsed_args


registry = Registry()
