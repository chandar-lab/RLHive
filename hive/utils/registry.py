import json
import inspect
import argparse


class Registrable:
    @classmethod
    def type_name(cls):
        raise ValueError


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
                    return None
                elif isinstance(object_or_config, type):
                    return object_or_config
                name = object_or_config["name"]
                kwargs = object_or_config["kwargs"]
                if name in self._registry[type.type_name()]:
                    object_class = self._registry[type.type_name()][name]
                    parsed_args = get_parsed_arguments(object_class, prefix=prefix)
                    kwargs.update(parsed_args)
                    kwargs = construct_objects(constructor, object_or_config, prefix)
                    return object_class(**kwargs)
                else:
                    raise ValueError(f"{name} class not found")

            setattr(self, f"get_{type.type_name()}", getter)
        self._registry[type.type_name()][name] = constructor

    def register_all(self, base_class, class_dict):
        for cls in class_dict:
            self.register(cls, class_dict[cls], base_class)


def construct_objects(object_constructor, config, prefix):
    signature = inspect.signature(object_constructor)
    prefix = "" if prefix is None else f"{prefix}."
    for argument in signature.parameters:
        if argument not in config:
            continue
        expected_type = signature.parameters[argument].annotation

        if issubclass(expected_type, Registrable):
            config[argument] = registry.__getattribute__(
                f"get_{expected_type.type_name()}"
            )(config[argument], f"{prefix}{argument}")

    return config


def get_parsed_arguments(callable, prefix=None):
    prefix = "" if prefix is None else f"{prefix}."
    signature = inspect.signature(callable)
    parser = argparse.ArgumentParser()
    for argument in signature.parameters:
        if argument == "self":
            continue
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
        expected_type = signature.parameters[argument].annotation
        if expected_type in [int, bool, str, float]:
            parsed_args[argument] = expected_type(parsed_args[argument])
        else:
            parsed_args[argument] = json.loads(parsed_args[argument])

    return parsed_args


registry = Registry()
