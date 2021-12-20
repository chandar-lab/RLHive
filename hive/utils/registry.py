import argparse
import inspect
from copy import deepcopy
from functools import partial, update_wrapper
from typing import List, Mapping, Sequence, _GenericAlias

import yaml


class Registrable:
    """Class used to denote which types of objects can be registered in the RLHive
    Registry. These objects can also be configured directly from the command line, and
    recursively built from the config, assuming type annotations are present.
    """

    @classmethod
    def type_name(cls):
        """This should represent a string that denotes the which type of class you are
        creating. For example, "logger", "agent", or "env".
        """
        raise ValueError


class CallableType(Registrable):
    """A wrapper that allows any callable to be registered in the RLHive Registry.
    Specifically, it maps the arguments and annotations of the wrapped function to the
    resulting callable, allowing any argument names and type annotations of the
    underlying function to be present for outer wrapper. When called with some
    arguments, this object returns a partial function with those arguments assigned.

    By default, the type_name is "callable", but if you want to create specific types
    of callables, you can simply create a subclass and override the type_name method.
    See :py:class:`hive.utils.utils.OptimizerFn`.
    """

    def __init__(self, fn):
        """
        Args:
            fn: callable to be wrapped.
        """
        self._fn = fn
        update_wrapper(self, self._fn)

    def __call__(self, *args, **kwargs):
        return partial(self._fn, *args, **kwargs)

    @classmethod
    def type_name(cls):
        return "callable"

    def __repr__(self):
        return f"<{type(self).__name__} {repr(self._fn)}>"


class Registry:
    """This is the Registry class for RLHive. It allows you to register different types
    of :py:class:`Registrable` classes and objects and generates constructors for those
    classes in the form of `get_{type_name}`.

    These constructors allow you to construct objects from dictionary configs. These
    configs should have two fields: `name`, which corresponds to the name used when
    registering a class in the registry, and `kwargs`, which corresponds to the keyword
    arguments that will be passed to the constructor of the object. These constructors
    can also build objects recursively, i.e. if a config contains the config for
    another `Registrable` object, this will be automatically created before being
    passed to the constructor of the original object. These constructors also allow you
    to directly specify/override arguments for object constructors directly from the
    command line. These parameters are specified in dot notation. They also are able
    to handle lists and dictionaries of Registrable objects.

    For example, let's consider the following scenario:
    Your agent class has an argument `arg1` which is annotated to be `List[Class1]`,
    `Class1` is `Registrable`, and the `Class1` constructor takes an argument `arg2`.
    In the passed yml config, there are two different Class1 object configs listed.
    the constructor will check to see if both `--agent.arg1.0.arg2` and
    `--agent.arg1.1.arg2` have been passed.

    The parameters passed in the command line will be parsed according to the type
    annotation of the corresponding low level constructor. If it is not one of
    `int`, `float`, `str`, or `bool`, it simply loads the string into python using a
    yaml loader.

    Each constructor returns the object, as well a dictionary config with all the
    parameters used to create the object and any Registrable objects created in the
    process of creating this object.
    """

    def __init__(self) -> None:
        self._registry = {}

    def register(self, name, constructor, type):
        """Register a Registrable class/object with RLHive.

        Args:
            name (str): Name of the class/object being registered.
            constructor (callable): Callable that will be passed all kwargs from
                configs and be analyzed to get type annotations.
            type (type): Type of class/object being registered. Should be subclass of
                Registrable.

        """
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
        """Bulk register function.

        Args:
            base_class (type): Corresponds to the `type` of the register function
            class_dict (dict[str, callable]): A dictionary mapping from name to
                constructor.
        """
        for cls in class_dict:
            self.register(cls, class_dict[cls], base_class)

    def __repr__(self):
        return str(self._registry)


def construct_objects(object_constructor, config, prefix=None):
    """Helper function that constructs any objects specified in the config that
    are registrable.

    Returns the object, as well a dictionary config with all the parameters used to
    create the object and any Registrable objects created in the process of creating
    this object.

    Args:
        object_constructor (callable): constructor of object that corresponds to
            config. The signature of this function will be analyzed to see if there
            are any :py:class:`Registrable` objects that might be specified in the
            config.
        config (dict): The kwargs for the object being created. May contain configs for
            other `Registrable` objects that need to be recursively created.
        prefix (str): Prefix that is attached to the argument names when looking for
            command line arguments.
    """
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
                (origin == List or origin == list)
                and len(args) == 1
                and isinstance(args[0], type)
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
                origin == dict
                and len(args) == 2
                and isinstance(args[1], type)
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
    """Helper function that extracts the command line arguments for a given function.

    Args:
        callable (callable): function whose arguments will be inspected to extract
            arguments from the command line.
        prefix (str): Prefix that is attached to the argument names when looking for
            command line arguments.
    """
    signature = inspect.signature(callable)
    arguments = {
        argument: signature.parameters[argument]
        for argument in signature.parameters
        if argument != "self"
    }
    return get_parsed_args(arguments, prefix)


def get_parsed_args(arguments, prefix=None):
    """Helper function that takes a dictionary mapping argument names to types, and
    extracts command line arguments for those arguments. If the dictionary contains
    a key-value pair "bar": int, and the prefix passed is "foo", this function will
    look for a command line argument "\-\-foo.bar". If present, it will cast it to an
    int.

    If the type for a given argument is not one of `int`, `float`, `str`, or `bool`,
    it simply loads the string into python using a yaml loader.

    Args:
        arguments (dict[str, type]): dictionary mapping argument names to types
        prefix (str): prefix that is attached to each argument name before searching
            for command line arguments.
    """
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
        if isinstance(expected_type, inspect.Parameter):
            expected_type = expected_type.annotation
        if expected_type in [int, str, float]:
            parsed_args[argument] = expected_type(parsed_args[argument])
        elif expected_type is bool:
            value = str(parsed_args[argument]).lower()
            parsed_args[argument] = not ("false".startswith(value) or value == "0")
        else:
            parsed_args[argument] = yaml.safe_load(parsed_args[argument])

    return parsed_args


registry = Registry()
