import argparse
import inspect
from copy import deepcopy
from functools import partial
from typing import _GenericAlias  # type: ignore
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import yaml

R = TypeVar("R", covariant=True)

SeqOrSingle = Union[Sequence[R], R]

from typing import Protocol


class Creates(Protocol[R]):
    def __call__(self, *args, **kwargs) -> R:
        ...


class PartialCreates(Creates[R]):
    # class Creates(Generic[R], Protocol):
    #     return super().__call__(*args, **kwds)
    def __init__(self, creator: Callable[..., R]):
        self._creator = creator

    def update_args(self, *args, **kwargs):
        self._creator = partial(self._creator, *args, **kwargs)

    def __call__(self, *args: Any, **kwds: Any) -> R:
        return self._creator(*args, **kwds)

    def signature(self):
        return inspect.signature(self._creator)


OCreates = Optional[Creates[R]]
import numpy as np

Float = Union[float, np.float32, np.float64]
Int = Union[int, np.int32, np.int64]


def default(fn: OCreates[R], default_fn: Callable) -> Creates[R]:
    if fn is None:
        return cast(Creates[R], default_fn)
    else:
        return fn


# Creates = NewType('Creates', Callable)
# [Callable[..., Union[R_co, Any]]]
T = TypeVar("T")
U = TypeVar("U")
import pprint
from typing import cast


class RegistryTree(Generic[T]):
    def __init__(self) -> None:
        self.creators: Dict[str, Callable[..., T]] = {}
        self.subtrees: Dict[Type, "RegistryTree"] = {}

    def get_tree(self, ty: Type[U]) -> "RegistryTree[U]":
        for registry_type in self.subtrees:
            if ty == registry_type:
                return self.subtrees[registry_type]
            elif issubclass(ty, registry_type):
                return self.subtrees[registry_type].get_tree(ty)
        self.subtrees[ty] = RegistryTree[ty]()
        return self.subtrees[ty]

    def register_creator(self, name: str, creator: Callable[..., T]) -> None:
        self.creators[name] = creator

    def get_constructors(self) -> Dict[str, Callable[..., T]]:
        constructors: Dict[str, Callable[..., T]] = {}
        for subtree in self.subtrees.values():
            constructors.update(subtree.get_constructors())
        constructors.update(self.creators)
        return constructors

    def __repr__(self) -> str:
        return pprint.pformat(
            (self.creators, self.subtrees),
            indent=2,
        )


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
        self._registry = RegistryTree[object]()

    def register(self, name: str, constructor: Callable[..., U], type: Type[U]) -> None:
        """Register a Registrable class/object with RLHive.

        Args:
            name (str): Name of the class/object being registered.
            constructor (callable): Callable that will be passed all kwargs from
                configs and be analyzed to get type annotations.
            type (type): Type of class/object being registered. Should be subclass of
                Registrable.

        """
        # if not issubclass(type, Registrable):
        #     raise ValueError(f"{type} is not Registrable")
        registry_tree = self._registry.get_tree(type)
        registry_tree.register_creator(name, constructor)
        # if type.type_name() not in self._registry:
        #     self._registry[type.type_name()] = {}

        #     def getter(self, object_or_config, prefix=None):
        #         if object_or_config is None:
        #             return None, {}
        #         elif isinstance(object_or_config, type):
        #             return object_or_config, {}
        #         name = object_or_config["name"]
        #         kwargs = object_or_config.get("kwargs", {})
        #         expanded_config = deepcopy(object_or_config)
        #         try:
        #             object_class = self._registry[type.type_name()][name]
        #             parsed_args = get_callable_parsed_args(object_class, prefix=prefix)
        #             kwargs.update(parsed_args)
        #             kwargs, kwargs_config = construct_objects(
        #                 object_class, kwargs, prefix
        #             )
        #             expanded_config["kwargs"] = kwargs_config
        #             return partial(object_class, **kwargs), expanded_config
        #         except:
        #             raise ValueError(f"Error creating {name} class")

        #     setattr(self.__class__, f"get_{type.type_name()}", getter)
        # self._registry[type.type_name()][name] = constructor

    def get(
        self, config: Dict[str, Any], type: Type[U], prefix: Optional[str] = None
    ) -> Tuple[PartialCreates[U], Dict[str, Any]]:
        if config is None:
            raise ValueError(f"Config for {type} is None")
        name = config["name"]
        kwargs = config.get("kwargs", {})
        expanded_config = deepcopy(config)
        try:
            object_creator = self._registry.get_tree(type).creators[name]
            object_creator = PartialCreates(object_creator)
            parsed_args, unused_args = get_callable_parsed_args(
                object_creator, prefix=prefix
            )
            kwargs.update(parsed_args)
            kwargs, kwargs_config = construct_objects(object_creator, kwargs)
            expanded_config["kwargs"] = kwargs_config
            object_creator.update_args(**kwargs)
            return object_creator, expanded_config
        except:
            raise ValueError(f"Error creating {name} class")

    def register_all(
        self, base_class: Type[U], class_dict: Dict[str, Callable[..., U]]
    ):
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


def construct_objects(
    object_constructor: PartialCreates,
    config: Dict[str, Any],
    prefix: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
    signature = object_constructor.signature()
    prefix = "" if prefix is None else f"{prefix}."
    expanded_config = deepcopy(config)
    for argument in signature.parameters:
        # if argument not in config:
        #     continue
        expected_type = signature.parameters[argument].annotation

        # if isinstance(expected_type, type) and issubclass(expected_type, Creates):
        #     config[argument], expanded_config[argument] = registry.__getattribute__(
        #         f"get_{expected_type.type_name()}"
        #     )(config[argument], f"{prefix}{argument}")
        if is_creates(expected_type):
            object_type = expected_type.__args__[0]
            config[argument], expanded_config[argument] = registry.get(
                config[argument], object_type, f"{prefix}{argument}"
            )
        if isinstance(expected_type, _GenericAlias):
            origin = expected_type.__origin__
            args = expected_type.__args__
            if (
                (origin == List or origin == list)
                and len(args) == 1
                and is_creates(args[0])
                and isinstance(config[argument], Sequence)
            ):
                objs = []
                expanded_config[argument] = []
                object_type = args[0].__args__[0]
                for idx, item in enumerate(config[argument]):
                    obj, obj_config = registry.get(
                        item, object_type, f"{prefix}{argument}.{idx}"
                    )
                    objs.append(obj)
                    expanded_config[argument].append(obj_config)
                config[argument] = objs
            elif (
                origin == dict
                and len(args) == 2
                and is_creates(args[1])
                and isinstance(config[argument], Mapping)
            ):
                objs = {}
                expanded_config[argument] = {}
                object_type = args[1].__args__[0]
                for key, val in config[argument].items():
                    obj, obj_config = registry.get(
                        val, object_type, f"{prefix}{argument}.{key}"
                    )
                    objs[key] = obj
                    expanded_config[argument][key] = obj_config
                config[argument] = objs

    return config, expanded_config


def is_creates(annotation) -> bool:
    return (
        isinstance(annotation, _GenericAlias)
        and annotation.__origin__ == PartialCreates
    )


def get_callable_parsed_args(
    callable: Callable, prefix=None
) -> Tuple[Dict[str, Any], List[str]]:
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


def get_parsed_args(
    arguments: Dict[str, inspect.Parameter], prefix=None
) -> Tuple[Dict[str, Any], List[str]]:
    """Helper function that takes a dictionary mapping argument names to types, and
    extracts command line arguments for those arguments. If the dictionary contains
    a key-value pair "bar": int, and the prefix passed is "foo", this function will
    look for a command line argument "--foo.bar". If present, it will cast it to an
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
    parsed_args, unused_args = parser.parse_known_args()
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

    return parsed_args, unused_args


registry = Registry()
