import argparse
import inspect
import logging
import pprint
from copy import deepcopy
from dataclasses import dataclass, field
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
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)
from typing import get_type_hints as _get_type_hints
from typing import runtime_checkable

import numpy as np
import yaml
from typing_extensions import Annotated

R = TypeVar("R", covariant=True)

SeqOrSingle = Union[Sequence[R], R]


@dataclass
class Config:
    name: str
    kwargs: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return 'Config(name="{}", kwargs={})'.format(
            self.name, pprint.pformat(self.kwargs, indent=2, compact=False), width=40
        )


def config_to_dict(config):
    if isinstance(config, Config):
        return {
            "name": config.name,
            "kwargs": {k: config_to_dict(v) for k, v in config.kwargs.items()},
        }
    elif isinstance(config, dict):
        return {k: config_to_dict(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_to_dict(v) for v in config]
    else:
        return config


def dict_to_config(dict_config: Mapping) -> Config:
    config = Config(
        name=dict_config["name"],
        kwargs={k: parse_item(v) for k, v in dict_config.get("kwargs", {}).items()},
    )
    return config


def parse_item(item):
    if is_config_dict(item):
        return dict_to_config(item)
    elif isinstance(item, dict):
        return {k: parse_item(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [parse_item(v) for v in item]
    else:
        return item


def is_config_dict(config):
    return isinstance(config, dict) and set(config.keys()).issubset({"name", "kwargs"})


T = TypeVar("T")

Creates = Annotated[Callable[..., R], "configured", "creates"]

C = TypeVar("C", bound=Callable)
Partial = Annotated[C, "configured", "partial"]


OCreates = Optional[Creates[R]]

Float = Union[float, np.float32, np.float64]
Int = Union[int, np.int32, np.int64]


def default(fn: Optional[T], default_fn: T) -> T:
    if fn is None:
        return default_fn
    else:
        return fn


T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True)
class Creator(Generic[T]):
    constructor: Callable[..., T]
    type: Type[T]


class RegistryStore:
    def __init__(self) -> None:
        self.creators: Dict[str, Set[Creator]] = {}

    def add_constructor(self, name: str, constructor: Callable[..., T], type: Type[T]):
        if name in self.creators:
            logging.warning(f"Multiple constructors registered with {name}.")
            self.creators[name].add(Creator(constructor, type))
        else:
            self.creators[name] = {Creator(constructor, type)}

    def get_constructors(self, name: str) -> Set[Creator]:
        if name in self.creators:
            return self.creators[name]
        else:
            raise KeyError(f"Name {name} not found in registry.")


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
        self._registry = RegistryStore()

    def register(self, name: str, constructor: Callable[..., U], type: Type[U]) -> None:
        """Register a Registrable class/object with RLHive.

        Args:
            name (str): Name of the class/object being registered.
            constructor (callable): Callable that will be passed all kwargs from
                configs and be analyzed to get type annotations.
            type (type): Type of class/object being registered. Should be subclass of
                Registrable.

        """
        self._registry.add_constructor(name, constructor, type)

    def register_class(self, name: str, type: Type) -> None:
        """Register a Registrable class/object with RLHive.

        Args:
            name (str): Name of the class/object being registered.
            constructor (callable): Callable that will be passed all kwargs from
                configs and be analyzed to get type annotations.
            type (type): Type of class/object being registered. Should be subclass of
                Registrable.

        """
        self._registry.add_constructor(name, type, type)

    def get(
        self, config: Config, type: Type[U], prefix: Optional[str] = None
    ) -> Tuple[Creates[U], Config]:
        return self._get(config, Creates[type], prefix)  # type: ignore

    def _get(
        self,
        config: Config,
        type: Type[Union[Creates[T], Partial[C]]],
        prefix: Optional[str] = None,
    ) -> Tuple[Union[Creates[T], Partial[C]], Config]:
        if config is None:
            raise ValueError(f"Config for {type} is None")
        name = config.name
        kwargs = config.kwargs
        # expanded_config = deepcopy(config)

        try:
            object_creators = self._registry.get_constructors(name)
            object_creator = resolve_creator(object_creators, type)
            # object_creator = PartialCreates(object_creator.constructor)
            parsed_args, unused_args = get_callable_parsed_args(
                object_creator.constructor, prefix=prefix
            )
            kwargs.update(parsed_args)
            kwargs, kwargs_config = construct_objects(object_creator, kwargs, prefix)
            # expanded_config["kwargs"] = kwargs_config
            constructor = partial(object_creator.constructor, **kwargs)
            return constructor, Config(name=config.name, kwargs=kwargs_config)
        except:
            logging.error(f"Error creating {name} class")
            raise

    def register_classes(self, class_dict: Dict[str, Type]):
        """Bulk register function.

        Args:
            base_class (type): Corresponds to the `type` of the register function
            class_dict (dict[str, callable]): A dictionary mapping from name to
                constructor.
        """
        for cls in class_dict:
            self.register_class(cls, class_dict[cls])

    def register_all_with_type(
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
        return pprint.pformat(self._registry, indent=2)


def resolve_creator(object_creators: Set[Creator], type: Type) -> Creator:
    create_types = get_configured_types(type)

    filtered_creators = {
        creator
        for creator in object_creators
        if any(check_subclass(creator.type, ct) for ct in create_types)
    }
    if len(filtered_creators) > 1:
        raise ValueError(f"Multiple creators for {type}")
    elif len(filtered_creators) == 0:
        raise ValueError(f"No creators for {type}")
    else:
        return filtered_creators.pop()


def get_configured_types(ty: Type):
    base_types = get_base_types(ty)
    configured_types = set()
    for base_type in base_types:
        if (
            type(base_type) is type(Creates)
            and hasattr(base_type, "__metadata__")
            and "configured" in base_type.__metadata__
        ):
            for t in get_args(base_type):
                if "partial" in base_type.__metadata__:
                    configured_types = configured_types.union(get_base_types(t))
                else:
                    create_type = get_args(t)[1]
                    configured_types = configured_types.union(
                        get_base_types(create_type)
                    )
    return configured_types


def intersect_generic_types(type1: Type, type2: Type) -> Set[Type]:
    base_types = get_base_types(type1)
    create_types = set()
    for base_type in base_types:
        if get_origin(base_type) and check_subclass(base_type, type2):  # type: ignore
            for t in get_args(base_type):
                create_types = create_types.union(get_base_types(t))
    return create_types


def get_base_types(type: Type) -> Sequence[Type]:
    base_types = []
    if get_origin(type) is Union:
        for t in get_args(type):
            base_types += get_base_types(t)
    else:
        base_types.append(type)
    return tuple(base_types)


def check_subclass(type1: Type, type2: Type) -> bool:
    try:
        if get_origin(type1):
            type1 = get_origin(type1)
        if get_origin(type2):
            type2 = get_origin(type2)
        return issubclass(type1, type2)
    except TypeError:
        return False


def get_type_hints(fn):
    if hasattr(fn, "__init__"):
        return _get_type_hints(fn.__init__)
    else:
        return _get_type_hints(fn)


def construct_objects(
    object_constructor: Creator,
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
    type_hints = get_type_hints(object_constructor.constructor)
    prefix = "" if prefix is None else f"{prefix}."
    expanded_config = deepcopy(config)
    for argument in type_hints:
        if argument not in config:
            continue
        expected_type = type_hints[argument]

        if isinstance(config[argument], Config):
            config[argument], expanded_config[argument] = registry._get(
                config[argument], expected_type, f"{prefix}{argument}"
            )
        elif isinstance(config[argument], Sequence) and not isinstance(
            config[argument], str
        ):
            sequence_type = tuple(intersect_generic_types(expected_type, Sequence))
            for idx, item in enumerate(config[argument]):
                if isinstance(item, Config):
                    if not sequence_type:
                        raise ValueError(
                            f"Could not find type match for {config[argument]} with {expected_type}"
                        )
                    (
                        config[argument][idx],
                        expanded_config[argument][idx],
                    ) = registry._get(
                        item, Union[sequence_type], f"{prefix}{argument}.{idx}"  # type: ignore
                    )
                else:
                    config[argument][idx] = item
        elif isinstance(config[argument], Mapping):
            mapping_type = tuple(intersect_generic_types(expected_type, Mapping))
            for key, item in config[argument].items():
                if isinstance(item, Config):
                    (
                        config[argument][key],
                        expanded_config[argument][key],
                    ) = registry._get(
                        item, Union[mapping_type], f"{prefix}{argument}.{key}"  # type: ignore
                    )
                else:
                    config[argument][key] = item

    return config, expanded_config


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
    callable = callable.__init__ if hasattr(callable, "__init__") else callable
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
