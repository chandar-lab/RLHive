import inspect
import os
import pickle
from collections import OrderedDict
from copy import deepcopy
from functools import partial

from torch import optim


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


class Chomp:
    def __init__(self, src_dict=None):
        self.__dict__["tparams"] = OrderedDict()
        if src_dict is not None:
            self.add_from_dict(src_dict)

    def __setattr__(self, name, array):
        tparams = self.__dict__["tparams"]
        tparams[name] = array

    def __setitem__(self, name, array):
        self.__setattr__(name, array)

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __getattr__(self, name):
        tparams = self.__dict__["tparams"]
        if name in tparams:
            return tparams[name]
        else:
            return None

    def remove(self, name):
        del self.__dict__["tparams"][name]

    def get(self):
        return self.__dict__["tparams"]

    def values(self):
        tparams = self.__dict__["tparams"]
        return list(tparams.values())

    def save(self, filename):
        tparams = self.__dict__["tparams"]
        pickle.dump({p: tparams[p] for p in tparams}, open(filename, "wb"), 2)

    def load(self, filename):
        tparams = self.__dict__["tparams"]
        loaded = pickle.load(open(filename, "rb"))
        for k in loaded:
            tparams[k] = loaded[k]

    def setvalues(self, values):
        tparams = self.__dict__["tparams"]
        for p, v in zip(tparams, values):
            tparams[p] = v

    def add_from_dict(self, src_dict):
        for key in src_dict.keys():
            self.__setattr__(key, src_dict[key])

    def __enter__(self):
        _, _, _, env_locals = inspect.getargvalues(inspect.currentframe().f_back)
        self.__dict__["_env_locals"] = list(env_locals.keys())

    def __exit__(self, type, value, traceback):
        _, _, _, env_locals = inspect.getargvalues(inspect.currentframe().f_back)
        prev_env_locals = self.__dict__["_env_locals"]
        del self.__dict__["_env_locals"]
        for k in list(env_locals.keys()):
            if k not in prev_env_locals:
                self.__setattr__(k, env_locals[k])
                env_locals[k] = self.__getattr__(k)
        return True

    def __deepcopy__(self, memo):
        new_container = Chomp()
        new_container.add_from_dict(src_dict=deepcopy(self.get()))
        return new_container


def create_class_constructor(base_class, class_dict):
    """Creates a constructor function for subclasses of base_class.
    
    The constructor function returned takes in either None, a object that is an 
    instance of base_class, or a dictionary config. If the argument is None or an
    instance of base_class, it is returned without modification. If it is a
    dictionary, the config should have two keys: name and kwargs. The name 
    parameter is used to lookup the correct class from class_dict and the object is
    created using kwargs as parameters.

    Args:
        base_class (type|"callable"): If base_class is a type, it is used to verify
            the type of the object passed to the constructor. If base_class is the
            string "callable", the object passed to the constructor is simply checked
            to see if it's callable.
        class_dict: A dictionary of class names to callables that can be passed kwargs
            to create the necessary objects.
    """

    def constructor(object_or_config):
        if object_or_config is None:
            return None
        if base_class == "callable":
            if callable(object_or_config):
                return object_or_config
        elif isinstance(object_or_config, base_class):
            return object_or_config
        name = object_or_config["name"]
        kwargs = object_or_config["kwargs"]
        if name in class_dict:
            object_class = class_dict[name]
            return object_class(**kwargs)
        else:
            raise ValueError(f"{name} class not found")

    return constructor


get_optimizer_fn = create_class_constructor(
    "callable",
    {
        "Adadelta": (lambda **kwargs: partial(optim.Adadelta, **kwargs)),
        "Adagrad": (lambda **kwargs: partial(optim.Adagrad, **kwargs)),
        "Adam": (lambda **kwargs: partial(optim.Adam, **kwargs)),
        "Adamax": (lambda **kwargs: partial(optim.Adamax, **kwargs)),
        "AdamW": (lambda **kwargs: partial(optim.AdamW, **kwargs)),
        "ASGD": (lambda **kwargs: partial(optim.ASGD, **kwargs)),
        "LBFGS": (lambda **kwargs: partial(optim.LBFGS, **kwargs)),
        "RMSprop": (lambda **kwargs: partial(optim.RMSprop, **kwargs)),
        "Rprop": (lambda **kwargs: partial(optim.Rprop, **kwargs)),
        "SGD": (lambda **kwargs: partial(optim.SGD, **kwargs)),
        "SparseAdam": (lambda **kwargs: partial(optim.SparseAdam, **kwargs)),
    },
)

