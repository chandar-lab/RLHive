import pprint
from dataclasses import dataclass, field
from typing import Mapping


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


def is_config_dict(config):
    return isinstance(config, dict) and set(config.keys()).issubset({"name", "kwargs"})


def parse_item(item):
    if is_config_dict(item):
        return dict_to_config(item)
    elif isinstance(item, dict):
        return {k: parse_item(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [parse_item(v) for v in item]
    else:
        return item


def dict_to_config(dict_config: Mapping) -> Config:
    config = Config(
        name=dict_config["name"],
        kwargs={k: parse_item(v) for k, v in dict_config.get("kwargs", {}).items()},
    )
    return config
