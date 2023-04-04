from hive.agents.world_models.base import WorldModel
from hive.agents.world_models.dyna_models import (
    ActionInMiddleDynaQModel,
    NetPerActionDynaQModel,
)
from hive.utils.registry import registry

registry.register_all(
    WorldModel,
    {
        "ActionInMiddleDynaQModel": ActionInMiddleDynaQModel,
        "NetPerActionDynaQModel": NetPerActionDynaQModel,
    },
)

get_wm = getattr(registry, f"get_{WorldModel.type_name()}")
