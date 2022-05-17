from hive.agents.world_models.base import WorldModel
from hive.agents.world_models.dyna_models import ActionInMiddleDynaQModel
from hive.utils.registry import registry

registry.register_all(
    WorldModel,
    {
        "ActionInMiddleDynaQModel": ActionInMiddleDynaQModel,
    },
)

get_wm = getattr(registry, f"get_{WorldModel.type_name()}")
