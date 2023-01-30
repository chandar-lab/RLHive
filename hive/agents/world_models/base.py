from hive.utils.registry import Registrable


class WorldModel(Registrable):
    """A wrapper for callables that produce world models.

    These wrapped callables can be partially initialized through configuration
    files or command line arguments.
    """

    @classmethod
    def type_name(cls):
        """
        Returns:
            "function"
        """
        return "function"
