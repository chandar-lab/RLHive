from hive.utils.registry import CallableType


class FunctionApproximator(CallableType):
    """A wrapper for callables that produce function approximators.

    For example, :obj:`FunctionApproximator(create_neural_network)` or
    :obj:`FunctionApproximator(MyNeuralNetwork)` where :obj:`create_neural_network` is a
    function that creates a neural network module and :obj:`MyNeuralNetwork` is a
    class that defines your function approximator.

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
