class EnvSpec:
    """Object used to store information about environment configuration. 
    Every environment should create an EnvSpec object.
    """

    def __init__(self, env_name, obs_dim, act_dim, env_info=None):
        """
        Args:
            env_name: Name of the environment
            obs_dim: Dimensionality of observations from environment. This can
                be a simple integer, or a complex object depending on the types
                of observations expected.
            act_dim: Dimensionality of action space.
            env_info: Any other info relevant to this environment. This can 
                include items such as random seeds or parameters used to create
                the environment
        """
        self.env_name = env_name
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        if env_info is None:
            self.env_info = {}
