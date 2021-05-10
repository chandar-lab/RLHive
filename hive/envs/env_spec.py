class EnvSpec:
    """Object used to store information about environment configuration. 
    Every environment should create an EnvSpec object.
    """

    def __init__(self, env_name, obs_dim, act_dim, num_disc_per_obs_dim=None, env_info=None):
        """
        Args:
            env_name: Name of the environment
            obs_dim: Dimensionality of observations from environment. This can
                be a simple integer, or a complex object depending on the types
                of observations expected.
            act_dim: Dimensionality of action space.
            num_disc_per_obs_dim: Number of discrete observations per dimension
                of the observation space. Each dimension of the obs space can
                be represented as a one hot encoding with this parameter as
                the max value for the encoding. If None (default) then this
                means that observations should be treated as continuous inputs
            env_info: Any other info relevant to this environment. This can 
                include items such as random seeds or parameters used to create
                the environment
        """
        self._env_name = env_name
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._num_disc_per_obs_dim = num_disc_per_obs_dim
        self._env_info = {} if env_info is None else env_info

    @property
    def env_name(self):
        return self._env_name

    @property
    def obs_dim(self):
        return self._obs_dim

    @property
    def act_dim(self):
        return self._act_dim

    @property
    def num_disc_per_obs_dim(self):
        return self._num_disc_per_obs_dim

    @property
    def env_info(self):
        return self._env_info
