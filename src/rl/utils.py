from gymnasium import spaces

from grid2op_env.observation_converter import NODES, EDGE_INDEX, EDGE_MASK


def assert_graph_obs_space_and_get_x_dim(obs_space: spaces.Dict) -> int:
    """
    Checks that the given dict space is a graph obs space and returns the node feature dimension.
    :param obs_space: the observation space to check,
    :return: the node feature dimension (x_dim) if the checks pass
    :raise AssertionError: if the obs_space does not contain node features, edge index and edge mask subspaces
    """

    assert NODES in obs_space.spaces, f"obs_space must contain '{NODES}' key for node features"
    assert EDGE_INDEX in obs_space.spaces, f"obs_space must contain '{EDGE_INDEX}' key for edge indices"
    assert EDGE_MASK in obs_space.spaces, f"obs_space must contain '{EDGE_MASK}' key for edge masks"

    _, x_dim = obs_space[NODES].shape
    return x_dim