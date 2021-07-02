import copy


def _get_recursive(dict_: dict, keys: list):
    v = dict_
    for k in keys:
        v = v[k]
    return v


def build_base_and_highway_networks(network_config, target_keys, highway_keys, highways, build_fn, extra_build_parameters):
    base_network = build_fn(network_config, *extra_build_parameters)
    highways_config = _get_recursive(network_config, highway_keys)
    highway_networks = []
    for highway in highways:
        highway_network_config = copy.deepcopy(network_config)
        target_config = _get_recursive(highway_network_config, target_keys)
        highway_config = highways_config[highway]
        target_config.update(highway_config)
        highway_networks.append(build_fn(highway_network_config, *extra_build_parameters))
    return base_network, *highway_networks
