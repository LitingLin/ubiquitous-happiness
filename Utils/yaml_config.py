import yaml


def _merge_dict(config: dict, default_config: dict):
    merged = {}
    for key, value in default_config.items():
        if key in config:
            if isinstance(config[key], dict) and isinstance(value, dict):
                value = _merge_dict(config[key], value)
            else:
                value = config[key]
        merged[key] = value
    for key, value in config.items():
        if key not in merged:
            merged[key] = value
    return merged


def load_config(default_config_path: str, path: str=None):
    with open(default_config_path, 'rb') as fid:
        default_config = yaml.safe_load(fid)
    if path is None:
        return default_config
    with open(path, 'rb') as fid:
        config = yaml.safe_load(fid)
    return _merge_dict(config, default_config)
