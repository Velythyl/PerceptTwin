def recursive_map(data, func):
    if isinstance(data, dict):
        return {k: recursive_map(v, func) for k, v in data.items()}
    elif isinstance(data, (list, tuple, set)):
        return type(data)(recursive_map(v, func) for v in data)
    else:
        return func(data)