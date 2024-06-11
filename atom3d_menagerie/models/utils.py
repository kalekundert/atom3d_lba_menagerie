def extract_state_dict(state_dict, prefix):
    i = len(prefix)
    return {
            k[i:]: v
            for k, v in state_dict.items()
            if k.startswith(prefix)
    }
