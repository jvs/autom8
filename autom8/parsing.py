def parse_number(obj):
    try:
        return int(obj)
    except Exception:
        pass

    try:
        return float(obj)
    except Exception:
        pass

    try:
        return int(obj, 16)
    except Exception:
        pass

    raise Exception(f'invalid number literal: {repr(obj)}')