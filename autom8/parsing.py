def parse_number(obj):
    obj = obj.replace(',', '')

    try:
        return int(obj)
    except Exception:
        pass

    try:
        return float(obj)
    except Exception:
        pass

    try:
        if obj.startswith(('0x', '0X')):
            return int(obj, 16)
    except Exception:
        pass

    raise Exception(f'invalid number literal: {repr(obj)}')
