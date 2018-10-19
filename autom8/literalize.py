from collections import Iterable, OrderedDict


def literalize(obj):
    if _is_literal(obj):
        return obj

    if hasattr(obj, 'tolist'):
        return literalize(obj.tolist())

    if isinstance(obj, tuple) and hasattr(obj, '_asdict'):
        result = literalize(dict(obj._asdict()))
        if '__class__' not in result:
            result['__class__'] = type(obj).__name__
        return result

    if isinstance(obj, dict):
        return {literalize(k): literalize(v) for k, v in obj.items()}

    if isinstance(obj, tuple):
        return tuple(literalize(i) for i in obj)

    if isinstance(obj, Iterable):
        return [literalize(i) for i in obj]

    return repr(obj)


def _is_literal(obj):
    if _is_primitive(obj):
        return True

    if type(obj) in (list, tuple) and all(_is_literal(i) for i in obj):
        return True

    if type(obj) in {dict, OrderedDict} and all(_is_literal(kv) for kv in obj.items()):
        return True

    return False


def _is_primitive(obj):
    return obj is None or type(obj) in {bool, int, float, str}

    if type(obj) in (list, tuple) and all(_is_primitive(i) for i in obj):
        return True
