def parse_number(string):
    """Takes a string and attempts to convert it to a number.

    This function simply removes commas and underscores from the string, and
    then tries to convert it to an int or a float.

    >>> parse_number('10')
    10
    >>> parse_number('1.4')
    1.4
    >>> parse_number('12,345')
    12345
    >>> parse_number('987_654_321')
    987654321
    >>> parse_number('0xff')
    255
    >>> parse_number('0XFE')
    254
    """

    string = string.replace(',', '').replace('_', '')

    try:
        return int(string)
    except Exception:
        pass

    try:
        return float(string)
    except Exception:
        pass

    try:
        if string.startswith(('0x', '0X')):
            return int(string, 16)
    except Exception:
        pass

    raise Exception(f'invalid number literal: {repr(obj)}')
