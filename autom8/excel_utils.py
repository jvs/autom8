from functools import reduce


def excel_column_index(name):
    """Takes the Excel-style name of a column and returns its 0-based index.

    >>> excel_column_index('A')
    0
    >>> excel_column_index('B')
    1
    >>> excel_column_index('Z')
    25
    >>> excel_column_index('AA')
    26
    >>> excel_column_index('AZ')
    51
    >>> excel_column_index('BC')
    54
    """

    A = ord('A')
    return reduce(lambda acc, c: acc * 26 + (ord(c) - A + 1), name, 0) - 1


def excel_column_name(num):
    """Takes the 0-based index of a column, and returns its Excel-style name.

    >>> excel_column_name(0)
    'A'
    >>> excel_column_name(1)
    'B'
    >>> excel_column_name(23)
    'X'
    >>> excel_column_name(26)
    'AA'
    >>> excel_column_name(57)
    'BF'
    """

    A = ord('A')
    result = []
    remaining = num + 1
    while remaining > 0:
        modulo = (remaining - 1) % 26
        result.append(chr(A + modulo))
        remaining = (remaining - modulo) // 26
    return ''.join(reversed(result))
