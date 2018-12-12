import csv
from functools import reduce
from io import StringIO

import chardet


def decode_csv(payload):
    """Takes the contents of a CSV file and returns a list of rows.

    Parameters:
        payload (bytes or str): The contents of a CSV file.

    Returns:
        list[row]: A list of rows. Each row is a list of strings and numbers.
    """

    def _convert(cell):
        try:
            return parse_number(cell)
        except Exception:
            return cell

    if isinstance(payload, bytes):
        payload = payload.decode(chardet.detect(payload)['encoding'])

    # SHOULD: Make this work for quoted values with newlines in the first row.
    dialect = csv.Sniffer().sniff(payload.split('\n')[0])
    reader = csv.reader(StringIO(payload), dialect)
    return [[_convert(i) for i in row] for row in reader]


def drop_empty_rows(rows):
    """Takes a list of rows and returns a new list without any blank rows.

    autom8 considers a row to be blank if it's totally empty, or if each item
    in the row is None, the empty string, or a string that only contain spaces.

    >>> drop_empty_rows([])
    []

    >>> drop_empty_rows([[], [1, 2, 3], [], [4, 5, 6], []])
    [[1, 2, 3], [4, 5, 6]]

    >>> drop_empty_rows([[9, 8, 7], [' ', '\t', ''], [6, 5, 4], [None, ' ']])
    [[9, 8, 7], [6, 5, 4]]
    """

    def _is_blank(obj):
        if isinstance(obj, str):
            return obj == '' or obj.isspace()
        else:
            return obj is None

    return [row for row in rows if any(not _is_blank(i) for i in row)]


def encode_csv(dataset):
    """Takes a list of rows and returns it as a CSV string.

    Parameters:
        dataset (list[row]): A list of rows. Each row is a list of primitive
            values.

    Returns:
        str: A CSV representation of the dataset.

    >>> encode_csv([[1, 2, 3], [4, 5, 6]])
    '1,2,3\\r\\n4,5,6\\r\\n'

    >>> encode_csv([['a, b, c', 'xyz', None, 1.0]])
    '"a, b, c",xyz,,1.0\\r\\n'

    >>> encode_csv([['he said, "well, ok" and then left', 'foo', 'bar']])
    '"he said, ""well, ok"" and then left",foo,bar\\r\\n'
    """

    result = StringIO()
    writer = csv.writer(result, quoting=csv.QUOTE_MINIMAL)
    for row in dataset:
        writer.writerow(row)
    return result.getvalue()


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

    >>> parse_number({})
    Traceback (most recent call last):
        ...
    TypeError: parse_number() argument must be a string
    """

    if not isinstance(string, str):
        raise TypeError('parse_number() argument must be a string')

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


def read_csv(path):
    """Reads the CSV file at the indicated path and returns a list of rows.

    Parameters:
        path (str): The path to a CSV file.

    Returns:
        list[row]: A list of rows. Each row is a list of strings and numbers.
    """

    with open(path, 'rb') as f:
        return decode_csv(f.read())
