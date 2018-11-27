import csv
from io import StringIO

import chardet
from .parsing import parse_number


def read_csv(path):
    """Reads the CSV file at the indicated path and returns a list of rows.

    Parameters:
        path (str): The path to a CSV file.

    Returns:
        list[row]: A list of rows. Each row is a list of strings and numbers.
    """

    with open(path, 'rb') as f:
        return decode_csv(f.read())


def decode_csv(payload):
    """Takes the contents of a CSV file and returns a list of rows.

    Parameters:
        payload (bytes or str): The contents of a CSV file.

    Returns:
        list[row]: A list of rows. Each row is a list of strings and numbers.
    """

    if isinstance(payload, bytes):
        payload = payload.decode(chardet.detect(payload)['encoding'])

    # SHOULD: Make this work for quoted values with newlines in the first row.
    dialect = csv.Sniffer().sniff(payload.split('\n')[0])
    reader = csv.reader(StringIO(payload), dialect)
    return [[_convert_csv_cell(i) for i in row] for row in reader]


def encode_csv(dataset):
    """Takes a list of rows and returns it as a CSV string.

    Parameters:
        dataset (list[row]): A list of rows. Each row is a list of primitive
            values.

    Returns:
        str: A CSV representation of the dataset.
    """

    result = StringIO()
    writer = csv.writer(result, quoting=csv.QUOTE_MINIMAL)
    for row in dataset:
        writer.writerow(row)
    return result.getvalue()


def _convert_csv_cell(cell):
    try:
        return parse_number(cell)
    except Exception:
        return cell
