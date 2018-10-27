import csv
from io import StringIO

import chardet
from .parsing import parse_number


def read_csv(path):
    with open(path, 'rb') as f:
        return decode_csv(f.read())


def decode_csv(payload):
    if isinstance(payload, bytes):
        payload = payload.decode(chardet.detect(payload)['encoding'])

    # SHOULD: Make this work for quoted values with newlines in the first row.
    dialect = csv.Sniffer().sniff(payload.split('\n')[0])
    reader = csv.reader(StringIO(payload), dialect)
    return [[_convert_csv_cell(i) for i in row] for row in reader]


def encode_csv(dataset):
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
