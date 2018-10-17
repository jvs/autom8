import csv
from io import StringIO
import chardet


def load_csv(path):
    with open(path) as f:
        return decode_csv(f.read('rb'))


def decode_csv(payload):
    if isinstance(payload, bytes):
        payload = payload.decode(chardet.detect(payload)['encoding'])
    # SHOULD: Make this work for quoted values with newlines in the first row.
    dialect = csv.Sniffer().sniff(payload.split('\n')[0])
    reader = csv.reader(StringIO(payload), dialect)
    return [[_convert_csv_cell(i) for i in row] for row in reader]


def _convert_csv_cell(cell):
    try:
        return int(cell)
    except ValueError:
        pass

    try:
        return float(cell)
    except ValueError:
        return cell
