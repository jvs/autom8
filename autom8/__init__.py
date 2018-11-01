__version__ = '0.0.1'

from .cleaning import clean_dataset
from .context import create_context
from .csv_utils import decode_csv, encode_csv, read_csv
from .evaluate import evaluate_pipeline
from .exceptions import Autom8Exception, Autom8Warning
from .fit import fit
from .inference import infer_roles

from .matrix import (
    create_matrix, excel_column_index, excel_column_name, Matrix,
)

from .pipeline import Pipeline

from .preprocessors import (
    add_column_of_ones,
    binarize_fractions,
    binarize_signs,
    coerce_columns,
    divide_columns,
    drop_duplicate_columns,
    drop_weak_columns,
    encode_categories,
    encode_text,
    logarithm_columns,
    multiply_columns,
    planner,
    preprocessor,
    scale_columns,
    square_columns,
    sqrt_columns,
)

from .receiver import Accumulator, Receiver
