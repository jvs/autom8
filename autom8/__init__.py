__version__ = '0.0.1'

from .cleaning import clean_dataset
from .csv_utils import load_csv, decode_csv
from .evaluate import evaluate_pipeline
from .exceptions import Autom8Exception, Autom8Warning
from .inference import infer_roles
from .matrix import create_matrix, Matrix

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
from .search import search
from .training import create_training_context
