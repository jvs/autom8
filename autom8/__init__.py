__version__ = '0.0.5'

from .cleaning import clean_dataset
from .context import create_context, RecordingContext, Labels
from .exceptions import Autom8Exception, Autom8Warning
from .formats import read_csv
from .inference import infer_roles
from .main import fit, run

from .matrix import (
    create_matrix,
    Column,
    Matrix,
)

from .packager import create_example_input, create_package
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
    preprocessor,
    scale_columns,
    square_columns,
    sqrt_columns,
)

from .receiver import Accumulator, Receiver
