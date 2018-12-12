import io
import json
import pickle
import pprint
from datetime import date
from string import Template
from zipfile import ZipFile

import numpy
import pandas
import sklearn
import scipy

try:
    import lightgbm
except ImportError:
    lightgbm = None

try:
    import xgboost
except ImportError:
    xgboost = None

from . import __version__
from .docstrings import render_docstring
from .matrix import create_matrix


@render_docstring
def create_example_input(pipeline, dataset, indices, receiver=None):
    """Creates an example input object that you can pass to `create_package`.

    Parameters:
        pipeline (Pipeline): An `autom8.Pipeline` object.
        dataset (list or Matrix): The dataset used to create the pipeline.
        indices (list[int]): A list of row indices. Indicates which rows to use
            in the example input.
        $receiver_parameter

    Returns:
        list: A list of rows that can be passed to ``autom8.create_package()``
        as the `example_input` argument.
    """

    matrix = create_matrix(dataset, receiver=receiver)
    target_column = matrix.column_names.index(pipeline.predicts_column)
    matrix.drop_columns_by_index(target_column)
    return [pipeline.input_columns] + matrix.select_rows(indices).tolist()


def create_package(package_name, pipeline, example_input, extra_notes=None):
    """Creates a Python package for an `autom8.Pipeline` object.

    This function creates a zip archive and returns it as a `bytes` object.
    The zip archive contains a Python project that runs a web service for
    making predictions with the provided pipeline.

    The zip archive contains a pickle file of the pipeline, along with a flask
    application, test script, Dockerfile, and Makefile.

    Parameters:
        package_name (str): The name to use for this package.

            This name will be used as the archive's top-level folder. It will
            also be used as the name of the Docker image and Docker container.

        pipeline (Pipeline): The pipeline. autom8 will serialize this object
            in the achive's `pipeline.pickle` file.

        example_input (list[row]): A list of rows to use as the example input.

            This list of rows will appear in the README file and in the
            unit tests.

        extra_notes (str or None): Optional content that will appear at the end
            of the generated README file. Defaults to None.

    Returns:
        bytes: The contents of the generated zip archive.

        You may write the bytes to disk as a zip file, or transmit the bytes
        over the network.
    """

    args = _template_args(package_name, pipeline, example_input, extra_notes)
    result = io.BytesIO()
    templates = {
        '.dockerignore': dockerignore,
        'Dockerfile': dockerfile,
        'LICENSE': license,
        'Makefile': makefile,
        'README.md': readme,
        'requirements.txt': requirements,
        'service.py': service,
        'tests.py': tests,
    }
    with ZipFile(result, 'w') as out:
        out.writestr(f'{package_name}/pipeline.pickle', pickle.dumps(pipeline))
        for file_name, template in templates.items():
            contents = Template(template).substitute(args).strip() + '\n'
            out.writestr(f'{package_name}/{file_name}', contents.encode('utf-8'))
    return result.getvalue()


def _template_args(package_name, pipeline, example_input, extra_notes):
    est = pipeline.estimator
    extra_packages = []

    if lightgbm is not None:
        if isinstance(est, (lightgbm.LGBMRegressor, lightgbm.LGBMClassifier)):
            extra_packages.append(f'lightgbm=={lightgbm.__version__}')

    if xgboost is not None:
        if isinstance(est, (xgboost.XGBRegressor, xgboost.XGBClassifier)):
            extra_packages.append(f'xgboost=={xgboost.__version__}')

    example_result = pipeline.run(example_input)
    example_output = {
        'columnName': f'Predicted {pipeline.predicts_column}',
        'predictions': example_result.predictions,
        'probabilities': example_result.probabilities,
    }

    return {
        'DOCKER_NAME': package_name,
        'ESTIMATOR_CLASS': type(pipeline.estimator).__name__,
        'EXTRA_NOTES': extra_notes or '',
        'EXTRA_PACKAGES': '\n'.join(extra_packages),
        'PACKAGE_NAME': package_name,
        'PACKAGE_NAME_REPR': repr(package_name),
        'PREDICTED_COLUMN': pipeline.predicts_column,
        'PREDICTED_COLUMN_REPR': repr(f'Predicted {pipeline.predicts_column}'),
        'README_INPUT_COLUMNS': '\n'.join(f'  -  {c}'
            for c in pipeline.input_columns),
        'README_INPUT_EXAMPLE': json
            .dumps({'rows': example_input}, sort_keys=True, indent=2)
            .replace("'", '\'\"\'\"\'').replace('\n', '\n        '),
        'README_OUTPUT_EXAMPLE': json
            .dumps(example_output, sort_keys=True, indent=2)
            .replace('\n', '\n    '),
        'TEST_INPUT': pformat(example_input),
        'TEST_OUTPUT_PREDICTIONS': pformat(example_result.predictions),
    }


def pformat(obj):
    return pprint.pformat(obj).replace('\n', '\n  ')


dockerfile = '''
FROM python:3.6.4

WORKDIR /deploy
COPY requirements.txt /deploy
RUN pip3 install --no-cache-dir --disable-pip-version-check -r requirements.txt

COPY LICENSE /deploy
COPY service.py /deploy
COPY pipeline.pickle /deploy

CMD ["gunicorn", "service:app", "--workers=1", "--bind", "0.0.0.0:80"]
'''


dockerignore = '''
.pytest_cache
.venv
__pycache__
'''


license = '''
MIT License

Copyright (c) 2018 Machinate, Inc

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''


makefile = '''
# This file defines commands for running and testing your web service. Use the
# `make` program to run them (for example, type `make test` to run the unit
# tests).


# Removes any compiled Python files that may be lying around.
clean:
    rm -rf __pycache__/*.pyc


# Runs your web service in a Docker container. Maps local port 55155 to the
# container's port 80.
container: image
    docker run \\
        --name $DOCKER_NAME \\
        --rm \\
        -p 0.0.0.0:55155:80 \\
        $DOCKER_NAME


# Builds the Docker image for your web service.
image:
    docker build -t $DOCKER_NAME .


# Stops the Docker cotainer.
stop:
    docker stop $DOCKER_NAME


# Runs the unit tests. (But first it runs `make clean` and `make venv`.)
test: clean venv
    .venv/bin/pytest -s -v -W "ignore::PendingDeprecationWarning" tests.py


# Creates the `activate` script for your virtual environment.
venv: .venv/bin/activate


# Creates your virtual environment and installs the required dependencies.
.venv/bin/activate: requirements.txt
    test -d .venv || python3 -m venv .venv
    .venv/bin/pip install -U -r requirements.txt
    .venv/bin/pip install -U pytest
    touch .venv/bin/activate


# Tells `make` which commands don't actaully produce any files.
.PHONY: clean container image stop test
'''.replace('    ', '\t')


readme = '''
# $PACKAGE_NAME

This project provides a stateless web service that hosts a machine learning model.
The web service provides a `/predict` URL for running the model and predicting
`$PREDICTED_COLUMN` values.


## Files

This package includes a handful of files:

- `.dockerignore` -- Prevents Docker from loading unnecessary files into its context.
- `Dockerfile` -- Defines the Docker image for your web service.
- `LICENSE` -- The MIT license for this software.
- `Makefile` -- Defines commands for starting and stopping the container, and running tests.
- `pipeline.pickle` -- The Python pickle file of your machine learning model.
- `README.md` -- This file. Provides some basic documentation.
- `requirements.txt` -- The Python dependencies for the web service.
- `service.py` -- Defines a flask application.
- `tests.py` -- Defines unit tests for your application. Run them with `make test`.


## Model Summary

The machine learning model is serialized in the `pipeline.pickle` file.

- Estimator: $ESTIMATOR_CLASS
- Predicts: $PREDICTED_COLUMN
- Inputs:
$README_INPUT_COLUMNS


## Requirements

- make
- docker
- python3


## Commands

- `make container` -- Runs the web service in a docker container.
- `make image` -- Builds the docker image.
- `make stop` -- Stops the container.
- `make test` -- Runs the unit tests in a virtual environment.


## URLs

- `GET /` -- Simply returns the string `$PACKAGE_NAME`. May be used as a health-check.
- `GET /describe` -- Returns some metadata about the machine learning model.
- `POST /predict` -- Runs the model on the provided rows and returns the predictions.


## Using the Container

Run the command `make container` to start the web service. This command maps
local port 55155 to the container's port 80.

To get predictions, send a POST request to `/predict`:

    curl --header "Content-Type: application/json" \\
        --data '$README_INPUT_EXAMPLE' \\
        http://localhost:55155/predict

This will return:

    $README_OUTPUT_EXAMPLE


$EXTRA_NOTES
'''


requirements = f'''
autom8=={__version__}
flask==1.*
gunicorn==19.*
numpy=={numpy.__version__}
pandas=={pandas.__version__}
scikit-learn=={sklearn.__version__}
scipy=={scipy.__version__}
$EXTRA_PACKAGES
'''


# TODO: Include an autom8 receiver when calling pipeline.run.
service = '''
import pickle
import flask

app = flask.Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

with open('pipeline.pickle', 'rb') as f:
    pipeline = pickle.load(f)


@app.route('/', methods=['GET'])
def health_check():
    return $PACKAGE_NAME_REPR


@app.route('/describe', methods=['GET'])
def describe():
    classes = pipeline.predicts_classes
    if classes is not None:
        classes = classes.tolist()

    return flask.jsonify({
        'inputColumns': pipeline.input_columns,
        'predictsColumn': pipeline.predicts_column,
        'predictsClasses': classes,
        'estimatorClass': type(pipeline.estimator).__name__,
        'estimator': repr(pipeline.estimator),
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        result = pipeline.run(flask.request.json['rows'])
        return flask.jsonify({
            'columnName': f'Predicted {pipeline.predicts_column}',
            'predictions': result.predictions,
            'probabilities': result.probabilities,
        })
    except Exception:
        app.logger.exception('Failed to make predictions')
        raise
'''


tests = '''
import pickle
import service


SAMPLE_INPUT = $TEST_INPUT

SAMPLE_PREDICTIONS = $TEST_OUTPUT_PREDICTIONS


def test_pipeline_run_method():
    with open('pipeline.pickle', 'rb') as f:
        pipeline = pickle.load(f)
    result = pipeline.run(SAMPLE_INPUT)
    assert result.predictions == SAMPLE_PREDICTIONS


def test_index():
    client = service.app.test_client()
    response = client.get('/')
    assert response.data.decode('utf-8') == $PACKAGE_NAME_REPR


def test_describe():
    client = service.app.test_client()
    response = client.get('/describe')
    doc = response.get_json()
    expected_keys = [
        'inputColumns',
        'predictsColumn',
        'predictsClasses',
        'estimatorClass',
        'estimator',
    ]
    for key in expected_keys:
        assert key in doc


def test_predict():
    client = service.app.test_client()
    response = client.post('/predict', json={'rows': SAMPLE_INPUT})
    doc = response.get_json()
    assert doc['columnName'] == $PREDICTED_COLUMN_REPR
    assert doc['predictions'] == SAMPLE_PREDICTIONS
'''
