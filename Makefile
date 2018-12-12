BIN := .venv/bin
COVERAGE := $(BIN)/coverage
PIP := $(BIN)/pip
PYTHON := $(BIN)/python -W "ignore::PendingDeprecationWarning"

TEST_FLAGS := -s -vv -W "ignore::PendingDeprecationWarning" \
	--doctest-modules --doctest-continue-on-failure --log-cli-level INFO
TEST := $(BIN)/pytest $(TEST_FLAGS)

blackbox-tests: clean venv
	$(TEST) tests/blackbox-tests

coverage: clean
	$(COVERAGE) run --source autom8 -m pytest $(TEST_FLAGS)
	$(COVERAGE) report
	$(COVERAGE) html
	open htmlcov/index.html

repl:
	$(PYTHON) -i -c "import autom8;import numpy as np"

test: unit-tests

unit-tests: clean venv
	$(TEST) autom8 tests/unit-tests

venv: .venv/bin/activate

.venv/bin/activate: setup.py
	test -d .venv || python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -U -e .
	$(PIP) install -U xgboost coverage pytest
	$(PIP) install -U sphinx recommonmark sphinxcontrib-napoleon sphinx_rtd_theme
	touch .venv/bin/activate


# Individual blackbox tests:

boston-test: clean venv
	$(TEST) tests/blackbox-tests/test_boston_dataset.py

iris-test: clean venv
	$(TEST) tests/blackbox-tests/test_iris_dataset.py

package-test: clean venv
	$(TEST) tests/blackbox-tests/test_create_package.py

wine-test: clean venv
	$(TEST) tests/blackbox-tests/test_wine_dataset.py


html-docs:
	$(BIN)/sphinx-build -M "html" docs docs/build


clean:
	rm -rf ./autom8/__pycache__/*.pyc ./tests/*/__pycache__/*.pyc

.PHONY: blackbox-tests boston-test clean coverage repl test unit-tests




tmp-test:
	$(TEST) tests/unit-tests/test_cleaning.py::test_column_of_ints_and_floats
