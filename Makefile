BIN := .virtualenv/bin
COVERAGE := $(BIN)/coverage
PIP := $(BIN)/pip
PYTHON := $(BIN)/python -W "ignore::PendingDeprecationWarning"

TEST_FLAGS := -s -vv -W "ignore::PendingDeprecationWarning" \
	--doctest-modules --doctest-continue-on-failure
TEST := $(BIN)/pytest $(TEST_FLAGS)

blackbox-tests: clean virtualenv
	$(TEST) tests/blackbox-tests

coverage: clean
	$(COVERAGE) run --source autom8 -m pytest $(TEST_FLAGS)
	$(COVERAGE) report
	$(COVERAGE) html
	open htmlcov/index.html

repl:
	$(PYTHON) -i -c "import autom8;import numpy as np"

test: unit-tests

unit-tests: clean virtualenv
	$(TEST) autom8 tests/unit-tests

virtualenv: .virtualenv/bin/activate

.virtualenv/bin/activate: setup.py
	test -d .virtualenv || virtualenv .virtualenv
	$(PIP) install -U -e .
	$(PIP) install -U xgboost coverage pytest
	$(PIP) install -U sphinx recommonmark sphinxcontrib-napoleon sphinx_rtd_theme
	touch .virtualenv/bin/activate


# Individual blackbox tests:

boston-test: clean virtualenv
	$(TEST) tests/blackbox-tests/test_boston_dataset.py

iris-test: clean virtualenv
	$(TEST) tests/blackbox-tests/test_iris_dataset.py

package-test: clean virtualenv
	$(TEST) tests/blackbox-tests/test_create_package.py

wine-test: clean virtualenv
	$(TEST) tests/blackbox-tests/test_wine_dataset.py


html-docs:
	$(BIN)/sphinx-build -M "html" docs docs/build


clean:
	rm -rf ./autom8/__pycache__/*.pyc ./tests/*/__pycache__/*.pyc

.PHONY: blackbox-tests boston-test clean coverage repl test unit-tests virtualenv
