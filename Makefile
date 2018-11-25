COVERAGE := .virtualenv/bin/coverage
PYTHON := .virtualenv/bin/python -W "ignore::PendingDeprecationWarning"
TEST_FLAGS := -s -vv -W "ignore::PendingDeprecationWarning"
TEST := .virtualenv/bin/pytest $(TEST_FLAGS)

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
	$(TEST) tests/unit-tests

virtualenv: .virtualenv/bin/activate

.virtualenv/bin/activate: setup.py
	test -d .virtualenv || virtualenv .virtualenv
	.virtualenv/bin/pip install -U -e .
	.virtualenv/bin/pip install -U xgboost coverage pytest
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


clean:
	rm -rf ./autom8/__pycache__/*.pyc ./tests/*/__pycache__/*.pyc

.PHONY: blackbox-tests boston-test clean coverage repl test unit-tests virtualenv
