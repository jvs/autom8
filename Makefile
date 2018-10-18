COVERAGE := .virtualenv/bin/coverage
PYTHON := .virtualenv/bin/python -W "ignore::PendingDeprecationWarning"
TEST_FLAGS = -m unittest discover -v -s
TEST := $(PYTHON) $(TEST_FLAGS)


blackbox-tests: virtualenv
	$(TEST) tests/blackbox-tests

# TODO: Make coverage ignore PendingDeprecationWarning.
coverage:
	$(COVERAGE) run --source autom8 $(TEST_FLAGS) tests/unit-tests
	$(COVERAGE) report
	$(COVERAGE) html
	open htmlcov/index.html

repl:
	$(PYTHON) -i -c "import autom8;import numpy as np"

test: unit-tests

unit-tests: virtualenv
	$(TEST) tests/unit-tests

virtualenv: .virtualenv/bin/activate

.virtualenv/bin/activate: setup.py
	test -d .virtualenv || virtualenv .virtualenv
	.virtualenv/bin/pip install -U -e .
	touch .virtualenv/bin/activate


# Individual blackbox tests:

boston-test: virtualenv
	$(TEST) tests/blackbox-tests -p test_boston_dataset.py

iris-test: virtualenv
	$(TEST) tests/blackbox-tests -p test_iris_dataset.py

wine-test: virtualenv
	$(TEST) tests/blackbox-tests -p test_wine_dataset.py


.PHONY: blackbox-tests boston-test coverage repl test unit-tests virtualenv
