PYTHON := .virtualenv/bin/python -W "ignore::PendingDeprecationWarning"
TEST := $(PYTHON) -m unittest discover -v -s

blackbox-tests:
	$(TEST) tests/blackbox-tests

repl:
	$(PYTHON) -i -c "import autom8;import numpy as np"

test: unit-tests

test-boston:
	$(TEST) tests/blackbox-tests -p test_boston_dataset.py

unit-tests: virtualenv
	$(TEST) tests/unit-tests

virtualenv: .virtualenv/bin/activate

.virtualenv/bin/activate: requirements.txt
	test -d .virtualenv || virtualenv .virtualenv
	.virtualenv/bin/pip install -Ur requirements.txt
	touch .virtualenv/bin/activate
