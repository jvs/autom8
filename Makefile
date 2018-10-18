PYTHON := .virtualenv/bin/python -W "ignore::PendingDeprecationWarning"
TEST := $(PYTHON) -m unittest discover -v -s

blackbox-tests:
	$(TEST) tests/blackbox-tests

boston-test:
	$(TEST) tests/blackbox-tests -p test_boston_dataset.py

repl:
	$(PYTHON) -i -c "import autom8;import numpy as np"

test: unit-tests

unit-tests: virtualenv
	$(TEST) tests/unit-tests

virtualenv: .virtualenv/bin/activate

.virtualenv/bin/activate: requirements.txt
	test -d .virtualenv || virtualenv .virtualenv
	.virtualenv/bin/pip install -Ur requirements.txt
	touch .virtualenv/bin/activate
