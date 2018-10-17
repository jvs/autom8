PYTHON := pipenv run python3 -W "ignore::PendingDeprecationWarning"
TEST := $(PYTHON) -m unittest discover -s


blackbox-tests:
	$(TEST) tests/blackbox-tests


repl:
	$(PYTHON) -i -c "import autom8;import numpy as np"


test: unit-tests


unit-tests:
	$(TEST) tests/unit-tests
