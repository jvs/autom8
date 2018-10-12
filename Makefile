repl:
	pipenv run python3 -W ignore::PendingDeprecationWarning -i -c "import autom8"

test:
	pipenv run python3 -W ignore::PendingDeprecationWarning -m unittest discover -s tests
