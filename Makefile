repl:
	pipenv run python3 -W ignore::PendingDeprecationWarning -i -c "import autom8"

unit-tests:
	pipenv run python3 \
		-W ignore::PendingDeprecationWarning \
		-m unittest discover \
		-s tests/unit-tests

test: unit-tests
