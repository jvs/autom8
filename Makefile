test:
	pipenv run python3 -W ignore::PendingDeprecationWarning -m unittest discover -s tests
