PY=python

.PHONY: setup test lint format precommit clean

setup:
	$(PY) run.py setup

test:
	$(PY) run.py test

lint:
	pip install -q flake8
	flake8 src

format:
	pip install -q black isort
	isort --profile black src
	black src

precommit:
	pip install -q pre-commit
	pre-commit run --all-files

clean:
	$(PY) run.py clean

