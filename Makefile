autoformat:
	black robustnessgym/ tests/
	isort --atomic robustnessgym/ tests/
	docformatter --in-place --recursive robustnessgym tests

lint:
	isort -c robustnessgym/ tests/
	black robustnessgym/ tests/ --check
	flake8 robustnessgym/ tests/

test:
	pytest

test-cov:
	pytest --cov=./ --cov-report=xml

docs:
	sphinx-build -b html docs/source/ docs/build/html/

docs-check:
	sphinx-build -b html docs/source/ docs/build/html/ -W

dev:
	pip install black isort flake8 docformatter pytest-cov

poetry-setup:
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py
	python get-poetry.py -y
	echo "$$HOME/.poetry/bin" >> $$GITHUB_PATH

poetry-install:
	poetry install --no-interaction
	poetry shell --no-interaction

all: autoformat lint docs test


