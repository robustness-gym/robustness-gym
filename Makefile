autoformat:
	black robustnessgym/ tests/
	isort --atomic -rc robustnessgym/ tests/
	docformatter --in-place --recursive robustnessgym tests

lint:
	isort -c -rc robustnessgym/ tests/
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

all: autoformat lint docs test


