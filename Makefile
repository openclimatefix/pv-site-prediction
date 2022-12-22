SRC=psp notebooks

.PHONY: init
init:
	poetry install

.PHONY: notebook
notebook:
	CWD=`pwd` poetry run jupyter notebook --notebook-dir notebooks --ip 0.0.0.0

.PHONY: test
test:
	poetry run pytest psp/tests $(ARGS)


.PHONY: format
format:
	poetry run black $(SRC)
	poetry run isort $(SRC)


.PHONY: lint
lint:
	poetry run flake8 $(SRC)
	poetry run mypy psp
	poetry run pydocstyle $(SRC)
