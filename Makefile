SRC=psp notebooks

.PHONY: notebook
notebook:
	CWD=`pwd` poetry run jupyter notebook --notebook-dir notebooks

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
