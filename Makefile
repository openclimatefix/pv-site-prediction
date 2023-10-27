SRC=psp notebooks
PORT_notebook=8868
PORT_lab=8869


.PHONY: notebook
notebook:
	CWD=`pwd` poetry run jupyter notebook --notebook-dir notebooks --ip 0.0.0.0 --port $(PORT_notebook) --no-browser


.PHONY: lab
lab:
	CWD=`pwd` poetry run jupyter lab --notebook-dir notebooks --ip 0.0.0.0 --port $(PORT_lab) --no-browser


.PHONY: test
test:
	poetry run pytest psp/tests \
		-n auto \
		--maxprocesses 8 \
		--verbose \
		--durations=10 \
		$(ARGS) 


.PHONY: format
format:
	poetry run black $(SRC)
	poetry run ruff --fix $(SRC)


# Same as `format` but without editing the files. Useful for CI.
.PHONY: lint
lint:
	poetry run black --check $(SRC)
	poetry run ruff $(SRC)
	poetry run mypy psp