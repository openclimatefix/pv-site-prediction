SRC=psp notebooks
PORT=8864

.PHONY: notebook
notebook:
	CWD=`pwd` poetry run jupyter notebook --notebook-dir notebooks --ip 0.0.0.0 --port $(PORT) --no-browser

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
