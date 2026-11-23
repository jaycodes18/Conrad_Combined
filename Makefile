.PHONY: setup check-ml dev-run notebooks check

setup:
	pip install -e "ml[dev]"

check-ml:
	cd ml && ruff check . && ruff format --check . && mypy driven_ml && python -m pytest -q

dev-run:
	cd ml && python -m driven_ml --env dev --no-discovery

notebooks:
	cd ml && python tools/build_notebook.py && python tools/build_business_notebook.py
	tmp=$$(mktemp -d) && \
	python ml/tools/run_notebook.py ml/notebooks/Driven_ML_Pipeline.ipynb $$tmp && \
	python ml/tools/run_notebook.py ml/notebooks/Driven_Product_and_Business.ipynb $$tmp && \
	cp $$tmp/Driven_ML_Pipeline.ipynb $$tmp/Driven_Product_and_Business.ipynb ml/notebooks/ && \
	rm -rf $$tmp

check: check-ml dev-run
