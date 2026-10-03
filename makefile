.PHONY: install lint security unit-tests integration-tests

install:
	uv sync --all-packages --all-extras --group dev
	uvx library-skills
	uv run prek install -f

lint:
	uv run --all-packages ruff check . --fix
	uv run --all-packages ruff format .
	uv run --all-packages ty check .

security:
	uv run --all-packages bandit -c pyproject.toml -r src/trustml projects/*/src --severity-level=medium --confidence-level=medium -f screen

unit-tests:
	uv run --all-packages coverage run -m pytest tests/unit projects/*/tests/unit -v --tb=short
	uv run --all-packages coverage report

# Downloads real model weights into the Hugging Face / noentenc caches on first run.
integration-tests:
	NOENTENC_RUN_INTEGRATION=1 uv run --all-packages pytest tests/integrations -v --tb=short
