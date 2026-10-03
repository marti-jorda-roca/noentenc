.PHONY: install lint security unit-tests integration-tests

install:
	uv sync --all-extras --group dev
	uvx library-skills
	uv run prek install -f

lint:
	uv run ruff check . --fix
	uv run ruff format .
	uv run ty check .

security:
	uv run bandit -c pyproject.toml -r src/noentenc --severity-level=medium --confidence-level=medium -f screen

unit-tests:
	uv run coverage run -m pytest tests/unit -v --tb=short
	uv run coverage report

# Downloads real model weights into the Hugging Face / noentenc caches on first run.
integration-tests:
	NOENTENC_RUN_INTEGRATION=1 uv run pytest tests/integrations -v --tb=short
