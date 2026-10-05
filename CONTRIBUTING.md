# Contributing

Report reproducible bugs through the [issue templates](https://github.com/marti-jorda-roca/noentenc/issues/new/choose). Use [Discussions](https://github.com/marti-jorda-roca/noentenc/discussions) for questions about installation, detection backends, translation models, and model weights.

For security issues, follow [SECURITY.md](SECURITY.md).

## Development

Use Python 3.14 for development (the package supports 3.11 and newer) and [uv](https://docs.astral.sh/uv/). Fork the repository, clone your fork, and create a branch for your change.

```bash
uv sync --locked --all-extras --group dev
uv run ruff check .
uv run ruff format --check .
uv run ty check .
uv run coverage run -m pytest tests/unit --strict-config --strict-markers -v --tb=short
```

The `heliport` extra is unavailable on Windows. Dependency markers skip it there.

CI runs the unit tests on Linux, macOS and Windows with every supported Python version, and runs the quickstart from the built wheel in a clean environment. To try another version locally, run `uv run --python 3.11 --isolated --all-extras --group dev pytest tests/unit`. `scripts/smoke_quickstart.py` has the commands for the wheel check.

Install [Trivy](https://trivy.dev/docs/latest/getting-started/installation/) to run the security check:

```bash
trivy fs --config trivy.yaml .
```

Integration tests download real model weights and can take substantial disk space and time. Run the relevant suite when changing model loading or inference:

```bash
NOENTENC_RUN_INTEGRATION=1 uv run pytest tests/integrations/language_detection -v --tb=short
NOENTENC_RUN_INTEGRATION=1 uv run pytest tests/integrations/translation -v --tb=short
```

## Pull requests

Target `main`. Explain the problem, the resulting behavior, and how you checked the change. Include a regression test for bug fixes and update examples or documentation when the public API changes.

Keep unrelated changes in separate pull requests. Do not commit model weights, caches, credentials, or generated build outputs. Use synthetic text in examples and tests when real data contains personal information.

Pull requests run linting, type checks, unit tests, and a security scan. A maintainer approves workflow runs from external contributors. Integration tests and distribution checks also run before releases.

The project uses Apache-2.0. Model weights have their own licenses, listed in the README. Include the source and license when proposing a new model backend or fixture.
