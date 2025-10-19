fmt:
    uv run ruff check src scripts --fix-only
    uv run ruff format src scripts

dump:
    u dump -i docs/assets -i data -i mkdocs.yml -i pyproject.toml -i justfile