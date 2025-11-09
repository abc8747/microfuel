fmt:
    uv run ruff check src scripts --fix-only
    uv run ruff format src scripts

dump:
    u dump -i docs/assets -i data -i mkdocs.yml -i pyproject.toml -i justfile

train:
    #!/usr/bin/env bash
    for seed in 19; do
        uv run scripts/main.py create-splits phase1 --seed $seed
        for beta in 0.9 0.999; do
            # uv run scripts/main.py evaluate data/checkpoints/gdn-all_ac-v0.0.9+seed${seed}+cb${beta}+dev1/best.pt
            uv run scripts/main.py train --exp-name gdn-all_ac-v0.0.9+seed${seed}+cb${beta}+dev1 --beta ${beta}
        done
    done
