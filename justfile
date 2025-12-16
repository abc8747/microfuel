fmt:
    uv run ruff check src scripts --fix-only
    uv run ruff format src scripts

dump:
    u dump -i docs/assets -i data -i mkdocs.yml -i pyproject.toml -i justfile

train:
    #!/usr/bin/env bash
    for seed in 19; do
        uv run scripts/main.py create-splits phase1 --seed $seed
        for beta in 0.99; do
            # uv run scripts/main.py evaluate data/checkpoints/gdn-all_ac-v0.0.12+seed${seed}+cb${beta}+dev3/best.pt
            uv run scripts/main.py train --exp-name gdn-all_ac-v0.0.12+seed${seed}+cb${beta}+dev4 --beta ${beta}
        done
    done

migrate:
    uv run scripts/migrate.py data/checkpoints/backup/gdn-all_ac-v0.0.10+seed19+cb0.99+dev1/step12800_220.21.pt data/checkpoints/v1.0-realtime/best_model.pt
    uv run scripts/migrate.py data/checkpoints/backup/gdn-all_ac-v0.0.12+seed19+cb0.99+dev2/step28725_198.07.pt data/checkpoints/v1.0-offline/best_model.pt
    uv run scripts/migrate.py data/checkpoints/backup/gdn-all_ac-v0.0.12+seed19+cb0.99+dev6/step62000_191.47.pt data/checkpoints/v1.0-offline-tas/best_model.pt
