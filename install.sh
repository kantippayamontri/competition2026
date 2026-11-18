#!/bin/sh
set -eu

CDPATH= cd -- "$(dirname -- "$0")"
if [ ! -f .venv/pyvenv.cfg ] || ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    python3 -m venv .venv
fi

.venv/bin/python -m pip install torch torchvision tqdm Pillow jupyterlab ipykernel
.venv/bin/python -m ipykernel install --user \
    --name competition2026 --display-name "Python (IDRiD 2026)"

cat <<EOF
From the repository root ($PWD), run:
  .venv/bin/jupyter-lab train.ipynb
If your class JupyterLab is already running under this account, refresh the page,
open train.ipynb, and select Python (IDRiD 2026).
EOF
