#!/bin/sh
set -eu

CDPATH= cd -- "$(dirname -- "$0")"
printf '%s\n' '[---] 0/3  Preparing the Python environment...'

# Large installs can be silent while pip resolves or unpacks packages.
python3 -u -c '
import time
start = time.monotonic()
while True:
    time.sleep(15)
    print(f"      Setup still running ({time.monotonic() - start:.0f}s elapsed).")
' >&2 &
heartbeat_pid=$!
trap 'kill "$heartbeat_pid" 2>/dev/null || true; wait "$heartbeat_pid" 2>/dev/null || true' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if [ ! -f .venv/pyvenv.cfg ] || ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    python3 -m venv .venv
fi

printf '%s\n' '[#--] 1/3  Installing training and Jupyter packages...'
.venv/bin/python -m pip install --progress-bar on torch torchvision tqdm Pillow jupyterlab ipykernel
printf '%s\n' '[##-] 2/3  Registering the Jupyter kernel...'
.venv/bin/python -m ipykernel install --user \
    --name competition2026 --display-name "Python (IDRiD 2026)"

cat <<EOF
[###] 3/3  Installation complete.
From the repository root ($PWD), run:
  .venv/bin/jupyter-lab train.ipynb
If your class JupyterLab is already running under this account, refresh the page,
open train.ipynb, and select Python (IDRiD 2026).
EOF
