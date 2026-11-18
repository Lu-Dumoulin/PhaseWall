#!/usr/bin/env bash
# Start the phasewall explorer (apps/explore.py) in the browser, from any folder:
#
#   ./explore.sh            dashboard: the controls and the results
#   ./explore.sh --edit     notebook: the code of every cell is visible and editable
#   ./explore.sh --port 2720
#
# The first time, it makes the environment .venv next to this file with Python 3.11 or newer and
# installs phasewall with the extra [app] (marimo, PyTorch, scikit-learn: a few minutes); after
# that it starts at once. No need to activate anything: the script uses .venv directly.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

if [ ! -x .venv/bin/python ]; then
    py=""
    for candidate in python3.13 python3.12 python3.11 python3; do
        if command -v "$candidate" >/dev/null 2>&1 &&
           "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
            py="$candidate"
            break
        fi
    done
    if [ -z "$py" ]; then
        echo "phasewall needs Python 3.11 or newer (for example: brew install python@3.12)" >&2
        exit 1
    fi
    echo "Making the environment .venv with $("$py" --version) ..."
    "$py" -m venv .venv
fi

if ! .venv/bin/python -c "import marimo, phasewall" >/dev/null 2>&1; then
    echo "Installing phasewall with the interface (marimo, PyTorch, scikit-learn) into .venv ..."
    .venv/bin/python -m pip install --quiet --upgrade pip
    .venv/bin/python -m pip install --quiet -e ".[app]"
fi

mode="run"
args=()
for a in "$@"; do
    if [ "$a" = "--edit" ]; then mode="edit"; else args+=("$a"); fi
done
exec .venv/bin/marimo "$mode" apps/explore.py ${args[@]+"${args[@]}"}
