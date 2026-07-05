#!/usr/bin/env bash
# Records examples/demo.py as an asciicast, then (optionally) converts it
# to a GIF for the README. Needs asciinema, and agg (or asciicast2gif) for
# the GIF step. Both are one-time installs:
#
#   pip install asciinema
#   # agg (fast, no docker/node required): https://github.com/asciinema/agg
#   cargo install --locked agg
#
# Usage:
#   ./examples/record_demo.sh            # records + writes demo.cast
#   ./examples/record_demo.sh --gif       # also renders demo.gif via agg
set -euo pipefail

cd "$(dirname "$0")/.."

CAST_FILE="examples/demo.cast"
GIF_FILE="examples/demo.gif"

if ! command -v asciinema >/dev/null 2>&1; then
    echo "asciinema not found. Install it with: pip install asciinema" >&2
    exit 1
fi

echo "Recording examples/demo.py to $CAST_FILE ..."
rm -f "$CAST_FILE"
asciinema rec "$CAST_FILE" \
    --title "llm-guardrails demo" \
    --idle-time-limit 2 \
    --command "python examples/demo.py"

echo "Recorded: $CAST_FILE"
echo "Preview it with: asciinema play $CAST_FILE"

if [[ "${1:-}" == "--gif" ]]; then
    if command -v agg >/dev/null 2>&1; then
        echo "Rendering GIF with agg ..."
        agg "$CAST_FILE" "$GIF_FILE"
        echo "GIF written to $GIF_FILE"
    else
        echo "agg not found - install it (cargo install --locked agg) or use" >&2
        echo "https://github.com/asciinema/asciicast2gif instead." >&2
        exit 1
    fi
fi
