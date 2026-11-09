#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m infra.scaling_fit --glob 'out/accept/*.json' --out out/fit/capacity
