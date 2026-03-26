#!/usr/bin/env bash
set -euo pipefail

mkdir -p build results

g++ -std=c++17 -O2 -Wall -Wextra -I../include main.cpp -o build/app

./build/app

python3 scripts/make_graph.py \
    --input results/vNPU.txt \
    --output results/motivation_allocation_scaling.png \
    --ghz 2.9

python3 scripts/make_alloc_compare.py \
    --vnpu results/vNPU.txt \
    --nas results/NAS.txt \
    --output results/eval_alloc_compare.png \
    --ghz 2.9

echo "Generated:"
echo "  - results/vNPU.txt"
echo "  - results/NAS.txt"
echo "  - results/motivation_allocation_scaling.png"
echo "  - results/eval_alloc_compare.png"
