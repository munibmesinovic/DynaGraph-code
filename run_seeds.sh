set -euo pipefail
dataset=$1
for seed in 42 1992 1709 250 213; do
  python -m dynagraph.train --config "configs/${dataset}.yaml" --seed "$seed" --out "runs/${dataset}/seed${seed}"
done
