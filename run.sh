#!/usr/bin/env bash

set -e
echo "training & predicting.."
for item in cpu ram gpu hdd screen; do
    python3 -m main \
    --item=$item \
    --train \
    --predict
done

echo "calculating metrics.."
python3 -m metrics