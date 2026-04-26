#!/usr/bin/env bash
# Deploy ALICE to Hugging Face Spaces via Docker SDK.
set -euo pipefail

SPACE_ID="${HF_SPACE_ID:-your-username/alice-rl-environment}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set}"

echo "Deploying ALICE to HF Spaces: $SPACE_ID"

# Create or update the Space
huggingface-cli repo create "$SPACE_ID" --type space --space-sdk docker || true

# Push all files to the Space
git remote add space "https://huggingface.co/spaces/$SPACE_ID" 2>/dev/null || true
git push space main --force

# Verify health endpoint
ENV_URL="https://${SPACE_ID/\//-}.hf.space"
echo "Waiting for Space to start..."
for i in $(seq 1 30); do
    if curl -sf "$ENV_URL/health" > /dev/null 2>&1; then
        echo "Space is healthy at $ENV_URL"
        exit 0
    fi
    sleep 10
done

echo "ERROR: Space did not become healthy within 5 minutes" >&2
exit 1
