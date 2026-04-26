#!/usr/bin/env bash
# Deploy ALICE RL Environment to Hugging Face Spaces.
#
# Usage:
#   HF_SPACE_ID=your-username/alice-rl-environment HF_TOKEN=hf_... bash scripts/deploy_spaces.sh
#
#   Rollback to a previous revision:
#   ROLLBACK_SHA=<commit-sha> HF_SPACE_ID=... HF_TOKEN=... bash scripts/deploy_spaces.sh --rollback
#
# Environment variables:
#   HF_SPACE_ID    (required) Format: <username>/<space-name>
#   HF_TOKEN       (required) Hugging Face write token
#   POLL_TIMEOUT   (optional) Seconds to wait for Space to become healthy (default: 600)
#   ENV_FILE       (optional) Path to .env file to write ALICE_ENV_URL (default: .env)
#   ROLLBACK_SHA   (required for --rollback) Commit SHA to roll back to
set -euo pipefail

ROLLBACK_MODE=false
for arg in "$@"; do
    [ "$arg" = "--rollback" ] && ROLLBACK_MODE=true
done

# ── Validate inputs ────────────────────────────────────────────────────────────
: "${HF_SPACE_ID:?HF_SPACE_ID must be set (format: username/space-name)}"
: "${HF_TOKEN:?HF_TOKEN must be set}"

POLL_TIMEOUT="${POLL_TIMEOUT:-600}"
ENV_FILE="${ENV_FILE:-.env}"

# Derive username and space name from HF_SPACE_ID
HF_USERNAME="${HF_SPACE_ID%%/*}"
SPACE_NAME="${HF_SPACE_ID##*/}"

# Canonical URL: HF converts slashes to dashes in the subdomain
SPACE_URL="https://${HF_USERNAME}-${SPACE_NAME}.hf.space"

# ── Helpers ────────────────────────────────────────────────────────────────────
log()  { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }
err()  { log "ERROR: $*" >&2; }
die()  { err "$*"; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "'$1' is not installed. Install it with: pip install huggingface_hub[cli]"
}

# ── Pre-flight checks ──────────────────────────────────────────────────────────
require_cmd huggingface-cli
require_cmd curl

log "Deploying ALICE RL Environment"
log "  Space ID : $HF_SPACE_ID"
log "  Space URL: $SPACE_URL"

# Authenticate
log "Authenticating with Hugging Face..."
alice_env/.venv/bin/hf auth login --token "$HF_TOKEN" 2>/dev/null || true

# ── Rollback mode ─────────────────────────────────────────────────────────────
if [ "$ROLLBACK_MODE" = "true" ]; then
    : "${ROLLBACK_SHA:?ROLLBACK_SHA must be set when using --rollback}"
    log "ROLLBACK MODE: reverting Space '$HF_SPACE_ID' to commit $ROLLBACK_SHA"
    TMPDIR=$(mktemp -d)
    log "Downloading revision $ROLLBACK_SHA to $TMPDIR..."
    alice_env/.venv/bin/hf download \
        "$HF_SPACE_ID" \
        --repo-type space \
        --revision "$ROLLBACK_SHA" \
        --token "$HF_TOKEN" \
        --local-dir "$TMPDIR"
    log "Re-uploading revision $ROLLBACK_SHA to Space..."
    alice_env/.venv/bin/hf upload \
        "$HF_SPACE_ID" \
        "$TMPDIR" \
        . \
        --repo-type space \
        --token "$HF_TOKEN" \
        --commit-message "Rollback to $ROLLBACK_SHA [$(date -u '+%Y-%m-%dT%H:%M:%SZ')]"
    rm -rf "$TMPDIR"
    log "Rollback push complete. Space is rebuilding..."
else

# ── Step 1: Create Space (idempotent) ─────────────────────────────────────────
log "Creating Space '$HF_SPACE_ID' (skips if already exists)..."
alice_env/.venv/bin/hf repo create "$HF_SPACE_ID" \
    --type space \
    --space-sdk docker \
    --token "$HF_TOKEN" \
    --exist-ok \
    && log "Space ready." || log "Space creation skipped (already exists)."

# ── Step 2: Capture current HEAD SHA for rollback reference ───────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALICE_ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PREV_SHA=$(alice_env/.venv/bin/python -c "
from huggingface_hub import HfApi
import os, sys
try:
    api = HfApi()
    commits = api.list_repo_commits(repo_id='$HF_SPACE_ID', repo_type='space', token='$HF_TOKEN')
    print(commits[0].commit_id if commits else '')
except Exception as e:
    print('', file=sys.stderr)
    print('')
" 2>/dev/null) || PREV_SHA=""

if [ -n "$PREV_SHA" ]; then
    log "Previous HEAD SHA: $PREV_SHA (use ROLLBACK_SHA=$PREV_SHA to roll back)"
fi

# ── Step 3: Push alice_env/ contents to the Space ─────────────────────────────
log "Pushing '$ALICE_ENV_DIR' to Space repository..."
alice_env/.venv/bin/hf upload \
    "$HF_SPACE_ID" \
    "$ALICE_ENV_DIR" \
    . \
    --repo-type space \
    --token "$HF_TOKEN" \
    --commit-message "Deploy ALICE RL Environment [$(date -u '+%Y-%m-%dT%H:%M:%SZ')]" \
    --exclude ".venv/**" \
    --exclude ".pytest_cache/**" \
    --exclude "__pycache__/**" \
    --exclude "*.pyc" \
    --exclude ".cache/**"

log "Push complete. Space is now building..."
fi  # end rollback / normal-deploy branch

# ── Step 4: Poll health endpoint ───────────────────────────────────────────────
log "Polling $SPACE_URL/health (timeout: ${POLL_TIMEOUT}s)..."
POLL_INTERVAL=20
ELAPSED=0
HEALTHY=false

while [ "$ELAPSED" -lt "$POLL_TIMEOUT" ]; do
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time 10 \
        "$SPACE_URL/health" 2>/dev/null) || HTTP_STATUS="000"

    if [ "$HTTP_STATUS" = "200" ]; then
        HEALTHY=true
        break
    fi

    log "  Health check: HTTP $HTTP_STATUS — waiting ${POLL_INTERVAL}s (${ELAPSED}/${POLL_TIMEOUT}s elapsed)..."
    sleep "$POLL_INTERVAL"
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
done

if [ "$HEALTHY" != "true" ]; then
    die "Space did not become healthy within ${POLL_TIMEOUT}s. Check build logs at https://huggingface.co/spaces/$HF_SPACE_ID"
fi

log "Space is healthy!"

# ── Step 5: Verify three-component model ──────────────────────────────────────
PASS=0
FAIL=0

check() {
    local label="$1"
    local result="$2"
    if [ "$result" = "ok" ]; then
        log "  [PASS] $label"
        PASS=$((PASS + 1))
    else
        log "  [FAIL] $label — $result"
        FAIL=$((FAIL + 1))
    fi
}

# Component 1 — Server: /health endpoint
log "Verifying Component 1: Server..."
HEALTH_BODY=$(curl -sf --max-time 10 "$SPACE_URL/health" 2>/dev/null) || HEALTH_BODY=""
if [ -n "$HEALTH_BODY" ]; then
    check "Server: GET $SPACE_URL/health → 200" "ok"
else
    check "Server: GET $SPACE_URL/health → 200" "no response or non-200"
fi

# Component 2 — Repository: pip-installable via git+https
PIP_URL="git+https://huggingface.co/spaces/$HF_SPACE_ID"
log "Verifying Component 2: Repository ($PIP_URL)..."
# Validate the URL is reachable (HEAD request to the HF git endpoint)
REPO_HTTP=$(curl -s -o /dev/null -w "%{http_code}" \
    --max-time 10 \
    "https://huggingface.co/spaces/$HF_SPACE_ID" 2>/dev/null) || REPO_HTTP="000"
if [ "$REPO_HTTP" = "200" ] || [ "$REPO_HTTP" = "301" ] || [ "$REPO_HTTP" = "302" ]; then
    check "Repository: pip install $PIP_URL (URL reachable)" "ok"
else
    check "Repository: pip install $PIP_URL (URL reachable)" "HTTP $REPO_HTTP"
fi

# Component 3 — Registry: Docker image URL well-formed and registry reachable
REGISTRY_IMAGE="registry.hf.space/${HF_SPACE_ID}:latest"
log "Verifying Component 3: Registry ($REGISTRY_IMAGE)..."
REGISTRY_HTTP=$(curl -s -o /dev/null -w "%{http_code}" \
    --max-time 10 \
    "https://registry.hf.space/v2/" 2>/dev/null) || REGISTRY_HTTP="000"
if [ "$REGISTRY_HTTP" = "200" ] || [ "$REGISTRY_HTTP" = "401" ]; then
    # 401 is expected (auth required) — registry is reachable
    check "Registry: registry.hf.space reachable (docker pull $REGISTRY_IMAGE)" "ok"
else
    check "Registry: registry.hf.space reachable (docker pull $REGISTRY_IMAGE)" "HTTP $REGISTRY_HTTP"
fi

# ── Step 6: Record canonical ALICE_ENV_URL ────────────────────────────────────
log ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "  ALICE_ENV_URL = $SPACE_URL"
log "  Components passed: $PASS / $((PASS + FAIL))"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Write/update ALICE_ENV_URL in the target .env file
if [ -f "$ENV_FILE" ]; then
    # Update existing entry if present, otherwise append
    if grep -q "^ALICE_ENV_URL=" "$ENV_FILE" 2>/dev/null; then
        # Use sed to replace in-place (portable: create temp file)
        TMP_ENV=$(mktemp)
        sed "s|^ALICE_ENV_URL=.*|ALICE_ENV_URL=$SPACE_URL|" "$ENV_FILE" > "$TMP_ENV"
        mv "$TMP_ENV" "$ENV_FILE"
        log "Updated ALICE_ENV_URL in $ENV_FILE"
    else
        echo "ALICE_ENV_URL=$SPACE_URL" >> "$ENV_FILE"
        log "Appended ALICE_ENV_URL to $ENV_FILE"
    fi
else
    echo "ALICE_ENV_URL=$SPACE_URL" > "$ENV_FILE"
    log "Created $ENV_FILE with ALICE_ENV_URL"
fi

# Also export for the current shell session
export ALICE_ENV_URL="$SPACE_URL"

if [ "$FAIL" -gt 0 ]; then
    err "$FAIL component(s) failed verification. Review logs above."
    exit 1
fi

log "Deployment complete."
