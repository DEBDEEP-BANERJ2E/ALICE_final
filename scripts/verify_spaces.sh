#!/usr/bin/env bash
# Smoke-test the three-component ALICE HF Space deployment.
#
# Usage:
#   ALICE_ENV_URL=https://your-username-alice-rl-environment.hf.space bash scripts/verify_spaces.sh
#
# Environment variables:
#   ALICE_ENV_URL  (required) Base URL of the deployed Space
#   HF_SPACE_ID    (optional) Format: <username>/<space-name> — used to verify Repository and Registry URLs
#   TIMEOUT        (optional) curl timeout per request in seconds (default: 15)
set -euo pipefail

# ── Validate inputs ────────────────────────────────────────────────────────────
: "${ALICE_ENV_URL:?ALICE_ENV_URL must be set (e.g. https://your-username-alice-rl-environment.hf.space)}"

TIMEOUT="${TIMEOUT:-15}"

# Strip trailing slash
ALICE_ENV_URL="${ALICE_ENV_URL%/}"

# ── Helpers ────────────────────────────────────────────────────────────────────
PASS=0
FAIL=0
RESULTS=()

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

# check_http <label> <url> <expected_status_codes_space_separated>
check_http() {
    local label="$1"
    local url="$2"
    shift 2
    local expected_codes=("$@")

    local http_status
    http_status=$(curl -s -o /tmp/verify_body.txt -w "%{http_code}" \
        --max-time "$TIMEOUT" \
        "$url" 2>/dev/null) || http_status="000"

    local matched=false
    for code in "${expected_codes[@]}"; do
        if [ "$http_status" = "$code" ]; then
            matched=true
            break
        fi
    done

    if [ "$matched" = "true" ]; then
        PASS=$((PASS + 1))
        RESULTS+=("  [PASS] $label (HTTP $http_status)")
    else
        FAIL=$((FAIL + 1))
        RESULTS+=("  [FAIL] $label (HTTP $http_status, expected: ${expected_codes[*]})")
    fi
}

# check_url_reachable <label> <url>  — accepts any 2xx/3xx
check_url_reachable() {
    local label="$1"
    local url="$2"

    local http_status
    http_status=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time "$TIMEOUT" \
        "$url" 2>/dev/null) || http_status="000"

    local first_digit="${http_status:0:1}"
    if [ "$first_digit" = "2" ] || [ "$first_digit" = "3" ] || [ "$http_status" = "401" ]; then
        PASS=$((PASS + 1))
        RESULTS+=("  [PASS] $label (HTTP $http_status)")
    else
        FAIL=$((FAIL + 1))
        RESULTS+=("  [FAIL] $label (HTTP $http_status)")
    fi
}

# ── Banner ─────────────────────────────────────────────────────────────────────
log "ALICE RL Environment — Smoke Test"
log "  ALICE_ENV_URL : $ALICE_ENV_URL"
log "  HF_SPACE_ID   : ${HF_SPACE_ID:-(not set)}"
log ""

# ── Component 1: Server ────────────────────────────────────────────────────────
log "── Component 1: Server ──────────────────────────────────────────────────────"

check_http "GET /health → 200" "$ALICE_ENV_URL/health" "200"

# Validate /health response body contains expected fields
HEALTH_BODY=$(cat /tmp/verify_body.txt 2>/dev/null || echo "")
if echo "$HEALTH_BODY" | grep -q "uptime\|status\|memory"; then
    PASS=$((PASS + 1))
    RESULTS+=("  [PASS] /health body contains expected fields")
else
    FAIL=$((FAIL + 1))
    RESULTS+=("  [FAIL] /health body missing expected fields (got: ${HEALTH_BODY:0:120})")
fi

check_http "GET /state → 200" "$ALICE_ENV_URL/state" "200"

# ── Component 2: Repository ────────────────────────────────────────────────────
log "── Component 2: Repository ──────────────────────────────────────────────────"

if [ -n "${HF_SPACE_ID:-}" ]; then
    REPO_URL="https://huggingface.co/spaces/$HF_SPACE_ID"
    check_url_reachable "HF Space page reachable ($REPO_URL)" "$REPO_URL"

    PIP_URL="git+https://huggingface.co/spaces/$HF_SPACE_ID"
    RESULTS+=("  [INFO] pip install URL: $PIP_URL")
    log "  pip install URL: $PIP_URL"
else
    FAIL=$((FAIL + 1))
    RESULTS+=("  [FAIL] Repository: HF_SPACE_ID not set — cannot verify pip install URL")
fi

# ── Component 3: Registry ──────────────────────────────────────────────────────
log "── Component 3: Registry ────────────────────────────────────────────────────"

if [ -n "${HF_SPACE_ID:-}" ]; then
    REGISTRY_IMAGE="registry.hf.space/${HF_SPACE_ID}:latest"
    # 401 = auth required (registry is up), 200 = open ping endpoint
    check_url_reachable "HF registry reachable (https://registry.hf.space/v2/)" \
        "https://registry.hf.space/v2/"
    RESULTS+=("  [INFO] docker pull $REGISTRY_IMAGE")
    log "  docker pull $REGISTRY_IMAGE"
else
    FAIL=$((FAIL + 1))
    RESULTS+=("  [FAIL] Registry: HF_SPACE_ID not set — cannot verify registry image URL")
fi

# ── Summary ────────────────────────────────────────────────────────────────────
TOTAL=$((PASS + FAIL))
log ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Results:"
for r in "${RESULTS[@]}"; do
    log "$r"
done
log ""
log "  Passed : $PASS / $TOTAL"
log "  Failed : $FAIL / $TOTAL"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$FAIL" -gt 0 ]; then
    log "RESULT: FAIL — $FAIL check(s) did not pass."
    exit 1
else
    log "RESULT: PASS — all checks passed."
    exit 0
fi
