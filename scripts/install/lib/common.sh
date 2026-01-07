#!/bin/bash
# =============================================================================
# common.sh - Color codes, logging, and UI helpers
# =============================================================================

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Current step tracking
CURRENT_STEP=0
TOTAL_STEPS=7

# Logging functions
log_step() {
    CURRENT_STEP=$1
    echo -e "${YELLOW}[$CURRENT_STEP/$TOTAL_STEPS] $2${NC}"
}

log_info() {
    echo -e "  $1"
}

log_success() {
    echo -e "  ${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "  ${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}ERROR:${NC} $1" >&2
}

log_fatal() {
    echo -e "${RED}FATAL:${NC} $1" >&2
    exit 1
}

# Print banner
print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║              Power Node Installation Script                   ║"
    echo "║         Gelotto Distributed GPU Compute Network               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Require a command to exist
require_command() {
    local cmd="$1"
    local msg="${2:-$cmd not found}"
    if ! command_exists "$cmd"; then
        log_fatal "$msg"
    fi
}
