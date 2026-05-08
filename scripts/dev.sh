#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"

usage() {
  cat <<'EOF'
Usage:
  scripts/dev.sh target status
  scripts/dev.sh target prune
  scripts/dev.sh target clean
  scripts/dev.sh cargo <cargo-args...>

Target commands:
  status  Show workspace target size, debug cache size, and deps file count.
  prune   Remove regenerable dev artifacts: target/debug, target/dev-fast,
          and xtask debug caches. Preserves release, container, doc, and package output.
  clean   Run cargo clean for a full Cargo artifact reset.

Cargo wrapper:
  If OX_CARGO_PROFILE is set, supported cargo commands are run with
  --profile "$OX_CARGO_PROFILE" unless a profile flag is already present.
EOF
}

size_of() {
  local path="$1"
  if [[ -e "$path" ]]; then
    du -sh "$path" 2>/dev/null | awk '{print $1}'
  else
    printf '-'
  fi
}

deps_file_count() {
  local deps="$TARGET_DIR/debug/deps"
  if [[ -d "$deps" ]]; then
    find "$deps" -maxdepth 1 -type f | wc -l | tr -d ' '
  else
    printf '0'
  fi
}

target_status() {
  printf '%-22s %s\n' 'target' "$(size_of "$TARGET_DIR")"
  printf '%-22s %s\n' 'debug' "$(size_of "$TARGET_DIR/debug")"
  printf '%-22s %s\n' 'debug/deps' "$(size_of "$TARGET_DIR/debug/deps")"
  printf '%-22s %s\n' 'debug/incremental' "$(size_of "$TARGET_DIR/debug/incremental")"
  printf '%-22s %s\n' 'dev-fast' "$(size_of "$TARGET_DIR/dev-fast")"
  printf '%-22s %s\n' 'release' "$(size_of "$TARGET_DIR/release")"
  printf '%-22s %s\n' 'container' "$(size_of "$TARGET_DIR/container")"
  printf '%-22s %s\n' 'xtask/debug' "$(size_of "$ROOT/xtask/target/debug")"
  printf '%-22s %s\n' 'xtask/release' "$(size_of "$ROOT/xtask/target/release")"
  printf '%-22s %s\n' 'deps files' "$(deps_file_count)"
}

remove_if_exists() {
  local path="$1"
  if [[ -e "$path" ]]; then
    rm -rf "$path"
    printf 'removed %s\n' "$path"
  fi
}

target_prune() {
  remove_if_exists "$TARGET_DIR/debug"
  remove_if_exists "$TARGET_DIR/dev-fast"
  remove_if_exists "$ROOT/xtask/target/debug"
  remove_if_exists "$ROOT/xtask/target/dev-fast"

  local child
  for child in "$TARGET_DIR"/*; do
    [[ -d "$child" ]] || continue
    remove_if_exists "$child/debug"
    remove_if_exists "$child/dev-fast"
  done
}

has_profile_flag() {
  local arg
  for arg in "$@"; do
    case "$arg" in
      --profile|--profile=*) return 0 ;;
    esac
  done
  return 1
}

run_cargo() {
  if [[ $# -eq 0 ]]; then
    usage
    exit 2
  fi

  local command="$1"
  shift
  case "${OX_CARGO_PROFILE:-}:$command" in
    :*) exec cargo "$command" "$@" ;;
    *:build|*:check|*:test|*:run|*:clippy|*:doc)
      if has_profile_flag "$@"; then
        exec cargo "$command" "$@"
      fi
      exec cargo "$command" --profile "$OX_CARGO_PROFILE" "$@"
      ;;
    *) exec cargo "$command" "$@" ;;
  esac
}

main() {
  if [[ $# -eq 0 ]]; then
    usage
    exit 2
  fi

  case "$1" in
    target)
      shift
      case "${1:-}" in
        status) target_status ;;
        prune) target_prune ;;
        clean) cargo clean --manifest-path "$ROOT/Cargo.toml" ;;
        *) usage; exit 2 ;;
      esac
      ;;
    cargo)
      shift
      run_cargo "$@"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      usage
      exit 2
      ;;
  esac
}

main "$@"
