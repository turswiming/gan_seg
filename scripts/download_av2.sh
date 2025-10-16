#!/usr/bin/env bash

set -euo pipefail

# Usage: ./download_av2.sh [target_directory]
# Default target directory: ./data/av2

TARGET_DIR="${1:-./data/av2}"

URLS=(
  "https://s3.amazonaws.com/argoverse/datasets/av2/tars/motion-forecasting/train.tar"
  "https://s3.amazonaws.com/argoverse/datasets/av2/tars/motion-forecasting/val.tar"
  "https://s3.amazonaws.com/argoverse/datasets/av2/tars/motion-forecasting/test.tar"
)

echo "Target directory: ${TARGET_DIR}"
mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl is required but not found in PATH." >&2
  exit 1
fi

download_with_retries() {
  local url="$1"
  local out_name
  out_name="$(basename "${url}")"

  echo "Downloading: ${out_name}"
  # Up to 5 attempts (curl also retries transient errors)
  for attempt in 1 2 3 4 5; do
    if curl -fL --retry 5 --retry-delay 2 --retry-all-errors -C - -O "${url}"; then
      # basic sanity check
      if [[ -s "${out_name}" ]]; then
        echo "Downloaded: ${out_name}"
        return 0
      fi
      echo "Warning: ${out_name} is empty after download, retrying..."
    else
      echo "Attempt ${attempt} failed for ${url}, retrying..."
    fi
    sleep 2
  done
  echo "Error: failed to download ${url} after multiple attempts." >&2
  return 1
}

for url in "${URLS[@]}"; do
  download_with_retries "${url}"
done

echo "All files downloaded to: ${TARGET_DIR}"


