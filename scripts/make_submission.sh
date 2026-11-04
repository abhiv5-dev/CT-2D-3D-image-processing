#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

SUBMISSION_DIR="$PROJECT_ROOT/submission"
OUTPUT_ZIP="$SUBMISSION_DIR/cpp_ct_project.zip"
MAX_SIZE_BYTES=20971520 # 20 MB

mkdir -p "$SUBMISSION_DIR"

echo "=== Building Submission Archive ==="

# Determine tree-ish to archive.
# If an argument is provided, use it.
# Otherwise, if working directory has modifications or staged files,
# generate a temporary tree object so the archive captures the exact working state
# without modifying git history or committing.
if [ $# -gt 0 ]; then
    TARGET_TREE="$1"
    echo "Using specified target ref/tree: $TARGET_TREE"
elif [ -n "$(git status --porcelain)" ]; then
    echo "Uncommitted changes detected in working tree."
    git add -A
    TARGET_TREE="$(git write-tree)"
    echo "Created tree object: $TARGET_TREE"
else
    TARGET_TREE="HEAD"
    echo "Working directory clean; archiving HEAD."
fi

# Build archive using git archive with working-tree attributes
echo "Archiving tree $TARGET_TREE to $OUTPUT_ZIP..."
git archive --format=zip --worktree-attributes -o "$OUTPUT_ZIP" "$TARGET_TREE"

# Check zip size
ZIP_SIZE=$(wc -c < "$OUTPUT_ZIP" | tr -d ' ')
ZIP_SIZE_MB=$(awk "BEGIN {printf \"%.2f\", $ZIP_SIZE / 1048576}")
echo "Archive created: $OUTPUT_ZIP ($ZIP_SIZE bytes, ${ZIP_SIZE_MB} MB)"

if [ "$ZIP_SIZE" -gt "$MAX_SIZE_BYTES" ]; then
    echo "ERROR: Archive size (${ZIP_SIZE_MB} MB) exceeds maximum allowed size of 20 MB!" >&2
    exit 1
fi
echo "[PASS] Archive size is <= 20 MB (${ZIP_SIZE_MB} MB)."

# Check contents
echo "Validating archive contents..."

# Check TIFF files: only data/sample/phantom_128x128.tif allowed
TIFF_FILES=$(unzip -l "$OUTPUT_ZIP" | awk '{print $4}' | grep -iE '\.(tif|tiff)$' || true)
while IFS= read -r f; do
    if [ -n "$f" ] && [ "$f" != "data/sample/phantom_128x128.tif" ]; then
        echo "ERROR: Prohibited TIFF file found in archive: $f" >&2
        exit 1
    fi
done <<< "$TIFF_FILES"
echo "[PASS] No unauthorized TIFF files found (only data/sample/phantom_128x128.tif)."

# Check for banned directories/patterns
BANNED_MATCHES=$(unzip -l "$OUTPUT_ZIP" | awk '{print $4}' | grep -E '(\.venv/|__pycache__/|build/|\.git/|data/real/)' || true)
if [ -n "$BANNED_MATCHES" ]; then
    echo "ERROR: Banned files/directories found in archive:" >&2
    echo "$BANNED_MATCHES" >&2
    exit 1
fi
echo "[PASS] No banned paths (.venv, __pycache__, build/, .git/, data/real/) found."

# Test unpacking and building in clean temporary directory
TMP_VERIFY_DIR="$(mktemp -d -t ct_verify_build_XXXXXX)"
echo "Testing clean unpack and build in $TMP_VERIFY_DIR..."
trap 'rm -rf "$TMP_VERIFY_DIR"' EXIT

unzip -q "$OUTPUT_ZIP" -d "$TMP_VERIFY_DIR"

# Build with CMake in Release mode
mkdir -p "$TMP_VERIFY_DIR/build"
cmake -S "$TMP_VERIFY_DIR" -B "$TMP_VERIFY_DIR/build" -DCMAKE_BUILD_TYPE=Release -DCT_BUILD_TESTS=ON -DCT_BUILD_PYTHON=OFF
cmake --build "$TMP_VERIFY_DIR/build" --target ctfilter ct_tests -j 4
echo "[PASS] Clean unpack and build succeeded."

# Print top-level files
echo "=== Archive Top-Level Contents ==="
unzip -l "$OUTPUT_ZIP" | awk 'NR>3 && NF>=4 {print $4}' | grep -v '^$' | awk -F'/' '{print $1}' | sort -u

echo "=== Submission package successfully verified! ==="
