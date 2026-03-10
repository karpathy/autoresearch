#!/bin/bash
# Download pretokenized TinyStories data for ANE training
# Format: flat uint16 token IDs (Llama2 BPE, 32K vocab)
# Source: enio/TinyStories on HuggingFace (pretokenized with karpathy/llama2.c)
#
# The tar.gz contains data00.bin..data49.bin (50 shards).
# We extract only data00.bin and rename it to tinystories_data00.bin.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="$SCRIPT_DIR/tinystories_data00.bin"

if [ -f "$OUTPUT" ]; then
    SIZE=$(stat -f%z "$OUTPUT" 2>/dev/null || stat -c%s "$OUTPUT" 2>/dev/null)
    TOKENS=$((SIZE / 2))
    echo "$OUTPUT already exists ($TOKENS tokens, $(echo "scale=1; $SIZE/1000000" | bc) MB)"
    exit 0
fi

TAR_URL="https://huggingface.co/datasets/enio/TinyStories/resolve/main/tok32000/TinyStories_tok32000.tar.gz?download=true"
TAR_FILE="$SCRIPT_DIR/TinyStories_tok32000.tar.gz"

echo "=== TinyStories Data Download ==="
echo "Downloading pretokenized TinyStories (32K vocab, ~993 MB)..."
echo "  Source: enio/TinyStories on HuggingFace"
echo "  This will take a few minutes depending on your connection."
echo ""

# Download the tar.gz
if [ ! -f "$TAR_FILE" ]; then
    if command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$TAR_FILE" "$TAR_URL"
    elif command -v wget &>/dev/null; then
        wget --show-progress -O "$TAR_FILE" "$TAR_URL"
    else
        echo "Error: need curl or wget"
        exit 1
    fi
else
    echo "Tar file already downloaded, skipping..."
fi

# Verify it's actually a gzip file (not an error page)
if ! file "$TAR_FILE" | grep -q "gzip"; then
    echo "Error: Downloaded file is not a valid gzip archive."
    echo "Content: $(head -c 100 "$TAR_FILE")"
    rm -f "$TAR_FILE"
    exit 1
fi

echo ""
echo "Extracting data00.bin from archive..."

# List what's in the archive to find the right path
DATA_FILE=$(tar tzf "$TAR_FILE" 2>/dev/null | grep 'data00\.bin' | head -1)
if [ -z "$DATA_FILE" ]; then
    echo "Error: data00.bin not found in archive. Contents:"
    tar tzf "$TAR_FILE" | head -20
    exit 1
fi
echo "  Found: $DATA_FILE"

# Extract just data00.bin
tar xzf "$TAR_FILE" -C "$SCRIPT_DIR" "$DATA_FILE"

# Move to expected location (might be in a subdirectory)
EXTRACTED="$SCRIPT_DIR/$DATA_FILE"
if [ "$EXTRACTED" != "$OUTPUT" ]; then
    mv "$EXTRACTED" "$OUTPUT"
    # Clean up any extracted subdirectories
    rmdir "$(dirname "$EXTRACTED")" 2>/dev/null || true
fi

# Clean up tar.gz to save disk space
echo "Cleaning up archive..."
rm -f "$TAR_FILE"

SIZE=$(stat -f%z "$OUTPUT" 2>/dev/null || stat -c%s "$OUTPUT" 2>/dev/null)
TOKENS=$((SIZE / 2))
echo ""
echo "Done: $OUTPUT"
echo "  $TOKENS tokens ($(echo "scale=1; $SIZE/1000000" | bc) MB)"

# Sanity check
python3 -c "
import struct
with open('$OUTPUT', 'rb') as f:
    tokens = struct.unpack('<10H', f.read(20))
    print(f'First 10 tokens: {tokens}')
" 2>/dev/null || true
