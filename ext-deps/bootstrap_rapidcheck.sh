#!/bin/bash
# Bootstrap script for RapidCheck - property-based testing for C++
#
# This script downloads and prepares RapidCheck for static linking
# with UltraGrid without requiring system-wide installation.

set -e

SCRIPT_DIR=$(dirname "$0")
RAPIDCHECK_DIR="$SCRIPT_DIR/rapidcheck"
# Pinned to a known-good commit so re-runs (and CI) get a reproducible
# build. The upstream has no tags; bump this when picking up a fix.
RAPIDCHECK_REF="b2d9ed2dddefc4b84318d664b4f221eb792d89c7"

echo "Bootstrapping RapidCheck..."

# Clean existing directory if present
if [ -d "$RAPIDCHECK_DIR" ]; then
    echo "Removing existing RapidCheck directory..."
    rm -rf "$RAPIDCHECK_DIR"
fi

# Clone RapidCheck
echo "Cloning RapidCheck @ $RAPIDCHECK_REF..."
git clone https://github.com/emil-e/rapidcheck.git "$RAPIDCHECK_DIR"
cd "$RAPIDCHECK_DIR"
git checkout --quiet "$RAPIDCHECK_REF"

# Build RapidCheck as a static library
echo "Building RapidCheck static library..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DRC_ENABLE_TESTS=OFF -DRC_ENABLE_EXAMPLES=OFF
make -j4

cd ..

# Create a pkg-config file for easier integration
cat > rapidcheck.pc << EOF
prefix=$RAPIDCHECK_DIR
includedir=\${prefix}/include
libdir=\${prefix}/build

Name: RapidCheck
Description: Property-based testing for C++
Version: 0.0.0
Cflags: -I\${includedir}
Libs: -L\${libdir} -lrapidcheck
EOF

echo "RapidCheck bootstrapped successfully!"
echo "To use in configure:"
echo "  PKG_CONFIG_PATH=$RAPIDCHECK_DIR:\$PKG_CONFIG_PATH ./configure"