#!/bin/sh -eux
#
# Builds rav1e but doesn't install it

rustup update
cargo install cargo-c
git clone --depth 1 https://github.com/xiph/rav1e.git
cd rav1e
cargo build --release
cargo cinstall --release --destdir=install

