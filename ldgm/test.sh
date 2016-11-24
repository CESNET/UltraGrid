#!/bin/bash
set -e

./matrix-gen/matrix-gen -c 5 -k 1024 -m 768 -r -s 1 -f /tmp/matrix.bin
./ldgm-encode -t /tmp/matrix.bin -k 1024 -m 768 -f 4000000 -w5 -c

