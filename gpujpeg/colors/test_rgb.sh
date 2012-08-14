#!/bin/bash

# Parameters
NAME="image_rgb_444"
EXTENSION="rgb"
MODE="--colorspace=rgb --sampling-factor=4:4:4"

# Run test
source `dirname $0`/test_common.sh
