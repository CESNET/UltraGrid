#!/bin/bash

# Parameters
NAME="image_rgb_444"
EXTENSION="yuv"
MODE="--colorspace=ycbcr-jpeg --sampling-factor=4:4:4"

# Get script folder
DIR=`dirname $0`

IMAGE=image_bt709_422.yuv
#IMAGE=camera_bt709_422.yuv

# Create an image from source in RGB
$DIR/../gpujpeg.sh --size=1920x1080 --colorspace=ycbcr-bt709 --sampling-factor=4:2:2 \
    --convert --colorspace=rgb --sampling-factor=4:4:4 $DIR/$IMAGE $DIR/original.rgb

# Convert image to specified mode and back
$DIR/../gpujpeg.sh --size=1920x1080 --colorspace=rgb --sampling-factor=4:4:4 \
    --convert $MODE $DIR/original.rgb $DIR/$NAME.$EXTENSION
$DIR/../gpujpeg.sh --size=1920x1080 $MODE \
    --convert --colorspace=rgb --sampling-factor=4:4:4 $DIR/$NAME.$EXTENSION $DIR/processed.rgb

# Display Left/Right Diff of the Original and the Processed Image
$DIR/display_diff.sh $DIR/original.rgb $DIR/processed.rgb

# Delete Created Files
rm -f $DIR/original.rgb $DIR/$NAME.$EXTENSION $DIR/processed.rgb

