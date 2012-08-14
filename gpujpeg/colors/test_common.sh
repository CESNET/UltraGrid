#!/bin/bash

# Get script folder
DIR=`dirname $0`

# Parameters
# NAME      = image name, e.g. "image_yuv_422"
# EXTENSION = image extension, e.g. "yuv"
# MODE      = image arguments for gpujpeg, e.g. "--colorspace=yuv --sampling-factor=4:2:2" 

IMAGE=image_bt709_422.yuv
#IMAGE=camera_bt709_422.yuv

# Create an image from source in specified mode ()
$DIR/../gpujpeg.sh --size=1920x1080 --colorspace=ycbcr-bt709 --sampling-factor=4:2:2 \
    --convert $MODE $DIR/$IMAGE $DIR/$NAME.$EXTENSION

# Encode and Decode the image
$DIR/../gpujpeg.sh --size 1920x1080 $MODE \
    --encode --quality 100 $DIR/$NAME.$EXTENSION $DIR/$NAME.encoded.jpg
$DIR/../gpujpeg.sh $MODE \
    --decode $DIR/$NAME.encoded.jpg $DIR/$NAME.decoded.$EXTENSION

# Convert the Original and the Processed Image to RGB444
$DIR/../gpujpeg.sh --size=1920x1080 $MODE \
    --convert --colorspace=rgb --sampling-factor=4:4:4 $DIR/$NAME.$EXTENSION $DIR/$NAME.rgb
$DIR/../gpujpeg.sh --size=1920x1080 $MODE \
    --convert --colorspace=rgb --sampling-factor=4:4:4 $DIR/$NAME.decoded.$EXTENSION $DIR/$NAME.decoded.rgb

# Display Left/Right Diff of the Original and the Processed Image
$DIR/display_diff.sh $DIR/$NAME.rgb $DIR/$NAME.decoded.rgb

# Delete Created Files
rm -f $DIR/$NAME.$EXTENSION $DIR/$NAME.rgb $DIR/$NAME.encoded.jpg $DIR/$NAME.decoded.$EXTENSION $DIR/$NAME.decoded.rgb

