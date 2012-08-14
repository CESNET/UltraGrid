#!/bin/bash

# Get script folder
DIR=`dirname $0`

convert -depth 8 -size 1920x1080  \
    $1 -crop $((1920/2))x1080+0+0 +repage $DIR/_tmp_left.rgb
    
convert -depth 8 -size 1920x1080  \
    $2 -crop $((1920/2))x1080+$((1920/2))+0 +repage $DIR/_tmp_right.rgb

convert -depth 8 -size $((1920/2))x1080 \
     $DIR/_tmp_left.rgb -depth 8 -size $((1920/2))x1080 \
     $DIR/_tmp_right.rgb +append $DIR/_tmp_diff.rgb
     
display -depth 8 -size 1920x1080 $DIR/_tmp_diff.rgb
#-equalize 

rm -f $DIR/_tmp_left.rgb $DIR/_tmp_right.rgb $DIR/_tmp_diff.rgb
