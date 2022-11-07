#!/bin/bash

#set correct glslc location
GLSLC=glslc

DEST_PATH=../../../../share/ultragrid/vulkan_shaders

declare -a SHADERS=("render.vert" "render.frag" "RGB10A2_conv.comp" "UYVA16_conv.comp" "UYVY8_conv.comp")

for shader in ${SHADERS[@]}; do
	echo "$GLSLC $SOURCE_PATH/$shader -o $DEST_PATH/$shader.spv"
	$GLSLC $SOURCE_PATH/$shader -o $DEST_PATH/$shader.spv
done


