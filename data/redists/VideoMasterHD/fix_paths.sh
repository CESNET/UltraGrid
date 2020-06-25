#!/bin/sh

ORIG_PREFIX=/Library/Frameworks

sudo install_name_tool -id @executable_path/../Frameworks/libVideoMasterHD_SP.dylib libVideoMasterHD_SP.dylib
sudo install_name_tool -id @executable_path/../Frameworks/VideoMasterHD_Audio.framework/Versions/A/VideoMasterHD_Audio VideoMasterHD_Audio.framework/Versions/A/VideoMasterHD_Audio
sudo install_name_tool -id @executable_path/../Frameworks/VideoMasterHD.framework/Versions/A/VideoMasterHD VideoMasterHD.framework/VideoMasterHD
sudo install_name_tool -change $ORIG_PREFIX/libVideoMasterHD_SP.dylib @executable_path/../Frameworks/libVideoMasterHD_SP.dylib VideoMasterHD.framework/VideoMasterHD
sudo install_name_tool -change $ORIG_PREFIX/VideoMasterHD.framework/Versions/A/VideoMasterHD @executable_path/../Frameworks/VideoMasterHD.framework/Versions/A/VideoMasterHD VideoMasterHD_Audio.framework/Versions/A/VideoMasterHD_Audio
