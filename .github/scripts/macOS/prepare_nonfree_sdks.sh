#!/bin/sh -eux

cd /tmp

# AJA
AJA_DIRECTORY=$SDK_NONFREE_PATH/ntv2sdk
if [ -f $SDK_NONFREE_PATH/ntv2sdkmac.zip ]; then
        unzip $SDK_NONFREE_PATH/ntv2sdkmac.zip
        mv /tmp/ntv2sdk* $AJA_DIRECTORY
        cd $AJA_DIRECTORY/ajalibraries/ajantv2/build
        xcodebuild -project ajantv2.xcodeproj
fi

# DELTACAST
DELTA_CACHE_INST=$SDK_NONFREE_PATH/VideoMasterHD_inst
if [ -f $SDK_NONFREE_PATH/VideoMaster_SDK_MacOSX.zip ]; then
        unzip $SDK_NONFREE_PATH/VideoMaster_SDK_MacOSX.zip
        sudo installer -pkg VideoMaster_SDK.pkg -target / || true
        cd /Library/Frameworks
        sudo install_name_tool -change /Library/Frameworks/VideoMasterHD.framework/Versions/A/VideoMasterHD @executable_path/../Frameworks/VideoMasterHD.framework/Versions/A/VideoMasterHD /Library/Frameworks/VideoMasterHD.framework/VideoMasterHD
        sudo install_name_tool -id @executable_path/../Frameworks/VideoMasterHD.framework/Versions/A/VideoMasterHD /Library/Frameworks/VideoMasterHD.framework/VideoMasterHD
        sudo install_name_tool -id @executable_path/../Frameworks/VideoMasterHD_Audio.framework/Versions/A/VideoMasterHD_Audio /Library/Frameworks/VideoMasterHD_Audio.framework/Versions/A/VideoMasterHD_Audio
        sudo install_name_tool -change /Library/Frameworks/VideoMasterHD.framework/Versions/A/VideoMasterHD @executable_path/../Frameworks/VideoMasterHD.framework/Versions/A/VideoMasterHD /Library/Frameworks/VideoMasterHD_Audio.framework/Versions/A/VideoMasterHD_Audio
        #sudo cp -a VideoMasterHD.framework VideoMasterHD_Audio.framework libVideoMasterHD_SP.dylib $(xcrun --show-sdk-path)/System/Library/Frameworks
        mkdir $DELTA_CACHE_INST
        sudo cp -a VideoMasterHD.framework VideoMasterHD_Audio.framework $DELTA_CACHE_INST
        cd -
        sudo rm -rf /Library/Frameworks/VideoMasterHD* # ensure that only the copy above is used
fi
