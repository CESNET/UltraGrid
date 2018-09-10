# see https://en.opensuse.org/openSUSE:Build_Service_Debian_builds#packageName.dsc
DEBTRANSFORM-TAR:	drivers.tar.gz 
DEBTRANSFORM-FILES-TAR:	debian.tar.gz
DEBTRANSFORM-SERIES:	debian-patches-Ubuntu_1404.series
Format: 1.0
Source: ultragrid-proprietary-drivers
Binary: ultragrid-proprietary-drivers
Architecture: any
Version: 20180511
Standards-Version: 3.9.6
Maintainer: 	Lukas Rucka <ultragrid-dev@cesnet.cz>
Build-Depends: 	debhelper (>= 7.0.50~), build-essential, linux-headers-generic, realpath, coreutils, libx11-dev, libgl1-mesa-dev, libglu1-mesa-dev, libglew-dev, libxext-dev, linux-libc-dev, libncurses5-dev, qtchooser, qt5-default, qtmultimedia5-dev
