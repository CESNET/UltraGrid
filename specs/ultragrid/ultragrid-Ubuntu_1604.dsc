# see https://en.opensuse.org/openSUSE:Build_Service_Debian_builds#packageName.dsc
DEBTRANSFORM-TAR:	ultragrid-1.3.tar.bz2
DEBTRANSFORM-FILES-TAR:	debian-Ubuntu_1604.tar.gz
DEBTRANSFORM-SERIES:	debian-patches-series
Format: 1.0
Source: ultragrid
Binary: ultragrid
Architecture: any
Standards-Version: 3.9.6
Version: 1.3-2016051001
Maintainer: 	Lukas Rucka <xrucka@fi.muni.cz>
Build-Depends: 	debhelper (>= 8.0), build-essential, make, autoconf, automake, libmagickwand-dev, libjpeg-dev, freeglut3-dev, libglew1.6-dev, libsdl-mixer1.2-dev, libsdl-ttf2.0-dev, libsdl1.2-dev, libqt4-dev, libqtgui4, qt4-dev-tools, libxxf86vm1, libx11-6, libxdamage1, portaudio19-dev, libjack-dev, libasound2-dev, libv4l-dev, zip, libavcodec-dev, libopencv-dev, libssl-dev, libgpujpeg-dev, libcairo2-dev, ultragrid-proprietary-drivers, libglib2.0-dev, libcurl4-openssl-dev, git, nvidia-cuda-toolkit (>= 6.0)
