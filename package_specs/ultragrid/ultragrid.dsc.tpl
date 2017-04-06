# see https://en.opensuse.org/openSUSE:Build_Service_Debian_builds#packageName.dsc
DEBTRANSFORM-TAR:	ultragrid-1.4.tar.bz2
DEBTRANSFORM-FILES-TAR:	debian.tar.gz
DEBTRANSFORM-SERIES:	debian-patches-series
Format: 1.0
Source: ultragrid
Binary: ultragrid
Architecture: any
Version: 1.4-2017040100
Maintainer: 	Lukas Rucka <xrucka@fi.muni.cz>
#Build-Depends: 	debhelper (>= 8.0), build-essential, make, autoconf, automake, autotools-dev, libmagickwand-dev, libjpeg-dev, freeglut3-dev, libglew1.6-dev, libsdl-mixer1.2-dev, libsdl-ttf2.0-dev, libsdl1.2-dev, libqt4-dev, libqt4-gui, qt4-dev-tools, libxxf86vm1, libx11-6, libxdamage1, portaudio19-dev, libjack-dev, libasound2-dev, libv4l-dev, zip, libavcodec-dev, liblivemedia-dev, libopencv-dev, libssl-dev, libgpujpeg-dev, libcairo2-dev, ultragrid-proprietary-drivers, cuda-core-7-0, cuda-command-line-tools-7-0, cuda-cudart-dev-7-0, libnvvm2, libglib2.0-dev, libcurl4-openssl-dev, git
# dummy, protoze jsem tomu asi smazal spatny soubor z srv
Build-Depends: 	debhelper (>= 8.0), build-essential, make, autoconf, automake, autotools-dev, libmagickwand-dev, libjpeg-dev, freeglut3-dev, libglew1.6-dev, libsdl-mixer1.2-dev, libsdl-ttf2.0-dev, libsdl1.2-dev, libqt4-dev, libqtgui4, qt4-dev-tools, libxxf86vm1, libx11-6, libxdamage1, portaudio19-dev, libjack-dev, libasound2-dev, libv4l-dev, zip, libavcodec-dev, liblivemedia-dev, libopencv-dev, libssl-dev, libgpujpeg-dev, libcairo2-dev, ultragrid-proprietary-drivers, libglib2.0-dev, libcurl4-openssl-dev, git, nvidia-cuda-toolkit (>= 5.0)
