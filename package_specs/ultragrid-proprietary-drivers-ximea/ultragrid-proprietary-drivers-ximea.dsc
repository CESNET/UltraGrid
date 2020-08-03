# see https://en.opensuse.org/openSUSE:Build_Service_Debian_builds#packageName.dsc
DEBTRANSFORM-TAR:	ximea-linux-upstream.tar.gz
DEBTRANSFORM-FILES-TAR:	debian.tar.gz
DEBTRANSFORM-SERIES:	debian-patches-series
Format: 1.0
Source: ultragrid-proprietary-drivers-ximea
Binary: ultragrid-proprietary-drivers-ximea
Architecture: any
Version: 20200109
Standards-Version: 3.9.6
Maintainer: 	Lukas Rucka <ultragrid-dev@cesnet.cz>
Build-Depends: 	debhelper (>= 7.0.50~), build-essential, linux-headers, realpath, coreutils, autoconf, automake, linux-libc-dev, python3, python3-distutils-extra, bash, libraw1394-11, libtiff-dev, libusb-1.0-0
