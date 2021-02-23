#!/bin/sh -e

SRC_DIR=ext-deps/gpujpeg
BUILD_DIR=$SRC_DIR/build
INSTALL_DIR=$SRC_DIR/install
CMAKE_ARGUMENTS="-DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DCMAKE_POSITION_INDEPENDENT_CODE=ON"
SUDO=
UPDATE=no
DOWNLOAD_ONLY=no

while getopts 'dfhsu' opt; do
        case "$opt" in
		'h'|'?')
			cat <<-EOF
			Downlads and builds GPUJPEG to be used with UltraGrid. It is downloaded to $SRC_DIR
			and statically built. UltraGrid configure then automatically finds the library with
			pkgconfig in $INSTALL_DIR/share/pkgconfig when called from same directory as this script.

			Usage:
			     $0 [-d|-f|-s|-u]

			Options:
			    -d - download only
			    -f - remove previously downloaded GPUJPEG
			    -s - causes GPUJPEG to be compiled as a dynamic library and installed system-wide
			    -u - update previously downloaded GPUJPEG

			EOF
			exit 0
			;;
		'd')
			DOWNLOAD_ONLY=yes
			;;
		'f')
			rm -rf $SRC_DIR
			;;
		's')
			CMAKE_ARGUMENTS="-DBUILD_SHARED_LIBS=ON"
			SUDO=$(command -v sudo || true)
			;;
		'u')
			UPDATE=yes
			;;
	esac
done

if [ $OPTIND -eq 1 ]; then
	echo "See also '$0 -h' for available options."
fi

if [ $UPDATE = yes ]; then
	( cd $SRC_DIR; git pull )
else
	git clone --depth 1 https://github.com/CESNET/GPUJPEG.git $SRC_DIR
fi
[ $DOWNLOAD_ONLY = yes ] && exit 0
cmake $CMAKE_ARGUMENTS $SRC_DIR -B $BUILD_DIR
cmake --build $BUILD_DIR --parallel
${SUDO}cmake --install $BUILD_DIR

# vim: set noexpandtab:
