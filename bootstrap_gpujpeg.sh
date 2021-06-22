#!/bin/sh -e

UG_SRC_PATH=$(dirname $0)
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

download_win_gpujpeg_from_github() {
	echo "MSYS2 detected - downloading prebuilt GPUJPEG instead"
	if [ $UPDATE = no -a -d $SRC_DIR ]; then
		echo "$SRC_DIR already exists and is a not empty directory!" >&2
		echo "Use -u (or -f) to update"
		exit 1
	fi
	mkdir -p ext-deps
	rm -f GPUJPEG.zip
	wget https://github.com/CESNET/GPUJPEG/releases/download/continuous/GPUJPEG.zip
	[ $UPDATE = yes ] && rm -rf $SRC_DIR
	unzip -d ext-deps GPUJPEG
	rm GPUJPEG.zip
	[ $DOWNLOAD_ONLY = yes ] && exit 0
	cp -r $SRC_DIR/bin $SRC_DIR/include $SRC_DIR/lib $MSYSTEM_PREFIX
	echo "Installed to $MSYSTEM_PREFIX"
	mkdir -p bin
	cp $SRC_DIR/bin/*dll bin
	exit 0
}

if [ $OPTIND -eq 1 ]; then
	echo "See also '$0 -h' for available options."
fi

if [ "${MSYSTEM:-}" = MSYS ]; then
	download_win_gpujpeg_from_github
fi

if [ $UPDATE = yes ]; then
	( cd $SRC_DIR; git pull )
else
	git clone --depth 1 https://github.com/CESNET/GPUJPEG.git $SRC_DIR
fi
[ $DOWNLOAD_ONLY = yes ] && exit 0
cmake $CMAKE_ARGUMENTS $SRC_DIR -B$BUILD_DIR
CMAKE_COMPILE_FLAGS=
CMAKE_VER=`cmake --version | head -n 1 | cut -f 3 -d\ `
if [ "$($UG_SRC_PATH/.github/scripts/Linux/utils/semver.sh $CMAKE_VER 3.12)" -ge 0 ]; then
	CMAKE_COMPILE_FLAGS=--parallel
fi
cmake --build $BUILD_DIR $CMAKE_COMPILE_FLAGS
${SUDO}cmake --install $BUILD_DIR

# vim: set noexpandtab:
