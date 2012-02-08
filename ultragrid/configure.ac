dnl Process this file with autoconf to produce a configure script.

AC_INIT([UltraGrid], [1.1-rc2], [ultragrid-dev@cesnet.cz])
AM_INIT_AUTOMAKE([1.10])
AC_PREREQ(2.61)
AC_CONFIG_SRCDIR([src/main.c])

PARENT=`echo $PWD | sed -e 's%/[[^/]]*$%%'`

AC_CANONICAL_HOST
AC_PROG_CC
AC_PROG_CPP
AC_PROG_CXX
AC_PROG_INSTALL
AC_HEADER_STDC
AC_C_CONST
AC_TYPE_SIZE_T
AC_C_BIGENDIAN
AC_C_CHAR_UNSIGNED

AC_CHECK_HEADERS(stropts.h sys/filio.h sys/wait.h)

# -------------------------------------------------------------------------------------------------
POST_COMPILE_MSG=""
AC_SUBST(LIBS)
AC_SUBST(OBJS)
AC_SUBST(POST_COMPILE_MSG)

AC_C_BIGENDIAN(
        AC_DEFINE([WORDS_BIGENDIAN], 1, [This is big endian system]),
        AC_DEFINE([WORDS_SMALLENDIAN], 1, [This is little endian system])
)

AC_DEFUN([HM_VERSION_WARNING], [
          AC_MSG_WARN([If you still want to try with unsupported version of dependency,
                       you may want to try --disable-depends-version-check option.
                       If you find out that some unsupported version is working,
                       please let us know at ultragrid-dev@cesnet.cz.
                               ])
          ])

case "$host" in
	*darwin9*)
		AC_DEFINE([HAVE_MACOSX_LEOPARD], [1], [This is Mac X OS Leopard])
	;;
  	x86_64-*-linux-*)
		AC_DEFINE([HAVE_64B_LINUX], [1], [We have 64-bit Linux])
	;;
	i*86-*-linux-*)
		AC_DEFINE([HAVE_32B_LINUX], [1], [We have 32-bit Linux])
	;;
esac	

AC_MSG_CHECKING([OS family])

if expr "$host_os" : ".*darwin.*" > /dev/null; then
        system=MacOSX
	AC_DEFINE([HAVE_MACOSX], [1], [This is Mac X OS])
	LINKER=$CXX
	CFLAGS+=" -arch i386 -I/opt/local/include"
	CPPFLAGS+=" -m32 -I/opt/local/include"
	CXXFLAGS+=" -m32 -I/opt/local/include"
	LDFLAGS+=" -m32 -mmacosx-version-min=10.5"
	X_CFLAGS+="-I/usr/X11R6/include"
        OBJS+=" src/utils/autorelease_pool.o"
else
        system=Linux
	LINKER=$CC
fi

AC_MSG_RESULT($system);

AC_MSG_CHECKING([OS kernel version (major)])
os_version_major=`uname -r |cut -d . -f 1`
AC_MSG_RESULT($os_version_major);
AC_MSG_CHECKING([OS kernel version (minor)])
os_version_minor=`uname -r |cut -d . -f 2`
AC_MSG_RESULT($os_version_minor);

AC_DEFINE_UNQUOTED([OS_VERSION_MAJOR], $os_version_major, [OS kernel major version])
AC_DEFINE_UNQUOTED([OS_VERSION_MINOR], $os_version_minor, [OS kernel minor version])

AC_SUBST(LINKER)
AC_SUBST(LDFLAGS)
AC_SUBST(CXXFLAGS)

AC_ARG_ENABLE(depends-version-check,
        AS_HELP_STRING([--disable-depends-version-check], [Disable libraries version check]), 
	[version_check=$enableval],
        [version_check=yes]
        )

if test $system = MacOSX -a \( $os_version_major -lt 9 -o $os_version_major -gt 11 \) -a $version_check = yes
then
        AC_MSG_WARN([*** Currently only Mac OS X versions 10.5 to 10.7 are supported.])
        HM_VERSION_WARNING
        AC_MSG_ERROR([Exiting due to unsatisfied dependency.])
fi

# -------------------------------------------------------------------------------------------------
# Check for standard size types.  The defaults are only valid on some
# systems so we hope that <inttypes.h> exists when they're wrong.

AC_CHECK_TYPE(int8_t,  signed char)
AC_CHECK_TYPE(int16_t, short)
AC_CHECK_TYPE(int32_t, long)
AC_CHECK_TYPE(int64_t, long long)

# Some systems have these in <stdint.h>, just to be difficult...
AC_CACHE_CHECK(for uint8_t in <stdint.h>, ucl_cv_uint8_t_in_stdint_h,
	AC_EGREP_HEADER(uint8_t,
			stdint.h,
			ucl_cv_uint8_t_in_stdint_h=yes,
			ucl_cv_uint8_t_in_stdint_h=no))
if test $ucl_cv_uint8_t_in_stdint_h = "no"
then
	 AC_CHECK_TYPE(uint8_t,  unsigned char)
fi

AC_CACHE_CHECK(for uint16_t in <stdint.h>, ucl_cv_uint16_t_in_stdint_h,
	AC_EGREP_HEADER(uint16_t,
			stdint.h,
			ucl_cv_uint16_t_in_stdint_h=yes,
			ucl_cv_uint16_t_in_stdint_h=no))
if test $ucl_cv_uint16_t_in_stdint_h = "no"
then
	 AC_CHECK_TYPE(uint16_t,  unsigned short)
fi

AC_CACHE_CHECK(for uint32_t in <stdint.h>, ucl_cv_uint32_t_in_stdint_h,
	AC_EGREP_HEADER(uint32_t,
			stdint.h,
			ucl_cv_uint32_t_in_stdint_h=yes,
			ucl_cv_uint32_t_in_stdint_h=no))
if test $ucl_cv_uint32_t_in_stdint_h = "no"
then
	 AC_CHECK_TYPE(uint32_t,  unsigned int)
fi

# -------------------------------------------------------------------------------------------------
# GCC-specific warning flags
if test "$GCC" = yes
then
    CFLAGS="$CFLAGS -pipe -W -Wall -Wcast-qual -Wcast-align -Wbad-function-cast -Wmissing-prototypes -Wmissing-declarations"
fi

# -------------------------------------------------------------------------------------------------
# Check for libraries

AC_SEARCH_LIBS(socket, socket)
AC_SEARCH_LIBS(inet_addr, nsl)
AC_CHECK_LIB(rt, timer_create)

AC_CHECK_FUNC(sin, MATHLIBS="", MATHLIBS="-lm")
AC_CHECK_LIB(ieee, main, [MATHLIBS="-lieee $MATHLIBS"])
AC_SUBST(MATHLIBS)

# -------------------------------------------------------------------------------------------------
# See if this system supports sched_setscheduler()

AC_CHECK_FUNCS(sched_setscheduler)
use_rt=no
AC_ARG_ENABLE(rt,
              AS_HELP_STRING([--enable-rt], [enable RT priority]), 
	[use_rt=$enableval])
if test "$use_rt" = "yes"; then
        AC_DEFINE([USE_RT], [1], [We use RT priority])
fi

# -------------------------------------------------------------------------------------------------
# Try to find the pthread library...

case "$host" in
	*-*-freebsd4*)
		CFLAGS="$CFLAGS -pthread"
		;;
	*)
		AC_CHECK_LIB(pthread, pthread_create, LIBS="$LIBS -lpthread", AC_MSG_ERROR([Require POSIX thread]))
		;;
esac

# -------------------------------------------------------------------------------------------------
debug_output=no
profile=no
ipv6=no

AC_ARG_ENABLE(profile,
	[  --enable-profile        enable profiling],
	[if test $enableval = yes
	then
		if test "$GCC" = yes
		then
			CFLAGS="$CFLAGS -pg"
                        profile=yes
		else
			AC_MSG_ERROR([Do not know how to enable profiling for $CC])
		fi
	fi])

AC_ARG_ENABLE(debug,
	[  --enable-debug          enable debug messages and code],
	[if test $enableval = yes
	then
		AC_DEFINE([DEBUG], [1], [We build with debug messages])
                debug_output=yes
	fi])

AC_ARG_ENABLE(ipv6,
	[  --enable-ipv6           enable ipv6],
	[if test $enableval = yes
	then
		AC_DEFINE([HAVE_IPv6], [1], [Enable IPv6 support])
                ipv6=YES
	fi])


# ------------------------------------------------------------------------------------------------
# Dynamic libraries stuff
#
build_libraries=no
AC_ARG_ENABLE(dlsym, AS_HELP_STRING([--enable-dlsym], [build moodules as libraries (default is off)]), 
	[build_libraries=$enableval])

if test "$build_libraries" = yes
then
        if test "$system" != "Linux"
        then
                AC_MSG_WARN([*** Building libraries is not supported with Mac OS X])
                build_libraries=no
        fi
        AC_DEFINE([BUILD_LIBRARIES], [1], [Build drivers as a standalone libraries])
fi

video_display_abi_version=1
video_capture_abi_version=1
video_compress_abi_version=1
video_decompress_abi_version=1
audio_capture_abi_version=1
audio_playback_abi_version=1
AC_DEFINE_UNQUOTED([VIDEO_DISPLAY_ABI_VERSION], $video_display_abi_version, [Specifies ABI version for video displays])
AC_DEFINE_UNQUOTED([VIDEO_CAPTURE_ABI_VERSION], $video_capture_abi_version, [Specifies ABI version for video capture devices])
AC_DEFINE_UNQUOTED([VIDEO_COMPRESS_ABI_VERSION], $video_compress_abi_version, [Specifies ABI version for video compression])
AC_DEFINE_UNQUOTED([VIDEO_DECOMPRESS_ABI_VERSION], $video_decompress_abi_version, [Specifies ABI version for video decompression])
AC_DEFINE_UNQUOTED([AUDIO_CAPTURE_ABI_VERSION], $audio_capture_abi_version, [Specifies ABI version for audio capture])
AC_DEFINE_UNQUOTED([AUDIO_PLAYBACK_ABI_VERSION], $audio_playback_abi_version, [Specifies ABI version for audio playback])
AC_SUBST(video_display_abi_version)
AC_SUBST(video_capture_abi_version)
AC_SUBST(video_compress_abi_version)
AC_SUBST(video_decompress_abi_version)
AC_SUBST(audio_capture_abi_version)
AC_SUBST(audio_playback_abi_version)


# -------------------------------------------------------------------------------------------------
# X Window System
#
# Use autoconf inbuilt X location.  Works v. nicely.  Substitution of X vars
# comes after broken X11 header check and attempted fix.
X_LIBS=
X_OBJ=

case "$host_os" in
	*darwin*)
        ;;
        *)
        AC_PATH_XTRA
        
        if test ! $no_x ; then 
                # Check if X headers are broken.  GCC 2.95 and up reject headers that don't
                # have omit types, X11 headers have implicit int declarations on some systems
                # (e.g. Solaris).
                AC_CACHE_CHECK(whether X11 headers are broken, x11_cv_broken_headers, [
                        SAVED_CFLAGS=$CFLAGS
                        CFLAGS="$X_CFLAGS $CFLAGS"
                        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <X11/Xlib.h>
                        #include <X11/Xutil.h>]], [[return 0]])],[x11_cv_broken_headers=no],[x11_cv_broken_headers=yes])
                        CFLAGS=$SAVED_CFLAGS
                ]);
        
                # If X headers look broken, check if we can use -istdinc.  If so replace 
                # -IXDIR with -istdincXDIR to suppress warnings.  NB it is possible to use
                # -fpermissive, but not in conjunction with -Werror since warnings are still
                # generated.  Could pull local headers through:
                #			 's/^\(extern\) \([[A-Za-z0-9]]*($\)/\1 int \2/' 
                if test $x11_cv_broken_headers = yes -a $GCC = "yes" ; then
                        SAVED_X_CFLAGS=$X_CFLAGS
                        X_CFLAGS=`echo $X_CFLAGS | sed 's%-I%-istdinc%'`
                        AC_CACHE_CHECK(whether -istdinc fixes X11 headers, x11_cv_gcc_istdinc_works, [
                                SAVED_CFLAGS=$CFLAGS
                                CFLAGS="$X_CFLAGS $CFLAGS"
                                AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <X11/Xlib.h>
                                #include <X11/Xutil.h>]], [[return 0]])],[x11_cv_gcc_istdinc_works=yes],[x11_cv_gcc_istdinc_works=no])
                                CFLAGS=$SAVED_CFLAGS
                        ])
                        if test $x11_cv_gcc_istdinc_works = no ; then
                                X_CFLAGS=$SAVED_X_CFLAGS
                        fi
                fi
                X_LIBS+=" -lX11"
                X_OBJ="src/x11_common.o"
                CFLAGS+=""
        fi
        ;;
esac

AC_SUBST(X_CFLAGS)
AC_SUBST(X_LIBS)
AC_SUBST(X_EXTRA_LIBS)
AC_SUBST(X_PRE_LIBS)
AC_SUBST(X_OBJ)

# -------------------------------------------------------------------------------------------------
# Do we have FireWire/DV support?

case "$host_os" in
*freebsd*)
        AC_CHECK_HEADER(dev/firewire/firewire.h,[
                AC_DEFINE([HAVE_FIREWIRE_DV_FREEBSD], [1], [We have Firewire DV])
        ])
	;;
esac

# This should be conditional on finding libdv 
AC_DEFINE([HAVE_DV_CODEC], [1], [We have libdv])

# -------------------------------------------------------------------------------------------------
# Try to find the SDK for the DVS cards
dvs=no
dvs_req=yes
AC_ARG_ENABLE(dvs,
              AS_HELP_STRING([--disable-dvs], [disable DVS (default is auto)]), 
	[dvs_req=$enableval])

case "$host" in
	*darwin*)
		DVS_SDK_SUBDIR="universal"
	;;
  	x86_64-*-linux-*)
		DVS_SDK_SUBDIR="linux-x86_64"
	;;
	i*86-*-linux-*)
		DVS_SDK_SUBDIR="linux-x86"
	;;
esac	

AC_ARG_WITH(dvs,           
	[  --with-dvs=DIR    specify location of DVS SDK],
	[DVS_INC=$withval/development/header
	 DVS_LIB=$withval/${DVS_SDK_SUBDIR}/lib],
	[DVS_INC=${PARENT}/sdk4.0.1.17/development/header
	 DVS_LIB=${PARENT}/sdk4.0.1.17/${DVS_SDK_SUBDIR}/lib])

AC_CHECK_HEADER(${DVS_INC}/dvs_clib.h, FOUND_DVS_H=yes, FOUND_DVS_H=no)

SAVED_LIBS=$LIBS
if test $system = MacOSX
then
	LIBS+=" -framework IOKit"
fi
AC_CHECK_LIB(dvsoem, sv_open, FOUND_DVS_L=yes, FOUND_DVS_L=no, -L${DVS_LIB})
LIBS=$SAVED_LIBS

if test $dvs_req = yes -a $FOUND_DVS_H = yes -a $FOUND_DVS_L = yes
then
	DVS_INC=-I${DVS_INC}
	DVS_LIB=${DVS_LIB}/libdvsoem.a
	DVS_CAP_OBJ="src/video_capture/dvs.o"
	DVS_DISP_OBJ="src/video_display/dvs.o"
        AC_DEFINE([HAVE_DVS], [1], [Build with DVS support])
        DVS_LIB+=" -ldl"
	if test $system = MacOSX
	then
		DVS_LIB+=" -framework IOKit"
	fi
        AC_SUBST(DVS_CAP_LIB_TARGET, "lib/ultragrid/vidcap_dvs.so.$video_display_abi_version")
        AC_SUBST(DVS_DISP_LIB_TARGET, "lib/ultragrid/display_dvs.so.$video_display_abi_version")
        dvs=yes
else
	DVS_INC=
	DVS_LIB=
	DVS_OBJ=
fi

AC_SUBST(DVS_INC)
AC_SUBST(DVS_LIB)
AC_SUBST(DVS_CAP_OBJ)
AC_SUBST(DVS_DISP_OBJ)

# -------------------------------------------------------------------------------------------------
# Blackmagic stuff

DECKLINK_OBJ=
DECKLINK_INC=
DECKLINK_LIB=
DECKLINK_PATH=
decklink=no
decklink_req=yes

AC_ARG_ENABLE(decklink, AS_HELP_STRING([--disable-decklink], [disable DeckLink (default is auto)]), 
	[decklink_req=$enableval])
AC_ARG_WITH(decklink,
        [  --with-decklink=DIR     specify location of DECKLINK],
        [DECKLINK_INC=$withval/include
        ],
        [DECKLINK_INC=""]
        )

case "$host_os" in
	*darwin*)
                if test -z $DECKLINK; then
                        DECKLINK_INC=${PARENT}/DeckLink/Mac
                fi
                OLD_LIBS=$LIBS
                LIBS+=" -framework CoreFoundation ../DeckLink/include/DeckLinkAPIDispatch.cpp"
                AC_LANG_PUSH(C++)
		AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include "../DeckLink/include/DeckLinkAPI.h"
                                   ]], 
		[[IDeckLinkIterator *deckLinkIter = CreateDeckLinkIteratorInstance();]])],FOUND_DECKLINK=yes,FOUND_DECKLINK=no)
                LIBS=$OLD_LIBS
                if test $FOUND_DECKLINK = yes; then
                        DECKLINK_LIB+=" -framework CoreFoundation"
                fi
                AC_LANG_POP(C++)
                ;;
        *)
                if test -z $DECKLINK; then
                        DECKLINK_INC=${PARENT}/DeckLink/include
                fi

                AC_CHECK_FILE(${DECKLINK_INC}/DeckLinkAPI.h, FOUND_DECKLINK_H=yes, FOUND_DECKLINK_H=no)
                AC_CHECK_FILE(${DECKLINK_INC}/DeckLinkAPIDispatch.cpp, FOUND_DECKLINK_SRC=yes, FOUND_DECKLINK_SRC=no)
                AC_CHECK_LIB(DeckLinkAPI, main, FOUND_DECKLINK_L=yes, FOUND_DECKLINK_L=no)
                #if test $FOUND_DECKLINK_H = yes -a $FOUND_DECKLINK_SRC = yes -a $FOUND_DECKLINK_L = yes; then
                if test $FOUND_DECKLINK_H = yes -a $FOUND_DECKLINK_SRC = yes; then
                        FOUND_DECKLINK=yes
                        DECKLINK_LIB+=" -ldl"
                else
                        FOUND_DECKLINK=no
                fi
                ;;
esac

if test $decklink_req = yes -a $FOUND_DECKLINK = yes
then
    DECKLINK_PATH=${DECKLINK_INC}
    DECKLINK_INC=-I${DECKLINK_INC}
    DECKLINK_CAP_OBJ="src/video_capture/decklink.o"
    DECKLINK_DISP_OBJ="src/video_display/decklink.o"
    DECKLINK_COMMON="src/video_capture/DeckLinkAPIDispatch.o"
    AC_DEFINE([HAVE_DECKLINK], [1], [Build with DeckLink support])
    LINKER=$CXX
    AC_SUBST(DECKLINK_CAP_LIB_TARGET, "lib/ultragrid/vidcap_decklink.so.$video_capture_abi_version")
    AC_SUBST(DECKLINK_DISP_LIB_TARGET, "lib/ultragrid/display_decklink.so.$video_display_abi_version")
    decklink=yes
else
    DECKLINK_PATH=
    DECKLINK_INC=
    DECKLINK_CAP_OBJ=
    DECKLINK_DISP_OBJ=
    DECKLINK_LIB=
fi

AC_SUBST(DECKLINK_PATH)
AC_SUBST(DECKLINK_COMMON)
AC_SUBST(DECKLINK_INC)
AC_SUBST(DECKLINK_LIB)
AC_SUBST(DECKLINK_CAP_OBJ)
AC_SUBST(DECKLINK_DISP_OBJ)
AC_SUBST(LINKER)

# -------------------------------------------------------------------------------------------------
# HD-SDI Master Quad stuff

QUAD_PATH=
QUAD_OBJ=
QUAD_INC=
quad=no

case $host in
    *-linux-*)

        AC_ARG_ENABLE(quad, 
              AS_HELP_STRING([--disable-quad], [disable Linsys Quad (default is auto)]), 
            [quad_req=$enableval],
            [quad_req=yes]
            )

        AC_ARG_WITH(decklink,
            [  --with-quad=DIR         specify location of QUAD],
            [QUAD_PATH=$withval
            ],
            [QUAD_PATH=${PARENT}/quad
            ])

        QUAD_INC=${QUAD_PATH}/include

        AC_CHECK_FILE(${QUAD_INC}/master.h, FOUND_QUAD_MASTER_H=yes, FOUND_QUAD_MASTER_H=no)
        AC_CHECK_FILE(${QUAD_INC}/sdi.h, FOUND_QUAD_SDI_H=yes, FOUND_QUAD_SDI_H=no)

        if test $quad_req = yes -a $FOUND_QUAD_MASTER_H = yes -a $FOUND_QUAD_SDI_H = yes
        then
            QUAD_INC="-I$QUAD_INC -I$QUAD_PATH/Examples"
            QUAD_OBJ="$QUAD_PATH/Examples/util.o src/video_capture/quad.o"
            AC_DEFINE([HAVE_QUAD], [1], [Build with Quad support])
            AC_SUBST(QUAD_LIB_TARGET, "lib/ultragrid/vidcap_quad.so.$video_display_abi_version")
            quad=yes
        else
            QUAD_PATH=
            QUAD_INC=
            QUAD_OBJ=
        fi
    ;;
esac

AC_SUBST(QUAD_PATH)
AC_SUBST(QUAD_INC)
AC_SUBST(QUAD_OBJ)

# -------------------------------------------------------------------------------------------------
# DELTACAST stuff

DELTACAST_PATH=
DELTACAST_CAP_OBJ=
DELTACAST_DISP_OBJ=
DELTACAST_INC=
DELTACAST_LIB=
deltacast_req=yes
deltacast=no

case $host in
    *-linux-*)

        AC_ARG_ENABLE(deltacast, 
              AS_HELP_STRING([--disable-deltacast], [disable DELTACAST suppport (default is auto)]), 
            [deltacast_req=$enableval],
            [deltacast_req=yes]
            )

        AC_ARG_WITH(deltacast,
            [  --with-deltacast=DIR         specify location of DELTACAST],
            [DELTACAST_PATH=$withval
            ],
            [DELTACAST_PATH=${PARENT}/VideoMasterHD
            ])

        DELTACAST_INC=${DELTACAST_PATH}/Include
        
        CXXFLAGS_SAVE=$CXXFLAGS
        CXXFLAGS+=" -I$DELTACAST_INC"
        CPPFLAGS_SAVE=$CPPFLAGS
        CPPFLAGS+=" -I$DELTACAST_INC"
        SAVED_LIBS=$LIBS
        
        AC_LANG_PUSH(C++)

        AC_CHECK_HEADERS([VideoMasterHD_Core.h VideoMasterHD_Sdi.h VideoMasterHD_Sdi_Audio.h],
                [], [],
                [[#ifdef HAVE_VIDEOMASTERHD_CORE_H
                # include <VideoMasterHD_Core.h>
                #endif
                #ifdef HAVE_VIDEOMASTERHD_SDI_H
                # include <VideoMasterHD_Sdi.h>
                #endif
                ]])
        AC_CHECK_LIB(videomasterhd, VHD_OpenBoardHandle)
        AC_CHECK_LIB(videomasterhd_audio, VHD_GetNbSamples, [], [], [-lvideomasterhd])
        LIBS=$SAVED_LIBS
        
        AC_LANG_POP(C++)
        
        CXXFLAGS=$CXXFLAGS_SAVE
        CPPFLAGS=$CPPFLAGS_SAVE

        if test $deltacast_req = yes -a "$ac_cv_header_VideoMasterHD_Core_h" = yes -a "$ac_cv_header_VideoMasterHD_Sdi_h" = yes -a "$ac_cv_header_VideoMasterHD_Sdi_Audio_h" = yes \
                -a "$ac_cv_lib_videomasterhd_VHD_OpenBoardHandle" = yes -a "$ac_cv_lib_videomasterhd_audio_VHD_GetNbSamples" = yes
        then
            DELTACAST_INC="-I$DELTACAST_INC"
            DELTACAST_CAP_OBJ="src/video_capture/deltacast.o"
            DELTACAST_DISP_OBJ="src/video_display/deltacast.o"
            DELTACAST_LIB="-lvideomasterhd -lvideomasterhd_audio"
            AC_DEFINE([HAVE_DELTACAST], [1], [Build with DELTACAST support])
            AC_SUBST(DELTACAST_CAP_LIB_TARGET, "lib/ultragrid/vidcap_deltacast.so.$video_display_abi_version")
            AC_SUBST(DELTACAST_DISP_LIB_TARGET, "lib/ultragrid/display_deltacast.so.$video_display_abi_version")
            deltacast=yes
        else
            DELTACAST_PATH=
            DELTACAST_LIB=
            DELTACAST_INC=
            DELTACAST_CAP_OBJ=
            DELTACAST_DISP_OBJ=
        fi
    ;;
esac

AC_SUBST(DELTACAST_PATH)
AC_SUBST(DELTACAST_INC)
AC_SUBST(DELTACAST_LIB)
AC_SUBST(DELTACAST_CAP_OBJ)
AC_SUBST(DELTACAST_DISP_OBJ)



# -------------------------------------------------------------------------------------------------
# QuickTime Stuff
QUICKTIME_INC=
QUICKTIME_LIB=
QUICKTIME_OBJ=
quicktime=no

AC_ARG_ENABLE(quicktime, 
      AS_HELP_STRING([--disable-quicktime], [disable QuickTime support(default is auto)]), 
    [quicktime_req=$enableval],
    [quicktime_req=yes]
    )

case $host_os in
	*darwin*)
		AC_CHECK_HEADERS([QuickTime/QuickTime.h], [FOUND_QUICKTIME_H=yes], [echo "QuickTime headers not found."])
		AC_CHECK_HEADERS([Carbon/Carbon.h], [FOUND_CARBON_H=yes], [echo "Carbon headers not found."])
	
		if test $quicktime_req = yes -a $FOUND_QUICKTIME_H = yes -a $FOUND_QUICKTIME_H = yes
		then
			QUICKTIME_INC=
			QUICKTIME_LIB="-framework QuickTime -framework Carbon -framework AppKit -framework CoreAudio -framework AudioUnit"
			QUICKTIME_OBJ="src/video_capture/quicktime.o src/video_display/quicktime.o"
                        quicktime=yes
		fi
    ;;
esac

AC_SUBST(QUICKTIME_INC)
AC_SUBST(QUICKTIME_LIB)
AC_SUBST(QUICKTIME_OBJ)

# -------------------------------------------------------------------------------------------------
# SDL Stuff
SDL_OBJ=
SDL_LIBS=
sdl=no

AC_ARG_ENABLE(sdl, 
      AS_HELP_STRING([--enable-sdl], [enable SDL(default is enable)]), 
    [sdl_req=$enableval],
    [sdl_req=yes]
    )
#
# version comparison function that works for "." delimited version numbers (no alpha please)
# returns 0 for logically equal version numbers (2 == 2.0.0)
# returns 1 if the first param is later then the second param
# returns 2 if the second param is later than the first param
#
# note the idiotic double brackets; m4 which is under the hood of autoconf swallows one pair of brackets
function cmpversion()
{
    # just in case we get whole numbers, we append a .0 to the end of everything
    p1="${1}.0"
    p2="${2}.0"
    val1=`echo $p1 | sed 's/\([[0-9]]*\)\..*/\1/'`
    val2=`echo $p2 | sed 's/\([[0-9]]*\)\..*/\1/'`

    winner=0
    index=1
    while ( test -n "$val1" ||  test -n "$val2" ) && test "$winner" -eq "0" && test "$index" -lt "10"
    do
      # null is the same as 0 if we are still trying to match something
      if test -z "$val1"
        then
              val1=0
      fi
      if test -z "$val2"
        then
              val2=0
      fi

      # the greater number is always the winner at any equal tuple
      if test "$val1" -gt "$val2"
      then
        winner=1;
      fi
      if test "$val2" -gt "$val1"
      then
        winner=2;
      fi

      # make sure we end this thing on bad input
      if (echo $p1 | grep -q '^[[0-9]]*\.')
      then
        p1=`echo $p1 | sed 's/[[0-9]]*\.\(.*\)/\1/'`
      else
        p1=""
      fi
      if (echo $p2 | grep -q '^[[0-9]]*\.')
      then
        p2=`echo $p2 | sed 's/[[0-9]]*\.\(.*\)/\1/'`
      else
        p2=""
      fi

      let "index = $index + 1"
      val1=`echo $p1 | sed 's/\([[0-9]]*\)\..*/\1/'`
      val2=`echo $p2 | sed 's/\([[0-9]]*\)\..*/\1/'`

    done

    echo $winner
}

case "$host_os" in
        *darwin*)
	# Mac workaround - redefining non-present macro to avoid errors
	AC_DEFUN([AM_PATH_SDL], [$2])
        AC_CHECK_HEADERS([SDL/SDL.h],
                         [AC_CHECK_LIB(SDL, SDL_Init, FOUND_SDL_L=yes, echo "SDL library not found or not usable.", -L/opt/local/lib)],
                         [echo "SDL.h not found."]
                        )
            if test $sdl_req = yes -a "$FOUND_SDL_L" = yes
            then
                CFLAGS="$CFLAGS -D_GNU_SOURCE=1 -D_THREAD_SAFE -I/opt/local/include/SDL"
                SDL_LIBS="-L/opt/local/lib -lSDLmain -lSDL -Wl,-framework,Cocoa"
                SDL_OBJ="$SDL_OBJ src/video_display/sdl.o"
                AC_DEFINE([HAVE_SDL], [1],  [Build with SDL support])
                SDL_VERSION_REQUIRED=1.2.10
                AC_CHECK_FILE(/opt/local/bin/sdl-config, FOUND_SDL_CONFIG=yes, echo "sdl-config not found.")
                if test "$FOUND_SDL_CONFIG" = yes
                then
                    SDL_VERSION_FOUND=`/opt/local/bin/sdl-config --version`
                    RESULT=`cmpversion $SDL_VERSION_FOUND $SDL_VERSION_REQUIRED`
                    if test $RESULT = 1
                    then
                        echo "checking for SDL - version >= $SDL_VERSION_REQUIRED... yes ($SDL_VERSION_FOUND)";
                        AC_DEFINE([HAVE_SDL_1210], [1], [We have SDL version >= 1.21])
                    else
                        echo "checking for SDL - version >= $SDL_VERSION_REQUIRED... no ($SDL_VERSION_FOUND)";
                    fi
                fi
                sdl=yes
            fi
        ;;
        *)
        AC_CHECK_HEADERS([SDL/SDL.h],
                         [AC_CHECK_LIB(SDL, SDL_Init, FOUND_SDL_L=yes, echo "SDL library not found or not usable.")],
                         [echo "SDL.h not found."]
                        )
            if test $sdl_req = yes -a "$FOUND_SDL_L" = yes
            then
                SDL_LIBS="-lSDL"
                SDL_OBJ="$SDL_OBJ src/video_display/sdl.o"
                AC_DEFINE([HAVE_SDL], [1],  [Build with SDL support])
                SDL_VERSION=1.2.10
                AM_PATH_SDL($SDL_VERSION, AC_DEFINE([HAVE_SDL_1210], [1], [We have SDL version >= 1.21]), :)
                sdl=yes
            fi
        ;;
esac

if test $sdl = yes
then
        AC_SUBST(SDL_LIB_TARGET, "lib/ultragrid/display_sdl.so.$video_display_abi_version")
fi

AC_SUBST(SDL_OBJ)
AC_SUBST(SDL_LIBS)

# -------------------------------------------------------------------------------------------------
# Testcard stuff
TESTCARD_OBJ="src/video_capture/testcard.o"
TESTCARD_LIB=
AC_SUBST(TESTCARD_OBJ)
AC_SUBST(TESTCARD_LIB)

SAVED_LIBS=$LIBS
AC_CHECK_LIB(SDL_mixer, Mix_PlayMusic)
LIBS=$SAVED_LIBS
if test "$ac_cv_lib_SDL_mixer_Mix_PlayMusic" = yes
then
        TESTCARD_LIB="-lSDL -lSDL_mixer"
fi

# we build testcard in any case
AC_SUBST(TESTCARD_LIB_TARGET, "lib/ultragrid/vidcap_testcard.so.$video_capture_abi_version")

TESTCARD2_OBJ=
TESTCARD2_LIB=
if test $sdl = yes
then
        TESTCARD2_OBJ="src/video_capture/testcard2.o"
        TESTCARD2_LIB=$SDL_LIB
        SAVED_LIBS=$LIBS
        AC_CHECK_LIB(SDL_ttf, TTF_Init)
        LIBS=$SAVED_LIBS
        if test "$ac_cv_lib_SDL_ttf_TTF_Init" = yes
        then
                TESTCARD2_LIB+=" -lSDL_ttf"
        fi
        AC_SUBST(TESTCARD2_LIB_TARGET, "lib/ultragrid/vidcap_testcard2.so.$video_capture_abi_version")
fi


AC_SUBST(TESTCARD2_OBJ)
AC_SUBST(TESTCARD2_LIB)



# -------------------------------------------------------------------------------------------------
# OpenGL stuff

GL_INC=
GL_OBJ=
GL_LIB=
gl=no

AC_ARG_ENABLE(gl, 
      AS_HELP_STRING([--disable-gl], [disable OpenGL (default is auto)]), 
    [gl_req=$enableval],
    [gl_req=yes]
    )

case "$host_os" in
	*darwin*)
	AC_CHECK_HEADERS([OpenGL/gl.h OpenGL/glext.h],
			 [FOUND_GL_H=yes],
			 [FOUND_GL_H=no], 
			 [
		 	  #include <OpenGL/gl.h>
			  #include <OpenGL/glext.h>
			 ]
			)

	if test $FOUND_GL_H = yes
	then
                AC_CHECK_FILE(/System/Library/Frameworks/OpenGL.framework/OpenGL, FOUND_GL_L=yes, echo "OpenGL framework not found.")

		if test $gl_req = yes -a $FOUND_GL_L = yes
		then
			case "$host_os" in
			*darwin8*)
				GL_LIB="-framework GLUT -framework OpenGL -framework Cocoa"
			;;
			*darwin9*)
				GL_LIB="-framework GLUT -framework OpenGL -framework Cocoa -dylib_file /System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib:/System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib"
			;;
			*)
				GL_LIB="-framework GLUT -framework OpenGL -framework Cocoa"
			;;
			esac

			#FOUND_GLEW_L=no
			#AC_CHECK_HEADERS([/opt/local/include/GL/glew.h],
			#		 [AC_CHECK_LIB(GLEW, main, FOUND_GLEW_L=yes , echo "GLEW library not found or not usable.", -L/opt/local/lib)],
			#		 [echo "GL/glew.h not found."]
			#		)
			AC_CHECK_HEADERS([GLUT/glut.h],
					 [AC_CHECK_FILE(/System/Library/Frameworks/GLUT.framework/GLUT, FOUND_GLUT=yes, echo "GLUT framework not found.")],
					 [echo "GLUT/glut.h not found."]
					)
			if test "$FOUND_GLUT" = yes
			then
				GL_INC="$GL_INC -I/opt/local/include"
				GL_OBJ="$GL_OBJ src/video_display/gl.o"
				AC_DEFINE([HAVE_GL], [1], [Build with OpenGL output])
                                gl=yes
			else
                                GL_LIB=
				echo "OpenGL (GLUT) libraries not found or not usable."
			fi
		else
			echo "OpenGL libraries not found or not usable."
		fi

	else
		echo "OpenGL headers not found or not usable."
	fi
	;;
	*) 
	AC_CHECK_HEADERS([GL/gl.h GL/glext.h],
			 [FOUND_GL_H=yes],
			 [FOUND_GL_H=no], 
			 [
		 	  #include <GL/gl.h>
			  #include <GL/glext.h>
			 ]
			)

	if test $FOUND_GL_H = yes
	then
		AC_CHECK_LIB(GL, glBindTexture, FOUND_GL_L=yes, FOUND_GL_L=no)

		if test $gl_req = yes -a $FOUND_GL_L = yes
		then
			GL_LIB+=" -lGL"
	
			#FOUND_GLEW_L=no
			#AC_CHECK_HEADERS([GL/glew.h],
			#		 [AC_CHECK_LIB(GLEW, main, FOUND_GLEW_L=yes , echo "GLEW library not found or not usable.")],
			#		 [echo "GL/glew.h not found."]
			#		)
			AC_CHECK_HEADERS([GL/glut.h],
					 [AC_CHECK_LIB(glut, glutInit, FOUND_GLUT=yes , echo "GLUT library not found or not usable.")],
					 [echo "GL/glut.h not found."]
					)
			if test "$FOUND_GLUT" = yes
			then
				GL_LIB+=" -lglut"
				GL_OBJ="$GL_OBJ src/video_display/gl.o"
				AC_DEFINE([HAVE_GL], [1], [Build with OpenGL output])
                                gl=yes
			else
				echo "OpenGL (GLUT) libraries not found or not usable."
			fi
		else
			echo "OpenGL libraries not found or not usable."
		fi

	else
		echo "OpenGL headers not found or not usable."
	fi
	;;
esac

if test $gl = yes
then
        AC_SUBST(GL_LIB_TARGET, "lib/ultragrid/display_gl.so.$video_display_abi_version")
fi

AC_SUBST(GL_INC)
AC_SUBST(GL_LIB)
AC_SUBST(GL_OBJ)


# -------------------------------------------------------------------------------------------------
# FastDXT stuff
fastdxt=no
fastdxt_req=yes
FASTDXT_OBJ=

AC_ARG_ENABLE(fastdxt, 
	[  --disable-fastdxt        disable FastDXT],
	[fastdxt_req=$enableval])

AC_ARG_WITH(fastdxt,
	[  --with-fastdxt=DIR      specify location of FastDXT],
	[FASTDXT_INC=$withval
	 FASTDXT_PATH=$withval],
	[FASTDXT_INC=${PARENT}/FastDXT
	 FASTDXT_PATH=${PARENT}/FastDXT])

AC_CHECK_HEADER(${FASTDXT_INC}/libdxt.h, FOUND_FASTDXT_H=yes, FOUND_FASTDXT_H=no)
AC_LANG_PUSH(C++)
AC_CHECK_LIB(dxt, DirectDXT1, FOUND_FASTDXT_L=yes, FOUND_FASTDXT_L=no, -I${FASTDXT_INC} -L${FASTDXT_PATH})
AC_LANG_POP(C++)

if test $fastdxt_req = yes -a $FOUND_FASTDXT_L = yes -a $FOUND_FASTDXT_H = yes
then
	FASTDXT_INC=-I${FASTDXT_INC}
	FASTDXT_LIB="-L${FASTDXT_PATH} -ldxt"
	FASTDXT_OBJ="$FASTDXT_OBJ src/video_compress/fastdxt.o"
        AC_DEFINE([HAVE_FASTDXT], [1], [Build with support for FastDXT])
	LINKER=$CXX
        AC_SUBST(FASTDXT_LIB_TARGET, "lib/ultragrid/vcompress_fastdxt.so.$video_compress_abi_version")
        fastdxt=yes
else
	FASTDXT_INC=
	FASTDXT_LIB=
fi

AC_SUBST(FASTDXT_INC)
AC_SUBST(FASTDXT_LIB)
AC_SUBST(FASTDXT_OBJ)

# -------------------------------------------------------------------------------------------------
# GLSL DXT
rtdxt=no

AC_ARG_ENABLE(rtdxt, 
      AS_HELP_STRING([--disable-rtdxt], [disable Real-Time OpenGL DXT compression (default is auto)]), 
	[rtdxt_req=$enableval],
        [rtdxt_req=yes])

SAVED_LIBS=$LIBS

AC_CHECK_HEADER(GL/glew.h, FOUND_GLEW_H=yes)
AC_CHECK_LIB(GLEW, glewIsSupported, FOUND_GLEW_L=yes)
AC_CHECK_HEADER(GL/glx.h, FOUND_GLX_H=yes)
AC_CHECK_HEADER(GL/gl.h, FOUND_GL_H=yes)
AC_CHECK_LIB(GL, glXCreateNewContext, FOUND_GLX_L=yes)
AC_CHECK_LIB(X11, XCreateWindow)

LIBS=$SAVED_LIBS

# Linux
if test $rtdxt_req = yes -a "$ac_cv_lib_X11_XCreateWindow" = yes -a "$FOUND_GLEW_L" = yes -a "$FOUND_GLEW_H" = yes -a "$FOUND_GLX_L" = yes -a "$FOUND_GLX_H" = yes \
        -a "$FOUND_GL_H" = yes -a "$system" = Linux
then
        RTDXT_COMMON_OBJ="src/glx_common.o"
        RTDXT_LIB=" -lGLEW -lGL -lX11"
        rtdxt=yes
fi

# Mac
if test $rtdxt_req = yes -a  "$system" = MacOSX
then
	RTDXT_LIB="-framework OpenGL -framework Cocoa"
        RTDXT_COMMON_OBJ="src/mac_gl_common.o"
        rtdxt=yes
fi

# common
if test $rtdxt = yes
then
        AC_DEFINE([HAVE_DXT_GLSL], [1], [Build with DXT_GLSL support])
        RTDXT_COMMON_OBJ+=" ../dxt_compress/dxt_util.o"
        RTDXT_COMPRESS_OBJ="../dxt_compress/dxt_encoder.o src/video_compress/dxt_glsl.o"
        RTDXT_DECOMPRESS_OBJ="../dxt_compress/dxt_decoder.o src/video_decompress/dxt_glsl.o"
        DXT_GLSL_CFLAGS="-std=gnu99"
        AC_SUBST(RTDXT_COMPRESS_LIB_TARGET, "lib/ultragrid/vcompress_rtdxt.so.$video_compress_abi_version")
        AC_SUBST(RTDXT_DECOMPRESS_LIB_TARGET, "lib/ultragrid/vdecompress_rtdxt.so.$video_decompress_abi_version")
fi    

AC_SUBST(RTDXT_COMMON_OBJ)
AC_SUBST(RTDXT_COMPRESS_OBJ)
AC_SUBST(RTDXT_DECOMPRESS_OBJ)
AC_SUBST(RTDXT_LIB)
AC_SUBST(DXT_GLSL_CFLAGS)

# -------------------------------------------------------------------------------------------------
# CUDA staff
CUDA_PATH=
CUDA_INC=
CUDA_LIB=
NVCC=
NVCCFLAGS=
CUDA_LIB_PATH=

AC_ARG_WITH(cuda,
            [  --with-cuda=DIR         specify location of CUDA Toolkit],
            [CUDA_PATH=$withval
            ],
            [CUDA_PATH=/usr/local/cuda/])

AC_CHECK_PROG([NVCC], [nvcc], [$CUDA_PATH/bin/nvcc],
                  [],
     [$CUDA_PATH/bin])

AC_CHECK_SIZEOF([int *])

if test -n "$NVCC"
then
	AC_MSG_CHECKING([CUDA Toolkit version - major])
        nvcc_major=`$NVCC --version |grep release|sed  's/^.*release \(@<:@0-9@:>@@<:@0-9@:>@*\).*$/\1/'`
	AC_MSG_RESULT($nvcc_major)
	AC_MSG_CHECKING([CUDA Toolkit version - minor])
        nvcc_minor=`$NVCC --version |grep release|sed  's/^.*release @<:@0-9@:>@@<:@0-9@:>@*\.\(@<:@0-9@:>@@<:@0-9@:>@*\).*$/\1/'`
	AC_MSG_RESULT($nvcc_minor)

        CUDA_INC=-I$CUDA_PATH/include
        if test $ac_cv_sizeof_int_p -eq 8 -a $system = Linux
        then
                CUDA_LIB_PATH="$CUDA_PATH/lib64"
        else
                CUDA_LIB_PATH="$CUDA_PATH/lib"
        fi

        CUDA_LIB="-L$CUDA_LIB_PATH -lcudart"
fi

AC_SUBST(NVCC)
AC_SUBST(NVCCFLAGS)
AC_SUBST(CUDA_PATH)
AC_SUBST(CUDA_LIB)
AC_SUBST(CUDA_INC)

# -------------------------------------------------------------------------------------------------
# JPEG
JPEG_OBJ=
JPEG_COMPRESS_OBJ=
JPEG_DECOMPRESS_OBJ=
JPEG_INC=
JPEG_LIB=

jpeg_req=yes
jpeg=no

AC_ARG_ENABLE(jpeg, 
      AS_HELP_STRING([--disable-jpeg], [disable JPEG compression (default is auto)]), 
	[jpeg_req=$enableval],
        [jpeg_req=yes])

SAVED_CXXFLAGS=$CXXFLAGS
SAVED_CPPFLAGS=$CPPFLAGS
CXXFLAGS+=" $CUDA_INC"
CPPFLAGS+=" $CUDA_INC"
AC_LANG_PUSH(C++)
AC_CHECK_HEADER(npp.h, FOUND_NPP_H=yes, FOUND_NPP_H=no)
AC_CHECK_FILE(../libgpujpeg/libgpujpeg.a)
AC_LANG_POP(C++)
CXXFLAGS=$SAVED_CXXFLAGS
CPPFLAGS=$SAVED_CPPFLAGS

if test -n "$NVCC"
then
        if test \( $nvcc_major -eq 4 -a $nvcc_minor -eq 1 \) -o $version_check = no
        then
                jpeg_cuda_version_ok=yes
        else
                AC_MSG_WARN([*** CUDA Toolkit other than 4.1 detected. Please install v4.1 to use JPEG.])
                HM_VERSION_WARNING
        fi
fi

if test "$jpeg_req" = yes -a -n "$NVCC" -a "$FOUND_NPP_H" = yes -a "$jpeg_cuda_version_ok" = yes -a $ac_cv_file____libgpujpeg_libgpujpeg_a = yes
then
        jpeg=yes
        JPEG_LIB="$CUDA_LIB -lnpp ../libgpujpeg/libgpujpeg.a"
        JPEG_INC=$CUDA_INC
        JPEG_COMMON_OBJ="src/video_compress/jpeg.o"
        JPEG_DECOMPRESS_OBJ="src/video_decompress/jpeg.o "
        LINKER=$CXX
        AC_DEFINE([HAVE_JPEG], [1], [Build with JPEG support])
        AC_SUBST(JPEG_COMPRESS_LIB_TARGET, "lib/ultragrid/vcompress_jpeg.so.$video_compress_abi_version")
        AC_SUBST(JPEG_DECOMPRESS_LIB_TARGET, "lib/ultragrid/vdecompress_jpeg.so.$video_decompress_abi_version")
        POST_COMPILE_MSG+="\n***\nYou have compiled in JPEG support.\nIn order to use use JPEG compression and \
decompression, you will need to have CUDA libraries visible to your OS.\n\
If not done so, you can accomplish this by adding line:\n\
        export LD_LIBRARY_PATH=$CUDA_LIB_PATH:\\\$\$LD_LIBRARY_PATH\n\
to your .bashrc file (in home directory). To take effect immediatelly, you will need to enter:\n\
        exec bash\n***\n"
fi

AC_SUBST(JPEG_COMMON_OBJ)
AC_SUBST(JPEG_COMPRESS_OBJ)
AC_SUBST(JPEG_DECOMPRESS_OBJ)
AC_SUBST(JPEG_INC)
AC_SUBST(JPEG_LIB)

# -------------------------------------------------------------------------------------------------
# SAGE stuff

SAGE_OBJ=
SAGE_INC=
SAGE_LIB=
sage=no

AC_ARG_ENABLE(sage, 
      AS_HELP_STRING([--disable-sage], [disable SAGE (default is auto)]), 
	[sage_req=$enableval],
        [sage_req=yes])

AC_ARG_WITH(sage,
	[  --with-sage=DIR         specify location of SAGE],
	[SAGE_INC=$withval/include
	 SAGE_LIB=$withval/lib],
	[SAGE_INC=${PARENT}/sage3.0/include
	 SAGE_LIB=${PARENT}/sage3.0/lib])

AC_CHECK_FILE(${SAGE_INC}/sail.h, FOUND_SAGE_H=yes, FOUND_SAGE_H=no)
AC_CHECK_LIB(sail, main, FOUND_SAGE_L=yes, FOUND_SAGE_L=no, -I${SAGE_INC} -L${SAGE_LIB})

if test $sage_req = yes -a $FOUND_SAGE_L = yes -a $FOUND_SAGE_H = yes
then
    SAGE_INC=-I${SAGE_INC}
    SAGE_LIB=-"L${SAGE_LIB} -lsail"
    SAGE_OBJ="src/video_display/sage.o src/video_display/sage_wrapper.o"
    AC_DEFINE([HAVE_SAGE], [1], [Build with SAGE support])
    LINKER=$CXX
    AC_SUBST(SAGE_LIB_TARGET, "lib/ultragrid/display_sage.so.$video_display_abi_version")
    sage=yes
else
    SAGE_INC=
    SAGE_LIB=
    SAGE_OBJ=
fi

AC_SUBST(SAGE_INC)
AC_SUBST(SAGE_LIB)
AC_SUBST(SAGE_OBJ)
AC_SUBST(LINKER)

# -------------------------------------------------------------------------------------------------
# AUDIO stuff

PORTAUDIO_INC=
PORTAUDIO_LIB=
PORTAUDIO_CAP_OBJ=
PORTAUDIO_PLAY_OBJ=
portaudio=no

AC_ARG_ENABLE(portaudio, 
      AS_HELP_STRING([--disable-portaudio], [disable Portaudio (default is auto)]), 
	[portaudio_req=$enableval],
        [portaudio_req=yes])

AC_ARG_WITH(portaudio,
	[  --with-portaudio=DIR    specify location of Portaudio],
	[PORTAUDIO_INC=$withval/include
	 PORTAUDIO_LIB=$withval/lib],
	[PORTAUDIO_INC=${PARENT}/portaudio/include
	 PORTAUDIO_LIB=${PARENT}/portaudio/lib/.libs])

AC_CHECK_FILE(${PORTAUDIO_INC}/portaudio.h, FOUND_PORTAUDIO_H=yes, FOUND_PORTAUDIO_H=no)
AC_CHECK_FILE(${PORTAUDIO_LIB}/libportaudio.a, PORTAUDIO_STATIC=yes, PORTAUDIO_STATIC=no)
#AC_CHECK_LIB(portaudio, Pa_Initialize, PORTAUDIO_STATIC=yes, PORTAUDIO_STATIC=no, -L${PORTAUDIO_LIB})

if test $portaudio_req = yes -a $FOUND_PORTAUDIO_H = yes -a $PORTAUDIO_STATIC = yes
then
    PORTAUDIO_INC=-I${PORTAUDIO_INC}
    # we have now portaudio linked statically
    # TODO: consider linking it dynamically
    PORTAUDIO_LIB="${PORTAUDIO_LIB}/libportaudio.a"
    # libs taken from .pc file won't work - there are missing some optional dependencies, eg. jack
    PORTAUDIO_LIB+=" `( . ../portaudio/lib/libportaudio.la ; echo $inherited_linker_flags $dependency_libs | sed 's/\/usr\/lib\/lib\(.*\)\.la/-l\1/')`"
    CFLAGS+=" `cat  ../portaudio/portaudio-2.0.pc|grep '^Cflags: '|sed 's/^Cflags: -I\${includedir}//'`"
    PORTAUDIO_CAP_OBJ="src/audio/capture/portaudio.o"
    PORTAUDIO_PLAY_OBJ="src/audio/playback/portaudio.o"
    AC_DEFINE([HAVE_PORTAUDIO], [1], [Build with Portaudio support])
    portaudio=yes
    AC_SUBST(PORTAUDIO_CAP_LIB_TARGET, "lib/ultragrid/acap_portaudio.so.$audio_capture_abi_version")
    AC_SUBST(PORTAUDIO_PLAY_LIB_TARGET, "lib/ultragrid/aplay_portaudio.so.$audio_playback_abi_version")
else
    PORTAUDIO_INC=
    PORTAUDIO_LIB=
    PORTAUDIO_CAP_OBJ=
    PORTAUDIO_PLAY_OBJ=
fi

AC_SUBST(PORTAUDIO_INC)
AC_SUBST(PORTAUDIO_LIB)
AC_SUBST(PORTAUDIO_CAP_OBJ)
AC_SUBST(PORTAUDIO_PLAY_OBJ)

# -------------------------------------------------------------------------------------------------
# SPEEX
speex=no
speex_req=yes
AC_ARG_ENABLE(speex,
              AS_HELP_STRING([--disable-speex], [disable speex (default is auto)]), 
	[speex_req=$enableval])

SPEEX_PATH=${PARENT}/speex-1.2rc1/
SAVED_CFLAGS=$CFLAGS
SAVED_CPPFLAGS=$CPPFLAGS
CFLAGS+=" -I${SPEEX_PATH}/include"
CPPFLAGS+=" -I${SPEEX_PATH}/include"
AC_CHECK_HEADER(${SPEEX_PATH}/include/speex/speex_resampler.h, FOUND_SPEEX_RES_H=yes, FOUND_SPEEX_RES_H=no)
CFLAGS=$SAVED_CFLAGS
CPPFLAGS=$SAVED_CPPFLAGS

if test $speex_req = yes -a $FOUND_SPEEX_RES_H = yes
then
	SPEEX_INC=-I${SPEEX_PATH}/include
	SPEEX_OBJ="src/audio/resample.o"
	SPEEX_LIB=
        AC_DEFINE([HAVE_SPEEX], [1], [Build with SPEEX support])
        speex=yes
else
	SPEEX_INC=
	SPEEX_LIB=
	SPEEX_OBJ=
fi

AC_SUBST(SPEEX_INC)
AC_SUBST(SPEEX_LIB)
AC_SUBST(SPEEX_OBJ)

# -------------------------------------------------------------------------------------------------
#
# Jack stuff
#

# transport
jack_trans=no
jack=no

AC_ARG_ENABLE(jack-transport, 
      AS_HELP_STRING([--enable-jack-transport], [enable JACK transport (default is disable)]), 
	[jack_trans_req=$enableval],
        [jack_trans_req=no])

AC_CHECK_HEADER(jack/jack.h, FOUND_JACK_H=yes, FOUND_JACK_H=no)
AC_CHECK_LIB(jack, jack_client_new, FOUND_JACK_L=yes, FOUND_JACK_L=no)
if test $jack_trans_req = yes -a $FOUND_JACK_H = yes -a $FOUND_JACK_L = yes
then
    JACK_TRANS_OBJ="src/audio/jack.o"
    JACK_TRANS_LIB="-ljack"
    AC_DEFINE([HAVE_JACK_TRANS], [1], [Build with JACK transport support])
    jack_trans=yes
fi
AC_SUBST(JACK_TRANS_OBJ)
AC_SUBST(JACK_TRANS_LIB)
AC_SUBST(JACK_TRANS_INC)

# sound system
AC_ARG_ENABLE(jack,
      AS_HELP_STRING([--enable-jack], [enable JACK (default is disable)]), 
	[jack_req=$enableval],
        [jack_req=yes])

if test $jack_req = yes -a $FOUND_JACK_H = yes -a $FOUND_JACK_L = yes
then
    JACK_CAP_OBJ="src/audio/capture/jack.o"
    JACK_PLAY_OBJ="src/audio/playback/jack.o"
    JACK_LIB="-ljack"
    AC_DEFINE([HAVE_JACK], [1], [Build with JACK support])
    jack=yes
    AC_SUBST(JACK_CAP_LIB_TARGET, "lib/ultragrid/acap_jack.so.$audio_capture_abi_version")
    AC_SUBST(JACK_PLAY_LIB_TARGET, "lib/ultragrid/aplay_jack.so.$audio_playback_abi_version")
fi

if test $jack = yes -a $speex = no
then
        AC_MSG_WARN([*** You are going to compile JACK but you do not have SPEEX compiled in.
                     It is recommended to compile also SPEEX library and rerun this script in
                     order to enable resampling (if needed).])
fi

AC_SUBST(JACK_CAP_OBJ)
AC_SUBST(JACK_PLAY_OBJ)
AC_SUBST(JACK_LIB)
AC_SUBST(JACK_INC)

# -------------------------------------------------------------------------------------------------
#
# ALSA
#
AC_ARG_ENABLE(alsa, 
      AS_HELP_STRING([--disable-alsa], [disable ALSA (default is auto)]), 
	[alsa_req=$enableval],
        [alsa_req=yes])
AC_CHECK_HEADER(alsa/asoundlib.h)
SAVED_LIBS=$LIBS
AC_CHECK_LIB(asound, snd_pcm_open)
LIBS=$SAVED_LIBS
if test "$alsa_req" = yes -a "$ac_cv_header_alsa_asoundlib_h" = yes -a "$ac_cv_lib_asound_snd_pcm_open" = yes
then
        alsa=yes
        AC_DEFINE([HAVE_ALSA], [1], [Build with ALSA support])
        ALSA_CAP_OBJ="src/audio/capture/alsa.o"
        ALSA_PLAY_OBJ="src/audio/playback/alsa.o"
        ALSA_LIB="-lasound"
        AC_SUBST(ALSA_CAP_LIB_TARGET, "lib/ultragrid/acap_alsa.so.$audio_capture_abi_version")
        AC_SUBST(ALSA_PLAY_LIB_TARGET, "lib/ultragrid/aplay_alsa.so.$audio_playback_abi_version")
else
        alsa=no
        ALSA_OBJ=
        ALSA_LIB=
fi

AC_SUBST(ALSA_CAP_OBJ)
AC_SUBST(ALSA_PLAY_OBJ)
AC_SUBST(ALSA_LIB)

# -------------------------------------------------------------------------------------------------
# CoreAudio Stuff
COREAUDIO_INC=
COREAUDIO_LIB=
COREAUDIO_OBJ=
coreaudio=no

AC_ARG_ENABLE(coreaudio, 
      AS_HELP_STRING([--disable-coreaudio], [disable CoreAudio support(default is auto)]), 
    [coreaudio_req=$enableval],
    [coreaudio_req=yes]
    )

case $host_os in
	*darwin*)
		AC_CHECK_HEADERS([AudioUnit/AudioUnit.h], [FOUND_AUDIOUNIT_H=yes], [echo "AudioUnit headers not found."])
	
		if test $coreaudio_req = yes -a $FOUND_AUDIOUNIT_H = yes
		then
			COREAUDIO_INC=
			COREAUDIO_LIB="-framework CoreAudio -framework AudioUnit"
			COREAUDIO_OBJ="src/audio/capture/coreaudio.o src/audio/playback/coreaudio.o"
                        coreaudio=yes
                        AC_DEFINE([HAVE_COREAUDIO], [1], [Build with ALSA support])
		fi
    ;;
esac

if test $coreaudio = yes -a $speex = no
then
        AC_MSG_WARN([*** You are going to compile CoreAudio but you do not have SPEEX compiled in.
                     It is recommended to compile also SPEEX library and rerun this script in
                     order to enable resampling (if needed).])
fi

AC_SUBST(COREAUDIO_INC)
AC_SUBST(COREAUDIO_LIB)
AC_SUBST(COREAUDIO_OBJ)


# -------------------------------------------------------------------------------------------------
# We need to add libraries then

# this is only needed when passing to "clean" make target
LIB_OBJS="$QUICKTIME_OBJ $SDL_OBJ $GL_OBJ $SAGE_OBJ $TESTCARD_OBJ $TESTCARD2_OBJ $DECKLINK_CAP_OBJ $DECKLINK_DISP_OBJ $DECKLINK_COMMON $QUAD_OBJ "\
"$DELTACAST_CAP_OBJ $DELTACAST_DISP_OBJ $DVS_CAP_OBJ $DVS_DISP_OBJ $FASTDXT_OBJ $RTDXT_COMPRESS_OBJ $JPEG_COMPRESS_OBJ $RTDXT_COMMON_OBJ $RTDXT_DECOMPRESS_OBJ $JPEG_COMMON_OBJ $JPEG_DECOMPRESS_OBJ "\
"$ALSA_CAP_OBJ $ALSA_PLAY_OBJ $PORTAUDIO_CAP_OBJ $PORTAUDIO_PLAY_OBJ $JACK_CAP_OBJ $JACK_PLAY_OBJ"

if test "$build_libraries" = yes
then
        LIB_TARGETS="$SDL_LIB_TARGET $GL_LIB_TARGET $TESTCARD_LIB_TARGET $TESTCARD2_LIB_TARGET $DECKLINK_CAP_LIB_TARGET $DECKLINK_DISP_LIB_TARGET $QUAD_LIB_TARGET "\
"$DELTACAST_CAP_LIB_TARGET $DELTACAST_DISP_LIB_TARGET $DVS_CAP_LIB_TARGET $DVS_DISP_LIB_TARGET $SAGE_LIB_TARGET $FASTDXT_LIB_TARGET $RTDXT_COMPRESS_LIB_TARGET $JPEG_COMPRESS_LIB_TARGET "\
"$RTDXT_DECOMPRESS_LIB_TARGET $JPEG_DECOMPRESS_LIB_TARGET $ALSA_CAP_LIB_TARGET $ALSA_PLAY_LIB_TARGET $PORTAUDIO_PLAY_LIB_TARGET $PORTAUDIO_CAP_LIB_TARGET $JACK_CAP_LIB_TARGET "\
"$JACK_PLAY_LIB_TARGET"
else
        LIBS+=" $GL_LIB $SDL_LIBS $DECKLINK_LIB $DELTACAST_LIB $QUICKTIME_LIB $SAGE_LIB $TESTCARD_LIB $TESTCARD2_LIB $DECKLINK_LIB $DELTACAST_LIB $DVS_LIB $FASTDXT_LIB $RTDXT_LIB $JPEG_LIB $ALSA_LIB "\
"$PORTAUDIO_LIB $JACK_LIB"
        OBJS+=" $LIB_OBJS"
        LIB_OBJS=
        LIB_TARGETS=
fi

AC_SUBST(LIB_OBJS)
AC_SUBST(LIB_TARGETS)


# -------------------------------------------------------------------------------------------------
# Finally, substitute things into the Makefile and config.h
AC_CONFIG_HEADER(src/config.h)
AC_CONFIG_FILES([Makefile])
AC_OUTPUT

AC_MSG_RESULT([
Configuration summary:

  Target ...................... $host
  Debug output ................ $debug_output
  Profiling support ........... $profile
  IPv6 support ................ $ipv6
  RT priority ................. $use_rt
  Standalone modules .......... $build_libraries

  DeckLink .................... $decklink
  DELTACAST ................... $deltacast
  DVS ......................... $dvs
  Linsys Quad ................. $quad
  OpenGL ...................... $gl
  Quicktime ................... $quicktime
  SAGE ........................ $sage
  SDL ......................... $sdl
  
  Portaudio ................... $portaudio
  ALSA ........................ $alsa
  CoreAudio ................... $coreaudio
  JACK  ....................... $jack
  JACK transport .............. $jack_trans
  speex resampler ............. $speex
  
  FastDXT ..................... $fastdxt
  Realtime DXT (OpenGL) ....... $rtdxt
  JPEG ........................ $jpeg
])

