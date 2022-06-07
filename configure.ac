dnl Process this file with autoconf to produce a configure script.

AC_INIT([UltraGrid],[1.7],[ultragrid-dev@cesnet.cz])
configure_flags="$*"
AC_CANONICAL_TARGET
AM_INIT_AUTOMAKE([1.10])
AC_PREREQ([2.61])
AC_CONFIG_SRCDIR([src/main.cpp])
AC_CONFIG_MACRO_DIR([m4])

PARENT=`echo $PWD | sed -e 's%/[[^/]]*$%%'`

AC_DEFINE_UNQUOTED([CONFIG_FLAGS], ["$configure_flags"], [Flags passed to configure])

AC_CANONICAL_HOST
AC_PROG_CC
AC_PROG_CPP
AC_PROG_CXX
AC_PROG_CXXCPP
AC_PROG_EGREP
AC_PROG_INSTALL
AC_C_CONST
AC_TYPE_SIZE_T
AC_C_BIGENDIAN
AC_C_CHAR_UNSIGNED
PKG_PROG_PKG_CONFIG

# -------------------------------------------------------------------------------------------------
POST_COMPILE_MSG=""
CFLAGS="$CFLAGS -I$srcdir"
CPPFLAGS="$CPPFLAGS -I$srcdir"
CXXFLAGS="$CXXFLAGS -I$srcdir -std=gnu++17"
LIB_MODULES=
AC_SUBST(INC)
AC_SUBST(LIBS)
AC_SUBST(OBJS)
AC_SUBST(POST_COMPILE_MSG)
AC_SUBST(COMMON_FLAGS)

AC_CHECK_SIZEOF([int *])

AC_C_BIGENDIAN(
        AC_DEFINE([WORDS_BIGENDIAN], 1, [This is big endian system]),
        AC_DEFINE([WORDS_SMALLENDIAN], 1, [This is little endian system])
)

if test "$ac_cv_prog_cc_c99" = no
then
        AC_MSG_ERROR([Your compiler does not accept C99.])
fi

AC_CHECK_SIZEOF([size_t])

if test $ac_cv_sizeof_size_t -eq 8
then
        WORD_LEN=64
else
        WORD_LEN=32
fi

AC_DEFUN([HM_VERSION_WARNING], [
          AC_MSG_WARN([If you still want to try with unsupported version of dependency,
                       you may want to try --disable-depends-version-check option.
                       If you find out that some unsupported version is working,
                       please let us know at ultragrid-dev@cesnet.cz.
                               ])
          ])

WARNINGS=
# variant of AC_MSG_WARN that in addition stores the message for repost at the end
m4_defun([UG_MSG_WARN], [
          AC_MSG_WARN([$1])
          WARNINGS="${WARNINGS}$1\n"
          ])

m4_defun([MY_CHECK_FRAMEWORK],
        [AC_CACHE_CHECK([if -framework ]$1[ works],[my_cv_framework_]$1,
                [save_LIBS="$LIBS"
                LIBS="$LIBS -framework ]$1["
                AC_LINK_IFELSE([AC_LANG_PROGRAM([],[])],
                        [my_cv_framework_]$1[=yes],
                        [my_cv_framework_]$1[=no])
                ])
        LIBS="$save_LIBS"
        if test "$my_cv_framework_]$1["="yes"; then
        AC_DEFINE(AS_TR_CPP([HAVE_FRAMEWORK_]$1),1,
                [Define if you have the ] $1 [ framework])
        AS_TR_CPP([FRAMEWORK_]$1)="-framework ]$1["
        AC_SUBST([FRAMEWORK_]AS_TR_CPP($1))
        fi])

AC_MSG_CHECKING([GIT presence])
if which git
then
        AC_MSG_CHECKING([checking GIT branch ])
        git_branch=`SRC=$(dirname "$0"); git -C "$SRC" name-rev --name-only HEAD`
        if test "$git_branch"; then
                AC_DEFINE_UNQUOTED([GIT_BRANCH], "$git_branch", [Current GIT branch])
                AC_MSG_RESULT($git_branch);
        else
                AC_MSG_RESULT(no);
        fi
        AC_MSG_CHECKING([checking GIT revision ])
        git_rev=`SRC=$(dirname "$0"); git -C "$SRC" rev-parse --short HEAD`
        if test "$git_rev"; then
                AC_DEFINE_UNQUOTED([GIT_REV], "$git_rev", [Current GIT revision])
                AC_MSG_RESULT($git_rev);
        else
                AC_MSG_RESULT(no);
        fi
fi

AC_MSG_CHECKING([OS family])
if test "$host_vendor" = "apple"; then
        system=MacOSX
	AC_DEFINE([HAVE_MACOSX], [1], [This is Mac X OS])
        APPEXT=.app
elif expr "$host_os" : ".*mingw32.*" > /dev/null || expr "$host_os" : ".*msys.*" > /dev/null; then
        system=Windows
        APPEXT=.exe
	AC_DEFINE([WIN32], [1], [This is an Windows OS])
else
        system=Linux
	AC_DEFINE([HAVE_LINUX], [1], [This is Linux])
fi
AC_MSG_RESULT($system);
AC_SUBST(system)

if test $build != $host; then
        cross_compile=yes # don't confuse with autoconf's $cross_compiling - ours is more correct
                          # because $cross_compiling=no if we are cross-compiling but are able to
                          # execute the compiled binary (eg. using wine-binfmt)
else
        cross_compile=no
fi

AC_MSG_CHECKING([OS kernel version (major)])
os_version_major=`uname -r |cut -d . -f 1`
AC_MSG_RESULT($os_version_major);
AC_MSG_CHECKING([OS kernel version (minor)])
os_version_minor=`uname -r |cut -d . -f 2`
AC_MSG_RESULT($os_version_minor);

if test $system = Linux; then
	LDFLAGS="$LDFLAGS -Wl,--dynamic-list-data"
fi

if test $system = MacOSX; then
        MACOS_LEGACY=no
        if test $os_version_major -lt 19; then # Darwin 19.0.0 is macOS 10.15
                MACOS_LEGACY=yes
                MIN_MACOS_VER=10.7
        else
                MIN_MACOS_VER=10.15
        fi
        COMMON_OSX_FLAGS="-iframework data/redists -mmacosx-version-min=$MIN_MACOS_VER -stdlib=libc++"
	CFLAGS="$CFLAGS $COMMON_OSX_FLAGS"
	CPPFLAGS="$CPPFLAGS $COMMON_OSX_FLAGS"
	CXXFLAGS="$CXXFLAGS $COMMON_OSX_FLAGS"
        if test $WORD_LEN = 32; then
                CUDA_FLAGS="$CUDA_FLAGS -Xcompiler -Wno-error=unused-command-line-argument-hard-error-in-future"
        fi
	LDFLAGS="$LDFLAGS $COMMON_OSX_FLAGS -headerpad_max_install_names"
        OBJS="$OBJS src/utils/autorelease_pool.o"
        AC_SUBST(MACOS_LEGACY)
fi

if test $system = Windows; then
	NET_LIBS="-lsetupapi -lws2_32 -liphlpapi -loleaut32"
	LIBS="$LIBS $NET_LIBS"
	AC_CHECK_FUNCS(SetThreadDescription)
fi
AC_SUBST(NET_LIBS)

LINKER=$CXX

AC_SUBST(LINKER)
AC_SUBST(LDFLAGS)
AC_SUBST(CXXFLAGS)

AC_ARG_ENABLE(depends-version-check,
        AS_HELP_STRING([--disable-depends-version-check], [Disable libraries version check]),
	[version_check=$enableval],
        [version_check=yes]
        )

build_default=auto
AC_ARG_ENABLE(all,
        AS_HELP_STRING([--enable-all], [Enable or disable all features (default is auto)]),
	[build_default=$enableval],
        [build_default=auto]
        )

if test $system = MacOSX -a \( $os_version_major -lt 11 \) -a $version_check = yes
then
        AC_MSG_WARN([*** Only OS X versions 10.7 and higher are supported.])
        HM_VERSION_WARNING
        AC_MSG_ERROR([Exiting due to unsatisfied dependency.])
fi

if test $system = MacOSX -o $system = Linux
then
        CFLAGS="$CFLAGS -fPIC"
        CXXFLAGS="$CXXFLAGS -fPIC"
        CUDA_FLAGS="$CUDA_FLAGS -Xcompiler -fPIC"
        CU_OBJ_SUFFIX="o"
else # Windows
        if test $WORD_LEN = 32; then
                CUDA_FLAGS="-m32"
        fi
        CU_OBJ_SUFFIX="lib"
fi
AC_SUBST(CUDA_FLAGS)

AC_ARG_VAR([ARCH], [Architecture to build for])
if test $target_cpu = x86_64 -o $target_cpu = i686
then
        test -z "$ARCH" && ARCH="-msse4.1"
fi
CFLAGS="$CFLAGS $ARCH"
CXXFLAGS="$CXXFLAGS $ARCH"
LDFLAGS="$LDFLAGS $ARCH"

if expr "$CFLAGS" : '.*-O' >/dev/null; then
        OFAST=""
else
        OFAST="-Ofast"
fi
AC_SUBST(OFAST)

AC_CHECK_HEADERS([termios.h])

# Adds the module to UltraGrid build system.
# @param $1 name of the module, should be in format <class>_<name> unless
#           sure that the modular build won't take place
# @param $2 object files
# @param $3 libraries
AC_DEFUN([ADD_MODULE], [
          TARGET="lib/ultragrid/ultragrid_"$1".so"
          LIB_OBJS=$LIB_OBJS" "$2
          LIB_MODULES=$LIB_MODULES" "$3
	  AM_CONDITIONAL([IS_TARGET], [test x$1 != x])
	  AM_COND_IF([IS_TARGET],
		                [
				 MODULES="$MODULES $TARGET"
				 TARGETS="$TARGETS
$TARGET: "$2"
	\$(MKDIR_P) lib/ultragrid
	\$(LINKER) \$(LDFLAGS) -shared -Wl,-soname,ultragrid_"$1.so" "$2" -o lib/ultragrid/ultragrid_"$1".so "$3"
"
				 ])
          ])

# ---------------------------------------------------------------------
# Dylib bundler
# ---------------------------------------------------------------------
if test $system = MacOSX; then
        AC_PATH_PROGS(DYLIBBUNDLER, dylibbundler, [true])
        if test "$DYLIBBUNDLER" = true; then
                UG_MSG_WARN([Cannot find dylibbundler. We cannot put libraries into a bundle so it probably will not be portable.])
        else
                AC_MSG_CHECKING([dylibbundler version])
                dyl_ver=$($DYLIBBUNDLER -V >/dev/null 2>&1 && dylibbundler -V | cut -d\  -f2 || echo unknown)
                AC_MSG_RESULT($dyl_ver);

                if test $dyl_ver = unknown || expr "$dyl_ver" \<= 2 >/dev/null; then
                        UG_MSG_WARN([Original dylib bundler detected, recommending v2 instead (https://github.com/SCG82/macdylibbundler)])
                fi
        fi
fi

AC_SUBST(DYLIBBUNDLER)

# -------------------------------------------------------------------------------------------------
# Additional macros
# -------------------------------------------------------------------------------------------------
AC_DEFUN([CUDA_MESSAGE], [
          if test -z "$cuda_msg_defined" -a $system != Windows; then
                  POST_COMPILE_MSG="$POST_COMPILE_MSG\n***\nYou have compiled in sort of CUDA code.\nIn order to use use it compression and \
decompression, you will need to have CUDA libraries visible to your OS.\n\
If not done so, you can accomplish this by adding line:\n\
        export LD_LIBRARY_PATH=$CUDA_LIB_PATH:\\\$\$LD_LIBRARY_PATH\n\
to your .bashrc file (in home directory). To take effect immediatelly, you will need to enter:\n\
        exec bash\n***\n"
                  cuda_msg_defined=yes
          fi
        ])

AC_DEFUN([DEFINE_CUDA], [
          if test -z "$cuda_var_defined"; then
                  AC_DEFINE([HAVE_CUDA], [1], [CUDA is present on the system])
                  if test $system = Windows; then
                          # It is twice by intent - as an object to force dependency
                          # (== build the object), as a lib to be at the end of the
                          # link list (it is a library) in MSW. TODO: make it prettier
                          # somehow. The same applies also for other CUDA objs/libs.
                          CUDA_COMMON_OBJ=src/cuda_wrapper.$CU_OBJ_SUFFIX
                          CUDA_COMMON_LIB=src/cuda_wrapper.$CU_OBJ_SUFFIX
                          DLL_LIBS="$DLL_LIBS src/cuda_wrapper.dll"
                  else
                          CUDA_COMMON_OBJ=src/cuda_wrapper.$CU_OBJ_SUFFIX
                          CUDA_COMMON_LIB=
                  fi
                  cuda_var_defined=yes
          fi
        ])
AC_SUBST(DLL_LIBS)

if test $system = Windows
then
        AC_MSG_CHECKING([if_nametoindex])
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
                           #include <windows.h>
                           #include <iphlpapi.h>
                           ]], [[
                                if_nametoindex("eth");]]
                           )], HAVE_IF_NAMETOINDEX=yes, HAVE_IF_NAMETOINDEX=no)
        AC_MSG_RESULT([$HAVE_IF_NAMETOINDEX])
        if test $HAVE_IF_NAMETOINDEX = yes; then
                AC_DEFINE([HAVE_IF_TONAMEINDEX], 1, [Function if_nametoindex is present.])
        fi
else
        AC_CHECK_FUNCS(if_nametoindex)
fi

AC_CHECK_FUNCS(usleep)
AC_CHECK_FUNCS(strtok_r)
AC_CHECK_FUNCS(timespec_get)

AC_CHECK_FUNCS(drand48)
if test $ac_cv_func_drand48 = no
then
	AC_DEFINE([NEED_DRAND48], 1, [We need custom implementation of drand48.])
fi

# ---------------------------------------------------------------------
# Common shell utility functions
# ---------------------------------------------------------------------
# Be careful about [] m4 processing - either use quadrigraphs or enclose macro with additional []

# remove redundant spaces from argument (multiplied, leading, trailing)
[compact_spaces_pipe() { sed -e 's/  */ /g' -e 's/^ //' -e 's/ $//' ;}]
[compact_spaces() { echo "$1" | compact_spaces_pipe ;}]
# remove "-mwindows" from lib flags obtained with pkg-config
# because it causes output to be scratched by cmd/pwsh (but
# not MSYS2 terminal)
remove_mwindows() {
        echo "$1" | sed 's/-mwindows//' | compact_spaces_pipe
}

# -------------------------------------------------------------------------------------------------
# GCC-specific warning flags
if test "$GCC" = yes
then
    CFLAGS="$CFLAGS -pipe -W -Wcast-qual -Wcast-align -Wbad-function-cast -Wmissing-prototypes -Wmissing-declarations"
    CXXFLAGS="$CXXFLAGS -pipe -W -Wcast-qual -Wcast-align -Wmissing-declarations"
fi

# ------------------------------------------------------------------------------------------------
# Dynamic libraries stuff
#
build_libraries=no
AC_ARG_ENABLE(plugins, AS_HELP_STRING([--enable-plugins], [build modules as libraries (default is off)]),
	[build_libraries=$enableval])

if test "$build_libraries" = yes
then
        if test "$system" = "Linux"
        then
                LIBS="$LIBS -ldl"
                LDFLAGS="$LDFLAGS -rdynamic"
        else
                UG_MSG_WARN([Building libraries is not supported with other system than Linux])
                build_libraries=no
        fi
        AC_DEFINE([BUILD_LIBRARIES], [1], [Build drivers as a standalone libraries])
fi


# -------------------------------------------------------------------------------------------------
# Check for libraries

AC_SEARCH_LIBS(socket, socket)
AC_SEARCH_LIBS(inet_addr, nsl)
AC_CHECK_LIB(rt, timer_create)

AC_CHECK_FUNC(sin, MATHLIBS="", MATHLIBS="-lm")
AC_SUBST(MATHLIBS)

# -------------------------------------------------------------------------------------------------
# See if this system supports sched_setscheduler()

use_rt=no
AC_ARG_ENABLE(rt,
              AS_HELP_STRING([--enable-rt], [enable RT priority]),
	[use_rt=$enableval])
if test "$use_rt" = "yes"; then
        AC_CHECK_FUNCS(sched_setscheduler)
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
ipv6=yes

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
              AS_HELP_STRING([--disable-ipv6], [disable ipv6]),
              [if test $enableval = no
               then
                       ipv6=no
               fi])

if test "$ipv6" != no
then
        AC_DEFINE([HAVE_IPv6], [1], [Enable IPv6 support])
fi

if command -v realpath > /dev/null; then
        srcdir_abs=`realpath $srcdir`
else
        srcdir_abs=`python -c 'import os; print(os.path.realpath("'$srcdir'"))'`
fi
AC_SUBST(srcdir_abs)

# ----------------------------------------------------------------------
# environment variables
# ----------------------------------------------------------------------
AC_ARG_VAR([AJA_DIRECTORY], [Directory to AJA NTV2 SDK.])
AC_ARG_VAR([BLUE_LATEST_SDK], [Placement of Bluefish444 SDK.])
AC_ARG_VAR([CINEFORM_DIRECTORY], [Directory to Cineform SDK.])
AC_ARG_VAR([CUDA_PATH], [Directory of your Nvidia toolkit instalation.])
AC_ARG_VAR([DELTACAST_DIRECTORY], [Placement of VideoMasterHD directory (Deltacast).])
AC_ARG_VAR([DVS_DIRECTORY], [Path to DVS installation.])
AC_ARG_VAR([GENICAM_GENTL64_PATH], [XIMEA SDK library path])
AC_ARG_VAR([NTV2_ROOT], [Directory to AJA NTV2 SDK (ends with ntv2projects).])
AC_ARG_VAR([SAGE_DIRECTORY], [Directory of your SAGE installation.])

# -------------------------------------------------------------------------------------------------
# GL common
# -------------------------------------------------------------------------------------------------
AC_ARG_ENABLE(gl,
[  --disable-gl            disable all OpenGL stuff (default is auto)]
[                          Requires:  glew gl (Linux + Win), none (macOS)],
    [gl_req=$enableval],
    [gl_req=$build_default]
    )
OPENGL=no

SAVED_LIBS=$LIBS
AC_CHECK_HEADER(GL/gl.h)
AC_CHECK_HEADER(GL/glew.h)
AC_CHECK_HEADER(GL/glx.h)
AC_CHECK_LIB(GL, glBindTexture)
AC_CHECK_LIB(GL, glXCreateNewContext)
AC_CHECK_LIB(GLEW, glewInit)
AC_CHECK_LIB(glew32, glewInit)
AC_CHECK_LIB(X11, XCreateWindow)
LIBS=$SAVED_LIBS

GL_COMMON_OBJ="src/gl_context.o"

if test "$ac_cv_header_GL_gl_h" = yes -a \
        "$ac_cv_lib_GL_glBindTexture" = yes
then
        HAVE_GL=yes
fi

if test "$ac_cv_header_GL_glew_h" = yes -a \
        \( "$ac_cv_lib_GLEW_glewInit" = yes -o "$ac_cv_lib_glew32_glewInit" = yes \)
then
        HAVE_GLEW=yes
fi

# Linux
if test "$system" = Linux -a \
        "$HAVE_GL" = yes -a \
        "$HAVE_GLEW" = yes -a \
        "$ac_cv_header_GL_glx_h" = yes -a \
        "$ac_cv_lib_GL_glXCreateNewContext" = yes -a \
        "$ac_cv_lib_X11_XCreateWindow" = yes
then
        OPENGL=yes
        OPENGL_LIB="-lGLEW -lGL -lX11"
        GL_COMMON_OBJ="$GL_COMMON_OBJ src/glx_common.o src/x11_common.o"
fi

# Mac
if test "$system" = MacOSX
then
        OPENGL=yes
        OPENGL_LIB="-framework Cocoa -framework GLUT -framework OpenGL"
        COMMON_FLAGS="$COMMON_FLAGS -DGL_SILENCE_DEPRECATION"
        GL_COMMON_OBJ="$GL_COMMON_OBJ src/mac_gl_common.o"
fi

# Win32
if test "$system" = Windows -a "$HAVE_GLEW" = yes
then
        OPENGL=yes
        GL_COMMON_OBJ="$GL_COMMON_OBJ src/win32_gl_common.o"
        OPENGL_LIB="-lopengl32 -lglew32 -lgdi32"
fi

if test $gl_req = no; then
        OPENGL=no
        GL_COMMON_OBJ=
        OPENGL_LIB=
fi

if test $gl_req = yes -a $OPENGL = no; then
        AC_MSG_ERROR([OpenGL stuff requested but dependencies were not found.])
fi

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
dvs_req=$build_default
AC_ARG_ENABLE(dvs,
[  --disable-dvs           disable DVS (default is auto)]
[                          Requires: DVS_SDK],
	[dvs_req=$enableval])

case "$system" in
	MacOSX)
		DVS_SDK_SUBDIR="universal"
	;;
  	Linux)
		DVS_SDK_SUBDIR="linux-x86_64"
	;;
	Windows)
		DVS_SDK_SUBDIR="win32"
        ;;
esac

if test -n "$DVS_DIRECTORY"
then
        DVS_HDR_PATH=$DVS_DIRECTORY/development/header
        DVS_LIB=$DVS_DIRECTORY/${DVS_SDK_SUBDIR}/lib
else
        DVS_HDR_PATH=sdk4.0.1.17/development/header
	DVS_LIB=sdk4.0.1.17/${DVS_SDK_SUBDIR}/lib
fi

AC_ARG_WITH(dvs,
	[  --with-dvs=DIR          specify location of DVS SDK],
	[DVS_HDR_PATH=$withval/development/header
	 DVS_LIB=$withval/${DVS_SDK_SUBDIR}/lib],
        )

SAVED_CFLAGS=$CFLAGS
SAVED_CPPFLAGS=$CPPFLAGS
CFLAGS="$CFLAGS -I${DVS_HDR_PATH}"
CPPFLAGS="$CPPFLAGS -I${DVS_HDR_PATH}"
AC_CHECK_HEADER(dvs_clib.h, FOUND_DVS_H=yes, FOUND_DVS_H=no)
CFLAGS=$SAVED_CFLAGS
CPPFLAGS=$SAVED_CPPFLAGS

SAVED_LIBS=$LIBS
if test $system = MacOSX
then
	LIBS="$LIBS -framework IOKit"
fi
AC_CHECK_LIB(dvsoem, sv_open, FOUND_DVS_L=yes, FOUND_DVS_L=no, -L${DVS_LIB})
LIBS=$SAVED_LIBS

if test $dvs_req != no -a $FOUND_DVS_H = yes -a $FOUND_DVS_L = yes
then
	DVS_INC=-I${DVS_HDR_PATH}
	DVS_LIB="-L${DVS_LIB} -ldvsoem"
        if test $system = Linux; then
                DVS_LIB="$DVS_LIB -ldl"
        fi
	DVS_CAP_OBJ="src/video_capture/dvs.o"
	DVS_DISP_OBJ="src/video_display/dvs.o"
        AC_DEFINE([HAVE_DVS], [1], [Build with DVS support])
        DVS_LIB="$DVS_LIB"
	if test $system = MacOSX
	then
		DVS_LIB="$DVS_LIB -framework IOKit"
	fi
        ADD_MODULE("vidcap_dvs", "$DVS_CAP_OBJ", "$DVS_LIB")
        ADD_MODULE("display_dvs", "$DVS_DISP_OBJ", "$DVS_LIB")
	INC="$INC $DVS_INC"
        dvs=yes
fi

if test $dvs_req = yes -a $dvs = no; then
        AC_MSG_ERROR([DVS not found.])
fi

# -------------------------------------------------------------------------------------------------
# Blackmagic stuff

DECKLINK_OBJ=
DECKLINK_INC=
DECKLINK_LIB=
DECKLINK_PATH=
decklink=no
decklink_req=$build_default

AC_ARG_ENABLE(decklink,
[  --disable-decklink      disable DeckLink (default is auto)]
[                          Requires: DesktopVideo],
	[decklink_req=$enableval])
DECKLINK_INC=$srcdir/ext-deps/DeckLink

case "$system" in
	MacOSX)
                DECKLINK_INC=$DECKLINK_INC/Mac
                OLD_LIBS=$LIBS
                LIBS="$LIBS -framework CoreFoundation"
                AC_LANG_PUSH(C++)
		AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include "$DECKLINK_INC/DeckLinkAPI.h"
                                   ]],
		[[IDeckLinkIterator *deckLinkIter = CreateDeckLinkIteratorInstance();]])],FOUND_DECKLINK=yes,FOUND_DECKLINK=no)
                LIBS=$OLD_LIBS
                if test $FOUND_DECKLINK = yes; then
                        DECKLINK_LIB="$DECKLINK_LIB -framework CoreFoundation"
                fi
                AC_LANG_POP(C++)
                ;;
	Windows)
                DECKLINK_INC=$DECKLINK_INC/Windows
                SAVED_LIBS=$LIBS
                LIBS="$LIBS -lole32"
                AC_LANG_PUSH(C++)
		AC_MSG_CHECKING([DeckLink usability])
		AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
#include "$DECKLINK_INC/DeckLinkAPI_h.h"
#include "$DECKLINK_INC/DeckLinkAPIVersion.h"
#include <objbase.h>
                                   ]],
		[[IDeckLinkIterator *deckLinkIter; CoCreateInstance(CLSID_CDeckLinkIterator, NULL, CLSCTX_ALL, IID_IDeckLinkIterator, (void **) deckLinkIter);]])],FOUND_DECKLINK=yes,FOUND_DECKLINK=no)
                LIBS=$SAVED_LIBS
                if test $FOUND_DECKLINK = yes; then
                        DECKLINK_LIB="$DECKLINK_LIB -lole32"
                fi
		AC_MSG_RESULT([$FOUND_DECKLINK])
                AC_LANG_POP(C++)
                ;;
        *)
                DECKLINK_INC=$DECKLINK_INC/Linux

                AC_CHECK_FILE(${DECKLINK_INC}/DeckLinkAPI.h, FOUND_DECKLINK_H=yes, FOUND_DECKLINK_H=no)
                AC_CHECK_FILE(${DECKLINK_INC}/DeckLinkAPIDispatch.cpp, FOUND_DECKLINK_SRC=yes, FOUND_DECKLINK_SRC=no)
                AC_CHECK_LIB(DeckLinkAPI, main, FOUND_DECKLINK_L=yes, FOUND_DECKLINK_L=no)
                #if test $FOUND_DECKLINK_H = yes -a $FOUND_DECKLINK_SRC = yes -a $FOUND_DECKLINK_L = yes; then
                if test $FOUND_DECKLINK_H = yes -a $FOUND_DECKLINK_SRC = yes; then
                        FOUND_DECKLINK=yes
                        DECKLINK_LIB="$DECKLINK_LIB -ldl"
                else
                        FOUND_DECKLINK=no
                fi
                ;;
esac

if test $decklink_req != no -a $FOUND_DECKLINK = yes
then
    DECKLINK_PATH=${DECKLINK_INC}
    DECKLINK_INC=-I${DECKLINK_INC}
    DECKLINK_CAP_OBJ="src/blackmagic_common.o src/video_capture/decklink.o"
    DECKLINK_DISP_OBJ="src/blackmagic_common.o src/video_display/decklink.o"
    DECKLINK_SOUND_PLAY_OBJ="src/blackmagic_common.o src/audio/playback/decklink.o"
    if test $system = "Windows"; then
        OBJS="$OBJS src/video_capture/DeckLinkAPI_i.o"
    else
        OBJS="$OBJS src/video_capture/DeckLinkAPIDispatch.o"
    fi
    AC_DEFINE([HAVE_DECKLINK], [1], [Build with DeckLink support])
    ADD_MODULE("vidcap_decklink", "$DECKLINK_CAP_OBJ", "$DECKLINK_LIB")
    ADD_MODULE("display_decklink", "$DECKLINK_DISP_OBJ", "$DECKLINK_LIB")
    ADD_MODULE("aplay_decklink", "$DECKLINK_SOUND_PLAY_OBJ", "$DECKLINK_LIB")
    decklink=yes
    INC="$INC $DECKLINK_INC"
else
    DECKLINK_PATH=
fi

if test $decklink_req = yes -a $decklink = no; then
        AC_MSG_ERROR([DeckLink not found.])
fi

AC_SUBST(DECKLINK_PATH)

# -------------------------------------------------------------------------------------------------
# gpustitch stuff
# -------------------------------------------------------------------------------------------------

GPUSTITCH_OBJ=
GPUSTITCH_INC=
GPUSTITCH_LIB=
GPUSTITCH_PATH=
gpustitch=no
gpustitch_req=auto

AC_ARG_ENABLE(gpustitch,
[  --disable-gpustitch      disable gpustitch (default is auto)]
[                          Requires: CUDA gpustitch],
	[gpustitch_req=$enableval])

PKG_CHECK_MODULES([LIBGPUSTITCH], [libgpustitch >= 0.0.1 ], [found_gpustitch=yes], [found_gpustitch=no])

SAVED_LIBS=$LIBS
LIBS="$LIBS -lgpustitch"
AC_CHECK_HEADER([libgpustitch/stitcher.hpp])
LIBS=$SAVED_LIBS

if test "$ac_cv_header_libgpustitch_stitcher_hpp" != yes
then
	found_gpustitch=no
else
	if test "$found_gpustitch" != yes
	then
		GPUSTITCH_LIB="$GPUSTITCH_LIB -lgpustitch"
		found_gpustitch=yes
	fi
fi

if test "$gpustitch_req" != no -a \
        "$found_gpustitch" = yes -a \
        "$FOUND_CUDA" = yes

then
        gpustitch=yes

        GPUSTITCH_INC="$GPUSTITCH_INC $LIBGPUSTITCH_CFLAGS"
        GPUSTITCH_LIB="$GPUSTITCH_LIB $LIBGPUSTITCH_LIBS"
        GPUSTITCH_OBJ="src/video_capture/gpustitch.o src/utils/cuda_pix_conv.$CU_OBJ_SUFFIX $CUDA_COMMON_OBJ"
        AC_DEFINE([HAVE_GPUSTITCH], [1], [Build with GPUSTITCH support])
        ADD_MODULE("vidcap_gpustitch", "$GPUSTITCH_OBJ", "$GPUSTITCH_LIB")

        INC="$INC $GPUSTITCH_INC"
        CUDA_MESSAGE
fi

if test $gpustitch_req = yes -a $gpustitch = no; then
        AC_MSG_ERROR([GPUSTITCH not found]);
fi

# -------------------------------------------------------------------------------------------------
# Bluefish444 stuff
# -------------------------------------------------------------------------------------------------
BLUEFISH444_PATH=
BLUEFISH444_CAP_OBJ=
BLUEFISH444_DISP_OBJ=
BLUEFISH444_INC=
BLUEFISH444_LIB=
bluefish444=no
blue_audio=no

AC_ARG_ENABLE(bluefish444,
[  --disable-bluefish444   disable Bluefish444 suppport (default is auto)]
[                          Requires: bluefish444_SDK],
              [bluefish444_req=$enableval],
              [bluefish444_req=$build_default]
              )

BLUEFISH444_PATH=$BLUE_LATEST_SDK

AC_ARG_WITH(bluefish444,
[  --with-bluefish444=DIR  specify location of Bluefish444],
            [BLUEFISH444_PATH=$withval
             ])


SAVED_LIBS=$LIBS
SAVED_CXXFLAGS=$CXXFLAGS
SAVED_CPPFLAGS_SAVE=$CPPFLAGS
if test -n "$BLUEFISH444_PATH"; then
        if test $system = Linux; then
                BLUEFISH444_INC="-I$BLUEFISH444_PATH/build/inc"
                BLUEFISH444_LIB="-L$BLUEFISH444_PATH/build/lib"
        elif test $system = Windows; then
                BLUEFISH444_INC="-I$BLUEFISH444_PATH/inc"
                BLUEFISH444_LIB="-L$BLUEFISH444_PATH/lib/win"
        fi
fi

if test $WORD_LEN -eq 64; then
        BLUEFISH444_LIB_NAME=BlueVelvetC64
else
        BLUEFISH444_LIB_NAME=BlueVelvetC
fi

AC_LANG_PUSH(C++)

BLUE_L=no
BLUE_H=no

CPPFLAGS="$CPPFLAGS $BLUEFISH444_INC"
CXXFLAGS="$CXXFLAGS $BLUEFISH444_INC"
LIBS="$LIBS $BLUEFISH444_LIB"
AC_CHECK_HEADERS([BlueVelvetC.h], [BLUE_H=yes], [BLUE_H=no])
AC_CHECK_LIB($BLUEFISH444_LIB_NAME, bfcFactory, [BLUE_L=yes], [BLUE_L=no])

if test $BLUE_H = yes -a $BLUE_L = yes -a $bluefish444_req != no
then
        BLUEFISH444_LIB="$BLUEFISH444_LIB -l$BLUEFISH444_LIB_NAME"
        BLUEFISH444_CAP_OBJ="src/video_capture/bluefish444.o"
        BLUEFISH444_DISP_OBJ="src/video_display/bluefish444.o"
        AC_DEFINE([HAVE_BLUEFISH444], [1], [Build with Bluefish444 support])
        ADD_MODULE("vidcap_bluefish444", "$BLUEFISH444_CAP_OBJ", "$BLUEFISH444_LIB")
        ADD_MODULE("display_bluefish444", "$BLUEFISH444_DISP_OBJ", "$BLUEFISH444_LIB")
	INC="$INC $BLUEFISH444_INC"
        bluefish444=yes
fi
AC_LANG_POP(C++)

LIBS=$SAVED_LIBS
CXXFLAGS=$SAVED_CXXFLAGS
CPPFLAGS=$SAVED_CPPFLAGS_SAVE

if test $bluefish444_req = yes -a $bluefish444 = no; then
        AC_MSG_ERROR([Bluefish444 not found]);
fi

# -------------------------------------------------------------------------------------------------
# AJA stuff
# -------------------------------------------------------------------------------------------------
AJA_CXXFLAGS=
AJA_INC=
AJA_LIB=
AJA_CAP_OBJ=
AJA_PATH=
aja=no

AC_ARG_ENABLE(aja,
[  --disable-aja           disable AJA suppport (default is auto)]
[                          Requires: NTV2_SDK],
[aja_req=$enableval],
[aja_req=$build_default])

if test -n "$NTV2_ROOT"; then
        AJA_PATH=$NTV2_ROOT/..
fi

if test -n "$AJA_DIRECTORY"; then
        AJA_PATH=$AJA_DIRECTORY
fi

AC_ARG_WITH(aja,
            [  --with-aja=DIR          specify location of NTV2],
            [AJA_PATH=$withval
             ])

# In lines below, there is a compatibility check for NTV2 12.x (Linux only).
# For macOS, only NTV2 13 is checked/supported.
if test $system = Linux
then
        if test -n "$AJA_PATH"
        then
                # NTV2 SDK 12.x
                AC_CHECK_FILES(${AJA_PATH}/lib/libaja.a ${AJA_PATH}/lib/libajastuff.a, FOUND_LIBAJA=yes, FOUND_LIBAJA=no)
                # NTV2 SDK 13.x
                if test "$FOUND_LIBAJA" = no; then
                        AC_CHECK_FILES(${AJA_PATH}/lib/libajantv2.a, FOUND_LIBAJA=yes, FOUND_LIBAJA=no)
                fi
        fi
elif test $system = Windows
then
        AC_CHECK_LIB(aja, display_aja_probe, [FOUND_LIBAJA=yes], [FOUND_LIBAJA=no])
else # MacOSX
        AC_CHECK_LIB(ajantv2, main, [FOUND_LIBAJA=yes], [FOUND_LIBAJA=no])
        if test "$FOUND_LIBAJA" = yes; then
                AC_CHECK_FILE(${AJA_PATH}/ajalibraries/ajantv2/includes/ajatypes.h, FOUND_LIBAJA=yes, FOUND_LIBAJA=no)
        fi
fi

if test "$aja_req" != no -a "$FOUND_LIBAJA" = yes
then
        if test $system = Windows; then
                aja=yes
        else # check version at least 16 (only mac+Lin, Windows would require a different check)
                AJA_VER=`sed -n 's/@%:@define AJA_NTV2_SDK_VERSION_MAJOR@<:@@<:@:space:@:>@@:>@*\(@<:@0-9@:>@*\).*/\1/p' <$AJA_PATH/ajalibraries/ajantv2/includes/ntv2enums.h`
                if test $AJA_VER -ge 16; then
                        aja=yes
                fi
        fi
fi

if test $aja = yes -a $system = Windows; then
        ADD_MODULE("aja", "src/aja_win32_stub.o", "-laja")
elif test $aja = yes # Linux and macOS
then
        AJA_INC="$AJA_INC -I${AJA_PATH}/ajalibraries/ajantv2/includes -I${AJA_PATH}/ajalibraries/ajantv2/src/mac -I${AJA_PATH}/ajaapps/crossplatform/demoapps -I${AJA_PATH}/ajalibraries -I${AJA_PATH}/ajaapps/crossplatform/demoapps/ntv2capture"
        if test $system = Linux
        then
                AJA_CXXFLAGS="-DAJALinux -DAJA_LINUX"
                AJA_INC="$AJA_INC -I${AJA_PATH}/ajalibraries/ajantv2/src/lin"
                AJA_LIB="${AJA_PATH}/lib/libajantv2.a"
        else # mac
                AJA_CXXFLAGS="-DAJAMac -DAJA_MAC"
                AJA_LIB="-lajantv2 -framework IOKit"
        fi
        AJA_CAP_OBJ="src/video_capture/aja.o"
        AJA_DISP_OBJ="src/video_display/aja.o"
        AC_DEFINE([HAVE_AJA], [1], [Build with AJA support])

        ADD_MODULE("aja", "src/aja_common.o $AJA_CAP_OBJ $AJA_DISP_OBJ", "$AJA_LIB")
        INC="$INC $AJA_INC"
        CXXFLAGS="$CXXFLAGS $AJA_CXXFLAGS"
fi

if test $aja_req = yes -a $aja = no; then
        AC_MSG_ERROR([AJA not found]);
fi

# -------------------------------------------------------------------------------------------------
# holepunch stuff

LIBJUICE_PATH=
LIBJUICE_INC=
LIBJUICE_LIB=
libjuice=no

AC_ARG_ENABLE(holepunch,
        AS_HELP_STRING([--disable-holepunch], [disable holepunch suppport (default is auto)]
        [Requires: libjuice]),
[libjuice_req=$enableval],
[libjuice_req=$build_default])

AC_CHECK_LIB(juice, juice_create, FOUND_LIBJUICE=yes, FOUND_LIBJUICE=no)
LIBJUICE_LIB="-ljuice"

if test "$libjuice_req" != no -a "$FOUND_LIBJUICE" = yes
then
        libjuice=yes
        ADD_MODULE("holepunch", "src/utils/udp_holepunch.o", "$LIBJUICE_LIB")
        INC="$INC $LIBJUICE_INC"
        COMMON_FLAGS="$COMMON_FLAGS -DHAVE_LIBJUICE"
fi

if test $libjuice_req = yes -a $libjuice = no; then
        AC_MSG_ERROR([LIBJUICE not found]);
fi

# -------------------------------------------------------------------------------------------------
# DELTACAST stuff

DELTACAST_PATH=
DELTACAST_CAP_OBJ=
DELTACAST_DISP_OBJ=
DELTACAST_INC=
DELTACAST_LIB=
deltacast=no
deltacast_found=no

AC_ARG_ENABLE(deltacast,
[  --disable-deltacast     disable DELTACAST suppport (default is auto)]
[                          Requires: VideoMasterHD_SDK],
              [deltacast_req=$enableval],
              [deltacast_req=$build_default]
              )
DELTACAST_PATH=$DELTACAST_DIRECTORY

AC_ARG_WITH(deltacast,
            [  --with-deltacast=DIR    specify location of DELTACAST],
            [DELTACAST_PATH=$withval
             ])

if test $system = Linux -o $system = Windows
then
        AC_LANG_PUSH(C++)

        if test -n "$DELTACAST_PATH"
        then
                CXXFLAGS_SAVE=$CXXFLAGS
                CXXFLAGS="$CXXFLAGS -I$DELTACAST_PATH/Include"
                CPPFLAGS_SAVE=$CPPFLAGS
                CPPFLAGS="$CPPFLAGS -I$DELTACAST_PATH/Include"
        fi

	if test $system = Windows; then
		EXTRA_HEADERS="#include <windows.h>"
	else
		EXTRA_HEADERS=
	fi

        AC_CHECK_HEADERS([VideoMasterHD_Core.h VideoMasterHD_Sdi.h VideoMasterHD_Sdi_Audio.h],
                         [], [],
                         [[$EXTRA_HEADERS
			#ifdef HAVE_VIDEOMASTERHD_CORE_H
                          # include <VideoMasterHD_Core.h>
                          #endif
                          #ifdef HAVE_VIDEOMASTERHD_SDI_H
                          # include <VideoMasterHD_Sdi.h>
                          #endif
                          ]])

        SAVED_LIBS=$LIBS
        DELTACAST_LIB="-lvideomasterhd_audio -lvideomasterhd"
        if test -n "$DELTACAST_PATH"; then
		if test $ac_cv_sizeof_int_p -eq 8 -a $system = Linux
		then
			DELTACAST_LIB="$DELTACAST_LIB -L$DELTACAST_PATH/Library/x64"
		else
			DELTACAST_LIB="$DELTACAST_LIB -L$DELTACAST_PATH/Library/x86"
		fi
	fi
        LIBS="$LIBS $DELTACAST_LIB"
		
        AC_MSG_CHECKING([DELTACAST library presence])
        AC_LINK_IFELSE([AC_LANG_PROGRAM([[$EXTRA_HEADERS
			  # include <VideoMasterHD_Core.h>
                          # include <VideoMasterHD_Sdi.h>
                          # include <VideoMasterHD_Sdi_Audio.h>
                           ]],
                           [[
			    VHD_OpenBoardHandle(0, (void **) 0, (void *) 0, 0);
			    VHD_SlotExtractAudio(0, 0);
                            VHD_LockSlotHandle(0, 0);
                            ]])],FOUND_VHD_L=yes,FOUND_VHD_L=no)
        #AC_CHECK_LIB(videomasterhd_audio, VHD_GetNbSamples, [], [], [-lvideomasterhd])
        LIBS=$SAVED_LIBS
        if test -n "$DELTACAST_PATH"
        then
                CXXFLAGS=$CXXFLAGS_SAVE
                CPPFLAGS=$CPPFLAGS_SAVE
        fi
        AC_MSG_RESULT([$FOUND_VHD_L])

        AC_LANG_POP(C++)

        if test "$ac_cv_header_VideoMasterHD_Core_h" = yes -a "$ac_cv_header_VideoMasterHD_Sdi_h" = yes -a "$ac_cv_header_VideoMasterHD_Sdi_Audio_h" = yes \
                -a "$FOUND_VHD_L" = yes; then
		deltacast_found=yes
	fi
else # OS X
	MY_CHECK_FRAMEWORK([VideoMasterHD])
	MY_CHECK_FRAMEWORK([VideoMasterHD_Audio])
        AC_CHECK_HEADERS([VideoMasterHD/VideoMasterHD_Core.h])
	if test "$ac_cv_header_VideoMasterHD_VideoMasterHD_Core_h" = yes -a $my_cv_framework_VideoMasterHD = yes -a $my_cv_framework_VideoMasterHD = yes; then
		deltacast_found=yes
		DELTACAST_LIB="$FRAMEWORK_VIDEOMASTERHD $FRAMEWORK_VIDEOMASTERHD_AUDIO"
                BIN_DEPS="${BIN_DEPS:+$BIN_DEPS }Frameworks/VideoMasterHD_Audio.framework Frameworks/VideoMasterHD.framework"
	fi
fi

if test $deltacast_req != no -a $deltacast_found = yes
then
	if test -n "$DELTACAST_PATH"
	then
		DELTACAST_INC="-I$DELTACAST_PATH/Include"
	fi
	DELTACAST_CAP_OBJ="src/video_capture/deltacast.o src/video_capture/deltacast_dvi.o"
	DELTACAST_DISP_OBJ="src/video_display/deltacast.o"
	AC_DEFINE([HAVE_DELTACAST], [1], [Build with DELTACAST support])
	ADD_MODULE("vidcap_deltacast", "$DELTACAST_CAP_OBJ", "$DELTACAST_LIB")
        ADD_MODULE("display_deltacast", "$DELTACAST_DISP_OBJ", "$DELTACAST_LIB")
	INC="$INC $DELTACAST_INC"
	deltacast=yes
fi

if test $deltacast_req = yes -a $deltacast = no; then
        AC_MSG_ERROR([Deltacast not found]);
fi

#----------------------------------------------------------------------
# XIMEA
# ---------------------------------------------------------------------
AC_ARG_ENABLE(ximea,
[  --disable-ximea         disable XIMEA suppport (default is auto)]
[                          Requires: XIMEA SDK],
              [ximea_req=$enableval],
              [ximea_req=$build_default]
              )

AC_ARG_ENABLE(ximea-runtime-linking,
[  --disable-ximea-runtime-linking disable XIMEA suppport (default is enable), disabling may result in non-portable build],
              [ximea_link_runtime=$enableval],
              [ximea_link_runtime=yes]
              )
ximea=no

if test $system = MacOSX; then
        MY_CHECK_FRAMEWORK([m3api])
        if test $my_cv_framework_m3api = yes; then
                if test $ximea_link_runtime = no; then
                        XIMEA_LIBS=$FRAMEWORK_M3API
                fi
                ximea=yes
        fi
        if test -z "$GENICAM_GENTL64_PATH"; then
                GENICAM_GENTL64_PATH="/Library/Frameworks/m3api.framework"
        fi
        XI_LIB_NAME="m3api"
else
        if test $system = Linux; then
                XIAPI_PATH=${GENICAM_GENTL64_PATH:-/opt/XIMEA/lib}
                XI_LIB_EXT=".so"
                XI_LIB_NAME="m3api"
                XI_LIB_PREF="lib"
                XIMEA_CFLAGS="-I$XIAPI_PATH/../include"
        else
                if test -d "$GENICAM_GENTL64_PATH/../../API/xiAPI"; then
                        # the env var contains actually something like "C:\XIMEA\GenTL Producer\64bit"
                        XIAPI_PATH=$(realpath "$GENICAM_GENTL64_PATH/../../API/xiAPI")
                else
                        XIAPI_PATH="C:/XIMEA/API/xiAPI"
                fi
                XI_LIB_EXT=".dll"
                XI_LIB_NAME="xiapi64"
                XIMEA_CFLAGS="-I$XIAPI_PATH"
        fi
        SAVED_CFLAGS=$CFLAGS
        SAVED_CPPFLAGS=$CPPFLAGS
        CFLAGS="$CFLAGS $XIMEA_CFLAGS"
        CPPFLAGS="$CPPFLAGS $XIMEA_CFLAGS"
        AC_CHECK_HEADERS([xiApi.h])
        CFLAGS=$SAVED_CFLAGS

        if test $ximea_link_runtime = no; then
                XIMEA_LIBS="-L$XIAPI_PATH"
                SAVED_LIBS=$LIBS
                LIBS="$LIBS $XIMEA_LIBS"
                CPPFLAGS=$SAVED_CPPFLAGS
                AC_CHECK_LIB($XI_LIB_NAME, xiGetImage, HAVE_XI_L=yes, HAVE_XI_L=no)
                XIMEA_LIBS="-l$XI_LIB_NAME -L$XIAPI_PATH"
                LIBS=$SAVED_LIBS
        else
                HAVE_XI_L=yes
        fi

        if test $ximea_req != no -a $HAVE_XI_L = yes -a $ac_cv_header_xiApi_h = yes; then
                ximea=yes
        fi
fi

if test $ximea = yes -a $ximea_link_runtime = yes; then
        AC_DEFINE([XIMEA_RUNTIME_LINKING], [1], [Dynamically open XIMEA lib on runtime])
        AC_DEFINE_UNQUOTED([XIMEA_LIBRARY_NAME], ["$XI_LIB_PREF$XI_LIB_NAME$XI_LIB_EXT"], [Name of the dynamic libarary])
        AC_DEFINE_UNQUOTED([XIMEA_LIBRARY_PATH], ["$GENICAM_GENTL64_PATH"], [Path to the XI dynamic libarary])
fi

if test $ximea = yes; then
        CFLAGS="$CFLAGS $XIMEA_CFLAGS"
        ADD_MODULE("vidcap_ximea", "src/video_capture/ximea.o", "$XIMEA_LIBS")
fi

if test $ximea_req = yes -a $ximea = no; then
        AC_MSG_ERROR([XIMEA SDK not found]);
fi

# -------------------------------------------------------------------------------------------------
# AV Foundation capture
# -------------------------------------------------------------------------------------------------
avfoundation=no
AC_ARG_ENABLE(avfoundation,
      AS_HELP_STRING([--disable-avfoundation], [enable AVFoundation support (default is enable on macOS)]),
    [avfoundation_req=$enableval],
    [avfoundation_req=auto]
    )

if test $system = MacOSX -a $avfoundation_req != no
then
	AVFOUNDATION_LIB="-framework AppKit -framework AVFoundation -framework CoreMedia -framework CoreVideo"
	AC_DEFINE([HAVE_AVFOUNDATION], [1],  [Build with AV Foundation support])
	AVFOUNDATION_OBJ="src/video_capture/avfoundation.o"
	ADD_MODULE("", "$AVFOUNDATION_OBJ", "$AVFOUNDATION_LIB")
	avfoundation=yes
fi

# ------------------------------------------------------------------------------
# SDL1/2 Stuff
# ------------------------------------------------------------------------------
sdl=no
sdl_version=1
AC_ARG_ENABLE(sdl,
              AS_HELP_STRING([--enable-sdl], [enable SDL (any version; default is auto)]),
              [sdl_req=$enableval],
              [sdl_req=$build_default]
              )

AC_ARG_ENABLE(sdl1,
[  --enable-sdl1           enable SDL1 (default is auto)]
[                          Requires: sdl >= 1.2.14 ],
    [sdl1_req=$enableval],
    [sdl1_req=$build_default]
    )

AC_ARG_ENABLE(sdl2,
[  --enable-sdl2           enable SDL2 (disabled if SDL1 is enabled, auto otherwise)]
[                          Requires: sdl2 ],
    [sdl2_req=$enableval],
    [sdl2_req=$build_default]
    )

if test $sdl1_req = yes -a $sdl2_req = yes; then
        AC_MSG_ERROR([Only one version of SDL (either 1 or 2) can be selected]);
fi

# if either of SDL versions was required, disable the other
if test $sdl1_req = yes; then
        sdl2_req=no
fi
if test $sdl2_req = yes; then
        sdl1_req=no
fi

PKG_CHECK_MODULES([SDL1], [sdl], [found_sdl1=yes], [found_sdl1=no])
PKG_CHECK_MODULES([SDL2], [sdl2], [found_sdl2=yes], [found_sdl2=no])
if test $sdl1_req = yes -a $found_sdl1 = no; then
        AC_MSG_ERROR([SDL1 not found]);
fi
if test $sdl2_req = yes -a $found_sdl2 = no; then
        AC_MSG_ERROR([SDL2 not found]);
fi

# prefer SDL2
if test $sdl2_req != no -a $found_sdl2 = yes
then
        SDL2_OBJ="src/video_display/sdl2.o"
        AC_DEFINE([HAVE_SDL2], [1],  [Build with SDL2 support])
        # on Win override libs from pkg-config because those contain
        # "-mwindows" which causes output to be scratched by cmd.exe
        # (but not MSYS2!)
        SDL2_LIBS=$(remove_mwindows "$SDL2_LIBS")
        ADD_MODULE("display_sdl2", "$SDL2_OBJ", "$SDL2_LIBS")
        SDL_LIB=$SDL2_LIBS
        INC="$INC $($PKG_CONFIG --cflags-only-I sdl2)"
        sdl_version=2
        sdl=yes
elif test $sdl1_req != no -a $found_sdl1 = yes
then
        SDL1_OBJ="src/video_display/sdl.o src/x11_common.o"
        AC_DEFINE([HAVE_SDL], [1],  [Build with SDL support])
        ADD_MODULE("display_sdl", "$SDL1_OBJ", "$SDL1_LIBS")
        SDL_LIB="$SDL1_LIBS -lX11" # X11 for x11_common.o
        INC="$INC $($PKG_CONFIG --cflags-only-I sdl)"
        sdl_version=1
        sdl=yes
fi

if test $sdl_req = yes -a $sdl = no; then
        AC_MSG_ERROR([SDL requested but found any version]);
fi

# ------------------------------------------------------------------------------
# GLM (OpenGL math library)
# ------------------------------------------------------------------------------
GLM_INC=
GLM_DIR=$srcdir/glm
glm=no

PKG_CHECK_MODULES([GLM], [glm], [FOUND_GLM=yes], [FOUND_GLM=no])

if test $FOUND_GLM = yes; then
        GLM_INC=$GLM_CFLAGS
        glm=yes
elif test $cross_compile = no; then
        AC_CHECK_FILE([$GLM_DIR/glm/glm.hpp], [found_glm_hpp=yes], [found_glm_hpp=no])
        if test $found_glm_hpp = yes; then
                glm=yes
                GLM_INC=-I$GLM_DIR
        fi
fi

# ------------------------------------------------------------------------------
# Panorama gl display
# ------------------------------------------------------------------------------
panogl_disp=no
AC_ARG_ENABLE(panogl_disp,
[  --disable-panogl_disp            disable gl panorama display (default is auto]
[                          Requires:  sdl2 gl],
    [panogl_disp_req=$enableval],
    [panogl_disp_req=$build_default]
    )

if test $panogl_disp_req = yes; then
        if test $sdl = no -o $sdl_version != 2; then
                AC_MSG_ERROR([SDL2 not found]);
        fi

        if test $glm = no; then
                AC_MSG_ERROR([GLM not found]);
        fi
fi

if test $panogl_disp_req != no -a $sdl = yes -a $sdl_version = 2 -a $glm = yes
then
        PANOGL_OBJ="src/video_display/pano_gl.o src/video_display/opengl_utils.o src/video_display/sdl_window.o"
        AC_DEFINE([HAVE_PANOGL_DISP], [1],  [Build with 360 panorama disp support])
        PANOGL_LIBS=$SDL_LIB
        ADD_MODULE("display_panogl", "$PANOGL_OBJ", "$PANOGL_LIBS")
        INC="$INC $GLM_INC"
        panogl_disp=yes
fi

# ------------------------------------------------------------------------------
# OpenXR VR gl display
# ------------------------------------------------------------------------------
xrgl_disp=no
AC_ARG_ENABLE(xrgl_disp,
[  --disable-xrgl_disp            disable OpenXR VR display (default is auto]
[                          Requires:  sdl2 gl openxr],
    [xrgl_disp_req=$enableval],
    [xrgl_disp_req=$build_default]
    )

define(xrgl_deps, [openxr])

PKG_CHECK_MODULES([XRGLDISP], [xrgl_deps], [FOUND_XRGL_DEPS=yes], [FOUND_XRGL_DEPS=no])

if test $sdl = no -o $sdl_version != 2 -o $glm = no; then
        FOUND_XRGL_DEPS=no
fi

if test $xrgl_disp_req = yes -a $FOUND_XRGL_DEPS = no; then
        AC_MSG_ERROR([XRGL deps not found]);
fi

if test $xrgl_disp_req != no -a $FOUND_XRGL_DEPS = yes
then
        XRGLDISP_OBJ="src/video_display/openxr_gl.o src/video_display/opengl_utils.o src/video_display/sdl_window.o"
        AC_DEFINE([HAVE_XRGL_DISP], [1],  [Build with OpenXR VR disp support])
        XRGLDISP_LIBS="$XRGLDISP_LIBS $SDL_LIB"
        ADD_MODULE("display_xrgl", "$XRGLDISP_OBJ", "$XRGLDISP_LIBS")
        INC="$INC $GLM_INC"
        xrgl_disp=yes
fi

# -------------------------------------------------------------------------------------------------
# OpenCV version detection
# -------------------------------------------------------------------------------------------------
[remove_all_libs_but_opencv_core() { echo "$1" | sed -e 's/-lopencv_[a-z0-9_]*//g' -e 's/$/ -lopencv_core/' | compact_spaces_pipe ; } ]
AC_ARG_ENABLE(opencv,
              AS_HELP_STRING([--disable-opencv], [disable all OpenCV code]),
              [opencv_req=$enableval],
              [opencv_req=$build_default])
PKG_CHECK_MODULES([OPENCV_4], [opencv4], [
                   AC_DEFINE([HAVE_OPENCV_4], [1], [Use OpenCV version 4])
                   OPENCV_MAJOR=4
                   OPENCV_LIBS=$OPENCV_4_LIBS
                   OPENCV_CFLAGS=$OPENCV_4_CFLAGS
                   ], [OPENCV_MAJOR=])
if test -z "$OPENCV_MAJOR"; then
        PKG_CHECK_MODULES([OPENCV_3], [opencv > 3.0], [
                           AC_DEFINE([HAVE_OPENCV_3], [1], [Use OpenCV version 3])
                           OPENCV_MAJOR=3
                           OPENCV_LIBS=$OPENCV_3_LIBS
                           OPENCV_CFLAGS=$OPENCV_3_CFLAGS
                           ], [OPENCV_MAJOR=])
fi
if test -z "$OPENCV_MAJOR"; then
        PKG_CHECK_MODULES([OPENCV_2], [opencv > 2.3], [
                           AC_DEFINE([HAVE_OPENCV_2], [1], [Use OpenCV version 2])
                           OPENCV_MAJOR=2
                           OPENCV_LIBS=$OPENCV_2_LIBS
                           OPENCV_CFLAGS=$OPENCV_2_CFLAGS
                           ], [OPENCV_MAJOR=])
fi
if test -n "$OPENCV_MAJOR" -a $opencv_req != "no"; then
        OPENCV_LIBS=$(remove_all_libs_but_opencv_core $OPENCV_LIBS)
        HAVE_OPENCV=yes
        opencv="yes"
        opencv_version=" (version: $OPENCV_MAJOR)"

        # opencv_imgproc is commonly needed so make the check here
        SAVED_LIBS=$LIBS
        LIBS="$LIBS $OPENCV_LIBS"
        AC_CHECK_LIB(opencv_imgproc, cvAcc)
        FOUND_OPENCV_IMGPROC=$ac_cv_lib_opencv_imgproc_cvAcc
        LIBS=$SAVED_LIBS
else
        HAVE_OPENCV=no
        opencv="no"
        opencv_version=
fi

# -------------------------------------------------------------------------------------------------
# RTSP capture stuff
# -------------------------------------------------------------------------------------------------
RTSP_INC=
RTSP_LIBS=
RTSP_OBJ=
rtsp=no

define(rtsp_deps, [libcurl])

AC_ARG_ENABLE(rtsp,
[  --disable-rtsp          disable RTSP support (default is auto)]
[                          Requires: rtsp_deps],
    [rtsp_req=$enableval],
    [rtsp_req=$build_default]
    )

PKG_CHECK_MODULES([RTSP], [rtsp_deps], FOUND_RTSP_DEPS=yes, FOUND_RTSP_DEPS=no)

if test $rtsp_req != no -a $FOUND_RTSP_DEPS = yes
then
        AC_DEFINE([HAVE_RTSP], [1], [RTSP capture build with curl])
        SAVED_CFLAGS=$CFLAGS
        CFLAGS="$CFLAGS ${RTSP_CFLAGS}"
        SAVED_CXXFLAGS=$CXXFLAGS
        CXXFLAGS="$CXXFLAGS ${RTSP_CFLAGS}"
        RTSP_INC=
        RTSP_OBJ="src/utils/h264_stream.o src/video_capture/rtsp.o src/rtp/rtpdec_h264.o"
        ADD_MODULE("vidcap_rtsp", "$RTSP_OBJ", "$RTSP_LIBS")
	INC="$INC $RTSP_INC"
        rtsp=yes
fi

if test $rtsp_req = yes -a $rtsp = no; then
        AC_MSG_ERROR([rtsp not found, check curl dependencies...]);
fi

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# RTSP server stuff
# TODO: create pkg-config search path for liblivemedia-dev (create live555.pc file)
# -------------------------------------------------------------------------------------------------
RTSP_SERVER_INC=
RTSP_SERVER_OBJ=
rtsp_server=no
livemedia=no
livemedia_prefix=/usr

AC_ARG_ENABLE(rtsp_server,
[  --enable-rtsp-server    enables RTSP server support (default is auto)]
[                          Requires: live555 (<= 2015)],
    [rtsp_server_req=$enableval],
    [rtsp_server_req=$build_default]
    )

AC_ARG_WITH(live555,
            [  --with-live555=DIR      specify location of live555],
            [livemedia_prefix=$withval
            ])

AC_LANG_PUSH(C++)
RTSP_SERVER_FLAGS="-I$livemedia_prefix/include/groupsock -I$livemedia_prefix/include/liveMedia -I$livemedia_prefix/include/BasicUsageEnvironment -I$livemedia_prefix/include/UsageEnvironment"
SAVED_CPPFLAGS=$CPPFLAGS
CPPFLAGS="$CPPFLAGS $RTSP_SERVER_FLAGS"
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
                   #include <liveMedia.hh>
                   #include <liveMedia_version.hh>
                   ]], [[
                        #if LIVEMEDIA_LIBRARY_VERSION_INT > 1438905600
                        #error "Unsuppored live555 version (too new)"
                        #endif
                        ]]
                   )], FOUND_LIVE_H=yes, [echo "liveMedia (live555 library) headers not found or wrong version (< 2015 needed)"; FOUND_LIVE_H=no])
CPPFLAGS=$SAVED_CPPFLAGS
AC_LANG_POP(C++)

if test $rtsp_server_req != no -a $FOUND_LIVE_H = yes
then
        AC_DEFINE([HAVE_RTSP_SERVER], [1], [RTSP server build with live555 support])
        CFLAGS="$CFLAGS $RTSP_SERVER_FLAGS"
        CXXFLAGS="$CXXFLAGS $RTSP_SERVER_FLAGS"
        RTSP_SERVER_LIBS="-lliveMedia -lBasicUsageEnvironment -lUsageEnvironment -lgroupsock"
        RTSP_SERVER_OBJ="src/rtsp/c_basicRTSPOnlyServer.o src/rtsp/BasicRTSPOnlyServer.o src/rtsp/BasicRTSPOnlySubsession.o src/video_rxtx/h264_rtp.o"
        ADD_MODULE("video_rxtx_h264", "$RTSP_SERVER_OBJ", "$RTSP_SERVER_LIBS")
        rtsp_server=yes
        livemedia=yes
fi

if test $rtsp_server_req = yes -a $rtsp_server = no; then
        AC_MSG_ERROR([rtsp server not found, check live555 -livemedia lib- dependencies...]);
fi

# ----------------------------------------------------------------------
# SDP over HTTP
# ----------------------------------------------------------------------
AC_ARG_ENABLE(sdp-http,
[  --enable-sdp-http       enables SDP/HTTP transport (default is auto)],
    [sdp_http_req=$enableval],
    [sdp_http_req=$build_default]
    )
sdp_http=no

AC_CHECK_HEADERS([EmbeddableWebServer.h])

if test $sdp_http_req != no && test $ac_cv_header_EmbeddableWebServer_h = yes
then
        AC_DEFINE([SDP_HTTP], 1, [Add support for SDP over HTTP])
        sdp_http=yes
fi

if test $sdp_http_req = yes -a $sdp_http = no; then
        AC_MSG_ERROR([SDP/HTTP dependency EmbeddableWebServer.h not found!]);
fi

# -------------------------------------------------------------------------------------------------
# Resize stuff
# -------------------------------------------------------------------------------------------------
RESIZE_LIBS=
RESIZE_OBJ=
resize=no

AC_ARG_ENABLE(resize,
[  --disable-resize        disable resize capture filter (default is auto)]
[                          Requires: opencv],
    [resize_req=$enableval],
    [resize_req=$build_default]
    )

if test $resize_req != no && test "$HAVE_OPENCV" = yes && test "$FOUND_OPENCV_IMGPROC" = yes
then
        CFLAGS="$CFLAGS ${OPENCV_CFLAGS}"
        CXXFLAGS="$CXXFLAGS ${OPENCV_CFLAGS}"
        RESIZE_OBJ="src/capture_filter/resize.o src/capture_filter/resize_utils.o"
        ADD_MODULE("vcapfilter_resize", "$RESIZE_OBJ", "$OPENCV_LIBS -lopencv_imgproc")
        resize=yes
fi

if test $resize_req = yes -a $resize = no; then
        AC_MSG_ERROR([Resize dependency not found, may install OpenCV...?]);
fi

# -------------------------------------------------------------------------------------------------
# Blank stuff
# -------------------------------------------------------------------------------------------------
BLANK_LIBS=
BLANK_OBJ=
blank=no

define(blank_dep, libswscale)

AC_ARG_ENABLE(blank,
[  --disable-blank         disable blank capture filter (default is auto)]
[                          Requires: blank_dep],
    [blank_req=$enableval],
    [blank_req=$build_default]
    )

PKG_CHECK_MODULES([BLANK], [blank_dep], FOUND_BLANK_DEP=yes, FOUND_BLANK_DEP=no)

if test $FOUND_BLANK_DEP = no; then
        AC_CHECK_HEADERS([libswscale/swscale.h], AC_CHECK_LIB(swscale, sws_scale, FOUND_BLANK_DEP=yes))
        if test $FOUND_BLANK_DEP = yes; then
                BLANK_LIBS='-lswscale'
        fi
fi

if test $blank_req != no -a $FOUND_BLANK_DEP = yes
then
        CFLAGS="$CFLAGS ${BLANK_CFLAGS}"
        CXXFLAGS="$CXXFLAGS ${BLANK_CFLAGS}"
        BLANK_OBJ="src/capture_filter/blank.o"
        BLANK_OBJ="$BLANK_OBJ src/libavcodec_common.o" # get_ug_to_av_pixfmt()
        ADD_MODULE("vcapfilter_blank", "$BLANK_OBJ", "$BLANK_LIBS")
        blank=yes
fi

if test $blank_req = yes -a $blank = no; then
        AC_MSG_ERROR([Blank dep not found (libswscale)]);
fi

# -------------------------------------------------------------------------------------------------
# Testcard stuff
# -------------------------------------------------------------------------------------------------
AC_ARG_ENABLE(testcard-extras,
      AS_HELP_STRING([--enable-testcard-extras[=mixer|ttf|all]], [enable testcard addons - audio and testcard2 (default is auto)]
      [This includes testcard2 (optionally with TTF) and sound support (mixer) for normal testcard.]
      [Requires: SDL(2)_mixer (optional) SDL(2)_ttf (optional)]),
    [testcard_extras_req=$enableval],
    [testcard_extras_req=$build_default]
    )

TESTCARD_OBJ="src/video_capture/testcard.o"
TESTCARD_LIB=
TESTCARD2_OBJ=
TESTCARD2_LIB=

testcard2_req=auto
testcard_sdl_mixer_req=auto
testcard_sdl_ttf_req=auto

testcard2=no
testcard_sdl_mixer=no
testcard_sdl_ttf=no

if test $testcard_extras_req = "no"; then
        testcard2_req=no
        testcard_sdl2_mixer_req=no
        testcard_sdl2_ttf_req=no
fi
if test $testcard_extras_req = "yes"; then
        testcard2_req=yes
fi
if expr $testcard_extras_req : '.*all' >/dev/null; then
        testcard_sdl_mixer_req=yes
        testcard_sdl_ttf_req=yes
fi
if expr $testcard_extras_req : '.*mixer' >/dev/null; then
        testcard_sdl_mixer_req=yes
fi
if expr $testcard_extras_req : '.*ttf' >/dev/null; then
        testcard_sdl_ttf_req=yes
fi

if test $testcard_sdl_mixer_req != no
then
        if test $sdl_version -eq 1; then
                PKG_CHECK_MODULES([SDL_MIXER], [SDL_mixer], [FOUND_SDL_MIXER=yes], [FOUND_SDL_MIXER=no])
        else
                PKG_CHECK_MODULES([SDL_MIXER], [SDL2_mixer], [FOUND_SDL_MIXER=yes], [FOUND_SDL_MIXER=no])
        fi
        SDL_MIXER_LIBS=$(remove_mwindows "$SDL_MIXER_LIBS")

        if test "$FOUND_SDL_MIXER" = yes
        then
                AC_DEFINE([HAVE_LIBSDL_MIXER], [1], [Build testcard with SDL Mixer support])
                TESTCARD_LIB="$SDL_MIXER_LIBS"
        fi
fi

TESTCARD_OBJ="src/video_capture/testcard.o"
ADD_MODULE("vidcap_testcard", "$TESTCARD_OBJ", "$TESTCARD_LIB")

if test $testcard2_req != no
then
        TESTCARD2_OBJ="src/video_capture/testcard2.o"
        if test $sdl_version -eq 1; then
                PKG_CHECK_MODULES([SDL_TTF], [SDL_ttf], [FOUND_SDL_TTF=yes], [FOUND_SDL_TTF=no])
        else
                PKG_CHECK_MODULES([SDL_TTF], [SDL2_ttf], [FOUND_SDL_TTF=yes], [FOUND_SDL_TTF=no])
                fi
                SDL_TTF_LIBS=$(remove_mwindows "$SDL_TTF_LIBS")
        if test "$FOUND_SDL_TTF" = yes
        then
                TESTCARD2_LIB="$TESTCARD2_LIB $SDL_TTF_LIBS"
                AC_DEFINE([HAVE_LIBSDL_TTF], [1], [Build testcard2 with TTF support])
        fi
        AC_DEFINE([HAVE_TESTCARD2], [1], [Build with testcard2 capture])
        ADD_MODULE("vidcap_testcard2", "$TESTCARD2_OBJ", "$TESTCARD2_LIB")
        testcard2=yes
fi

if { test $testcard2_req = yes && test $testcard2 = no ;} \
                || { test $testcard_sdl_mixer_req = yes && test $FOUND_SDL_MIXER = no ;} \
                || { test $testcard_sdl_ttf_req = yes && test $FOUND_SDL_TTF = no ;} ; then
        AC_MSG_ERROR([Testcard2 dependencies not found])
fi


# -------------------------------------------------------------------------------------------------
# libavcodec hw-accelerated decoding support
# -------------------------------------------------------------------------------------------------
lavc_hwacc_common=no
lavc_hwacc_vdpau=no
lavc_hwacc_vaapi=no
lavc_hwacc_rpi4=no

define(lavc_hwacc_common_dep, libavcodec libavutil >= 55.22.1)
define(lavc_hwacc_vdpau_dep, lavc_hwacc_common_dep vdpau)
define(lavc_hwacc_vaapi_dep, lavc_hwacc_common_dep libva)

AC_ARG_ENABLE(lavc-hw-accel-vdpau,
[  --disable-lavc-hw-accel-vdpau   disable lavc-hw-accel-vdpau (default is auto)]
[                          Requires: lavc_hwacc_vdpau_dep],
    [lavc_hwacc_vdpau_req=$enableval],
    [lavc_hwacc_vdpau_req=$build_default]
)

AC_ARG_ENABLE(lavc-hw-accel-vaapi,
[  --disable-lavc-hw-accel-vaapi   disable lavc-hw-accel-vaapi (default is auto)]
[                          Requires: lavc_hwacc_vaapi_dep],
    [lavc_hwacc_vaapi_req=$enableval],
    [lavc_hwacc_vaapi_req=$build_default]
)

AC_ARG_ENABLE(lavc-hw-accel-rpi4,
[  --disable-lavc-hw-accel-rpi4          disable lavc-hw-accel-rpi4 (default is auto)]
[                          Requires: Raspbian-patched ffmpeg libmmal],
    [lavc_hwacc_rpi4_req=$enableval],
    [lavc_hwacc_rpi4_req=$build_default]
)


PKG_CHECK_MODULES([LAVC_HWACC_COMMON], [lavc_hwacc_common_dep], [FOUND_HWACC_COMMON_DEP=yes], [FOUND_HWACC_COMMON_DEP=no])
PKG_CHECK_MODULES([LAVC_HWACC_VDPAU], [lavc_hwacc_vdpau_dep], [FOUND_HWACC_VDPAU_DEP=yes], [FOUND_HWACC_VDPAU_DEP=no])
PKG_CHECK_MODULES([LAVC_HWACC_VAAPI], [lavc_hwacc_vaapi_dep], [FOUND_HWACC_VAAPI_DEP=yes], [FOUND_HWACC_VAAPI_DEP=no])

SAVED_PKG_CONFIG_PATH=$PKG_CONFIG_PATH
PKG_CONFIG_PATH="$PKG_CONFIG_PATH:/opt/vc/lib/pkgconfig/"
export PKG_CONFIG_PATH

PKG_CHECK_MODULES([MMAL], [mmal], [ FOUND_HWACC_RPI4_DEP=yes ], [ FOUND_HWACC_RPI4_DEP=no ])
PKG_CHECK_MODULES([BCM_HOST], [bcm_host], [ ], [ FOUND_HWACC_RPI4_DEP=no ])

PKG_CONFIG_PATH=$SAVED_PKG_CONFIG_PATH
export PKG_CONFIG_PATH

AC_CHECK_HEADER([libavcodec/rpi_zc.h], [  ], [FOUND_HWACC_RPI4_DEP=no], [#include <libavcodec/avcodec.h>])


LAVC_HWACC_FLAGS=
LAVC_HWACC_LIBS=
HW_ACC_OBJ="src/hwaccel_libav_common.o"

if test $lavc_hwacc_vdpau_req != no -a $FOUND_HWACC_VDPAU_DEP = yes
then
        lavc_hwacc_vdpau=yes
        lavc_hwacc_common=yes
fi
if test $lavc_hwacc_vaapi_req != no -a $FOUND_HWACC_VAAPI_DEP = yes
then
        lavc_hwacc_vaapi=yes
        lavc_hwacc_common=yes
fi
if test $lavc_hwacc_rpi4_req != no -a $FOUND_HWACC_RPI4_DEP = yes
then
        lavc_hwacc_rpi4=yes
        lavc_hwacc_common=yes
fi
if test $system = MacOSX; then
        lavc_hwacc_common=yes
fi

if test $lavc_hwacc_common = yes
then
        LAVC_HWACC_FLAGS="${LAVC_HWACC_FLAGS} -DHWACC_COMMON_IMPL"
fi

if test $lavc_hwacc_vdpau = yes
then
        LAVC_HWACC_FLAGS="${LAVC_HWACC_FLAGS} -DHWACC_VDPAU ${LAVC_HWACC_VDPAU_CFLAGS}"
        LAVC_HWACC_LIBS="${LAVC_HWACC_LIBS} ${LAVC_HWACC_VDPAU_LIBS}"
        HW_ACC_OBJ="${HW_ACC_OBJ} src/hwaccel_vdpau.o"
fi

if test $lavc_hwacc_vaapi = yes
then
        LAVC_HWACC_FLAGS="${LAVC_HWACC_FLAGS} -DHWACC_VAAPI ${LAVC_HWACC_VAAPI_CFLAGS}"
        LAVC_HWACC_LIBS="${LAVC_HWACC_LIBS} ${LAVC_HWACC_VAAPI_LIBS}"
        HW_ACC_OBJ="${HW_ACC_OBJ} src/hwaccel_vaapi.o"
fi

if test $lavc_hwacc_rpi4 = yes
then
	LAVC_HWACC_FLAGS="${LAVC_HWACC_FLAGS} -DHWACC_RPI4 ${MMAL_CFLAGS} ${BCM_HOST_CFLAGS}"
	LAVC_HWACC_LIBS="${LAVC_HWACC_LIBS} ${MMAL_LIBS} ${BCM_HOST_LIBS}"
	ADD_MODULE("rpi4_hw_accel", "src/video_display/rpi4_out.o", "${MMAL_LIBS} ${BCM_HOST_LIBS}")
fi

COMMON_FLAGS="$COMMON_FLAGS $LAVC_HWACC_FLAGS"

if test $lavc_hwacc_vdpau_req = yes -a $lavc_hwacc_vdpau = no; then
	AC_MSG_ERROR([Could not find hwacc vdpau dependencies!]);
fi

if test $lavc_hwacc_vaapi_req = yes -a $lavc_hwacc_vaapi = no; then
	AC_MSG_ERROR([Could not find hwacc vaapi dependencies!]);
fi

if test $lavc_hwacc_rpi4_req = yes -a $lavc_hwacc_rpi4 = no; then
	AC_MSG_ERROR([Could not find hwacc rpi4 dependencies!]);
fi

# -------------------------------------------------------------------------------------------------
# Libswscale
# -------------------------------------------------------------------------------------------------
libswscale=no
AC_ARG_ENABLE(libswscale,
[  --disable-libswscale    disable libswscale support (default is auto)]
[                          Requires: libswscale libavutil],
[libswscale_req=$enableval],
[libswscale_req=$build_default]
)

if test $system = Windows; then
AC_CHECK_HEADERS([libswscale/swscale.h])
AC_CHECK_LIB(swscale, sws_getContext)
if test $ac_cv_header_libswscale_swscale_h = yes -a \
        $ac_cv_lib_swscale_sws_getContext = yes
then
        found_libswscale=yes
        LIBSWSCALE_LIBS="-lswscale"
else
        found_libswscale=no
fi
else # Linux, OS X
PKG_CHECK_MODULES([LIBSWSCALE], [libswscale], [found_libswscale=yes], [found_libswscale=no])
fi

if test $libswscale_req != no -a $found_libswscale = yes
then
        AC_DEFINE([HAVE_SWSCALE], [1], [Build with Swscale support])

        COMMON_FLAGS="$COMMON_FLAGS $LIBSWSCALE_CFLAGS"
        libswscale=yes
        #LIBSWSCALE_LIBS="$LIBSWSCALE_LIBS $LIBAVUTIL_LIBS" # added libavutil explicitly
fi

if test $libswscale_req = yes -a $libswscale = no; then
        AC_MSG_ERROR([Libswscale not found]);
fi

# -------------------------------------------------------------------------------------------------
# Libav
# -------------------------------------------------------------------------------------------------
libavcodec=no
libavcodec_audio=no
AC_ARG_ENABLE(libavcodec,
[  --disable-libavcodec    disable libavcodec support (default is auto)]
[                          Requires: libavcodec libavutil],
[libavcodec_req=$enableval],
[libavcodec_req=$build_default]
)

PKG_CHECK_MODULES([LIBAVCODEC], [libavcodec >= 53.35.0 ], [found_libavcodec=yes], [found_libavcodec=no])
PKG_CHECK_MODULES([LIBAVUTIL], [libavutil >= 51.22.1 ], [found_libavutil=yes], [found_libavutil=no])

if test $found_libavcodec = no -o $found_libavutil = no; then
        AC_CHECK_HEADERS([libavcodec/avcodec.h libavutil/imgutils.h libavutil/opt.h])
        AC_CHECK_LIB(avcodec, avcodec_open2)
        AC_CHECK_LIB(avutil, av_free)
        if test $ac_cv_header_libavcodec_avcodec_h = yes -a \
                        $ac_cv_header_libavutil_imgutils_h = yes -a \
                        $ac_cv_header_libavutil_opt_h = yes -a \
                        $ac_cv_lib_avcodec_avcodec_open2 = yes -a \
                        $ac_cv_lib_avutil_av_free = yes; then
                found_libavcodec=yes
                found_libavutil=yes
                libavcodec_ge_55=yes # TODO
                LIBAVCODEC_LIBS="-lavcodec"
                LIBAVUTIL_LIBS="-lavutil"
        fi
fi

if test $libavcodec_req != no -a $found_libavcodec = yes -a $found_libavutil = yes
then
        LIBAVCODEC_COMMON="src/libavcodec_common.o"
        LIBAVCODEC_VIDEO="src/libavcodec/lavc_video.o src/libavcodec/from_lavc_vid_conv.o src/libavcodec/to_lavc_vid_conv.o"
        AC_DEFINE([HAVE_LAVC], [1], [Build with LAVC support])
        LIBAVCODEC_AUDIO_CODEC_OBJ="$LIBAVCODEC_COMMON src/audio/codec/libavcodec.o"

        LIBAVCODEC_COMPRESS_OBJ="$LIBAVCODEC_COMMON $LIBAVCODEC_VIDEO src/video_compress/libavcodec.o"
        LIBAVCODEC_DECOMPRESS_OBJ="$LIBAVCODEC_COMMON $LIBAVCODEC_VIDEO $HW_ACC_OBJ src/video_decompress/libavcodec.o"
        if test $system = MacOSX; then
                LIBAVCODEC_DECOMPRESS_OBJ="$LIBAVCODEC_DECOMPRESS_OBJ src/hwaccel_videotoolbox.o"
        fi
        COMMON_FLAGS="$COMMON_FLAGS $LIBAVCODEC_CFLAGS $LIBAVUTIL_CFLAGS"
        libavcodec=yes
        LIBAVCODEC_LIBS="$LIBAVCODEC_LIBS $LIBAVUTIL_LIBS" # added libavutil explicitly
        ADD_MODULE("vcompress_libavcodec", "$LIBAVCODEC_COMPRESS_OBJ", "$LIBAVCODEC_LIBS $LIBSWSCALE_LIBS")
        ADD_MODULE("vdecompress_libavcodec", "$LIBAVCODEC_DECOMPRESS_OBJ", "$LIBAVCODEC_LIBS $LAVC_HWACC_LIBS")
        ADD_MODULE("acompress_libavcodec", "$LIBAVCODEC_AUDIO_CODEC_OBJ", "$LIBAVCODEC_LIBS")
else
        HW_ACC_OBJ=
fi

if test $libavcodec_req = yes -a $libavcodec = no; then
        AC_MSG_ERROR([Libavcodec not found]);
fi

# -------------------------------------------------------------------------------------------------
# File input
# -------------------------------------------------------------------------------------------------
file=no
AC_ARG_ENABLE(file,
[  --disable-file          disable file input support (default is auto)]
[                          Requires: libavcodec >= 57.107.100 libavformat libavutil libswscale],
[file_req=$enableval],
[file_req=$build_default]
)

PKG_CHECK_MODULES([LIBAVFORMAT], [libavformat], [found_libavformat=yes], [found_libavformat=no])
PKG_CHECK_MODULES([LIBAVCODEC_MIN_VERSION], [libavcodec >= 57.107.100], [libavcodec_min_version=yes], [libavcodec_min_version=no])
# Windows doesn't use pkg-config so we assume that libavcodec is new enough
if test $system = Windows; then
        libavcodec_min_version=yes
fi
if test "$found_libavformat" != yes; then
        SAVED_LIBS=$LIBS
        AC_CHECK_HEADERS([libavformat/avformat.h])
        AC_CHECK_LIB(avformat, avformat_open_input)
        LIBS=$SAVED_LIBS

        if test $ac_cv_header_libavformat_avformat_h = yes -a \
                $ac_cv_lib_avformat_avformat_open_input = yes
        then
                found_libavformat=yes
                LIBAVFORMAT_LIBS="-lavformat"
        else
                found_libavformat=no
        fi
fi

if test $file_req != no -a $libavcodec = yes -a $libswscale = yes -a $found_libavformat = yes -a "$libavcodec_min_version" = yes
then
        AC_DEFINE([HAVE_FILE], [1], [Build with file input support])

        COMMON_FLAGS="$COMMON_FLAGS $LIBAVFORMAT_CFLAGS"
        file=yes
        ADD_MODULE("vidcap_file", "src/libavcodec_common.o src/video_capture/file.o", "$LIBAVCODEC_LIBS $LIBAVFORMAT_LIBS $LIBSWSCALE_LIBS")
fi

if test $file_req = yes -a $file = no; then
        AC_MSG_ERROR([File input dependencies not found]);
fi


# -------------------------------------------------------------------------------------------------
# OpenGL display
# -------------------------------------------------------------------------------------------------
GL_INC=
GL_OBJ=$GL_COMMON_OBJ
GL_LIB=$OPENGL_LIB
gl_display=no

AC_ARG_ENABLE(gl-display,
[  --disable-gl-display    disable OpenGL display (default is auto)]
[                          Requires: OpenGL GLFW3],
    [gl_display_req=$enableval],
    [gl_display_req=$build_default]
    )

GL_OBJ="$GL_OBJ src/video_display/gl.o"

PKG_CHECK_MODULES([GLFW], [glfw3], [ FOUND_GLFW=yes ], [ FOUND_GLFW=no ])

if test "$OPENGL" = yes -a "$FOUND_GLFW" = yes -a $gl_display_req != no
then
        GL_LIB="$GLFW_LIBS $OPENGL_LIB"
        gl_display=yes
        AC_DEFINE([HAVE_GL], [1], [Build with OpenGL output])
	INC="$INC $GLFW_INC"
	ADD_MODULE("display_gl", "$GL_OBJ $HW_ACC_OBJ", "$GL_LIB $LAVC_HWACC_LIBS")
fi

if test $gl_display_req = yes -a $gl_display = no; then
        AC_MSG_ERROR([OpenGL not found]);
fi

# -------------------------------------------------------------------------------------------------
# SW Mix Stuff
# -------------------------------------------------------------------------------------------------
SWMIX_INC=
SWMIX_LIB=
SWMIX_OBJ=
swmix=no

AC_ARG_ENABLE(swmix,
[  --disable-swmix         disable SW mix (default is auto)]
[                          Requires: gl],
    [swmix_req=$enableval],
    [swmix_req=$build_default]
    )

if test $swmix_req != no -a $OPENGL = yes
then
        swmix=yes
        SWMIX_LIB="$OPENGL_LIB"
        SWMIX_OBJ="$SWMIX_OBJ $GL_COMMON_OBJ src/video_capture/swmix.o"
        ADD_MODULE("vidcap_swmix", "$SWMIX_OBJ", "$SWMIX_LIB")
        AC_DEFINE([HAVE_SWMIX], [1], [Build SW mix capture])
fi

if test $swmix_req = yes -a $swmix = no; then
        AC_MSG_ERROR([SW mix was not found (OpenGL libraries missing)]);
fi

# -----------------------------------------------------------------------------
# DirectShow
# -----------------------------------------------------------------------------
dshow=no
AC_ARG_ENABLE(dshow,
      AS_HELP_STRING([--disable-dshow], [disable DirectShow support(default is auto)]),
    [dshow_req=$enableval],
    [dshow_req=$build_default]
    )

if test $system = Windows -a $dshow_req != no; then
        AC_DEFINE([HAVE_DSHOW], [1], [Build with DirectShow support])
        ADD_MODULE("", "src/video_capture/DirectShowGrabber.o",
                   "-lstrmiids -luuid -lole32 -loleaut32") #-lstrmbase -ldxguid -ldmoguids
        dshow=yes
fi

if test $dshow_req = yes -a $dshow = no; then
        AC_MSG_ERROR([DirectShow not found]);
fi

# -------------------------------------------------------------------------------------------------
# Screen capture stuff
# -------------------------------------------------------------------------------------------------
SCREEN_CAP_INC=
SCREEN_CAP_OBJ=
SCREEN_CAP_LIB=
screen_cap=no

AC_ARG_ENABLE(screen,
[  --disable-screen        disable screen capture (default is auto)]
[                          Requires: none (OSX) x11 (Linux) dshow (Win)],
    [screen_cap_req=$enableval],
    [screen_cap_req=$build_default]
    )

case "$system" in
	MacOSX)
                screen_cap=yes
                SCREEN_CAP_LIB="-framework CoreFoundation"
	;;
        Windows)
                screen_cap=$dshow
        ;;
	*)
                # Linux
                SAVED_LIBS=$LIBS
                AC_CHECK_HEADER(X11/Xlib.h)
                AC_CHECK_HEADER(X11/Xutil.h)
                AC_CHECK_LIB(X11, XGetImage)
                #PKG_CHECK_MODULES([XFIXES], [xfixes], 
                #                  [AC_DEFINE([HAVE_XFIXES], [1], [Build with XFixes support])],
                #                  [HAVE_XFIXES=no]
                #                  )
                AC_CHECK_LIB(Xfixes, XFixesGetCursorImage)
                AC_CHECK_HEADER(X11/extensions/Xfixes.h)
                LIBS=$SAVED_LIBS

		if test $screen_cap_req != no -a $ac_cv_lib_X11_XGetImage = yes -a \
                         $ac_cv_header_X11_Xlib_h = yes -a $ac_cv_header_X11_Xutil_h = yes
		then
                        screen_cap=yes
                        SCREEN_CAP_LIB="$SCREEN_CAP_LIB -lX11"
                        if test $ac_cv_lib_Xfixes_XFixesGetCursorImage = yes -a \
                                $ac_cv_header_X11_extensions_Xfixes_h = yes
                        then
                                AC_DEFINE([HAVE_XFIXES], [1], [Build with XFixes support])
                                SCREEN_CAP_LIB="$SCREEN_CAP_LIB -lXfixes"
                        fi

		else
                        screen_cap=no
		fi
	;;
esac

if test $screen_cap_req != no -a $screen_cap = yes
then
        AC_DEFINE([HAVE_SCREEN_CAP], [1], [Build with screen capture])
        if test $system = MacOSX; then
                SCREEN_CAP_OBJ="src/video_capture/screen_osx.o"
        elif test $system = Linux; then
                SCREEN_CAP_OBJ="src/video_capture/screen_x11.o src/x11_common.o"
        else
                SCREEN_CAP_OBJ="src/video_capture/screen_win.o"
                DLL_LIBS="${DLL_LIBS:+$DLL_LIBS }${srcdir}/data/screen-capture-recorder-x64.dll"
        fi
        ADD_MODULE("vidcap_screen", "$SCREEN_CAP_OBJ", "$SCREEN_CAP_LIB")
else
        screen_cap=no
fi

if test $screen_cap_req = yes -a $screen_cap = no; then
        AC_MSG_ERROR([Screen capture not found]);
fi

# -------------------------------------------------------------------------------------------------
# GLSL DXT
# -------------------------------------------------------------------------------------------------
rtdxt=no

AC_ARG_ENABLE(rtdxt,
[  --disable-rtdxt         disable Real-Time OpenGL DXT compression (default is auto)]
[                          Requires: gl],
	[rtdxt_req=$enableval],
        [rtdxt_req=$build_default])


AC_DEFINE([USE_PBO_DXT_ENCODER], [1], [We want to use OpenGL pixel buffer objects])
# We probably want this only if there is need to maximalize bandwidth
# With NVidia cards, it increases latency up about 1.5x frame time
#AC_DEFINE([USE_PBO_DXT_ECODER], [1], [We want to use OpenGL pixel buffer objects])

# Linux
if test $rtdxt_req != no -a $OPENGL = yes
then
        rtdxt=yes
        RTDXT_LIB="$OPENGL_LIB"
        AC_DEFINE([HAVE_DXT_GLSL], [1], [Build with DXT_GLSL support])
        RTDXT_COMMON_OBJ="$RTDXT_COMMON_OBJ $GL_COMMON_OBJ dxt_compress/dxt_util.o"
        RTDXT_COMMON_HEADERS="$RTDXT_COMMON_HEADERS dxt_compress/dxt_glsl.h"
        RTDXT_COMPRESS_OBJ="$RTDXT_COMMON_OBJ dxt_compress/dxt_encoder.o src/video_compress/dxt_glsl.o"
        RTDXT_DECOMPRESS_OBJ="$RTDXT_COMMON_OBJ dxt_compress/dxt_decoder.o src/video_decompress/dxt_glsl.o"
        DXT_GLSL_CFLAGS="-std=gnu99"
        COMMON_FLAGS="$COMMON_FLAGS -Idxt_compress"
        ADD_MODULE("vcompress_rtdxt", "$RTDXT_COMPRESS_OBJ ",
                  "$RTDXT_LIB")
        ADD_MODULE("vdecompress_rtdxt", "$RTDXT_DECOMPRESS_OBJ ",
                  "$RTDXT_LIB")
        LIB_GENERATED_HEADERS="$RTDXT_COMMON_HEADERS"
fi

if test $rtdxt_req = yes -a $rtdxt = no; then
        AC_MSG_ERROR([RTDXT not found]);
fi

AC_SUBST(DXT_GLSL_CFLAGS)

# -------------------------------------------------------------------------------------------------
# UYVY
uyvy=no

AC_ARG_ENABLE(uyvy,
[  --disable-uyvy          disable dummy RGB to UYVY compression (default is auto)]
[                          Requires: CUDA],
	[uyvy_req=$enableval],
        [uyvy_req=$build_default])

SAVED_LIBS=$LIBS

if test $uyvy_req != no -a "$OPENGL" = yes
then
        uyvy=yes
fi

# common
if test $uyvy = yes
then
        AC_DEFINE([HAVE_COMPRESS_UYVY], [1], [Build with dummy UYVY compression])
        UYVY_COMPRESS_OBJ="$GL_COMMON_OBJ src/video_compress/uyvy.o"
        ADD_MODULE("vcompress_uyvy", "$UYVY_COMPRESS_OBJ", "$OPENGL_LIB")
fi

if test $uyvy_req = yes -a $uyvy = no; then
        AC_MSG_ERROR([UYVY not found]);
fi

# -------------------------------------------------------------------------------------------------
# CUDA stuff
#
# CUDA compiler choice - defaults to nvcc
# 
# CUDA lookup order:
# 1) --with-cuda
# 2) CUDA_DIRECTORY env variable
# 3) default system path (PATH, LD_LIBRARY_PATH, C_INCLUDE_PATH/CPATH)
# 4) default CUDA path

CUDA_COMPUTE_ARGS=""
CUDA_COMPILER=""

CUDA_INC=
CUDA_LIB=
NVCC=
CUDA_LIB_PATH=
FOUND_CUDA=no
CUDA_PATH=$CUDA_DIRECTORY

AC_ARG_ENABLE(cuda,
              AS_HELP_STRING([--disable-cuda], [disable all CUDA code]),
              [cuda_req=$enableval],
              [cuda_req=$build_default])

AC_ARG_WITH(cuda,
            [  --with-cuda=DIR         specify location of CUDA Toolkit],
            [CUDA_PATH=$withval
            ])

if test -n "$CUDA_PATH"; then
        CUDA_PATH_SEP="$CUDA_PATH/bin$PATH_SEPARATOR"
fi

# nvcc is intentional here
AC_PATH_PROG(NVCC, nvcc, [], [$CUDA_PATH_SEP$PATH]dnl
        [$PATH_SEPARATOR/opt/cuda/bin$PATH_SEPARATOR/usr/local/cuda/bin])

if test -n "$NVCC" -a $cross_compile = no -a $cuda_req != no
then
        CUDA_PATH=`dirname "$NVCC"`
        CUDA_PATH=`dirname "$CUDA_PATH"`

        if test $system = Windows; then
                AC_CHECK_PROG([CL], [cl], [cl],
                              [])
                if test -n "$CL"; then
                        FOUND_CUDA=yes
                fi
        else
                FOUND_CUDA=yes
        fi
	AC_MSG_CHECKING([CUDA Toolkit version - major])
        nvcc_major=`"$NVCC" --version |grep release|sed  's/^.*release \(@<:@0-9@:>@@<:@0-9@:>@*\).*$/\1/'`
	AC_MSG_RESULT($nvcc_major)
	AC_MSG_CHECKING([CUDA Toolkit version - minor])
        nvcc_minor=`"$NVCC" --version |grep release|sed  's/^.*release @<:@0-9@:>@@<:@0-9@:>@*\.\(@<:@0-9@:>@@<:@0-9@:>@*\).*$/\1/'`
	AC_MSG_RESULT($nvcc_minor)

        CUDA_INC="\"-I$CUDA_PATH/include\""
	CUDA_COMPILER="$NVCC"
        INC="$INC $CUDA_INC"

        if test $ac_cv_sizeof_int_p -eq 8 -a $system = Linux
        then
                CUDA_LIB_PATH="$CUDA_PATH/lib64"
        else
                CUDA_LIB_PATH="$CUDA_PATH/lib"
        fi

        if test $system = Windows -a $nvcc_major -ge 9;
        then
                if test $WORD_LEN = 64; then
                        CUDA_LIB_PATH="$CUDA_LIB_PATH/x64"
                else
                        CUDA_LIB_PATH="$CUDA_LIB_PATH/Win32"
                fi
        fi

        CUDA_LIB="-L\"$CUDA_LIB_PATH\" -lcudart"

        if test $system = MacOSX; then
                LDFLAGS="$LDFLAGS${LDFLAGS:+ }-rpath $CUDA_LIB_PATH"
                DYLIBBUNDLER_FLAGS="${DYLIBBUNDER_FLAGS:+$DYLIBBUNDLER_FLAGS }-s $CUDA_LIB_PATH"
        fi
fi

AC_SUBST(DYLIBBUNDLER_FLAGS)

# Pick compiler for cuda device code
AC_ARG_WITH(cuda-compiler,
              AS_HELP_STRING([--with-cuda-compiler=compiler], [select compiler to compile cuda code with; suported: clang, nvcc; default: nvcc]),
              [CUDA_COMPILER="$withval"],
              [])

HOST_CC_REPORT=" ("
case "$CUDA_COMPILER" in
	*nvcc*)
		;;
	*clang*)
		CUDA_COMPUTE_ARGS="--cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35"
		;;
	"")
		;;
	*)
		UG_MSG_WARN([Unsupported cuda compiler $CUDA_COMPILER])
		;;
esac
HOST_CC_REPORT="$HOST_CC_REPORT"$(basename "$CUDA_COMPILER")

# CUDA host compiler
AC_ARG_WITH(cuda-host-compiler,
	AS_HELP_STRING([--with-cuda-host-compiler=BINARY],[specify compiler used for building cuda host code]))
if test "x${with_cuda_host_compiler}" != "x" ; then
	AC_CHECK_PROG([cuda_host_cc], [${with_cuda_host_compiler}],[yes])
        # note AC_CHECK_PROG always fails on path with spaces in MSW
        if test "x${cuda_host_cc}" = "xyes" -o $system = Windows; then
                CUDA_FLAGS+=" -ccbin \"${with_cuda_host_compiler}\""
	else
		AC_MSG_FAILURE([Binary ${with_cuda_host_compiler} given as cuda host compiler, yet not found!])
	fi
else
	with_cuda_host_compiler="default"
fi

if test "x$FOUND_CUDA" = "xyes" ; then
	HOST_CC_REPORT="$HOST_CC_REPORT, ${with_cuda_host_compiler})"
else
	HOST_CC_REPORT=""
fi

AC_SUBST(FOUND_CUDA)
AC_SUBST(CUDA_PATH)
AC_SUBST(CUDA_LIB)
AC_SUBST(CUDA_INC)
AC_SUBST(CUDA_COMPILER)
AC_SUBST(CUDA_COMPUTE_ARGS)

# -------------------------------------------------------------------------------------------------
# GPUJPEG
GPUJPEG_OBJ=
GPUJPEG_COMPRESS_OBJ=
GPUJPEG_DECOMPRESS_OBJ=
GPUJPEG_INC=
GPUJPEG_LIB=

gpujpeg=no

AC_ARG_ENABLE(gpujpeg,
[  --disable-gpujpeg          disable GPUJPEG compression (auto)]
[                          Requires: libgpujpeg ],
	[gpujpeg_req=$enableval],
        [gpujpeg_req=$build_default])


SAVED_PKG_CONFIG_PATH=$PKG_CONFIG_PATH
PKG_CONFIG_PATH="ext-deps/gpujpeg/install/share/pkgconfig:$PKG_CONFIG_PATH"
PKG_CHECK_MODULES([LIBGPUJPEG_ANY], [libgpujpeg], [ found_gpujpeg_any=yes ], [ found_gpujpeg_any=no ])
PKG_CHECK_MODULES([LIBGPUJPEG], [libgpujpeg >= 0.12.0], [ found_gpujpeg=yes ], [ found_gpujpeg=no ])
PKG_CONFIG_PATH=$SAVED_PKG_CONFIG_PATH

if test "$found_gpujpeg_any" = yes -a "$found_gpujpeg" = no; then
        UG_MSG_WARN([Old GPUJPEG version found, please upgrade])
fi

# try to find without pkg-config (but only if old version was not detected - to avoid side-loading
# that version without pkg-config version check)
if test "$found_gpujpeg_any" != yes -a "$found_gpujpeg" != yes
then
        GPUJPEG_LIB="$GPUJPEG_LIB -lgpujpeg"
        SAVED_LIBS=$LIBS
        LIBS="$LIBS $GPUJPEG_LIB"
        AC_CHECK_LIB([gpujpeg], [gpujpeg_encoder_create])
        AC_CHECK_HEADER([libgpujpeg/gpujpeg_decoder.h])
        LIBS=$SAVED_LIBS
        if test "$ac_cv_lib_gpujpeg_gpujpeg_encoder_create" = yes -a \
                "$ac_cv_header_libgpujpeg_gpujpeg_decoder_h" = yes
        then
                found_gpujpeg=yes
        else
                found_gpujpeg=no
        fi
fi

if test "$gpujpeg_req" != no -a \
        "$found_gpujpeg" = yes
then
        gpujpeg=yes

        GPUJPEG_INC="$GPUJPEG_INC $LIBGPUJPEG_CFLAGS"
        GPUJPEG_LIB="$GPUJPEG_LIB $LIBGPUJPEG_LIBS"
        GPUJPEG_COMPRESS_OBJ="src/video_compress/gpujpeg.o"
        GPUJPEG_DECOMPRESS_OBJ="src/video_decompress/gpujpeg.o "
        AC_DEFINE([HAVE_GPUJPEG], [1], [Build with GPUJPEG support])
        ADD_MODULE("vcompress_gpujpeg", "$GPUJPEG_COMPRESS_OBJ", "$GPUJPEG_LIB")
        ADD_MODULE("vdecompress_gpujpeg", "$GPUJPEG_DECOMPRESS_OBJ", "$GPUJPEG_LIB")

	INC="$INC $GPUJPEG_INC"
fi

if test $gpujpeg_req = yes -a $gpujpeg = no; then
        AC_MSG_ERROR([GPUJPEG not found]);
fi

# -------------------------------------------------------------------------------------------------
# CUDA DXT
# -------------------------------------------------------------------------------------------------
CUDA_DXT_OBJ=
CUDA_DXT_INC=
CUDA_DXT_LIB=

cuda_dxt=no

AC_DEFUN([DEFINE_CUDA_DXT], [
          if test -z "$included_shared_cuda_dxt_cu"; then
                  if test $system = Windows; then
                          CUDA_DXT_COMMON_OBJ="cuda_dxt/cuda_dxt.$CU_OBJ_SUFFIX"
                          CUDA_DXT_COMMON_LIB="cuda_dxt/cuda_dxt.$CU_OBJ_SUFFIX"
                          DLL_LIBS="$DLL_LIBS cuda_dxt/cuda_dxt.dll"
                  else
                          CUDA_DXT_COMMON_OBJ="cuda_dxt/cuda_dxt.$CU_OBJ_SUFFIX"
                          CUDA_DXT_COMMON_LIB=
                  fi
                  included_shared_cuda_dxt_cu=yes
          fi
        ])

AC_ARG_ENABLE(cuda-dxt,
[  --disable-cuda-dxt      disable CUDA DXT compression (auto)]
[                          Requires: CUDA],
	[cuda_dxt_req=$enableval],
        [cuda_dxt_req=$build_default])

LIBS=$SAVED_LIBS

if test "$cuda_dxt_req" != no -a $FOUND_CUDA = yes
then
        cuda_dxt=yes

        DEFINE_CUDA
        DEFINE_CUDA_DXT
        CUDA_DXT_LIB="$CUDA_COMMON_LIB $CUDA_DXT_COMMON_LIB $CUDA_LIB"
        CUDA_DXT_INC=$CUDA_INC
        CUDA_DXT_OBJ="src/video_compress/cuda_dxt.o $CUDA_DXT_COMMON_OBJ $CUDA_COMMON_OBJ"
        AC_DEFINE([HAVE_CUDA_DXT], [1], [Build with CUDA DXT support])
        ADD_MODULE("vcompress_cuda_dxt", $CUDA_DXT_OBJ, "$CUDA_DXT_LIB")
        CUDA_MESSAGE
fi

if test $cuda_dxt_req = yes -a $cuda_dxt = no; then
        AC_MSG_ERROR([CUDA DXT not found]);
fi

# -------------------------------------------------------------------------------------------------
# GPUJPEG transcode to DXT
# -------------------------------------------------------------------------------------------------
gpujpeg_to_dxt=no
AC_ARG_ENABLE(gpujpeg_to_dxt,
[  --disable-jpeg-to-dxt   disable GPUJPEG DXT transcoder (default is disable)]
[                          Requires: CUDA libgpujpeg],
              [gpujpeg_to_dxt_req=$enableval],
              [gpujpeg_to_dxt_req=no])

if test $gpujpeg_to_dxt_req != no -a $FOUND_CUDA = yes -a \
        "$found_gpujpeg" = yes
then
        DEFINE_CUDA
        DEFINE_CUDA_DXT
        gpujpeg_to_dxt=yes
        GPUJPEG_TO_DXT_INC=" $CUDA_INC"
        GPUJPEG_TO_DXT_LIB="$CUDA_DXT_COMMON_LIB $CUDA_COMMON_LIB $CUDA_LIB $GPUJPEG_LIB"
        GPUJPEG_TO_DXT_OBJ="src/video_decompress/gpujpeg_to_dxt.o $CUDA_COMMON_OBJ $CUDA_DXT_COMMON_OBJ"
        ADD_MODULE("vdecompress_gpujpeg_to_dxt", $GPUJPEG_TO_DXT_OBJ, "$GPUJPEG_TO_DXT_LIB")
        AC_DEFINE([HAVE_GPUJPEG_TO_DXT], [1], [Build with GPUJPEG DXT transcode support])
        CUDA_MESSAGE
fi


if test $gpujpeg_to_dxt = no -a $gpujpeg_to_dxt_req = yes
then
        AC_MSG_ERROR([GPUJPEG DXT transcoder not found])
fi

# -------------------------------------------------------------------------------------------------
# SAGE stuff

SAGE_OBJ=
SAGE_INC=
SAGE_LIB=
sage=no

AC_ARG_ENABLE(sage,
[  --disable-sage          disable SAGE (default is auto)]
[                          Requires: SAGE],
	[sage_req=$enableval],
        [sage_req=$build_default])

AC_ARG_ENABLE(sage-native-dxt5ycocg,
      AS_HELP_STRING([--enable-sage-native-dxt5ycocg], [Assume that SAGE supports native DXT5 YCoCg (default is disable)]),
	[native_ycocg=$enableval],
        [native_ycocg=no])

AC_ARG_WITH(sage,
	[  --with-sage=DIR         specify location of SAGE],
	[SAGE_DIRECTORY=$withval])

if test $native_ycocg = yes
then
        AC_DEFINE([SAGE_NATIVE_DXT5YCOCG], [1], [Assume that SAGE supports native DXT5 YCoCg])
fi

# if directory not set manually, use default installation location
if test -z "$SAGE_DIRECTORY"
then
        SAGE_DIRECTORY=/usr/local/sage
fi

SAGE_INC=$SAGE_DIRECTORY/include
SAGE_LIB=$SAGE_DIRECTORY/lib64

AC_CHECK_LIB(sail, main, FOUND_SAGE_L=yes, FOUND_SAGE_L=no, -lquanta -I${SAGE_INC} -L${SAGE_LIB})
# if library not found in lib64/ try lib/
if test $FOUND_SAGE_L = no
then
        SAGE_LIB=$SAGE_DIRECTORY/lib
        AC_CHECK_LIB(sail, main, FOUND_SAGE_L=yes, FOUND_SAGE_L=no, -lquanta -I${SAGE_INC} -L${SAGE_LIB})
fi

AC_CHECK_HEADERS(${SAGE_INC}/sail.h, FOUND_SAGE_H=yes, FOUND_SAGE_H=no)

if test $sage_req != no -a $FOUND_SAGE_L = yes -a $FOUND_SAGE_H = yes
then
    SAGE_INC=-I${SAGE_INC}
    SAGE_LIB=-"L${SAGE_LIB} -lsail -lquanta"
    SAGE_OBJ="src/video_display/sage.o"
    AC_DEFINE([HAVE_SAGE], [1], [Build with SAGE support])
    CXXFLAGS="$CXXFLAGS -DQUANTA_USE_PTHREADS -DQUANTA_THREAD_SAFE -DGLSL_YUV"
    ADD_MODULE("display_sage", "$SAGE_OBJ", "$SAGE_LIB")
    INC="$INC $SAGE_INC"
    sage=yes
fi

if test $sage_req = yes -a $sage = no; then
        AC_MSG_ERROR([SAGE not found]);
fi

# -------------------------------------------------------------------------------------------------
# AUDIO stuff

portaudio=no

AC_ARG_ENABLE(portaudio,
[  --disable-portaudio     disable Portaudio (default is auto)]
[                          Requires: portaudio-2.0],
	[portaudio_req=$enableval],
        [portaudio_req=$build_default])


SAVED_LIBS=$LIBS

AC_CHECK_HEADER(portaudio.h)
AC_CHECK_LIB(portaudio, Pa_GetDeviceCount) # present in v19
LIBS=$SAVED_LIBS

if test $ac_cv_lib_portaudio_Pa_GetDeviceCount = yes -a $ac_cv_header_portaudio_h = yes -a $portaudio_req != no
then
        PORTAUDIO_LIB='-lportaudio'
        portaudio=yes
fi


if test $portaudio = yes
then
        AC_CHECK_LIB(portaudio, Pa_GetVersionInfo,
                     AC_DEFINE([HAVE_PA_GETVERSIONINFO], [1], [Portaudio
                                has Pa_GetVersionInfo]))
        PORTAUDIO_CAP_OBJ="src/audio/portaudio_common.o src/audio/capture/portaudio.o"
        PORTAUDIO_PLAY_OBJ="src/audio/portaudio_common.o src/audio/playback/portaudio.o"
        AC_DEFINE([HAVE_PORTAUDIO], [1], [Build with Portaudio support])
        ADD_MODULE("acap_portaudio", "$PORTAUDIO_CAP_OBJ", "$PORTAUDIO_LIB")
        ADD_MODULE("aplay_portaudio", "$PORTAUDIO_PLAY_OBJ", "$PORTAUDIO_LIB")
fi

if test $portaudio_req = yes -a $portaudio = no; then
        AC_MSG_ERROR([PortAudio not found]);
fi

# ---------------------------------------------------------------------------
# SpeexDSP
# ---------------------------------------------------------------------------
speexdsp=no

AC_ARG_ENABLE(speexdsp,
              AS_HELP_STRING([--disable-speexdsp], [disable SpeexDSP (default is auto)]),
              [speexdsp_req=$enableval],
              [speexdsp_req=$build_default])
PKG_CHECK_MODULES([LIBSPEEXDSP], [speexdsp], [found_speexdsp=yes], [found_speexdsp=no])

if test "$found_speexdsp" = yes -a "$speexdsp_req" != no; then
        OBJS="$OBJS src/audio/echo.o"
        LIBS="$LIBS $LIBSPEEXDSP_LIBS"
        COMMON_FLAGS="$COMMON_FLAGS${SPEEXDSP_COMMON_FLAGS:+${COMMON_FLAGS:+ }}$SPEEXDSP_COMMON_FLAGS"
        AC_DEFINE([HAVE_SPEEXDSP], [1], [Build with SpeexDSP support])
        speexdsp=yes
elif test "$speexdsp_req" = yes; then
        AC_MSG_ERROR([SpeexDSP not found]);
elif test "$found_speexdsp" = no -a "$speexdsp_req" != no; then
        UG_MSG_WARN([SpeexDSP was not found. Strongly recommending installing that, otherwise audio part of UG will be crippled.])
fi

# ---------------------------------------------------------------------------
# Zfec
# ---------------------------------------------------------------------------
zfec=no
AC_ARG_ENABLE(zfec,
              AS_HELP_STRING([--disable-zfec], [disable zfec (default is auto)]),
              [zfec_req=$enableval],
              [zfec_req=$build_default])
AC_ARG_WITH(zfec,
        AS_HELP_STRING([--with-zfec=DIR], [path to zfec sources]),
        [ZFEC_PREFIX=$withval],
        [ZFEC_PREFIX=$srcdir/ext-deps/zfec/zfec]
)
found_zfec=no
if test $cross_compile = no; then
        AC_CHECK_FILES([$ZFEC_PREFIX/fec.c $ZFEC_PREFIX/fec.h], [found_zfec=yes])
        if test $found_zfec = no; then
                ZFEC_PREFIX=/usr/lib/python3/dist-packages/zfec
                if test ! -d "$ZFEC_PREFIX"; then
                        ZFEC_PREFIX=$(ls -d /usr/lib/python* | tail -n 1)/site-packages/zfec
                fi
                if test ! -d "$ZFEC_PREFIX"; then
                        ZFEC_PREFIX=/usr/local/src/zfec
                fi
                if test ! -d "$ZFEC_PREFIX"; then
                        ZFEC_PREFIX=/usr/src/zfec
                fi
                AC_CHECK_FILES([$ZFEC_PREFIX/fec.c $ZFEC_PREFIX/fec.h], [found_zfec=yes])
        fi
fi

if test "$found_zfec" = yes -a "$zfec_req" != no; then
        OBJS="$OBJS src/zfec.o"
        COMMON_FLAGS="$COMMON_FLAGS -I$ZFEC_PREFIX"
        AC_DEFINE([HAVE_ZFEC], [1], [Build with zfec support])
        AC_SUBST(ZFEC_PREFIX)
        zfec=yes
elif test "$zfec_req" = yes; then
        AC_MSG_ERROR([Zfec not found]);
fi

# -------------------------------------------------------------------------------------------------
#
# Jack stuff
#

# transport
jack_trans=no
jack=no

AC_ARG_ENABLE(jack-transport,
[  --enable-jack-transport[=force] enable JACK transport (default is auto)]
[                          Requires: jack],
	[jack_trans_req=$enableval],
        [jack_trans_req=$build_default])


AC_CHECK_HEADER(jack/jack.h, FOUND_JACK_H=yes, FOUND_JACK_H=no)

if test $jack_trans_req != no -a $FOUND_JACK_H = yes
then
    JACK_TRANS_OBJ="src/audio/jack.o"
    AC_DEFINE([HAVE_JACK_TRANS], [1], [Build with JACK transport support])
    jack_trans=yes
fi
AC_SUBST(JACK_TRANS_OBJ)

# sound system
AC_ARG_ENABLE(jack,
[  --disable-jack          disable JACK (default is auto)]
[                          Requires: jack],
	[jack_req=$enableval],
        [jack_req=$build_default])

if test $jack_req != no -a $FOUND_JACK_H = yes
then
    JACK_CAP_OBJ="src/audio/capture/jack.o"
    JACK_PLAY_OBJ="src/audio/playback/jack.o"
    AC_DEFINE([HAVE_JACK], [1], [Build with JACK support])
    jack=yes
    ADD_MODULE("acap_jack", "$JACK_CAP_OBJ", "")
    ADD_MODULE("aplay_jack", "$JACK_PLAY_OBJ", "")
fi

if test $jack_req = yes -a $jack = no; then
        AC_MSG_ERROR([JACK not found]);
fi

# -------------------------------------------------------------------------------------------------
#
# ALSA
#
AC_ARG_ENABLE(alsa,
[  --disable-alsa          disable ALSA (default is auto)]
[                          Requires: alsa],
	[alsa_req=$enableval],
        [alsa_req=$build_default])
AC_CHECK_HEADER(alsa/asoundlib.h)
SAVED_LIBS=$LIBS
AC_CHECK_LIB(asound, snd_pcm_open)
LIBS=$SAVED_LIBS
if test "$alsa_req" != no -a "$ac_cv_header_alsa_asoundlib_h" = yes -a "$ac_cv_lib_asound_snd_pcm_open" = yes
then
        alsa=yes
        AC_DEFINE([HAVE_ALSA], [1], [Build with ALSA support])
        ALSA_CAP_OBJ="src/audio/capture/alsa.o"
        ALSA_PLAY_OBJ="src/audio/playback/alsa.o"
        ALSA_LIB="-lasound"
        ADD_MODULE("acap_alsa", "$ALSA_CAP_OBJ", "$ALSA_LIB")
        ADD_MODULE("aplay_alsa", "$ALSA_PLAY_OBJ", "$ALSA_LIB")
else
        alsa=no
fi

if test $alsa_req = yes -a $alsa = no; then
        AC_MSG_ERROR([ALSA not found]);
fi

# -------------------------------------------------------------------------------------------------
# CoreAudio Stuff
COREAUDIO_LIB=
COREAUDIO_OBJ=
coreaudio=no

AC_ARG_ENABLE(coreaudio,
      AS_HELP_STRING([--disable-coreaudio], [disable CoreAudio support(default is auto)]),
    [coreaudio_req=$enableval],
    [coreaudio_req=$build_default]
    )

case $host_os in
	*darwin*)
		AC_CHECK_HEADERS([AudioUnit/AudioUnit.h], [FOUND_AUDIOUNIT_H=yes], [echo "AudioUnit headers not found."])

		if test $coreaudio_req != no -a $FOUND_AUDIOUNIT_H = yes
		then
			COREAUDIO_LIB="-framework CoreAudio -framework AudioUnit -framework AVFoundation"
			COREAUDIO_OBJ="src/audio/capture/coreaudio.o src/audio/playback/coreaudio.o"
                        coreaudio=yes
                        AC_DEFINE([HAVE_COREAUDIO], [1], [Build with CoreAudio support])
                        ADD_MODULE("", "$COREAUDIO_OBJ", "$COREAUDIO_LIB")
		fi
    ;;
esac

if test $coreaudio_req = yes -a $coreaudio = no; then
        AC_MSG_ERROR([Core Audio not found]);
fi

# -------------------------------------------------------------------------------------------------
# Scale Stuff
# -------------------------------------------------------------------------------------------------
SCALE_INC=
SCALE_LIB=
SCALE_OBJ=
scale=no

AC_ARG_ENABLE(scale,
[  --disable-scale         disable scale postprocessor (default is auto)]
[                          Requires: gl],
    [scale_req=$enableval],
    [scale_req=$build_default]
    )

if test $scale_req != no -a $OPENGL = yes
then
        scale=yes
        SCALE_LIB="$SCALE_LIB $OPENGL_LIB"
        SCALE_OBJ="$SCALE_OBJ $GL_COMMON_OBJ src/vo_postprocess/scale.o"
        ADD_MODULE("vo_pp_scale", "$SCALE_OBJ", "$SCALE_LIB")
        AC_DEFINE([HAVE_SCALE], [1], [Build scale postprocessor])
fi

if test $scale_req = yes -a $scale = no; then
        AC_MSG_ERROR([Scale not found]);
fi

# -------------------------------------------------------------------------------------------------
# Text Stuff
# -------------------------------------------------------------------------------------------------
TEXT_INC=
TEXT_LIB=
TEXT_OBJ=
text=no

define(text_dep, MagickWand)

AC_ARG_ENABLE(text,
[  --disable-text          disable text postprocessor (default is auto)]
[                          Requires: text_dep ],
    [text_req=$enableval],
    [text_req=$build_default]
    )

PKG_CHECK_MODULES([TEXT], [text_dep], FOUND_TEXT_DEPS=yes, FOUND_TEXT_DEPS=no)

if test $text_req != no -a "$FOUND_TEXT_DEPS" = yes
then
        text=yes
        COMMON_FLAGS="$COMMON_FLAGS $TEXT_CFLAGS"
        TEXT_LIB="$TEXT_LIBS"
        TEXT_OBJ="$TEXT_OBJ src/vo_postprocess/text.o"
        CFLAGS="$CFLAGS $MAGICKWAND_CFLAGS"
        ADD_MODULE("vo_pp_text", "$TEXT_OBJ", "$TEXT_LIB")
        AC_DEFINE([HAVE_TEXT], [1], [Build text postprocessor])
        PKG_CHECK_MODULES([WAND7], [MagickWand > 7], [FOUND_WAND7=yes], [FOUND_WAND7=NO])
        if test "$FOUND_WAND7" = yes; then
                AC_DEFINE([WAND7], [1], [MagickWand is v7 or later])
        fi
fi

if test $text_req = yes -a $text = no; then
        AC_MSG_ERROR([text not found]);
fi

# -------------------------------------------------------------------------------------------------
# QT GUI

qt_gui=no
static_qt=no

GUI_EXE=
GUI_TARGET=

QMAKE=
#qmake # default for most distributions

define(qt5_gui_dep, Qt5Core >= 5.4 Qt5Widgets Qt5Gui)

AC_ARG_ENABLE(qt,
[  --enable-qt             build Qt GUI (default is disabled)]
[                          Requires: qt5_gui_dep | Qt6 ],
    [qt_req=$enableval],
    [qt_req=no]
    )

AC_ARG_ENABLE(static-qt,
    AS_HELP_STRING([--enable-static-qt], [use statically built Qt (default is disabled)]),
    [static_qt=$enableval],
    [static_qt=no]
    )

# Check for Qt libraries
SAVED_PKG_CONFIG=$PKG_CONFIG
if test $static_qt = yes; then
        PKG_CONFIG='pkg-config --static'
fi
PKG_CHECK_MODULES(QT, [qt5_gui_dep], [FOUND_QT_DEP=yes], [FOUND_QT_DEP=no])
PKG_CONFIG=$SAVED_PKG_CONFIG

# TODO: fix Qt 6 lookup (and precedence?)
AC_PATH_PROGS_FEATURE_CHECK([QTCHOOSER], [qtchooser],
        [AS_IF([$ac_path_QTCHOOSER --run-tool=qmake --qt=5 | grep QMake  > /dev/null],
                [QMAKE="$ac_path_QTCHOOSER --run-tool=qmake --qt=5"])])

#AC_MSG_NOTICE("Qmake: $QMAKE")
if test "x$QMAKE" = "x"; then
        AC_PATH_PROGS(QMAKE, [qmake6 qmake-qt5 qmake], [:], [/usr/lib/qt5/bin:$PATH])
fi

if test "$QMAKE" -a "$QMAKE" != ":"
then
        QMAKE_VERSION=`$QMAKE -v | sed -n 's/.*Qt version \(@<:@^ @:>@*\).*/\1/p'`
fi

# Qt deps not found with pkg-config but qmake found (eg. in Qt 6)
if test $FOUND_QT_DEP = no && test -n "$QMAKE_VERSION" && expr "$QMAKE_VERSION" \>= 6 >/dev/null
then
        touch qt.pro
        MAKEFILE=Make.qt
        $QMAKE qt.pro -o $MAKEFILE
        if test -f Make.qt.Release; then
                MAKEFILE=Make.qt.Release
        fi
        printf 'print-libs:\n\t@echo $(LIBS)\n\n' >> $MAKEFILE
        printf 'print-cxxflags:\n\t@echo $(INCPATH) $(CXXFLAGS)\n\n' >> $MAKEFILE
        QT_CFLAGS=$(make -f $MAKEFILE print-cxxflags)
        QT_LIBS=$(make -f $MAKEFILE print-libs)
        rm qt.pro Make.qt*
        FOUND_QT_DEP=yes
fi

if test $qt_req != no -a $FOUND_QT_DEP = yes
then
	#AC_MSG_NOTICE("Qmake: $QMAKE")
        GUI_TARGET=gui/QT/uv-qt$APPEXT
        GUI_EXE=$GUI_TARGET
        if test $system = MacOSX; then
                GUI_EXE=$GUI_TARGET/Contents/MacOS/uv-qt
        fi

	if test ":" = "$QMAKE"; then
		qt_gui=no
	else
		qt_gui=yes
	fi
fi

AC_SUBST(GUI_EXE)
AC_SUBST(GUI_TARGET)
AC_SUBST(QMAKE)
AC_SUBST(QT_CFLAGS)

if test $qt_req = yes -a $qt_gui = no; then
		AC_MSG_ERROR([Qt libraries are required.])
fi

# -------------------------------------------------------------------------------------------------
# V4L2
# -------------------------------------------------------------------------------------------------
v4l2=no
found_v4l2=no

AC_ARG_ENABLE(v4l2,
    AS_HELP_STRING([--disable-v4l2], [disable V4L2 support (default is auto); requires: Linux, v4lconvert (optional)]),
    [v4l2_req=$enableval],
    [v4l2_req=$build_default]
    )

if test "$system" = Linux && test "$v4l2_req" != no
then
        PKG_CHECK_MODULES([V4LCONVERT], [libv4lconvert], [found_v4lconvert=yes], [found_v4lconvert=no])
        if test $found_v4lconvert = yes; then
                AC_DEFINE([HAVE_LIBV4LCONVERT], [1], [Build with libv4lconvert support])
        elif test if "$v4l2_req" = yes; then
                UG_MSG_WARN([Compiling V4L2 capture without v4lconvert support (not found).])
        fi
        V4L2_CAP_OBJ="src/video_capture/v4l2.o"
        V4L2_DISP_OBJ="src/video_display/v4l2.o"
        ADD_MODULE("vidcap_v4l2", "$V4L2_CAP_OBJ", "$V4LCONVERT_LIBS")
        ADD_MODULE("display_v4l2", "$V4L2_DISP_OBJ", "")
        CFLAGS="$CFLAGS $V4LCONVERT_CFLAGS"
        v4l2=yes
fi

if test $v4l2_req = yes -a $v4l2 = no; then
        AC_MSG_ERROR([V4L2 not found]);
fi

# -----------------------------------------------------------------------------
# WASAPI
# -----------------------------------------------------------------------------
wasapi=no
AC_ARG_ENABLE(wasapi,
      AS_HELP_STRING([--disable-wasapi], [disable WASAPI support (default is auto)]),
    [wasapi_req=$enableval],
    [wasapi_req=$build_default]
    )

if test $system = Windows -a $wasapi_req != no; then
        AC_DEFINE([HAVE_WASAPI], [1], [Build with WASAPI support])
        ADD_MODULE("aplay_wasapi", "src/audio/capture/wasapi.o src/audio/playback/wasapi.o", "")
        wasapi=yes
fi

if test $wasapi_req = yes -a $wasapi = no; then
        AC_MSG_ERROR([WASAPI not found]);
fi

# -------------------------------------------------------------------------------------------------
# iHDTV
# -------------------------------------------------------------------------------------------------
AC_ARG_ENABLE(ihdtv,
      AS_HELP_STRING([--enable-ihdtv], [enable iHDTV support (default is disabled)]),
    [ihdtv_req=$enableval],
    [ihdtv_req=no]
    )
ihdtv=no

if test $ihdtv_req = yes
then
        AC_DEFINE([HAVE_IHDTV], [1], [Build with iHDTV support])
        ihdtv=yes
fi

# -------------------------------------------------------------------------------------------------
# use shared decoder
# -------------------------------------------------------------------------------------------------

# the shared behaviour is faulty, but it saves resources when restarting sender (different SSRC).
shared_decoder=no
if test $shared_decoder = yes
then
        AC_DEFINE([SHARED_DECODER], [1], [use shared decoder for all participants])
fi

# -------------------------------------------------------------------------------------------------
# OpenSSL-libcrypto
# -------------------------------------------------------------------------------------------------
crypto=no
AC_ARG_ENABLE(openssl,
[  --disable-openssl       disable OpenSSL support (default is auto)]
[                          Requires: crypto],
    [crypto_req=$enableval],
    [crypto_req=$build_default]
    )

PKG_CHECK_MODULES([CRYPTO], [libcrypto], [FOUND_CRYPTO_PC=yes], [FOUND_CRYPTO_PC=no])

if test "$FOUND_CRYPTO_PC" = yes; then
        SAVED_CFLAGS=$CFLAGS
        CFLAGS="$CFLAGS $CRYPTO_CFLAGS"
        AC_CHECK_HEADER(openssl/aes.h, FOUND_CRYPTO_H=yes, FOUND_CRYPTO_H=no)
        if test $FOUND_CRYPTO_H = no; then
                UG_MSG_WARN([libcrypto.pc found but openssl/aes.h is not present. This is usually caused by presence of /usr/lib/pkgconfig/libcrypto.pc on macOS. If encryption is requested, install openssl from Homebrew and set PKG_CONFIG_PATH accordingly (see 'brew ln openssl').])
        fi
        CFLAGS=$SAVED_CFLAGS
fi

if test $crypto_req != no -a "$FOUND_CRYPTO_PC" = yes -a "$FOUND_CRYPTO_H" = yes; then
        crypto=yes
        OPENSSL_OBJ="src/crypto/openssl_encrypt.o src/crypto/openssl_decrypt.o"
        CFLAGS="$CFLAGS $CRYPTO_CFLAGS"
        CXXFLAGS="$CXXFLAGS $CRYPTO_CFLAGS"
        CPPFLAGS="$CPPFLAGS $CRYPTO_CFLAGS"
        ADD_MODULE("openssl", "$OPENSSL_OBJ", "$CRYPTO_LIBS")
        SAVED_LIBS=$LIBS
        LIBS="$LIBS $CRYPTO_LIBS"
        AC_CHECK_FUNCS(AES_ctr128_encrypt)
        LIBS=$SAVED_LIBS
fi

if test $crypto_req = yes -a $crypto = no; then
        AC_MSG_ERROR([OpenSSL not found]);
fi

# -------------------------------------------------------------------------------------------------
# GPU accelerated LDGM
# -------------------------------------------------------------------------------------------------
ldgm_gpu=no
AC_ARG_ENABLE(ldgm-gpu,
[  --disable-ldgm-gpu      disable GPU accelerated LDGM (default is auto)]
[                          Requires: CUDA],
    [ldgm_gpu_req=$enableval],
    [ldgm_gpu_req=$build_default]
    )

if test $ldgm_gpu_req != no -a $FOUND_CUDA = yes
then
        DEFINE_CUDA
        CUDA_MESSAGE
        AC_DEFINE([HAVE_LDGM_GPU], [1], [Build with GPU accelerated LDGM])
        LDGM_GPU_OBJS="ldgm/src/ldgm-session-gpu.o src/rtp/ldgm_gpu.o $CUDA_COMMON_OBJ"
        LDGM_GPU_LIBS="$CUDA_COMMON_LIB $CUDA_LIB"
        if test $system = Windows; then
                DLL_LIBS="$DLL_LIBS ldgm/src/gpu.dll"
                LDGM_GPU_OBJS="$LDGM_GPU_OBJS ldgm/src/gpu.$CU_OBJ_SUFFIX"
                LDGM_GPU_LIBS="$LDGM_GPU_LIBS ldgm/src/gpu.$CU_OBJ_SUFFIX"
        else
                LDGM_GPU_OBJS="$LDGM_GPU_OBJS ldgm/src/gpu.$CU_OBJ_SUFFIX"
        fi
        ADD_MODULE("ldgm_gpu", "$LDGM_GPU_OBJS", "$LDGM_GPU_LIBS")
        if test $WORD_LEN = 32 -a $system = Linux; then
                CUDA_FLAGS="$CUDA_FLAGS -Xcompiler -msse2"
        fi
        ldgm_gpu=yes
else
        ldgm_gpu=no
fi

if test $ldgm_gpu_req = yes -a $ldgm_gpu = no; then
        AC_MSG_ERROR([LDGM accelerated GPU cannot be enabled (CUDA not found?)]);
fi

# -------------------------------------------------------------------------------------------------
# Syphon
# -------------------------------------------------------------------------------------------------
syphon=no
AC_ARG_ENABLE(syphon,
[  --enable-syphon         enables Syphon support (default is auto)]
[                          Requires: Syphon],
    [syphon_req=$enableval],
    [syphon_req=$build_default]
    )

MY_CHECK_FRAMEWORK([Syphon])

if test $syphon_req != no -a $my_cv_framework_Syphon = yes -a $host_cpu = x86_64
then
        AC_DEFINE([HAVE_SYPHON], [1], [Build with Syphon support])
        LIBS="$LIBS -framework Syphon"
        OBJS="$OBJS src/syphon_server.o src/video_capture/syphon.o"
        BIN_DEPS="${BIN_DEPS:+$BIN_DEPS }Frameworks/Syphon.framework"
        syphon=yes
fi

if test $syphon_req = yes -a $syphon = no; then
        AC_MSG_ERROR([Syphon framework cannot be found!]);
fi


# ------------------------------------------------------------------------------
# Spout
# ------------------------------------------------------------------------------
spout=no
AC_ARG_ENABLE(spout,
[  --enable-spout          enables Spout support (default is auto)]
[                          Requires: OpenGL Spout],
    [spout_req=$enableval],
    [spout_req=$build_default]
    )

FOUND_SPOUT=no
if test $system = Windows
then
        AC_LANG_PUSH([C++])
        SAVED_LIBS="$LIBS"
        LIBS="$LIBS -lSpout"
        AC_LINK_IFELSE([AC_LANG_PROGRAM([[#include <SpoutSDK/Spout.h>]],
                       [[SpoutSender *sender = new SpoutSender;]])],
                       FOUND_SPOUT=yes, FOUND_SPOUT=no)
        LIBS=$SAVED_LIBS
        AC_LANG_POP([C++])
fi

if test $spout_req != no -a "$OPENGL" = yes -a $FOUND_SPOUT = yes
then
        AC_DEFINE([HAVE_SPOUT], [1], [Build with Spout support])
        LIBS="$LIBS -lSpout"
        OBJS="$OBJS src/spout_sender.o src/video_capture/spout.o"
        spout=yes
fi

if test $spout_req = yes -a $spout = no; then
        AC_MSG_ERROR([Could not found Spout dependencies!]);
fi

# -------------------------------------------------------------------------------------------------
# MCU-like video mixer
# -------------------------------------------------------------------------------------------------
video_mix=no

AC_ARG_ENABLE(video-mixer,
[  --disable-video-mixer   disable MCU-like video mixer (default is auto)]
[                          Requires: opencv],
    [video_mix_req=$enableval],
    [video_mix_req=$build_default]
    )

if test x$HAVE_OPENCV != xno && test "$FOUND_OPENCV_IMGPROC" = yes; then
	FOUND_VIDEO_MIX_DEP=yes
	VIDEO_MIX_LIBS="$OPENCV_LIBS -lopencv_imgproc"
	VIDEO_MIX_CFLAGS=$OPENCV_CFLAGS
else
	FOUND_VIDEO_MIX_DEP=no
fi
MIXER_CPU_ONLY=no

if test $video_mix_req != no -a $FOUND_VIDEO_MIX_DEP = yes -a x$OPENCV_MAJOR = x3
then
	AC_CHECK_LIB([opencv_cudawarping], [main], [
		VIDEO_MIX_LIBS="$VIDEO_MIX_LIBS -lopencv_cudawarping "
		AC_DEFINE([HAVE_OPENCV_CUDA], [1], [Whether the opencv installation supports cuda modules])
	], [
		MIXER_CPU_ONLY=yes
	], [
		$VIDEO_MIX_LIBS
	])
elif test x$OPENCV_MAJOR != x3
then
	AC_CHECK_HEADER(opencv2/gpu/gpu.hpp, AC_DEFINE([HAVE_OPENCV_CUDA], [1],
			[Whether the opencv installation supports cuda modules]),
			MIXER_CPU_ONLY=yes)
fi

if test $video_mix_req != no -a $FOUND_VIDEO_MIX_DEP = yes
then
        if test $MIXER_CPU_ONLY = yes; then
            if test $video_mix_req = yes; then
		UG_MSG_WARN([Video mixer will be compiled as CPU-only])
            else
		AC_MSG_WARN([Video mixer will be compiled as CPU-only])
            fi
        fi
        CFLAGS="$CFLAGS ${VIDEO_MIX_CFLAGS}"
        CXXFLAGS="$CXXFLAGS ${VIDEO_MIX_CFLAGS}"
        VIDEO_MIX_OBJ="src/video_display/conference.o"
        ADD_MODULE("display_video_mix", "$VIDEO_MIX_OBJ", "$VIDEO_MIX_LIBS")
        video_mix=yes
fi

if test $video_mix_req = yes -a $video_mix = no; then
        AC_MSG_ERROR([Could not found OpenCV needed for video mixer!]);
fi

# ------------------------------------------------------------------------------------------------
# BitFlow
# -------------------------------------------------------------------------------------------------
AC_ARG_ENABLE(bitflow,
      AS_HELP_STRING([--enable-bitflow], [Compile BitFlow grabber support]),
    [bitflow_req=$enableval],
    [bitflow_req=no]
    )

bitflow=no

if test $bitflow_req = yes
then
        LIBS="$LIBS -lBFSOciLib -lBFciLib -lBFML"
        OBJS="$OBJS src/video_capture/bitflow.o"
        bitflow=yes
fi

# ------------------------------------------------------------------------------
# Comprimato J2K
# ------------------------------------------------------------------------------
cmpto_j2k=no
AC_ARG_ENABLE(cmpto-j2k,
[  --enable-cmpto-j2k      enables Comprimato J2K support (default is auto)]
[                          Requires: cmpto_j2k],
    [cmpto_j2k_req=$enableval],
    [cmpto_j2k_req=$build_default]
    )

AC_CHECK_HEADER(cmpto_j2k_enc.h, FOUND_CMPTO_J2K_ENC_H=yes, FOUND_CMPTO_J2K_ENC_H=no)
AC_CHECK_HEADER(cmpto_j2k_dec.h, FOUND_CMPTO_J2K_DEC_H=yes, FOUND_CMPTO_J2K_DEC_H=no)
AC_CHECK_LIB(cmpto_j2k_enc, cmpto_j2k_enc_ctx_cfg_create, FOUND_CMPTO_J2K_ENC_L=yes, FOUND_CMPTO_J2K_ENC_L=no)
AC_CHECK_LIB(cmpto_j2k_dec, cmpto_j2k_dec_ctx_cfg_create, FOUND_CMPTO_J2K_DEC_L=yes, FOUND_CMPTO_J2K_DEC_L=no)

if test $cmpto_j2k_req != no -a "$FOUND_CMPTO_J2K_ENC_H" = yes -a "$FOUND_CMPTO_J2K_DEC_H" = yes -a "$FOUND_CMPTO_J2K_ENC_L" = yes -a "$FOUND_CMPTO_J2K_DEC_L" = yes
then
        ADD_MODULE("vcompress_cmpto_j2k", "src/video_compress/cmpto_j2k.o", "-lcmpto_j2k_enc")
        ADD_MODULE("vdecompress_cmpto_j2k", "src/video_decompress/cmpto_j2k.o", "-lcmpto_j2k_dec")
        cmpto_j2k=yes
fi

if test $cmpto_j2k_req = yes -a $cmpto_j2k = no; then
        AC_MSG_ERROR([Comprimato J2K not found!]);
fi

# ------------------------------------------------------------------------------
# NDI
# ------------------------------------------------------------------------------
ndi=no
AC_ARG_ENABLE(ndi,
      AS_HELP_STRING([--enable-ndi@<:@=ver@:>@], [enables NDI support (default is auto)]),
    [ndi_req=$enableval],
    [ndi_req=$build_default]
    )

AC_CHECK_HEADER(Processing.NDI.Lib.h, FOUND_NDI_H=yes, FOUND_NDI_H=no)

NDI_VER=
if expr $ndi_req : '@<:@0-9@:>@' >/dev/null; then
        NDI_VER=$ndi_req
        ndi_req=yes
fi

# if NDI_SDK_DIR is defined in Windows, ignore autoconf tests
if test $ndi_req != no && ( test "$FOUND_NDI_H" = yes || ( test -n "$NDI_SDK_DIR" && test $system = Windows ) )
then
        if test -n "$NDI_SDK_DIR" && test $system = Windows; then
                COMMON_FLAGS="$COMMON_FLAGS -I\"$NDI_SDK_DIR/Include\""
        fi
        if test -n "$NDI_VER"; then
                AC_DEFINE_UNQUOTED([USE_NDI_VERSION], [$NDI_VER], [NDI version to use])
        fi
        ADD_MODULE("vidcap_ndi", "src/video_capture/ndi.o", "")
        ADD_MODULE("display_ndi", "src/video_display/ndi.o", "")
        ndi=yes
fi

if test $ndi_req = yes -a $ndi = no; then
        AC_MSG_ERROR([NDI not found!]);
fi

# ------------------------------------------------------------------------------
# Cineform 
# ------------------------------------------------------------------------------
CINEFORM_PATH=cineform-sdk
CINEFORM_INC=
CINEFORM_LIB=

cineform=no

AC_ARG_ENABLE(cineform,
              [  --disable-cineform      disable cineform suppport (default is auto)]
              [            Requires: CINEFORM_SDK],
              [cineform_req=$enableval],
              [cineform_req=$build_default]
              )

if test -n "$CINEFORM_DIRECTORY"; then
        CINEFORM_PATH=$CINEFORM_DIRECTORY
fi

AC_ARG_WITH(cineform,
            [  --with-cineform=DIR     specify location of Cineform SDK],
            [CINEFORM_PATH=$withval
             ])

AC_LANG_PUSH([C++])
AC_CHECK_LIB(CFHDCodec, CFHD_OpenDecoder, [FOUND_CINEFORM=yes; CINEFORM_LIB=-lCFHDCodec], FOUND_CINEFORM=no)
AC_LANG_POP([C++])

if test "$FOUND_CINEFORM" = no
then

        if test $system = Windows
        then
                CINEFORM_LIB="${CINEFORM_PATH}/Release/CFHDCodec.lib"
        else
                CINEFORM_LIB="${CINEFORM_PATH}/libCFHDCodec.a"
        fi

        if test -n "$CINEFORM_PATH" -a $cross_compile = no
        then
                AC_CHECK_FILES(${CINEFORM_LIB}, FOUND_CINEFORM=yes, FOUND_CINEFORM=no)
        fi
fi

if test "$FOUND_CINEFORM" = yes; then
        CINEFORM_INC="$CINEFORM_INC -I${CINEFORM_PATH}/Common/"

        if test $system != Windows
        then
                CINEFORM_LIB="${CINEFORM_LIB} -luuid"
        fi
else # if not found, look for libcineform with pkg-config configuration
        PKG_CHECK_MODULES([CINEFORM], [libcineformsdk], [FOUND_CINEFORM=yes], [FOUND_CINEFORM=no])
        if test "$FOUND_CINEFORM" = no; then
                PKG_CHECK_MODULES([CINEFORM], [libcineform], [FOUND_CINEFORM=yes], [FOUND_CINEFORM=no])
        fi
        CINEFORM_INC=$CINEFORM_CFLAGS
        CINEFORM_LIB=$CINEFORM_LIBS
fi

if test "$cineform_req" != no -a "$FOUND_CINEFORM" = yes
then
        cineform=yes
        CINEFORM_COMPRESS_OBJ="src/video_compress/cineform.o"
        CINEFORM_DECOMPRESS_OBJ="src/video_decompress/cineform.o"

        INC="$INC $CINEFORM_INC"
        
        ADD_MODULE("vcompress_cineform", "$CINEFORM_COMPRESS_OBJ", "$CINEFORM_LIB")
        ADD_MODULE("vdecompress_cineform", "$CINEFORM_DECOMPRESS_OBJ", "$CINEFORM_LIB")
fi

if test $cineform_req = yes -a $cineform = no; then
        AC_MSG_ERROR([CINEFORM not found]);
fi

# ---------------------------------------------------------------------
# NAT-PMP
# -----------------------------------
natpmp=no
AC_ARG_ENABLE(natpmp,
              [  --disable-natpmp       disable NAT-PMP suppport (default is auto)]
              [            Requires: natpmp],
              [natpmp_req=$enableval],
              [natpmp_req=$build_default]
              )
AC_CHECK_HEADER(natpmp.h, FOUND_NATPMP_H=yes, FOUND_NATPMP_H=no)
AC_CHECK_LIB(natpmp, initnatpmp,  FOUND_NATPMP_L=yes, FOUND_NATPMP_L=no)
if test "$FOUND_NATPMP_H" = yes -a "$FOUND_NATPMP_L" -a $natpmp_req != no; then
        LIBS="-lnatpmp $LIBS"
        AC_DEFINE([HAVE_NATPMP], [1], [Build with NAT-PMP support])
        if test $system = Windows; then
                COMMON_FLAGS="$COMMON_FLAGS -DNATPMP_STATICLIB"
        fi
        natpmp=yes
fi

if test $natpmp_req = yes -a $natpmp = no; then
        AC_MSG_ERROR([NAT-PMP not found]);
fi

# ---------------------------------------------------------------------
# PCP
# -----------------------------------
pcp=no
AC_ARG_ENABLE(pcp,
              [  --disable-pcp       disable PCP suppport (default is auto)]
              [            Requires: libpcp],
              [pcp_req=$enableval],
              [pcp_req=$build_default]
              )
PKG_CHECK_MODULES([PCP], [libpcp-client], [FOUND_PCP=yes], [FOUND_PCP=no])
if test "$FOUND_PCP" = yes -a $pcp_req != no; then
        LIBS="$PCP_LIBS $LIBS"
        AC_DEFINE([HAVE_PCP], [1], [Build with PCP support])
        pcp=yes
fi

if test $pcp_req = yes -a $pcp = no; then
        AC_MSG_ERROR([PCP not found]);
fi

# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
AC_ARG_ENABLE(cppunit, AS_HELP_STRING([--disable-cppunit], [disable cppunit tests (default is auto)]),
	[cppunit_req=$enableval], [cppunit_req=$build_default])

PKG_CHECK_MODULES([CPPUNIT], [cppunit], [cppunit=yes], [cppunit=no])
if test "$cppunit" = yes; then
        AC_DEFINE([HAVE_CPPUNIT], [1], [Build with CppUnit tests])
        cppunit=yes
else
        CPPUNIT_LIBS=
        cppunit=no
fi
AC_SUBST(CPPUNIT_LIBS)

if test $cppunit_req = yes -a $cppunit = no; then
        AC_MSG_ERROR([cppunit not found]);
fi

# -------------------------------------------------------------------------------------------------
# We need to add libraries then
# -------------------------------------------------------------------------------------------------
TEST_LIBS="$LIB_MODULES $LIBS"
TEST_OBJS="$OBJS $LIB_OBJS $X_OBJ"

# this is only needed when passing to "clean" make target
if test "$build_libraries" != yes
then
        LIBS="$LIB_MODULES $LIBS"
        OBJS="$OBJS $LIB_OBJS $X_OBJ"
        HEADERS="$HEADERS $LIB_GENERATED_HEADERS"
        LIB_OBJS=
        LIB_GENERATED_HEADERS=
        MODULES=
	TARGETS=
fi

# remove duplicite objs
OBJS=`echo $OBJS | tr ' ' '\n' | sort | uniq | tr '\n' ' '`
TEST_OBJS=`echo $TEST_OBJS | tr ' ' '\n' | sort | uniq | tr '\n' ' '`

AC_SUBST(BIN_DEPS)
AC_SUBST(GENERATED_HEADERS)
AC_SUBST(LIB_GENERATED_HEADERS)
AC_SUBST(LIB_MODULES)
AC_SUBST(LIB_OBJS)
AC_SUBST(MODULES)
AC_SUBST(TARGETS)
AC_SUBST(TEST_LIBS)
AC_SUBST(TEST_OBJS)

# -------------------------------------------------------------------------------------------------
# Finally, substitute things into the Makefile and config.h

start_section() {
        printf "$1\n$2:"
        true # set $? to zero -> causes add_column to start on new line
}

end_section() {
        printf "$1
        "
}

# params: <result> <feature> <bool> 0|1
# 0 - print left column; 1 - print right column
add_column() {
        PAD_DOTS=31
        # left column
        if [ test $4 = 0 ]; then
                printf "$1\n  $2 $(for n in `seq ${#2} $PAD_DOTS`; do printf .; done) $(printf %-3s $3)"
                return 1
        else
                # right column
                printf "$1   $2 $(for n in `seq ${#2} $PAD_DOTS`; do printf .; done) $(printf %-3s $3)\n"
                return 0
        fi
}

RESULT="Target: $target\n"

# features
RESULT=`start_section "$RESULT" "Features"`
RESULT=`add_column "$RESULT" "CppUnit tests" $cppunit $?`
RESULT=`add_column "$RESULT" "CUDA support$HOST_CC_REPORT" $FOUND_CUDA $?`
RESULT=`add_column "$RESULT" "Debug output" $debug_output $?`
RESULT=`add_column "$RESULT" "iHDTV support" $ihdtv $?`
RESULT=`add_column "$RESULT" "IPv6 support" $ipv6 $?`
RESULT=`add_column "$RESULT" "Library live555" $livemedia $?`
RESULT=`add_column "$RESULT" "OpenCV$opencv_version" $opencv $?`
RESULT=`add_column "$RESULT" "OpenSSL-libcrypto" $crypto $?`
RESULT=`add_column "$RESULT" "Profiling support" $profile $?`
RESULT=`add_column "$RESULT" "Qt GUI" $qt_gui $?`
RESULT=`add_column "$RESULT" "RT priority" $use_rt $?`
RESULT=`add_column "$RESULT" "SpeexDSP" $speexdsp $?`
RESULT=`add_column "$RESULT" "Standalone modules" $build_libraries $?`
RESULT=`add_column "$RESULT" "zfec" $zfec $?`
RESULT=`end_section "$RESULT"`

# audio
RESULT=`start_section "$RESULT" "Audio"`
RESULT=`add_column "$RESULT" "ALSA" $alsa $?`
RESULT=`add_column "$RESULT" "CoreAudio" $coreaudio $?`
RESULT=`add_column "$RESULT" "JACK" $jack $?`
RESULT=`add_column "$RESULT" "JACK transport" $jack_trans $?`
RESULT=`add_column "$RESULT" "Portaudio" $portaudio $?`
RESULT=`add_column "$RESULT" "WASAPI" $wasapi $?`
RESULT=`end_section "$RESULT"`

# video
RESULT=`start_section "$RESULT" "Video"`
RESULT=`add_column "$RESULT" "AJA" $aja $?`
RESULT=`add_column "$RESULT" "AV Foundation" $avfoundation $?`
RESULT=`add_column "$RESULT" "BitFlow" $bitflow $?`
RESULT=`add_column "$RESULT" "Bluefish444" $bluefish444 $?`
RESULT=`add_column "$RESULT" "DeckLink" $decklink $?`
RESULT=`add_column "$RESULT" "DELTACAST" $deltacast $?`
RESULT=`add_column "$RESULT" "DirectShow" $dshow $?`
RESULT=`add_column "$RESULT" "DVS" $dvs $?`
RESULT=`add_column "$RESULT" "File input" $file $?`
RESULT=`add_column "$RESULT" "Gpustitch" $gpustitch $?`
RESULT=`add_column "$RESULT" "NDI" $ndi $?`
RESULT=`add_column "$RESULT" "OpenGL" $gl_display $?`
RESULT=`add_column "$RESULT" "OpenXR VR Display" $xrgl_disp $?`
RESULT=`add_column "$RESULT" "Panorama Gl Display" $panogl_disp $?`
RESULT=`add_column "$RESULT" "RTSP capture client" $rtsp $?`
RESULT=`add_column "$RESULT" "SAGE" $sage $?`
RESULT=`add_column "$RESULT" "Screen capture" $screen_cap $?`
RESULT=`add_column "$RESULT" "SDL (ver. $sdl_version)" $sdl $?`
RESULT=`add_column "$RESULT" "SW video mix" $swmix $?`
RESULT=`add_column "$RESULT" "V4L2" $v4l2 $?`
RESULT=`add_column "$RESULT" "XIMEA" $ximea $?`
RESULT=`end_section "$RESULT"`

# compressions
RESULT=`start_section "$RESULT" "Compressions"`
RESULT=`add_column "$RESULT" "Cineform" $cineform $?`
RESULT=`add_column "$RESULT" "Comprimato J2K" $cmpto_j2k $?`
RESULT=`add_column "$RESULT" "CUDA DXT" $cuda_dxt $?`
RESULT=`add_column "$RESULT" "GPUJPEG" $gpujpeg $?`
RESULT=`add_column "$RESULT" "GPUJPEG transcode to DXT" $gpujpeg_to_dxt $?`
RESULT=`add_column "$RESULT" "Lavc (VDP $lavc_hwacc_vdpau, VA $lavc_hwacc_vaapi, RPI4 $lavc_hwacc_rpi4)" $libavcodec $?`
RESULT=`add_column "$RESULT" "Realtime DXT" $rtdxt $?`
RESULT=`add_column "$RESULT" "UYVY dummy compression" $uyvy $?`
RESULT=`end_section "$RESULT"`

# other
RESULT=`start_section "$RESULT" "Others"`
RESULT=`add_column "$RESULT" "Blank capture filter" $blank $?`
RESULT=`add_column "$RESULT" "GPU accelerated LDGM" $ldgm_gpu $?`
RESULT=`add_column "$RESULT" "Hole punching" $libjuice $?`
RESULT=`add_column "$RESULT" "iHDTV support" $ihdtv $?`
RESULT=`add_column "$RESULT" "MCU-like video mixer" $video_mix $?`
RESULT=`add_column "$RESULT" "NAT-PMP traversal" $natpmp $?`
RESULT=`add_column "$RESULT" "PCP NAT traversal" $pcp $?`
RESULT=`add_column "$RESULT" "Resize capture filter" $resize $?`
RESULT=`add_column "$RESULT" "RTSP server" $rtsp_server $?`
RESULT=`add_column "$RESULT" "Scale postprocessor" $scale $?`
RESULT=`add_column "$RESULT" "SDP over HTTP" $sdp_http $?`
RESULT=`add_column "$RESULT" "Spout" $spout $?`
RESULT=`add_column "$RESULT" "Syphon" $syphon $?`
RESULT=`add_column "$RESULT" "Testcard extras" $testcard2 $?`
RESULT=`add_column "$RESULT" "Text postprocess and filter" $text $?`
RESULT=`add_column "$RESULT" "Swscale support" $libswscale $?`
RESULT=`end_section "$RESULT"`

AC_MSG_RESULT(
[
UltraGrid configuration summary
===============================])
AC_MSG_RESULT([$RESULT])

if test -n "$WARNINGS"; then
        printf "*** Following WARNINGS were issued:\n$WARNINGS\n"
fi

RES_STR=`printf "$RESULT" | sed 's/$/\\\\n/' | tr -d '\n'`

AC_DEFINE_UNQUOTED([AUTOCONF_RESULT], "$RES_STR", [Autoconf result])

AC_CONFIG_HEADERS([src/config.h])
AC_CONFIG_FILES([Makefile])
AC_OUTPUT

