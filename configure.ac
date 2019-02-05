dnl Process this file with autoconf to produce a configure script.

AC_INIT([UltraGrid], [1.5], [ultragrid-dev@cesnet.cz])
configure_flags="$*"
AC_CANONICAL_SYSTEM
AM_INIT_AUTOMAKE([1.10])
AC_PREREQ(2.61)
AC_CONFIG_SRCDIR([src/main.cpp])
AC_CONFIG_MACRO_DIR([m4])

PARENT=`echo $PWD | sed -e 's%/[[^/]]*$%%'`

AC_DEFINE_UNQUOTED([CONFIG_FLAGS], ["$configure_flags"], [Flags passed to configure])

AC_CANONICAL_HOST
AC_PROG_CC
AC_PROG_CC_C99
AC_PROG_CPP
AC_PROG_CXX
AC_PROG_INSTALL
AC_HEADER_STDC
AC_C_CONST
AC_TYPE_SIZE_T
AC_C_BIGENDIAN
AC_C_CHAR_UNSIGNED
AC_HEADER_STDBOOL

AH_BOTTOM([
#ifndef __cplusplus
#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#else
# ifndef HAVE__BOOL
#  ifdef __cplusplus
typedef bool _Bool;
#  else
#   define _Bool signed char
#  endif
# endif
# define bool _Bool
# define false 0
# define true 1
# define __bool_true_false_are_defined 1
#endif
#endif // ! defined __cplusplus
])

AC_CHECK_HEADERS(stropts.h sys/filio.h sys/wait.h)

# -------------------------------------------------------------------------------------------------
POST_COMPILE_MSG=""
CFLAGS="$CFLAGS -I$srcdir"
CPPFLAGS="$CPPFLAGS -I$srcdir"
CXXFLAGS="$CXXFLAGS -I$srcdir -std=gnu++11"
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
        git_branch=`git name-rev --name-only HEAD`
        if test "$git_branch"; then
                AC_DEFINE_UNQUOTED([GIT_BRANCH], "$git_branch", [Current GIT branch])
                AC_MSG_RESULT($git_branch);
        else
                AC_MSG_RESULT(no);
        fi
        AC_MSG_CHECKING([checking GIT revision ])
        git_rev=`git rev-parse --short HEAD`
        if test "$git_rev"; then
                AC_DEFINE_UNQUOTED([GIT_REV], "$git_rev", [Current GIT revision])
                AC_MSG_RESULT($git_rev);
        else
                AC_MSG_RESULT(no);
        fi
fi

AC_MSG_CHECKING([OS family])

if expr "$host_os" : ".*darwin.*" > /dev/null; then
        system=MacOSX
	AC_DEFINE([HAVE_MACOSX], [1], [This is Mac X OS])
else if expr "$host_os" : ".*mingw32.*" > /dev/null || expr "$host_os" : ".*msys.*" > /dev/null; then
        system=Windows
	AC_DEFINE([WIN32], [1], [This is an Windows OS])
	NET_LIBS="-lsetupapi -lWs2_32 -lIphlpapi -lOleAut32"
	LIBS="$LIBS $NET_LIBS"
else
        system=Linux
	LDFLAGS="$LDFLAGS -Wl,--dynamic-list-data"
	AC_DEFINE([HAVE_LINUX], [1], [This is Linux])
fi
fi

AC_SUBST(system)
AC_SUBST(NET_LIBS)

AC_MSG_RESULT($system);

AC_MSG_CHECKING([OS kernel version (major)])
os_version_major=`uname -r |cut -d . -f 1`
AC_MSG_RESULT($os_version_major);
AC_MSG_CHECKING([OS kernel version (minor)])
os_version_minor=`uname -r |cut -d . -f 2`
AC_MSG_RESULT($os_version_minor);

if test $system = MacOSX
then
        COMMON_OSX_FLAGS="-mmacosx-version-min=10.7 -stdlib=libc++"
	CFLAGS="$CFLAGS $COMMON_OSX_FLAGS"
	CPPFLAGS="$CPPFLAGS $COMMON_OSX_FLAGS"
	CXXFLAGS="$CXXFLAGS $COMMON_OSX_FLAGS"
        if test $WORD_LEN = 32; then
                CUDA_FLAGS="$CUDA_FLAGS -Xcompiler -Wno-error=unused-command-line-argument-hard-error-in-future"
        fi
	LDFLAGS="$LDFLAGS $COMMON_OSX_FLAGS -headerpad_max_install_names"
	X_CFLAGS="$X_CFLAGS -I/usr/X11R6/include"
        OBJS="$OBJS src/utils/autorelease_pool.o"
fi

AC_DEFINE_UNQUOTED([OS_VERSION_MAJOR], $os_version_major, [OS kernel major version])
AC_DEFINE_UNQUOTED([OS_VERSION_MINOR], $os_version_minor, [OS kernel minor version])

LINKER=$CXX

AC_SUBST(LINKER)
AC_SUBST(LDFLAGS)
AC_SUBST(CXXFLAGS)

AC_ARG_ENABLE(depends-version-check,
        AS_HELP_STRING([--disable-depends-version-check], [Disable libraries version check]),
	[version_check=$enableval],
        [version_check=yes]
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
        CU_OBJ_SUFFIX="cu.o"
else # Windows
        if test $WORD_LEN = 32; then
                CUDA_FLAGS="-m32"
        fi
        CU_OBJ_SUFFIX="cu.lib"
fi
AC_SUBST(CUDA_FLAGS)

if test "$system" = MacOSX; then
        GL_COMMON_OBJ="src/mac_gl_common.o src/gl_context.o"
else if test "$system" = Linux; then
        GL_COMMON_OBJ="src/glx_common.o src/gl_context.o src/x11_common.o"
else
        GL_COMMON_OBJ="src/win32_gl_common.o src/gl_context.o"
fi
fi

AC_ARG_VAR([ARCH], [Architecture to build for])
if test $target_cpu = x86_64 -o $target_cpu = i686
then
        test -z "$ARCH" && ARCH="-msse4"
fi
CFLAGS="$CFLAGS $ARCH"
CXXFLAGS="$CXXFLAGS $ARCH"

AC_CHECK_HEADERS([termios.h])

AH_BOTTOM([
/*
 * Mac OS X Snow Leopard does not have posix_memalign
 * so supplying a fake one (Macs allocate always aligned pointers
 * to a 16 byte boundry.
 */
#if defined HAVE_MACOSX && OS_VERSION_MAJOR <= 9
#include <errno.h>
#include <stdlib.h>
#ifndef POSIX_MEMALIGN
#define POSIX_MEMALIGN
static inline int posix_memalign(void **memptr, size_t alignment, size_t size);

static inline int posix_memalign(void **memptr, size_t alignment, size_t size)
{
       if(!alignment || (alignment & (alignment - 1)) || alignment > 16) return EINVAL;
       *memptr=malloc(size);
       if(!*memptr) return ENOMEM;
       return 0;
}
#endif // POSIX_MEMALIGN
#endif // defined HAVE_MACOSX && OS_VERSION_MAJOR <= 9

#undef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#undef MAX
#define MAX(a,b) (((a)>(b))?(a):(b))
])

# @param $1 name of the module
function MODULE_NAME() {
        echo "lib/ultragrid/ultragrid_"$1".so"
}

# Adds the module to UltraGrid build system.
# @param $1 name of the module
# @param $2 object files
# @param $3 libraries
# @param $4 static libraries (?)
AC_DEFUN([ADD_MODULE], [
          TARGET=$(MODULE_NAME "$1")
          LIB_OBJS=$LIB_OBJS" "$2
          LIB_MODULES=$LIB_MODULES" "$3
	  AM_CONDITIONAL([IS_TARGET], [test x$1 != x])
	  AM_COND_IF([IS_TARGET],
		                [
				 MODULES="$MODULES $TARGET"
				 TARGETS="$TARGETS
$TARGET: "$2" $4
	\$(LINKER) \$(LDFLAGS) -shared -Wl,-soname,ultragrid_"$1.so" "$2" -o lib/ultragrid/ultragrid_"$1".so "$3" $4
"
				 ])
          ])

# -------------------------------------------------------------------------------------------------
# Additional macros
# -------------------------------------------------------------------------------------------------
AC_DEFUN([CUDA_MESSAGE], [
          if test -z "$cuda_msg_defined"; then
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
                          DLL_LIBS="$DLL_LIBS src/cuda_wrapper.cu.dll"
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

AC_CHECK_FUNCS(strtok_r)

AC_CHECK_FUNCS(drand48)
if test $ac_cv_func_drand48 = no
then
	AC_DEFINE([NEED_DRAND48], 1, [We need custom implementation of drand48.])
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
                AC_MSG_WARN([*** Building libraries is not supported with other system than Linux])
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
                DEBUG_FLAGS="-Werror -Wno-error=cast-qual -Wno-error=cast-align -Wno-error=strict-aliasing"
                CFLAGS="$CFLAGS $DEBUG_FLAGS"
                CXXFLAGS="$CXXFLAGS $DEBUG_FLAGS"
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

# ------------------------------------------------------------------------------------------------
# environment variables
#
AC_ARG_VAR([CUDA_PATH], [Directory of your Nvidia toolkit instalation.])
AC_ARG_VAR([SAGE_DIRECTORY], [Directory of your SAGE installation.])
AC_ARG_VAR([DVS_DIRECTORY], [Path to DVS installation.])
AC_ARG_VAR([DELTACAST_DIRECTORY], [Placement of VideoMasterHD directory (Deltacast).])
AC_ARG_VAR([BLUE_LATEST_SDK], [Placement of Bluefish444 SDK.])
AC_ARG_VAR([NTV2_ROOT], [Directory to AJA NTV2 SDK (ends with ntv2projects).])
AC_ARG_VAR([AJA_DIRECTORY], [Directory to AJA NTV2 SDK.])
AC_ARG_VAR([CINEFORM_DIRECTORY], [Directory to Cineform SDK.])

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
                if test $x11_cv_broken_headers = yes -a "$GCC" = "yes" ; then
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
                X_OBJ="src/x11_common.o"
                LIBS="$LIBS -ldl"
                CFLAGS="$CFLAGS"
                AC_DEFINE([HAVE_X], [1], [Build with X11 support])
        fi
        ;;
esac

AC_SUBST(X_CFLAGS)
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
dvs_req=auto
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
decklink_req=auto

AC_ARG_ENABLE(decklink,
[  --disable-decklink      disable DeckLink (default is auto)]
[                          Requires: DesktopVideo],
	[decklink_req=$enableval])
DECKLINK_INC=

case "$system" in
	MacOSX)
                if test -z $DECKLINK_INC; then
                        DECKLINK_INC=$srcdir/DeckLink/Mac
                fi
                OLD_LIBS=$LIBS
                LIBS="$LIBS -framework CoreFoundation"
                AC_LANG_PUSH(C++)
		AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include "DeckLink/Mac/DeckLinkAPI.h"
                                   ]],
		[[IDeckLinkIterator *deckLinkIter = CreateDeckLinkIteratorInstance();]])],FOUND_DECKLINK=yes,FOUND_DECKLINK=no)
                LIBS=$OLD_LIBS
                if test $FOUND_DECKLINK = yes; then
                        DECKLINK_LIB="$DECKLINK_LIB -framework CoreFoundation"
                fi
                AC_LANG_POP(C++)
                ;;
	Windows)
                if test -z $DECKLINK_INC; then
                        DECKLINK_INC=$srcdir/DeckLink/Windows
                fi
                SAVED_LIBS=$LIBS
                LIBS="$LIBS -lOle32"
                AC_LANG_PUSH(C++)
		AC_MSG_CHECKING([DeckLink usability])
		AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
#include "DeckLink/Windows/DeckLinkAPI_h.h"
#include "DeckLink/Windows/DeckLinkAPIVersion.h"
#include <objbase.h>
                                   ]],
		[[IDeckLinkIterator *deckLinkIter; CoCreateInstance(CLSID_CDeckLinkIterator, NULL, CLSCTX_ALL, IID_IDeckLinkIterator, (void **) deckLinkIter);]])],FOUND_DECKLINK=yes,FOUND_DECKLINK=no)
                LIBS=$SAVED_LIBS
                if test $FOUND_DECKLINK = yes; then
                        DECKLINK_LIB="$DECKLINK_LIB -lOle32"
                fi
		AC_MSG_RESULT([$FOUND_DECKLINK])
                AC_LANG_POP(C++)
                ;;
        *)
                if test -z $DECKLINK_INC; then
                        DECKLINK_INC=$srcdir/DeckLink/Linux
                fi

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
# Bluefish444 stuff
# -------------------------------------------------------------------------------------------------
BLUEFISH444_PATH=
BLUEFISH444_CAP_OBJ=
BLUEFISH444_DISP_OBJ=
BLUEFISH444_INC=
BLUEFISH444_LIB=
bluefish444_req=auto
bluefish444=no
blue_audio_req=auto
blue_audio=no

AC_ARG_ENABLE(bluefish444,
[  --disable-bluefish444   disable Bluefish444 suppport (default is auto)]
[                          Requires: bluefish444_SDK],
              [bluefish444_req=$enableval],
              [bluefish444_req=auto]
              )
AC_ARG_ENABLE(blue-audio,
              AS_HELP_STRING([--disable-blue-audio], [disable Bluefish444 audio suppport (default is auto)]),
              [blue_audio_req=$enableval],
              [blue_audio_req=auto]
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
                BLUEFISH444_INC="-I$BLUEFISH444_PATH/release/driver/inc"
        else
                BLUEFISH444_INC="-I$BLUEFISH444_PATH/Inc -IBlueVelvetC/BlueVelvetC"
        fi
        
        CXXFLAGS="$CXXFLAGS $BLUEFISH444_INC "
        CPPFLAGS="$CPPFLAGS $BLUEFISH444_INC"
        if test $system = Windows; then
                BLUEFISH444_LIB="-L$BLUEFISH444_PATH/Lib -LBlueVelvetC/Release $BLUEFISH444_LIB"
        fi
	LIBS="$LIBS $BLUEFISH444_LIB"
fi

AC_LANG_PUSH(C++)

BLUE_L=no
BLUE_H=no

if test $system = Windows
then
        AC_CHECK_HEADERS([BlueVelvetC_UltraGrid.h],
                        [BLUE_H=yes], [BLUE_H=no],
                        [[
#include <objbase.h>
                        ]]
                        )
        if test $WORD_LEN -eq 64; then
                BLUE_AUDIO_LIB="BlueHancUtils64"
                BLUEFISH444_LIB_NAME=BlueVelvetC64
                BLUE_AUX=BlueVelvet64
        else
                BLUE_AUDIO_LIB="BlueHancUtils"
                BLUEFISH444_LIB_NAME=BlueVelvetC_UltraGrid
                BLUE_AUX=BlueVelvet3
        fi
        
        AC_CHECK_LIB($BLUEFISH444_LIB_NAME, bfcFactory, [BLUE_L=yes],
                        [BLUE_L=no])
        BLUEFISH444_LIB="$BLUEFISH444_LIB -l$BLUE_AUX -l$BLUEFISH444_LIB_NAME"
else
if test $system = Linux
then
        AC_CHECK_HEADERS([BlueVelvet.h], [BLUE_H=yes], [BLUE_H=no])
        if test $WORD_LEN -eq 64; then
                AC_CHECK_LIB(BlueVelvet64, BlueVelvetFactory, [BLUE_L=yes], [BLUE_L=no])
                BLUE_AUDIO_LIB="BlueANCUtils64"
                BLUEFISH444_LIB="$BLUEFISH444_LIB -lBlueVelvet64"
        else
                AC_CHECK_LIB(BlueVelvet, BlueVelvetFactory, [BLUE_L=yes], [BLUE_L=no])
                BLUE_AUDIO_LIB="BlueANCUtils"
                BLUEFISH444_LIB="$BLUEFISH444_LIB -lBlueVelvet"
        fi
fi
fi

if test $system = Windows -o $system = Linux;then
        AC_CHECK_LIB($BLUE_AUDIO_LIB, hanc_decoder_ex, [HAVE_BLUE_AUDIO_L=yes],
                     [HAVE_BLUE_AUDIO_L=no])
        if test $HAVE_BLUE_AUDIO_L = yes -a $blue_audio_req != no; then
                BLUEFISH444_LIB="$BLUEFISH444_LIB -l$BLUE_AUDIO_LIB"
                blue_audio=yes
        fi
fi

if test $BLUE_H = yes -a $BLUE_L = yes -a $bluefish444_req != no
then
        BLUEFISH444_CAP_OBJ="src/video_capture/bluefish444.o"
        BLUEFISH444_DISP_OBJ="src/video_display/bluefish444.o"
        AC_DEFINE([HAVE_BLUEFISH444], [1], [Build with Bluefish444 support])
        ADD_MODULE("vidcap_bluefish444", "$BLUEFISH444_CAP_OBJ", "$BLUEFISH444_LIB")
        ADD_MODULE("display_bluefish444", "$BLUEFISH444_DISP_OBJ", "$BLUEFISH444_LIB")
        if test x$blue_audio = xyes; then
                AC_DEFINE([HAVE_BLUE_AUDIO], [1], [Build with Bluefish444 audio support])
        fi
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
[aja_req=auto])

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
                        NTV2_13=yes
                fi
        fi
else
if test $system = Windows
then
        if test "$aja_req" = yes
        then
                ADD_MODULE("aja", "src/aja_win32_stub.o", "-laja")
                aja=yes
        fi
else # MacOSX
        AC_CHECK_LIB(ajantv2, main, [FOUND_LIBAJA=yes], [FOUND_LIBAJA=no])
        if test "$FOUND_LIBAJA" = yes; then
                AC_CHECK_FILE(${AJA_PATH}/ajalibraries/ajantv2/includes/ajatypes.h, FOUND_LIBAJA=yes, FOUND_LIBAJA=no)
        fi
fi
fi

# this is used only for Mac and Linux
if test "$aja_req" != no -a "$FOUND_LIBAJA" = yes
then
        aja=yes
        if test $system = Linux -a "$NTV2_13" != yes
	then
                AJA_INC="$AJA_INC -I${AJA_PATH}/ntv2projects/classes -I${AJA_PATH}/ajaapi -I${AJA_PATH}/ntv2projects/includes -I${AJA_PATH}/ntv2projects/democlasses"
                AJA_LIB="${AJA_PATH}/lib/libaja.a ${AJA_PATH}/lib/libajastuff.a"
        else # ntv2 13.x
                AJA_INC="$AJA_INC -I${AJA_PATH}/ajalibraries/ajantv2/includes -I${AJA_PATH}/ajalibraries/ajantv2/src/mac -I${AJA_PATH}/ajaapps/crossplatform/demoapps -I${AJA_PATH}/ajalibraries -I${AJA_PATH}/ajaapps/crossplatform/demoapps/ntv2capture"
        fi
        if test $system = Linux
        then
                AJA_CXXFLAGS="-DAJALinux -DAJA_LINUX"
                if test "$NTV2_13" = yes
                then
                        AJA_INC="$AJA_INC -I${AJA_PATH}/ajalibraries/ajantv2/src/lin"
                        AJA_LIB="${AJA_PATH}/lib/libajantv2.a"
                fi
        else # mac
                AJA_CXXFLAGS="-DAJAMac -DAJA_MAC"
                AJA_LIB="-lajantv2"
        fi
        AJA_CAP_OBJ="src/video_capture/aja.o"
        AJA_DISP_OBJ="src/video_display/aja.o"
        AC_DEFINE([HAVE_AJA], [1], [Build with AJA support])

        ADD_MODULE("aja", "$AJA_CAP_OBJ $AJA_DISP_OBJ", "$AJA_LIB")
        INC="$INC $AJA_INC"
        CXXFLAGS="$CXXFLAGS $AJA_CXXFLAGS"
fi

if test $aja_req = yes -a $aja = no; then
        AC_MSG_ERROR([AJA not found]);
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
              [deltacast_req=auto]
              )
DELTACAST_PATH=$DELTACAST_DIRECTORY

AC_ARG_WITH(deltacast,
            [  --with-deltacast=DIR    specify location of DELTACAST],
            [DELTACAST_PATH=$withval
             ])

if test $system = Linux -o $system = Windows
then
        AC_LANG_PUSH(C++)

	if test $system = Windows; then
		HEADER_SUBDIR="Header"
	else
		HEADER_SUBDIR="Include"
	fi

        if test -n "$DELTACAST_PATH"
        then
                CXXFLAGS_SAVE=$CXXFLAGS
                CXXFLAGS="$CXXFLAGS -I$DELTACAST_PATH/$HEADER_SUBDIR"
                CPPFLAGS_SAVE=$CPPFLAGS
                CPPFLAGS="$CPPFLAGS -I$DELTACAST_PATH/$HEADER_SUBDIR"
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
	fi
fi

if test $deltacast_req != no -a $deltacast_found = yes
then
	if test -n "$DELTACAST_PATH"
	then
		DELTACAST_INC="-I$DELTACAST_PATH/$HEADER_SUBDIR"
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

# -------------------------------------------------------------------------------------------------
# QuickTime Stuff
# -------------------------------------------------------------------------------------------------
QUICKTIME_INC=
QUICKTIME_LIB=
QUICKTIME_OBJ=
quicktime=no

AC_ARG_ENABLE(quicktime,
      AS_HELP_STRING([--disable-quicktime], [enable QuickTime support(default is disable)]),
    [quicktime_req=$enableval],
    [quicktime_req=no]
    )

if test $WORD_LEN = 64 -a $quicktime_req = yes; then
        AC_MSG_ERROR([QuickTime not supported in 64-bit builds, please build 32-bit UltraGrid (CFLAGS/CXXFLAGS/LDFLAGS/CUDA_FLAGS=-m32)]);
fi

case $host_os in
	*darwin*)
		AC_CHECK_HEADERS([QuickTime/QuickTime.h], [FOUND_QUICKTIME_H=yes], [echo "QuickTime headers not found."])
		AC_CHECK_HEADERS([Carbon/Carbon.h], [FOUND_CARBON_H=yes], [echo "Carbon headers not found."])

		if test $quicktime_req != no -a $FOUND_QUICKTIME_H = yes -a $FOUND_CARBON_H = yes -a $WORD_LEN = 32
		then
                        AC_DEFINE([HAVE_QUICKTIME], [1],  [Build with QuickTime support])
			QUICKTIME_INC=
			QUICKTIME_LIB="-framework QuickTime -framework Carbon -framework AppKit -framework CoreAudio -framework AudioUnit"
			QUICKTIME_OBJ="src/video_capture/quicktime.o src/video_display/quicktime.o"
                        ADD_MODULE("", "$QUICKTIME_OBJ", "$QUICKTIME_LIB")
			INC="$INC $QUICKTIME_INC"
                        quicktime=yes
		fi
    ;;
esac

if test $quicktime_req = yes -a $quicktime = no; then
        AC_MSG_ERROR([QuickTime not found]);
fi

# -------------------------------------------------------------------------------------------------
# AV Foundation capture
# -------------------------------------------------------------------------------------------------
avfoundation=no

if test $system = MacOSX
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
              [],
              AC_MSG_ERROR([Use either "--enable-sdl1" or "--enable-sdl2" option.])
              )

AC_ARG_ENABLE(sdl1,
[  --enable-sdl1           enable SDL1 (default is auto)]
[                          Requires: sdl >= 1.2.14 ],
    [sdl1_req=$enableval],
    [sdl1_req=auto]
    )

AC_ARG_ENABLE(sdl2,
[  --enable-sdl2           enable SDL2 (disabled if SDL1 is enabled, auto otherwise)]
[                          Requires: sdl2 ],
    [sdl2_req=$enableval],
    [sdl2_req=auto]
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
        AC_MSG_ERROR([SDL not found]);
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
        if test $system = Windows; then
                SDL2_LIBS="-lSDL2"
        fi
        ADD_MODULE("display_sdl2", "$SDL2_OBJ", "$SDL2_LIBS")
        SDL_LIB=$SDL2_LIB
        sdl_version=2
        sdl=yes
else
if test $sdl1_req != no -a $found_sdl1 = yes
then
        SDL1_OBJ="src/video_display/sdl.o"
        AC_DEFINE([HAVE_SDL], [1],  [Build with SDL support])
        ADD_MODULE("display_sdl", "$SDL1_OBJ", "$SDL1_LIBS")
        SDL_LIB=$SDL1_LIB
        sdl_version=1
        sdl=yes
fi
fi

# -------------------------------------------------------------------------------------------------
# OpenCV version detection
# -------------------------------------------------------------------------------------------------

AC_ARG_ENABLE(opencv,
              AS_HELP_STRING([--disable-opencv], [disable all OpenCV code]),
              [opencv_req=$enableval],
              [opencv_req=auto])

PKG_CHECK_MODULES([OPENCV_3], [opencv > 3.0], [
		AC_DEFINE([HAVE_OPENCV_3], [1], [Use OpenCV version 3])
		OPENCV_MAJOR=3
		OPENCV_LIBS=$OPENCV_3_LIBS
		OPENCV_CFLAGS=$OPENCV_3_CFLAGS
	], [
		PKG_CHECK_MODULES([OPENCV_2], [opencv > 2.3], [
			AC_DEFINE([HAVE_OPENCV_2], [1], [Use OpenCV version 2])
			OPENCV_MAJOR=2
			OPENCV_LIBS=$OPENCV_2_LIBS
			OPENCV_CFLAGS=$OPENCV_2_CFLAGS
			], [
				PKG_CHECK_MODULES([OPENCV_4], [opencv4], [
					AC_DEFINE([HAVE_OPENCV_4], [1], [Use OpenCV version 4])
					OPENCV_MAJOR=4
					OPENCV_LIBS=$OPENCV_4_LIBS
					OPENCV_CFLAGS=$OPENCV_4_CFLAGS
					], [OPENCV_MAJOR=]
					)
			]
		)
	]
)

if test -n "$OPENCV_MAJOR" -a $opencv_req != "no"; then
        HAVE_OPENCV=yes
        opencv="yes (version: $OPENCV_MAJOR)"
else
        HAVE_OPENCV=no
        opencv="no"
fi

# -------------------------------------------------------------------------------------------------
# RTSP capture stuff
# -------------------------------------------------------------------------------------------------
RTSP_INC=
RTSP_LIBS=
RTSP_OBJ=
rtsp=no

define(rtsp_deps, [glib-2.0 >= 2.24.1 libcurl])

AC_ARG_ENABLE(rtsp,
[  --disable-rtsp          disable RTSP support (default is auto)]
[                          Requires: rtsp_deps],
    [rtsp_req=$enableval],
    [rtsp_req=auto]
    )

PKG_CHECK_MODULES([RTSP], [rtsp_deps], FOUND_RTSP_DEPS=yes, FOUND_RTSP_DEPS=no)

if test $rtsp_req != no -a $FOUND_RTSP_DEPS = yes
then
        AC_DEFINE([HAVE_RTSP], [1], [RTSP capture build with curl ang glib support])
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
        AC_MSG_ERROR([rtsp not found, check curl or glib dependencies...]);
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
[  --enable-rtsp-server    enables RTSP server support (default is disable)]
[                          Requires: live555],
    [rtsp_server_req=$enableval],
    [rtsp_server_req=no]
    )

AC_ARG_WITH(live555,
            [  --with-live555=DIR      specify location of live555],
            [livemedia_prefix=$withval
            ])


AC_CHECK_FILE($livemedia_prefix/include/liveMedia/liveMedia.hh, [FOUND_LIVE_H=yes], [echo "liveMedia (live555 library) headers not found; may apt-get install liblivemedia-dev"; FOUND_LIVE_H=no])

if test $rtsp_server_req != no -a $FOUND_LIVE_H = yes
then
        AC_DEFINE([HAVE_RTSP_SERVER], [1], [RTSP server build with live555 support])
        RTSP_SERVER_FLAGS="-I$livemedia_prefix/include/groupsock -I$livemedia_prefix/include/liveMedia -I$livemedia_prefix/include/BasicUsageEnvironment -I$livemedia_prefix/include/UsageEnvironment"
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
sdp_http=no
AC_ARG_ENABLE(sdp-http,
[  --enable-sdp-http       enables SDP/HTTP transport (default is enabled)],
    [sdp_http_req=$enableval],
    [sdp_http_req=yes]
    )

if test $sdp_http_req != no
then
        AC_DEFINE([SDP_HTTP], 1, [Add support for SDP over HTTP])
        sdp_http=yes
fi

if test $sdp_http_req = yes -a $sdp_http = no; then
        AC_MSG_ERROR([SDP over HTTP is currently not supported under MSW]);
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
    [resize_req=auto]
    )

if test $resize_req != no -a $HAVE_OPENCV = yes
then
        CFLAGS="$CFLAGS ${OPENCV_CFLAGS}"
        CXXFLAGS="$CXXFLAGS ${OPENCV_CFLAGS}"
        RESIZE_OBJ="src/capture_filter/resize.o src/capture_filter/resize_utils.o"
        ADD_MODULE("vcapfilter_resize", "$RESIZE_OBJ", "$OPENCV_LIBS")
        resize=yes
fi

if test $resize_req = yes -a $resize = no; then
        AC_MSG_ERROR([OpenCV not found, may install OpenCV...?]);
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
    [blank_req=auto]
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
      AS_HELP_STRING([--enable-testcard-extras], [enable testcard addons - audio and testcard2 (default is disabled)]),
    [testcard_extras_req=$enableval],
    [testcard_extras_req=no]
    )

TESTCARD_OBJ="src/video_capture/testcard.o"
TESTCARD_LIB=
TESTCARD2_OBJ=
TESTCARD2_LIB=

TESTCARD_COMMON="src/video_capture/testcard_common.o"

if test $testcard_extras_req != no
then
        if test $sdl_version -eq 1; then
                PKG_CHECK_MODULES([SDL_MIXER], [SDL_mixer], [FOUND_SDL_MIXER=yes], [FOUND_SDL_MIXER=no])
        else
                PKG_CHECK_MODULES([SDL_MIXER], [SDL2_mixer], [FOUND_SDL_MIXER=yes], [FOUND_SDL_MIXER=no])
        fi

        if test "$FOUND_SDL_MIXER" = yes
        then
                TESTCARD_LIB="$SDL_MIXER_LIBS"
        fi

        TESTCARD2_OBJ=
        if test $sdl = yes
        then
                TESTCARD2_OBJ="$TESTCARD_COMMON src/video_capture/testcard2.o"
                TESTCARD2_LIB="$TESTCARD2_LIB $SDL_LIB $X11_LIB"
                if test $sdl_version -eq 1; then
                        PKG_CHECK_MODULES([SDL_TTF], [SDL_ttf], [FOUND_SDL_TTF=yes], [FOUND_SDL_TTF=no])
                else
                        PKG_CHECK_MODULES([SDL_TTF], [SDL2_ttf], [FOUND_SDL_TTF=yes], [FOUND_SDL_TTF=no])
                fi
                if test "$FOUND_SDL_TTF" = yes
                then
                        TESTCARD2_LIB="$TESTCARD2_LIB $SDL_TTF_LIBS"
                        AC_DEFINE([HAVE_LIBSDL_TTF], [1], [Build testcard2 with TTF support])
                fi
                AC_DEFINE([HAVE_TESTCARD2], [1], [Build with testcard2 capture])
                ADD_MODULE("vidcap_testcard2", "$TESTCARD2_OBJ", "$TESTCARD2_LIB")
        fi
        testcard_extras=yes
else
        testcard_extras=no
fi

TESTCARD_OBJ="$TESTCARD_COMMON src/video_capture/testcard.o"
ADD_MODULE("vidcap_testcard", "$TESTCARD_OBJ", "$TESTCARD_LIB")

if test $testcard_extras_req = yes -a $testcard_extras = no; then
        AC_MSG_ERROR([Testcard2 not found]);
fi


# -------------------------------------------------------------------------------------------------
# libavcodec hw-accelerated decoding support
# -------------------------------------------------------------------------------------------------
lavc_hwacc_common=no
lavc_hwacc_vdpau=no
lavc_hwacc_vaapi=no

define(lavc_hwacc_common_dep, libavutil >= 55.22.1)
define(lavc_hwacc_vdpau_dep, lavc_hwacc_common_dep vdpau)
define(lavc_hwacc_vaapi_dep, lavc_hwacc_common_dep libva)

AC_ARG_ENABLE(lavc-hw-accel-vdpau,
[  --disable-lavc-hw-accel-vdpau   disable lavc-hw-accel-vdpau (default is auto)]
[                          Requires: lavc_hwacc_vdpau_dep],
    [lavc_hwacc_vdpau_req=$enableval],
    [lavc_hwacc_vdpau_req=auto]
    )

AC_ARG_ENABLE(lavc-hw-accel-vaapi,
[  --disable-lavc-hw-accel-vaapi   disable lavc-hw-accel-vaapi (default is auto)]
[                          Requires: lavc_hwacc_vaapi_dep],
    [lavc_hwacc_vaapi_req=$enableval],
    [lavc_hwacc_vaapi_req=auto]
    )

PKG_CHECK_MODULES([LAVC_HWACC_COMMON], [lavc_hwacc_common_dep], [FOUND_HWACC_COMMON_DEP=yes], [FOUND_HWACC_COMMON_DEP=no])
PKG_CHECK_MODULES([LAVC_HWACC_VDPAU], [lavc_hwacc_vdpau_dep], [FOUND_HWACC_VDPAU_DEP=yes], [FOUND_HWACC_VDPAU_DEP=no])
PKG_CHECK_MODULES([LAVC_HWACC_VAAPI], [lavc_hwacc_vaapi_dep], [FOUND_HWACC_VAAPI_DEP=yes], [FOUND_HWACC_VAAPI_DEP=no])

LAVC_HWACC_FLAGS=
LAVC_HWACC_LIBS=
HW_ACC_OBJ=
HW_ACC_DISPLAY_OBJ=

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

if test $lavc_hwacc_common = yes
then
        LAVC_HWACC_FLAGS="${LAVC_HWACC_FLAGS} -DHWACC_COMMON"
        HW_ACC_OBJ="${HW_ACC_OBJ} src/hwaccel_libav_common.o"
        HW_ACC_DISPLAY_OBJ="${HW_ACC_DISPLAY_OBJ} src/hwaccel_libav_common.o"
fi

if test $lavc_hwacc_vdpau = yes
then
        LAVC_HWACC_FLAGS="${LAVC_HWACC_FLAGS} -DHWACC_VDPAU ${LAVC_HWACC_VDPAU_CFLAGS}"
        LAVC_HWACC_LIBS="${LAVC_HWACC_LIBS} ${LAVC_HWACC_VDPAU_LIBS}"
        HW_ACC_OBJ="${HW_ACC_OBJ} src/hwaccel_vdpau.o"
        HW_ACC_DISPLAY_OBJ="${HW_ACC_DISPLAY_OBJ} src/hwaccel_vdpau.o"
fi

if test $lavc_hwacc_vaapi = yes
then
        LAVC_HWACC_FLAGS="${LAVC_HWACC_FLAGS} -DHWACC_VAAPI ${LAVC_HWACC_VAAPI_CFLAGS}"
        LAVC_HWACC_LIBS="${LAVC_HWACC_LIBS} ${LAVC_HWACC_VAAPI_LIBS}"
        HW_ACC_OBJ="${HW_ACC_OBJ} src/hwaccel_vaapi.o"
        HW_ACC_DISPLAY_OBJ="${HW_ACC_DISPLAY_OBJ} src/hwaccel_vaapi.o"
fi

COMMON_FLAGS="$COMMON_FLAGS $LAVC_HWACC_FLAGS"

if test -n "$HW_ACC_OBJ"
then
        ADD_MODULE("hw_accel", "$HW_ACC_OBJ", "$LAVC_HWACC_LIBS")
fi

if test $lavc_hwacc_vdpau_req = yes -a $lavc_hwacc_vdpau = no; then
	AC_MSG_ERROR([Could not find hwacc vdpau dependencies!]);
fi

if test $lavc_hwacc_vaapi_req = yes -a $lavc_hwacc_vaapi = no; then
	AC_MSG_ERROR([Could not find hwacc vaapi dependencies!]);
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
[libavcodec_req=auto]
)

if test $system = Windows; then
AC_CHECK_HEADERS([libavcodec/avcodec.h libavutil/imgutils.h libavutil/opt.h])
AC_CHECK_LIB(avcodec, avcodec_open2)
AC_CHECK_LIB(avutil, av_free)
if test $ac_cv_header_libavcodec_avcodec_h = yes -a \
        $ac_cv_header_libavutil_imgutils_h = yes -a \
        $ac_cv_header_libavutil_opt_h = yes -a \
        $ac_cv_lib_avcodec_avcodec_open2 = yes -a \
        $ac_cv_lib_avutil_av_free = yes
then
        found_libavcodec=yes
        found_libavutil=yes
        libavcodec_ge_55=yes # TODO
        LIBAVCODEC_LIBS="-lavcodec"
        LIBAVUTIL_LIBS="-lavutil"
else
        found_libavcodec=no
        found_libavutil=no
fi
else # Linux, OS X
PKG_CHECK_MODULES([LIBAVCODEC], [libavcodec >= 53.35.0 ], [found_libavcodec=yes], [found_libavcodec=no])
PKG_CHECK_MODULES([LIBAVUTIL], [libavutil >= 51.22.1 ], [found_libavutil=yes], [found_libavutil=no])
fi

if test $libavcodec_req != no -a $found_libavcodec = yes -a $found_libavutil = yes
then
        LIBAVCODEC_OBJ="src/video_compress/libavcodec.o"
        AC_DEFINE([HAVE_LAVC], [1], [Build with LAVC support])
        AC_DEFINE([HAVE_LAVC_AUDIO], [1], [Build with LAVC audio support])
        LIBAVCODEC_AUDIO_CODEC_OBJ=src/audio/codec/libavcodec.o

        LIBAVCODEC_COMPRESS_OBJ=src/video_compress/libavcodec.o
        LIBAVCODEC_DECOMPRESS_OBJ=src/video_decompress/libavcodec.o
        COMMON_FLAGS="$COMMON_FLAGS $LIBAVCODEC_CFLAGS $LIBAVUTIL_CFLAGS"
        libavcodec=yes
        LIBAVCODEC_LIBS="$LIBAVCODEC_LIBS $LIBAVUTIL_LIBS" # added libavutil explicitly
        ADD_MODULE("vcompress_libavcodec", "$LIBAVCODEC_COMPRESS_OBJ", "$LIBAVCODEC_LIBS")
        ADD_MODULE("vdecompress_libavcodec", "$LIBAVCODEC_DECOMPRESS_OBJ", "$LIBAVCODEC_LIBS")
        ADD_MODULE("acompress_libavcodec", "$LIBAVCODEC_AUDIO_CODEC_OBJ", "$LIBAVCODEC_LIBS")
fi

if test $libavcodec_req = yes -a $libavcodec = no; then
        AC_MSG_ERROR([Libavcodec not found]);
fi

# -------------------------------------------------------------------------------------------------
# OpenGL stuff
# -------------------------------------------------------------------------------------------------
GL_INC=
GL_OBJ=$GL_COMMON_OBJ
GL_LIB=
gl=no

AC_ARG_ENABLE(gl,
[  --disable-gl            disable OpenGL (default is auto]
[                          Requires:  glew GLUT gl],
    [gl_req=$enableval],
    [gl_req=auto]
    )

# this variable is shared multiple times in code, so make sure that change in this
# lines does not inferre with something below
#SAVED_LIBS=$LIBS
#if test $system = Windows; then
#	AC_MSG_CHECKING([GLEW presence (library and headers)])
#	SAVED_LIBS=$LIBS
#	LIBS="$LIBS -lglew32"
#	AC_LINK_IFELSE([ AC_LANG_PROGRAM([
##define _STDCALL_SUPPORTED 1
##include <GL/glew.h>
#], [
#glewInit();
#		])],
#		[FOUND_GLEW=yes],[FOUND_GLEW=no])
#	LIBS=$SAVED_LIBS
#	AC_MSG_RESULT([$FOUND_GLEW])
#        #AC_CHECK_HEADERS([GL/glew.h], AC_CHECK_LIB(GLEW, glewInit, FOUND_GLEW=yes, FOUND_GLEW=no), FOUND_GLEW=no)
#else
#        AC_CHECK_HEADERS([GL/glew.h], AC_CHECK_LIB(GLEW, glewInit, FOUND_GLEW=yes, FOUND_GLEW=no), FOUND_GLEW=no)
#        #PKG_CHECK_MODULES([GLEW], [glew >= 1.5.2 ], [FOUND_GLEW=yes], [FOUND_GLEW=no])
#fi
#LIBS=$SAVED_LIBS
#AC_SUBST(FOUND_GLEW)
#
case "$system" in
	MacOSX)
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

		if test $gl_req != no -a $FOUND_GL_L = yes
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

			AC_CHECK_HEADERS([GL/glew.h], AC_CHECK_LIB(GLEW, glewInit, FOUND_GLEW=yes, FOUND_GLEW=no), FOUND_GLEW=no)

			AC_CHECK_HEADERS([GLUT/glut.h],
					 [AC_CHECK_FILE(/System/Library/Frameworks/GLUT.framework/GLUT, FOUND_GLUT=yes, echo "GLUT framework not found.")],
					 [echo "GLUT/glut.h not found."]
					)
			if test "$FOUND_GLUT" = yes
			then
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
	Windows)
	AC_MSG_CHECKING([OpenGL support])
	SAVED_LIBS=$LIBS

	#AC_MSG_CHECKING([GLEW presence (library and headers)])
	#SAVED_LIBS=$LIBS
	LIBS="$LIBS -lglew32"
	AC_LINK_IFELSE([ AC_LANG_PROGRAM([
#define _STDCALL_SUPPORTED 1
#include <GL/glew.h>
], [
glewInit();
		])],
		[FOUND_GLEW=yes],[FOUND_GLEW=no])

	LIBS="$LIBS -lfreeglut -lglew32 -lopengl32"
	AC_LINK_IFELSE([ AC_LANG_PROGRAM([
#define _STDCALL_SUPPORTED 1
#include <GL/glew.h>
#include <GL/glut.h>
], [glutMainLoop();
glewInit();
glBindTexture(GL_TEXTURE_2D, 0);
		])],
		[HAVE_GL=yes],[HAVE_GL=no])
	LIBS=$SAVED_LIBS
	AC_MSG_RESULT([$HAVE_GL])

	if test $gl_req != no -a "$HAVE_GL" = yes; then
		GL_LIB="$GL_LIB -lfreeglut -lglew32 -lopengl32"
		GL_OBJ="$GL_OBJ src/video_display/gl.o"
		AC_DEFINE([HAVE_GL], [1], [Build with OpenGL output])
		gl=yes	
	fi
	;;
	*)
	AC_CHECK_HEADERS([GL/gl.h GL/glx.h ],
			 [FOUND_GL_H=yes],
			 [FOUND_GL_H=no],
			 [
		 	  #include <GL/gl.h>
			  #include <GL/glx.h>
			 ]
			)

	if test $FOUND_GL_H = yes
	then
		AC_CHECK_LIB(GL, glBindTexture, FOUND_GL_L=yes, FOUND_GL_L=no)

		if test $gl_req != no -a $FOUND_GL_L = yes
		then
			GL_LIB="$GL_LIB -lGL"

			AC_CHECK_HEADERS([GL/glew.h], AC_CHECK_LIB(GLEW, glewInit, FOUND_GLEW=yes, FOUND_GLEW=no, $GL_LIB), FOUND_GLEW=no)


			AC_CHECK_HEADERS([GL/glut.h],
					 [AC_CHECK_LIB(glut, glutInit, FOUND_GLUT=yes , echo "GLUT library not found or not usable.", $GL_LIB)],
					 [echo "GL/glut.h not found."]
					)
			if test "$FOUND_GLUT" = yes -a $FOUND_GLEW = yes
			then
				GL_LIB="$GL_LIB -lglut -lGLEW"
				GL_OBJ="$GL_OBJ src/video_display/gl.o"
				AC_DEFINE([HAVE_GL], [1], [Build with OpenGL output])
                                gl=yes
			else
                                if test "$FOUND_GLUT" = no
                                then
                                        AC_MSG_WARN([OpenGL (GLUT) libraries not found or not usable.])
                                fi

                                if test $FOUND_GLEW = no
                                then
                                        AC_MSG_WARN([OpenGL GLEW libraries or headers not found or not usable.])
                                fi
			fi
		else
			AC_MSG_WARN([OpenGL libraries not found or not usable.])
		fi

	else
		AC_MSG_WARN([OpenGL headers (gl.h, glx.h) not found or not usable.])
	fi
	;;
esac
AC_SUBST(FOUND_GLEW)

if test $gl = yes
then
	INC="$INC $GL_INC"
	ADD_MODULE("display_gl", "$GL_OBJ $HW_ACC_DISPLAY_OBJ", "$GL_LIB")
fi

if test $gl_req = yes -a $gl = no; then
        AC_MSG_ERROR([OpenGL not found]);
fi

# -------------------------------------------------------------------------------------------------
# These tests are shared for OpenGL processing
# -------------------------------------------------------------------------------------------------
SAVED_LIBS=$LIBS

AC_CHECK_HEADER(GL/glx.h, FOUND_GLX_H=yes)
AC_CHECK_HEADER(GL/gl.h, FOUND_GL_H=yes)
AC_CHECK_LIB(GL, glBindTexture)
AC_CHECK_LIB(GL, glXCreateNewContext, FOUND_GLX_L=yes)
AC_CHECK_LIB(X11, XCreateWindow)

LIBS=$SAVED_LIBS

OPENGL=no

# Linux
if test "$ac_cv_lib_X11_XCreateWindow" = yes -a $FOUND_GLEW = yes -a "$FOUND_GLX_L" = yes -a "$FOUND_GLX_H" = yes \
        -a "$FOUND_GL_H" = yes -a $ac_cv_lib_GL_glBindTexture = yes -a "$system" = Linux
then
        OPENGL=yes
        OPENGL_LIB="-lGLEW -lGL -lX11"
fi

# Mac
if test "$system" = MacOSX
then
        OPENGL=yes
        OPENGL_LIB="-framework OpenGL -framework Cocoa"
fi

# Win32
if test "$system" = Windows -a "$FOUND_GLEW" = yes
then
        OPENGL=yes
        OPENGL_LIB="-lopengl32 -lglew32 -lgdi32"
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
    [swmix_req=auto]
    )

if test $swmix_req != no -a $OPENGL = yes
then
        swmix=yes
        SWMIX_LIB="$OPENGL_LIB $X11_LIB"
        SWMIX_OBJ="$SWMIX_OBJ $GL_COMMON_OBJ src/video_capture/swmix.o"
        ADD_MODULE("vidcap_swmix", "$SWMIX_OBJ", "$SWMIX_LIB")
        AC_DEFINE([HAVE_SWMIX], [1], [Build SW mix capture])
fi

if test $swmix_req = yes -a $swmix = no; then
        AC_MSG_ERROR([SW mix was not found (OpenGL libraries missing)]);
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
[                          Requires: none (OSX) x11 (Linux)],
    [screen_cap_req=$enableval],
    [screen_cap_req=auto]
    )

case "$system" in
	MacOSX)
		if test $screen_cap_req != no -a $os_version_major -ge 10
		then
                        screen_cap=yes
                        SCREEN_CAP_LIB="-framework CoreFoundation"
		else
                        if test $screen_cap_req != no
                        then
                                AC_MSG_WARN([*** Screen capture supported for Mac version 10.6 and higher.])
                        fi
                        screen_cap=no
		fi

	;;
        Windows)
                screen_cap=no
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

if test $screen_cap = yes
then
        AC_DEFINE([HAVE_SCREEN_CAP], [1], [Build with screen capture])
        if test $system = MacOSX; then
                SCREEN_CAP_OBJ="src/video_capture/screen_osx.o"
        else if test $system = Linux; then
                SCREEN_CAP_OBJ="src/video_capture/screen_x11.o"
        fi
        fi
        ADD_MODULE("vidcap_screen", "$SCREEN_CAP_OBJ", "$SCREEN_CAP_LIB")
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
        [rtdxt_req=auto])


AC_DEFINE([USE_PBO_DXT_ENCODER], [1], [We want to use OpenGL pixel buffer objects])
# We probably want this only if there is need to maximalize bandwidth
# With NVidia cards, it increases latency up about 1.5x frame time
#AC_DEFINE([USE_PBO_DXT_ECODER], [1], [We want to use OpenGL pixel buffer objects])

# Linux
if test $rtdxt_req != no -a $OPENGL = yes
then
        rtdxt=yes
        RTDXT_LIB="$OPENGL_LIB $X_LIBS"
        AC_DEFINE([HAVE_DXT_GLSL], [1], [Build with DXT_GLSL support])
        RTDXT_COMMON_OBJ="$RTDXT_COMMON_OBJ $GL_COMMON_OBJ dxt_compress/dxt_util.o"
        RTDXT_COMMON_HEADERS="$RTDXT_COMMON_HEADERS dxt_compress/dxt_glsl.h"
        RTDXT_COMPRESS_OBJ="$RTDXT_COMMON_OBJ dxt_compress/dxt_encoder.o src/video_compress/dxt_glsl.o"
        RTDXT_DECOMPRESS_OBJ="$RTDXT_COMMON_OBJ dxt_compress/dxt_decoder.o src/video_decompress/dxt_glsl.o"
        DXT_GLSL_CFLAGS="-std=gnu99"
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
        [uyvy_req=auto])

SAVED_LIBS=$LIBS

# Linux
if test $uyvy_req != no -a "$ac_cv_lib_X11_XCreateWindow" = yes -a $FOUND_GLEW = yes -a "$FOUND_GLX_L" = yes -a "$FOUND_GLX_H" = yes \
        -a "$FOUND_GL_H" = yes -a "$system" = Linux
then
        UYVY_LIB=" -lGLEW -lGL -lX11"
        uyvy=yes
fi

# Mac
if test $uyvy_req != no -a "$system" != Linux -a "$gl" = yes
then
        uyvy=yes
fi

# common
if test $uyvy = yes
then
        AC_DEFINE([HAVE_COMPRESS_UYVY], [1], [Build with dummy UYVY compression])
        UYVY_COMPRESS_OBJ="$GL_COMMON_OBJ src/video_compress/uyvy.o"
        ADD_MODULE("vcompress_uyvy", "$UYVY_COMPRESS_OBJ", "$UYVY_LIB")
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
              [cuda_req=auto])

AC_ARG_WITH(cuda,
            [  --with-cuda=DIR         specify location of CUDA Toolkit],
            [CUDA_PATH=$withval
            ])

# nvcc is intentional here
AC_PATH_PROG(NVCC, nvcc, [], [$CUDA_PATH/bin$PATH_SEPARATOR$PATH]dnl
        [$PATH_SEPARATOR/opt/cuda/bin$PATH_SEPARATOR/usr/local/cuda/bin])

if test -n "$NVCC" -a $cuda_req != no
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
fi

# Pick compiler for cuda device code
AC_ARG_WITH(cuda-compiler,
              AS_HELP_STRING([--with-cuda-compiler=compiler], [select compiler to compile cuda code with; suported: clang, nvcc; default: nvcc]),
              [CUDA_COMPILER="$withval"],
              [])

HOST_CC_REPORT=" ("
case "$CUDA_COMPILER" in
	*nvcc*)
		CUDA_COMPUTE_ARGS="-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=compute_35"
		;;
	*clang*)
		CUDA_COMPUTE_ARGS="--cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35"
		;;
	*)
		AC_MSG_WARN([Unsupported cuda compiler $CUDA_COMPILER])
		;;
esac
HOST_CC_REPORT="$HOST_CC_REPORT"$(basename "$CUDA_COMPILER")

# CUDA host compiler
AC_ARG_WITH(cuda-host-compiler,
	AC_HELP_STRING([--with-cuda-host-compiler=BINARY], [specify compiler used for building cuda host code]))
if test "x${with_cuda_host_compiler}" != "x" ; then
	AC_CHECK_PROG([cuda_host_cc], [${with_cuda_host_compiler}],[yes])
	if test "x${cuda_host_cc}" = "xyes" ; then
		CUDA_FLAGS+=" -ccbin ${with_cuda_host_compiler}"
	else
		AC_MSG_FAILURE([Binary ${with_cuda_host_compiler} given as cuda host compiler, yet not found!])
	fi
else
	with_cuda_host_compiler="default"
fi

if test "x$NVCC" != "X" ; then
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
# JPEG
JPEG_OBJ=
JPEG_COMPRESS_OBJ=
JPEG_DECOMPRESS_OBJ=
JPEG_INC=
JPEG_LIB=

jpeg=no

AC_ARG_ENABLE(jpeg,
[  --disable-jpeg          disable JPEG compression (auto)]
[                          Requires: CUDA libgpujpeg ],
	[jpeg_req=$enableval],
        [jpeg_req=auto])

SAVED_CXXFLAGS=$CXXFLAGS
SAVED_CPPFLAGS=$CPPFLAGS
CXXFLAGS="$CXXFLAGS $CUDA_INC"
CPPFLAGS="$CPPFLAGS $CUDA_INC"
SAVED_CFLAGS=$CFLAGS
CFLAGS="$CFLAGS $CUDA_INC"
CFLAGS=$SAVED_CFLAGS
#AC_CHECK_FILE(libgpujpeg/libgpujpeg.a)
CXXFLAGS=$SAVED_CXXFLAGS
CPPFLAGS=$SAVED_CPPFLAGS

if test $system = Windows
then
        JPEG_LIB="$JPEG_LIB -Lgpujpeg/Release -lgpujpeg"
        SAVED_LIBS=$LIBS
        LIBS="$LIBS $JPEG_LIB"
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
else
        PKG_CHECK_MODULES([LIBGPUJPEG], [libgpujpeg >= 0.0.2 ], [found_gpujpeg=yes], [found_gpujpeg=no])
fi

if test "$jpeg_req" != no -a \
        "$FOUND_CUDA" = yes -a \
        "$found_gpujpeg" = yes
then
        jpeg=yes

        JPEG_INC="$JPEG_INC $LIBGPUJPEG_CFLAGS"
        JPEG_LIB="$JPEG_LIB $LIBGPUJPEG_LIBS"
        JPEG_COMPRESS_OBJ="src/video_compress/jpeg.o"
        JPEG_DECOMPRESS_OBJ="src/video_decompress/jpeg.o "
        AC_DEFINE([HAVE_JPEG], [1], [Build with JPEG support])
        ADD_MODULE("vcompress_jpeg", "$JPEG_COMPRESS_OBJ", "$JPEG_LIB")
        ADD_MODULE("vdecompress_jpeg", "$JPEG_DECOMPRESS_OBJ", "$JPEG_LIB")

	INC="$INC $JPEG_INC"
        CUDA_MESSAGE
fi

if test $jpeg_req = yes -a $jpeg = no; then
        AC_MSG_ERROR([JPEG not found]);
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
                          DLL_LIBS="$DLL_LIBS cuda_dxt/cuda_dxt.cu.dll"
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
        [cuda_dxt_req=auto])

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
# JPEG to DXT plugin
# -------------------------------------------------------------------------------------------------
jpeg_to_dxt=no
AC_ARG_ENABLE(jpeg_to_dxt,
[  --disable-jpeg-to-dxt   disable JPEG to DXT transcoder (default is auto)]
[                          Requires: CUDA libgpujpeg],
              [jpeg_to_dxt_req=$enableval],
              [jpeg_to_dxt_req=auto])

if test $jpeg_to_dxt_req != no -a $FOUND_CUDA = yes -a \
        "$found_gpujpeg" = yes
then
        DEFINE_CUDA
        DEFINE_CUDA_DXT
        jpeg_to_dxt=yes
        JPEG_TO_DXT_INC=" $CUDA_INC"
        JPEG_TO_DXT_LIB="$CUDA_DXT_COMMON_LIB $CUDA_COMMON_LIB $CUDA_LIB $JPEG_LIB"
        JPEG_TO_DXT_OBJ="src/video_decompress/jpeg_to_dxt.o $CUDA_COMMON_OBJ $CUDA_DXT_COMMON_OBJ"
        ADD_MODULE("vdecompress_jpeg_to_dxt", $JPEG_TO_DXT_OBJ, "$JPEG_TO_DXT_LIB")
        AC_DEFINE([HAVE_JPEG_TO_DXT], [1], [Build with JPEG to DXT transcode support])
        CUDA_MESSAGE
fi


if test $jpeg_to_dxt = no -a $jpeg_to_dxt_req = yes
then
        AC_MSG_ERROR([JPEG to DXT transcoder not found])
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
        [sage_req=auto])

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

AC_CHECK_FILE(${SAGE_INC}/sail.h, FOUND_SAGE_H=yes, FOUND_SAGE_H=no)

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
        [portaudio_req=auto])


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

# -------------------------------------------------------------------------------------------------
# SPEEX

speex=yes
SPEEX_PATH=speex-1.2rc1
SPEEX_INC=-I$srcdir/${SPEEX_PATH}/include
SPEEX_OBJ="src/audio/resample.o src/audio/preprocess.o src/audio/filterbank.o src/audio/fftwrap.o src/audio/smallft.o src/audio/mdf.o"
SPEEX_OBJ="$SPEEX_OBJ src/audio/echo.o"
SPEEX_LIB=
AC_DEFINE([HAVE_SPEEX], [1], [Build with SPEEX support])

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
[  --enable-jack-transport enable JACK transport (default is disable)]
[                          Requires: jack],
	[jack_trans_req=$enableval],
        [jack_trans_req=no])

AC_CHECK_HEADER(jack/jack.h, FOUND_JACK_H=yes, FOUND_JACK_H=no)
AC_CHECK_LIB(jack, jack_client_new, FOUND_JACK_L=yes, FOUND_JACK_L=no)
if test $jack_trans_req != no -a $FOUND_JACK_H = yes -a $FOUND_JACK_L = yes
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
[  --disable-jack          disable JACK (default is auto)]
[                          Requires: jack],
	[jack_req=$enableval],
        [jack_req=auto])

if test $jack_req != no -a $FOUND_JACK_H = yes -a $FOUND_JACK_L = yes
then
    JACK_CAP_OBJ="src/audio/capture/jack.o"
    JACK_PLAY_OBJ="src/audio/playback/jack.o"
    JACK_LIB="-ljack"
    AC_DEFINE([HAVE_JACK], [1], [Build with JACK support])
    jack=yes
    ADD_MODULE("acap_jack", "$JACK_CAP_OBJ", "$JACK_LIB")
    ADD_MODULE("aplay_jack", "$JACK_PLAY_OBJ", "$JACK_LIB")
fi

if test $jack = yes -a $speex = no
then
        AC_MSG_WARN([*** You are going to compile JACK but you do not have SPEEX compiled in.
                     It is recommended to compile also SPEEX library and rerun this script in
                     order to enable resampling (if needed).])
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
        [alsa_req=auto])
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
    [coreaudio_req=auto]
    )

case $host_os in
	*darwin*)
		AC_CHECK_HEADERS([AudioUnit/AudioUnit.h], [FOUND_AUDIOUNIT_H=yes], [echo "AudioUnit headers not found."])

		if test $coreaudio_req != no -a $FOUND_AUDIOUNIT_H = yes
		then
			COREAUDIO_LIB="-framework CoreAudio -framework AudioUnit"
			COREAUDIO_OBJ="src/audio/capture/coreaudio.o src/audio/playback/coreaudio.o"
                        coreaudio=yes
                        AC_DEFINE([HAVE_COREAUDIO], [1], [Build with CoreAudio support])
		fi
    ;;
esac

if test $coreaudio = yes -a $speex = no
then
        AC_MSG_WARN([*** You are going to compile CoreAudio but you do not have SPEEX compiled in.
                     It is recommended to compile also SPEEX library and rerun this script in
                     order to enable resampling (if needed).])
fi

if test $coreaudio_req = yes -a $coreaudio = no; then
        AC_MSG_ERROR([Core Audio not found]);
fi

AC_SUBST(COREAUDIO_LIB)
AC_SUBST(COREAUDIO_OBJ)


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
    [scale_req=auto]
    )

if test $scale_req != no -a $OPENGL = yes
then
        scale=yes
        SCALE_LIB="$SCALE_LIB $OPENGL_LIB $X11_LIB"
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
    [text_req=auto]
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

GUI_TARGET=
QMAKE=
#qmake # default for most distributions

AC_ARG_ENABLE(qt,
[  --enable-qt             build Qt GUI (default is disabled)]
[                          Requires: Qt5Core >= 5.4 Qt5Widgets Qt5Gui (for Linux also v4l2) ],
    [qt_req=$enableval],
    [qt_req=no]
    )

AC_ARG_ENABLE(static-qt,
    AS_HELP_STRING([--enable-static-qt], [use statically built Qt (default is disabled)]),
    [static_qt=$enableval],
    [static_qt=no]
    )

# Check for Qt libraries
QT_GUI_DEP="Qt5Core >= 5.4 Qt5Widgets Qt5Gui"

SAVED_PKG_CONFIG=$PKG_CONFIG
if test $static_qt = yes; then
        PKG_CONFIG='pkg-config --static'
fi
PKG_CHECK_MODULES(QT, [$QT_GUI_DEP], [FOUND_QT_DEP=yes], [FOUND_QT_DEP=no])
PKG_CONFIG=$SAVED_PKG_CONFIG

if test $qt_req != no -a $FOUND_QT_DEP = yes
then
	# Retrieve Qt compilation flags
	CXXFLAGS="$QT_CFLAGS $CXXFLAGS"
	CFLAGS="$QT_CFLAGS $CFLAGS"

	ADD_MODULE("display_preview", "src/video_display/preview.o src/shared_mem_frame.o", "$QT_LIBS")
	ADD_MODULE("capture_filter_preview", "src/capture_filter/preview.o src/shared_mem_frame.o", "$QT_LIBS $GL_LIB")

	AC_PATH_PROGS_FEATURE_CHECK([QTCHOOSER], [qtchooser], 
		[AS_IF([$ac_path_QTCHOOSER --run-tool=qmake --qt=5 | grep QMake  > /dev/null],
			[QMAKE="$ac_path_QTCHOOSER --run-tool=qmake --qt=5"])])

	#AC_MSG_NOTICE("Qmake: $QMAKE")
	if test "x$QMAKE" = "x"; then
		AC_PATH_PROGS(QMAKE, [qmake-qt5 qmake], [:], [/usr/lib/qt5/bin:$PATH])
	fi

	#AC_MSG_NOTICE("Qmake: $QMAKE")
	GUI_TARGET=gui/QT/uv-qt

	if test ":" = "$QMAKE"; then
		qt_gui=no
	else
		qt_gui=yes
	fi
fi

AC_SUBST(GUI_TARGET)
AC_SUBST(QMAKE)

if test $qt_req = yes -a $qt_gui = no; then
		AC_MSG_ERROR([Qt libraries are required.])
fi

# -------------------------------------------------------------------------------------------------
# V4L2
# -------------------------------------------------------------------------------------------------
v4l2=no
found_v4l2=no

define(v4l2_deps, libv4l2 libv4lconvert)

AC_ARG_ENABLE(v4l2,
[  --disable-v4l2          disable V4L2 support (default is auto)]
[                          Requires: v4l2_deps],
    [v4l2_req=$enableval],
    [v4l2_req=auto]
    )

if test $system = Linux; then
        PKG_CHECK_MODULES([V4L2], [libv4l2 libv4lconvert], [found_v4l2=yes], [found_v4l2=no])
fi

if test $v4l2_req != no -a $found_v4l2 = yes
then
        V4L2_OBJ="src/video_capture/v4l2.o"
        AC_DEFINE([HAVE_V4L2], [1], [Build with V4L2 support])
        ADD_MODULE("vidcap_v4l2", "$V4L2_OBJ", "$V4L2_LIBS")
        CFLAGS="$CFLAGS $V4L2_CFLAGS"
        v4l2=yes
fi

if test $v4l2_req = yes -a $v4l2 = no; then
        AC_MSG_ERROR([V4L2 not found]);
fi

# -----------------------------------------------------------------------------
# DirectShow
# -----------------------------------------------------------------------------
dshow=no
dshow_req=auto
AC_ARG_ENABLE(dshow,
      AS_HELP_STRING([--disable-dshow], [disable DirectShow support(default is auto)]),
    [dshow_req=$enableval],
    [dshow_req=auto]
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
# Import control - keyboard
# -------------------------------------------------------------------------------------------------
FOUND_CURSES=no

if test $system = Linux -o $system = MacOSX; then
        PKG_CHECK_MODULES([CURSES], [ncursesw], [FOUND_CURSES=yes], [FOUND_CURSES=no])
        if test $FOUND_CURSES = no; then
                PKG_CHECK_MODULES([CURSES], [ncurses], [FOUND_CURSES=yes], [FOUND_CURSES=no])
        fi
else # Windows
        AC_CHECK_HEADER(curses.h)
        AC_CHECK_LIB(pdcurses, initscr)
        if test $ac_cv_lib_pdcurses_initscr = yes -a \
                $ac_cv_header_curses_h = yes; then
                FOUND_CURSES=yes
                CURSES_LIBS="-lpdcurses"
        fi
fi

if test $FOUND_CURSES = yes; then
        AC_DEFINE([HAVE_CURSES], [1], [Build with curses support])
fi

AC_SUBST(CURSES_LIBS)

# -------------------------------------------------------------------------------------------------
# OpenSSL-libcrypto
# -------------------------------------------------------------------------------------------------
crypto=no
AC_ARG_ENABLE(openssl,
[  --disable-openssl       disable OpenSSL support (default is auto)]
[                          Requires: crypto],
    [crypto_req=$enableval],
    [crypto_req=auto]
    )

PKG_CHECK_MODULES([CRYPTO], [libcrypto], [crypto=yes], [crypto=no])

if test $crypto_req != no -a $crypto = yes; then
        OPENSSL_OBJ="src/crypto/openssl_encrypt.o src/crypto/openssl_decrypt.o"
        CFLAGS="$CFLAGS $CRYPTO_CFLAGS"
        CXXFLAGS="$CXXFLAGS $CRYPTO_CFLAGS"
        CPPFLAGS="$CPPFLAGS $CRYPTO_CFLAGS"
        ADD_MODULE("openssl", "$OPENSSL_OBJ", "$CRYPTO_LIBS")
        SAVED_LIBS=$LIBS
        LIBS="$LIBS $CRYPTO_LIBS"
        AC_CHECK_FUNCS(AES_ctr128_encrypt)
        LIBS=$SAVED_LIBS
else
        crypto=no
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
    [ldgm_gpu_req=auto]
    )

if test $ldgm_gpu_req != no -a $FOUND_CUDA = yes
then
        DEFINE_CUDA
        CUDA_MESSAGE
        AC_DEFINE([HAVE_LDGM_GPU], [1], [Build with GPU accelerated LDGM])
        LDGM_GPU_OBJS="ldgm/src/ldgm-session-gpu.o src/rtp/ldgm_gpu.o $CUDA_COMMON_OBJ"
        LDGM_GPU_LIBS="$CUDA_COMMON_LIB $CUDA_LIB"
        if test $system = Windows; then
                DLL_LIBS="$DLL_LIBS ldgm/src/gpu.cu.dll"
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
    [syphon_req=auto]
    )

MY_CHECK_FRAMEWORK([Syphon])

if test $syphon_req != no -a $my_cv_framework_Syphon = yes
then
        AC_DEFINE([HAVE_SYPHON], [1], [Build with Syphon support])
        LIBS="$LIBS -framework Syphon"
        OBJS="$OBJS src/syphon_server.o src/video_capture/syphon.o"
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
[  --enable-spout          enables Spout support (default is disabled)]
[                          Requires: Spout],
    [spout_req=$enableval],
    [spout_req=no]
    )

if test $spout_req == yes
then
        AC_DEFINE([HAVE_SPOUT], [1], [Build with Spout support])
        LIBS="$LIBS -lspout_wrapper"
        OBJS="$OBJS src/video_capture/spout.o"
        spout=yes
fi


# -------------------------------------------------------------------------------------------------
# MCU-like video mixer
# -------------------------------------------------------------------------------------------------
video_mix=no

AC_ARG_ENABLE(video-mixer,
[  --disable-video-mixer   disable MCU-like video mixer (default is auto)]
[                          Requires: opencv],
    [video_mix_req=$enableval],
    [video_mix_req=auto]
    )

if test x$HAVE_OPENCV != xno; then
	FOUND_VIDEO_MIX_DEP=yes
	VIDEO_MIX_LIBS=$OPENCV_LIBS
	VIDEO_MIX_CFLAGS=$OPENCV_CFLAGS
else
	FOUND_VIDEO_MIX_DEP=no
fi

if test $video_mix_req != no -a $FOUND_VIDEO_MIX_DEP = yes -a x$OPENCV_MAJOR = x3
then
	AC_CHECK_LIB([opencv_cudawarping], [main], [
		VIDEO_MIX_LIBS="$VIDEO_MIX_LIBS -lopencv_cudawarping "
		AC_DEFINE([HAVE_OPENCV_CUDA], [1], [Whether the opencv installation supports cuda modules])
	], [
		AC_MSG_WARN([Video mixer will be compiled as CPU-only])
	], [
		$VIDEO_MIX_LIBS
	])
elif test x$OPENCV_MAJOR != x3
then
	AC_CHECK_HEADER(opencv2/gpu/gpu.hpp, AC_DEFINE([HAVE_OPENCV_CUDA], [1],
			[Whether the opencv installation supports cuda modules]),
			AC_MSG_WARN([Video mixer will be compiled as CPU-only]))
fi

if test $video_mix_req != no -a $FOUND_VIDEO_MIX_DEP = yes
then
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
    [cmpto_j2k_req=auto]
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
        AC_MSG_ERROR([Comprimato J2K not found!)]);
fi

# ------------------------------------------------------------------------------
# NDI
# ------------------------------------------------------------------------------
ndi=no
AC_ARG_ENABLE(ndi,
[  --enable-ndi            enables NDI support (default is auto)],
    [ndi_req=$enableval],
    [ndi_req=auto]
    )

AC_CHECK_HEADER(Processing.NDI.Lib.h, FOUND_NDI_H=yes, FOUND_NDI_H=no)
AC_CHECK_LIB(ndi, NDIlib_initialize,  FOUND_NDI_L=yes, FOUND_NDI_L=no)

if test $ndi_req != no -a "$FOUND_NDI_H" = yes -a "$FOUND_NDI_L"
then
        ADD_MODULE("vidcap_ndi", "src/video_capture/ndi.o", "-lndi")
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

CINEFORM_STATIC_PATH=

cineform=no

AC_ARG_ENABLE(cineform,
              [  --disable-cineform      disable cineform suppport (default is auto)]
              [            Requires: CINEFORM_SDK],
              [cineform_req=$enableval],
              [cineform_req=auto]
              )

if test -n "$CINEFORM_DIRECTORY"; then
        CINEFORM_PATH=$CINEFORM_DIRECTORY
fi

AC_ARG_WITH(cineform,
            [  --with-cineform=DIR     specify location of Cineform SDK],
            [CINEFORM_PATH=$withval
             ])

if test $system = Windows
then
	CINEFORM_STATIC_PATH="${CINEFORM_PATH}/Release/CFHDCodec.lib"
else
	CINEFORM_STATIC_PATH="${CINEFORM_PATH}/libCFHDCodec.a"
fi


if test -n "$CINEFORM_PATH"
then
        AC_CHECK_FILES(${CINEFORM_STATIC_PATH}, FOUND_CINEFORM=yes, FOUND_CINEFORM=no)
fi

if test "$cineform_req" != no -a "$FOUND_CINEFORM" = yes
then
        cineform=yes
        CINEFORM_INC="$CINEFORM_INC -I${CINEFORM_PATH}/Common/"

		CINEFORM_LIB="${CINEFORM_STATIC_PATH}"
		if test $system != Windows
		then
			CINEFORM_LIB="${CINEFORM_LIB} -luuid"
		fi

        CINEFORM_COMPRESS_OBJ="src/video_compress/cineform.o"
        CINEFORM_DECOMPRESS_OBJ="src/video_decompress/cineform.o"

        INC="$INC $CINEFORM_INC"
        
        ADD_MODULE("vcompress_cineform", "$CINEFORM_COMPRESS_OBJ", "$CINEFORM_LIB")
        ADD_MODULE("vdecompress_cineform", "$CINEFORM_DECOMPRESS_OBJ", "$CINEFORM_LIB")
fi

if test $cineform_req = yes -a $cineform = no; then
        AC_MSG_ERROR([CINEFORM not found]);
fi

# -------------------------------------------------------------------------------------------------
# We need to add libraries then
# -------------------------------------------------------------------------------------------------

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

AC_SUBST(GENERATED_HEADERS)
AC_SUBST(LIB_GENERATED_HEADERS)
AC_SUBST(LIB_OBJS)
AC_SUBST(MODULES)
AC_SUBST(TARGETS)

# -------------------------------------------------------------------------------------------------
# Finally, substitute things into the Makefile and config.h

RESULT=\
"  Target ...................... $target
  Debug output ................ $debug_output
  Profiling support ........... $profile
  IPv6 support ................ $ipv6
  RT priority ................. $use_rt
  Standalone modules .......... $build_libraries
  iHDTV support ............... $ihdtv
  OpenSSL-libcrypto ........... $crypto
  CUDA support ................ $FOUND_CUDA$HOST_CC_REPORT
  Library live555 ............. $livemedia
  Qt GUI ...................... $qt_gui
  OpenCV ...................... $opencv
"

# params: <result> <feature> <bool> 0|1
# 0 - print left column; 1 - print right column
add_column() {
        # left column
        if [ test $4 = 0 ]; then
                printf "$1\n  $2 $(for n in `seq ${#2} 27`; do printf .; done) $(printf %-3s $3)"
                return 1
        else
                # right column
                printf "$1   $2 $(for n in `seq ${#2} 27`; do printf .; done) $(printf %-3s $3)\n"
                return 0
        fi
}

true # set $? to zero
RESULT=`add_column "$RESULT" "AJA" $aja $?`
RESULT=`add_column "$RESULT" "AV Foundation" $avfoundation $?`
RESULT=`add_column "$RESULT" "BitFlow" $bitflow $?`
RESULT=`add_column "$RESULT" "Bluefish444 (audio: $blue_audio)" $bluefish444 $?`
RESULT=`add_column "$RESULT" "DeckLink" $decklink $?`
RESULT=`add_column "$RESULT" "DELTACAST" $deltacast $?`
RESULT=`add_column "$RESULT" "DirectShow" $dshow $?`
RESULT=`add_column "$RESULT" "DVS" $dvs $?`
RESULT=`add_column "$RESULT" "NDI" $ndi $?`
RESULT=`add_column "$RESULT" "OpenGL" $gl $?`
RESULT=`add_column "$RESULT" "QuickTime" $quicktime $?`
RESULT=`add_column "$RESULT" "RTSP capture client" $rtsp $?`
RESULT=`add_column "$RESULT" "SAGE" $sage $?`
RESULT=`add_column "$RESULT" "Screen capture" $screen_cap $?`
RESULT=`add_column "$RESULT" "SDL (ver. $sdl_version)" $sdl $?`
RESULT=`add_column "$RESULT" "SW video mix" $swmix $?`
RESULT=`add_column "$RESULT" "V4L2" $v4l2 $?`

RESULT="$RESULT
"
true # set $? to zero
RESULT=`add_column "$RESULT" "ALSA" $alsa $?`
RESULT=`add_column "$RESULT" "CoreAudio" $coreaudio $?`
RESULT=`add_column "$RESULT" "JACK" $jack $?`
RESULT=`add_column "$RESULT" "JACK transport" $jack_trans $?`
RESULT=`add_column "$RESULT" "Portaudio" $portaudio $?`

RESULT="$RESULT
"
true # set $? to zero
RESULT=`add_column "$RESULT" "Comprimato J2K" $cmpto_j2k $?`
RESULT=`add_column "$RESULT" "CUDA DXT" $cuda_dxt $?`
RESULT=`add_column "$RESULT" "JPEG" $jpeg $?`
RESULT=`add_column "$RESULT" "JPEG to DXT" $jpeg_to_dxt $?`
RESULT=`add_column "$RESULT" "Libavcodec (VDP $lavc_hwacc_vdpau, VA $lavc_hwacc_vaapi)" $libavcodec $?`
RESULT=`add_column "$RESULT" "Realtime DXT" $rtdxt $?`
RESULT=`add_column "$RESULT" "UYVY dummy compression" $uyvy $?`
RESULT=`add_column "$RESULT" "Cineform" $cineform $?`

RESULT="$RESULT
"
true # set $? to zero
RESULT=`add_column "$RESULT" "Blank capture filter" $blank $?`
RESULT=`add_column "$RESULT" "GPU accelerated LDGM" $ldgm_gpu $?`
RESULT=`add_column "$RESULT" "iHDTV support" $ihdtv $?`
RESULT=`add_column "$RESULT" "MCU-like video mixer" $video_mix $?`
RESULT=`add_column "$RESULT" "Resize capture filter" $resize $?`
RESULT=`add_column "$RESULT" "RTSP server" $rtsp_server $?`
RESULT=`add_column "$RESULT" "Scale postprocessor" $scale $?`
RESULT=`add_column "$RESULT" "SDP over HTTP" $sdp_http $?`
RESULT=`add_column "$RESULT" "Spout" $spout $?`
RESULT=`add_column "$RESULT" "Syphon" $syphon $?`
RESULT=`add_column "$RESULT" "Testcard extras" $testcard_extras $?`
RESULT=`add_column "$RESULT" "Text postprocess and filter" $text $?`

RESULT="$RESULT
"

AC_MSG_RESULT(
[
UltraGrid configuration summary:
])
AC_MSG_RESULT([$RESULT])

RES_STR=`printf "$RESULT" | sed 's/$/\\\\n/' | tr -d '\n'`

AC_DEFINE_UNQUOTED([AUTOCONF_RESULT], "$RES_STR", [Autoconf result])

AC_CONFIG_HEADER(src/config.h)
AC_CONFIG_FILES([Makefile])
AC_OUTPUT

