dnl Process this file with autoconf to produce a configure script.

AC_INIT
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
LINKER=

AC_C_BIGENDIAN(
        AC_DEFINE([WORDS_BIGENDIAN], 1, [This is big endian system]),
        AC_DEFINE([WORDS_SMALLENDIAN], 1, [This is little endian system])
)

case "$host" in
	*darwin9*)
		AC_DEFINE([HAVE_MACOSX_LEOPARD], [1], [This is Mac X OS Leopard])
	;;
  	x86_64-*-linux-*)
		DVS_SDK_SUBDIR="linux-x86_64"
		AC_DEFINE([HAVE_64B_LINUX], [1], [We have 64-bit Linux])
                LIBS+=" -lrt"
	;;
	i*86-*-linux-*)
		DVS_SDK_SUBDIR="linux-x86"
		AC_DEFINE([HAVE_32B_LINUX], [1], [We have 32-bit Linux])
                LIBS+=" -lrt"
	;;
esac	

if expr "$host_os" : ".*darwin.*" > /dev/null; then
	AC_MSG_CHECKING([Mac X OS kernel version (major)])
	AC_DEFINE([HAVE_MACOSX], [1], [This is Mac X OS])
	LINKER=$CXX
	version_major="`echo \"$host_os\" | sed 's/^.*darwin\([[^.]]*\)..*$/\1/'`"
	AC_DEFINE_UNQUOTED([MACOSX_VERSION_MAJOR], $version_major, [Major number of Mac X OS version])
	AC_MSG_RESULT($version_major);
	CFLAGS+=" -m32 -I/opt/local/include"
	CPPFLAGS+=" -m32 -I/opt/local/include"
	CXXFLAGS+=" -m32 -I/opt/local/include"
	LDFLAGS+=" -m32 -L/opt/local/lib -mmacosx-version-min=10.5"
	X_CFLAGS+="-I/usr/X11R6/include"
else
	LINKER=$CC
fi

AC_SUBST(LINKER)
AC_SUBST(LDFLAGS)
AC_SUBST(CXXFLAGS)

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

AC_CHECK_LIB(rt, main, LIBS="$LIBS -lrt", LIBS="$LIBS")

AC_CHECK_FUNC(sin, MATHLIBS="", MATHLIBS="-lm")
AC_CHECK_LIB(ieee, main, [MATHLIBS="-lieee $MATHLIBS"])
AC_SUBST(MATHLIBS)

# -------------------------------------------------------------------------------------------------
# See if this system supports sched_setscheduler()

AC_CHECK_FUNCS(sched_setscheduler)
use_rt=no
AC_ARG_ENABLE(dvs,
              AS_HELP_STRING([--enable-rt], [enable RT priority]), 
	[use_rt=$enableval])
if [ "$use_rt" = "yes" ]; then
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
			AC_MSG_ERROR([Don't know how to enable profiling for $CC])
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
                HAVE_X11=yes
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
              AS_HELP_STRING([--enable-dvs], [enable DVS (default is enable)]), 
	[dvs_req=$enableval])

AC_ARG_WITH(dvs,           
	[  --with-dvs=DIR    specify location of DVS SDK],
	[DVS_INC=$withval/development/header
	 DVS_LIB=$withval/linux/lib],
	[DVS_INC=${PARENT}/sdk4.0.1.17/development/header
	 DVS_LIB=${PARENT}/sdk4.0.1.17/${DVS_SDK_SUBDIR}/lib])

AC_CHECK_HEADER(${DVS_INC}/dvs_clib.h, FOUND_DVS_H=yes, FOUND_DVS_H=no)
AC_CHECK_LIB(dvsoem, sv_open, FOUND_DVS_L=yes, FOUND_DVS_L=no, -L${DVS_LIB})

if test $dvs_req = yes -a $FOUND_DVS_H = yes -a $FOUND_DVS_L = yes
then
	DVS_INC=-I${DVS_INC}
	DVS_LIB=${DVS_LIB}/libdvsoem.a
	DVS_OBJ="src/video_capture/dvs.o src/video_display/dvs.o"
        DVS_UV_LIB="lib/uv_dvs_lib.so"
        AC_DEFINE([HAVE_DVS], [1], [Build with DVS support])
        LIBS+=" -ldl"
        dvs=yes
else
	DVS_INC=
	DVS_LIB=
	DVS_OBJ=
fi

AC_SUBST(DVS_INC)
AC_SUBST(DVS_LIB)
AC_SUBST(DVS_OBJ)
AC_SUBST(DVS_UV_LIB)

# -------------------------------------------------------------------------------------------------
# Blackmagic stuff

DECKLINK_OBJ=
DECKLINK_INC=
DECKLINK_PATH=
decklink=no
decklink_req=yes

AC_ARG_ENABLE(decklink, AS_HELP_STRING([--enable-decklink], [enable DeckLink (default is enable)]), 
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
                        LIBS+=" -framework CoreFoundation"
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
                        LIBS+=" -ldl"
                else
                        FOUND_DECKLINK=no
                fi
                ;;
esac

if test $decklink_req = yes -a $FOUND_DECKLINK = yes
then
    DECKLINK_PATH=${DECKLINK_INC}
    DECKLINK_INC=-I${DECKLINK_INC}
    DECKLINK_OBJ="src/video_capture/DeckLinkAPIDispatch.o src/video_capture/decklink.o src/video_display/decklink.o"
    AC_DEFINE([HAVE_DECKLINK], [1], [Build with DeckLink support])
    LINKER=$CXX
    decklink=yes
else
    DECKLINK_PATH=
    DECKLINK_INC=
    DECKLINK_OBJ=
fi

AC_SUBST(DECKLINK_PATH)
AC_SUBST(DECKLINK_INC)
AC_SUBST(DECKLINK_OBJ)
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
              AS_HELP_STRING([--enable-quad], [enable Linsys Quad (default is enable)]), 
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
# QuickTime Stuff
QUICKTIME_INC=
QUICKTIME_LIB=
QUICKTIME_OBJ=
quicktime=no

AC_ARG_ENABLE(quicktime, 
      AS_HELP_STRING([--enable-quicktime], [enable QuickTime support(default is enable)]), 
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
		fi
                quicktime=yes
    ;;
esac

AC_SUBST(QUICKTIME_INC)
AC_SUBST(QUICKTIME_LIB)
AC_SUBST(QUICKTIME_OBJ)

# -------------------------------------------------------------------------------------------------
# SDL Stuff
SDL_OBJ=
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
                LIBS="$LIBS -L/opt/local/lib -lSDLmain -lSDL -Wl,-framework,Cocoa"
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
                AC_CHECK_LIB(SDL_mixer, Mix_PlayMusic)
                AC_CHECK_LIB(SDL_ttf, TTF_Init)
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
                LIBS="$LIBS -lSDL"
                SDL_OBJ="$SDL_OBJ src/video_display/sdl.o"
                AC_DEFINE([HAVE_SDL], [1],  [Build with SDL support])
                SDL_VERSION=1.2.10
                AM_PATH_SDL($SDL_VERSION, AC_DEFINE([HAVE_SDL_1210], [1], [We have SDL version >= 1.21]), :)
                AC_CHECK_LIB(SDL_mixer, Mix_PlayMusic)
                AC_CHECK_LIB(SDL_ttf, TTF_Init)
                sdl=yes
            fi
        ;;
esac

AC_SUBST(SDL_OBJ)

# -------------------------------------------------------------------------------------------------
# OpenGL stuff

GL_INC=
GL_OBJ=
gl=no

AC_ARG_ENABLE(gl, 
      AS_HELP_STRING([--enable-gl], [enable OpenGL (default is enable)]), 
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
				LIBS="$LIBS -framework GLUT -framework OpenGL -framework Cocoa"
			;;
			*darwin9*)
				LIBS="$LIBS -framework GLUT -framework OpenGL -framework Cocoa -dylib_file /System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib:/System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib"
			;;
			*)
				LIBS="$LIBS -framework GLUT -framework OpenGL -framework Cocoa"
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
			LIBS="$LIBS -lGL"
	
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
				LIBS="$LIBS -lglut"
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

AC_SUBST(GL_INC)
AC_SUBST(GL_OBJ)


# -------------------------------------------------------------------------------------------------
# FastDXT stuff
fastdxt=no
fastdxt_req=yes
COMPRESS_OBJ=

AC_ARG_ENABLE(fastdxt, 
	[  --enable-fastdxt        enable FastDXT],
	[fastdxt_req=$enableval])

AC_ARG_WITH(fastdxt,
	[  --with-fastdxt=DIR      specify location of FastDXT],
	[DXT_INC=$withval
	 DXT_LIB=$withval],
	[DXT_INC=${PARENT}/FastDXT
	 DXT_LIB=${PARENT}/FastDXT])

AC_CHECK_HEADER(${DXT_INC}/libdxt.h, FOUND_DXT_H=yes, FOUND_DXT_H=no)
AC_CHECK_LIB(dxt, main, FOUND_DXT_L=yes, FOUND_DXT_L=no, -I${DXT_INC} -L${DXT_LIB})

if test $fastdxt_req = yes -a $FOUND_DXT_L = yes -a $FOUND_DXT_H = yes
then
	DXT_INC=-I${DXT_INC}
	DXT_LIB="-L${DXT_LIB} -ldxt"
	COMPRESS_OBJ="$COMPRESS_OBJ src/video_compress/fastdxt.o"
        AC_DEFINE([HAVE_FASTDXT], [1], [Build with support for FastDXT])
	LINKER=$CXX
        fastdxt=yes
else
	DXT_INC=
	DXT_LIB=
fi

AC_SUBST(DXT_INC)
AC_SUBST(DXT_LIB)
AC_SUBST(COMPRESS_OBJ)

# -------------------------------------------------------------------------------------------------
# GLSL DXT
SAVED_CFLAGS=$CFLAGS
SAVED_CPPFLAGS=$CPPFLAGS
CFLAGS+=" $X_CFLAGS"
CPPFLAGS+=" $X_CFLAGS"
AC_SUBST(CFLAGS)
SAVED_LIBS=$LIBS
LIBS+=" $X_LIBS"
rtdxt=no

AC_ARG_ENABLE(rtdxt, 
      AS_HELP_STRING([--enable-rtdxt], [enable Real-Time OpenGL DXT compression (default is enable)]), 
	[rtdxt_req=$enableval],
        [rtdxt_req=yes])

AC_CHECK_LIB(GLUT, glutInit)
AC_CHECK_HEADER(GL/glew.h, FOUND_GLEW_H=yes)
AC_CHECK_LIB(GLEW, glewIsSupported, FOUND_GLEW_L=yes)
AC_CHECK_HEADER(GL/glx.h, FOUND_GLX_H=yes)
AC_CHECK_HEADER(GL/gl.h, FOUND_GL_H=yes)
AC_CHECK_LIB(GL, glXCreateNewContext, FOUND_GLX_L=yes)

if test $rtdxt_req = yes -a "$HAVE_X11" = yes -a "$FOUND_GLEW_L" = yes -a "$FOUND_GLEW_H" = yes -a "$FOUND_GLX_L" = yes -a "$FOUND_GLX_H" = yes \
        -a "$FOUND_GL_H" = yes \
        -a `expr "$host_os" : ".*darwin.*"` -eq 0 # not mac, just for now
then
        LIBS+=" -lGLEW -lGL -lX11"
        AC_DEFINE([HAVE_DXT_GLSL], [1], [Build with DXT_GLSL support])
        DXT_GLSL_OBJS="../dxt_compress/dxt_encoder.o ../dxt_compress/dxt_decoder.o  \
../dxt_compress/dxt_common.o ../dxt_compress/dxt_util.o src/video_compress/dxt_glsl.o src/video_decompress/dxt_glsl.o"
        DXT_GLSL_CFLAGS="-std=gnu99"
        rtdxt=yes
else
        CFLAGS=$SAVED_CFLAGS
        CPPFLAGS=$SAVED_CPPFLAGS
        LIBS=$SAVED_LIBS
fi
AC_SUBST(DXT_GLSL_OBJS)
AC_SUBST(DXT_GLSL_CFLAGS)

# -------------------------------------------------------------------------------------------------
# SAGE stuff

SAGE_OBJ=
SAGE_INC=
SAGE_LIB=
sage=no

AC_ARG_ENABLE(sage, 
      AS_HELP_STRING([--enable-sage], [enable SAGE (default is enable)]), 
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
    SAGE_OBJ="$SAGE_OBJ src/video_display/sage.o src/video_display/sage_wrapper.o"
    AC_DEFINE([HAVE_SAGE], [1], [Build with SAGE support])
    LINKER=$CXX
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
PORTAUDIO_OBJ=
portaudio=no

AC_ARG_ENABLE(portaudio, 
      AS_HELP_STRING([--enable-portaudio], [enable Portaudio (default is enable)]), 
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

if test $portaudio_req = yes -a $FOUND_PORTAUDIO_H = yes -a $PORTAUDIO_STATIC = yes
then
    PORTAUDIO_INC=-I${PORTAUDIO_INC}
    # we have now portaudio linked statically
    # TODO: consider linking it dynamically
    PORTAUDIO_LIB="${PORTAUDIO_LIB}/libportaudio.a"
    # libs taken from .pc file won't work - there are missing some optional dependencies, eg. jack
    PORTAUDIO_LIB+=" `( . ../portaudio/lib/libportaudio.la ; echo $inherited_linker_flags $dependency_libs | sed 's/\/usr\/lib\/lib\(.*\)\.la/-l\1/')`"
    CFLAGS+=" `cat  ../portaudio/portaudio-2.0.pc|grep '^Cflags: '|sed 's/^Cflags: -I\${includedir}//'`"
    PORTAUDIO_OBJ="src/audio/portaudio.o"
    AC_DEFINE([HAVE_PORTAUDIO], [1], [Build with Portaudio support])
    portaudio=yes
else
    PORTAUDIO_INC=
    PORTAUDIO_LIB=
    PORTAUDIO_OBJ=
fi

AC_SUBST(PORTAUDIO_INC)
AC_SUBST(PORTAUDIO_LIB)
AC_SUBST(PORTAUDIO_OBJ)

# -------------------------------------------------------------------------------------------------
#
# Jack stuff
#
jack=no

AC_ARG_ENABLE(jack, 
      AS_HELP_STRING([--enable-jack], [enable JACK (default is enable)]), 
	[jack_req=$enableval],
        [jack_req=yes])

AC_CHECK_HEADER(jack/jack.h, FOUND_JACK_H=yes, FOUND_JACK_H=no)
AC_CHECK_LIB(jack, jack_client_new, FOUND_JACK_L=yes, FOUND_JACK_L=no)
if test $jack_req = yes -a $FOUND_JACK_H = yes -a $FOUND_JACK_L = yes
then
    JACK_OBJ="src/audio/jack.o"
    JACK_LIB="-ljack"
    AC_DEFINE([HAVE_JACK], [1], [Build with JACK support])
    jack=yes
fi
AC_SUBST(JACK_OBJ)
AC_SUBST(JACK_LIB)
AC_SUBST(JACK_INC)


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

  DeckLink .................... $decklink
  DVS ......................... $dvs
  Linsys Quad ................. $quad
  OpenGL ...................... $gl
  Quicktime ................... $quicktime
  SAGE ........................ $sage
  SDL ......................... $sdl
  
  Portaudio ................... $portaudio
  Jack transport .............. $jack
  
  FastDXT ..................... $fastdxt
  Realtime DXT (OpenGL) ....... $rtdxt
])

