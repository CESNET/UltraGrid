dnl Process this file with autoconf to produce a configure script.

AC_INIT
AC_CONFIG_SRCDIR([src/main.c])

PARENT=`echo $PWD | sed -e 's%/[[^/]]*$%%'`

AC_CANONICAL_HOST
AC_PROG_CC
AC_PROG_CPP
AC_PROG_CXX
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

case "$host_os" in
	*darwin8*)
		AC_DEFINE([HAVE_MACOSX], [1], [This is Mac X OS])
		LINKER=$CXX
	;;
	*darwin9*)
		AC_DEFINE([HAVE_MACOSX], [1], [This is Mac X OS])
		AC_DEFINE([HAVE_MACOSX_LEOPARD], [1], [This is Mac X OS Leopard])
		LINKER=$CXX
	;;
	*)
		LINKER=$CC
	;;
esac

case "$host" in
  	x86_64-*-linux-*)
		DVS_SDK_SUBDIR="linux-x86_64"
		AC_DEFINE([HAVE_64B_LINUX], [1], [We have 64-bit Linux])
	;;
	i*86-*-linux-*)
		DVS_SDK_SUBDIR="linux-x86"
		AC_DEFINE([HAVE_32B_LINUX], [1], [We have 32-bit Linux])
	;;
esac	

AC_SUBST(LINKER)

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
AC_ARG_ENABLE(profile,
	[  --enable-profile        enable profiling],
	[if test $enableval = yes
	then
		if test "$GCC" = yes
		then
			CFLAGS="$CFLAGS -pg"
		else
			AC_MSG_ERROR([Don't know how to enable profiling for $CC])
		fi
	fi])

AC_ARG_ENABLE(debug,
	[  --enable-debug          enable debug messages and code],
	[if test $enableval = yes
	then
		AC_DEFINE([DEBUG], [1], [We build with debug messages])
	fi])

AC_ARG_ENABLE(ipv6,
	[  --enable-ipv6           enable ipv6],
	[if test $enableval = yes
	then
		AC_DEFINE([HAVE_IPv6], [1], [Enable IPv6 support])
	fi])

# -------------------------------------------------------------------------------------------------
# X Window System
#
# Use autoconf inbuilt X location.  Works v. nicely.  Substitution of X vars
# comes after broken X11 header check and attempted fix.
X_LIBS=
X_OBJ=

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
        X_LIBS+=" -lX11 -lXext"
        CFLAGS+=""

fi

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
# Try to find the SDK for the DVS HDstation card...
AC_ARG_ENABLE(hdstation,
	[  --enable-hdstation      enable HDStation or Centaurus cards],
	[if test $enableval = yes
	then
		AC_DEFINE([HAVE_HDSTATION], [1], [Build with HDStation support])
	fi])

AC_ARG_WITH(hdstation,           
	[  --with-hdstation=DIR    specify location of DVS HDstation SDK],
	[DVS_INC=$withval/development/header
	 DVS_LIB=$withval/linux/lib],
	[DVS_INC=${PARENT}/sdk3.2.0.0/development/header
	 DVS_LIB=${PARENT}/sdk3.2.0.0/${DVS_SDK_SUBDIR}/lib])

AC_CHECK_HEADER(${DVS_INC}/dvs_clib.h, FOUND_DVS=yes, FOUND_DVS=no)
AC_CHECK_LIB(dvsoem, sv_open, FOUND_DVS=yes, FOUND_DVS=no, -L${DVS_LIB})

if test $FOUND_DVS = yes
then
	DVS_INC=-I${DVS_INC}
	DVS_LIB=${DVS_LIB}/libdvsoem.a
	DVS_OBJ="src/video_capture/hdstation.o src/video_display/hdstation.o"
        AC_DEFINE([HAVE_HDSTATION], [1], [Build with HDStation support])
else
	DVS_INC=
	DVS_LIB=
	DVS_OBJ=
fi

AC_SUBST(DVS_INC)
AC_SUBST(DVS_LIB)
AC_SUBST(DVS_OBJ)

# -------------------------------------------------------------------------------------------------
# Blackmagic stuff

DECKLINK_OBJ=
DECKLINK_INC=
DECKLINK_PATH=

AC_ARG_ENABLE(decklink, 
	[  --enable-decklink       enable Blackmagic DeckLink support],
	[if test $enableval = yes
	then
		AC_DEFINE([HAVE_DECKLINK], [1], [Build with DeckLink support])
	fi])

AC_ARG_WITH(decklink,
	[  --with-decklink=DIR     specify location of DECKLINK],
	[DECKLINK_INC=$withval/include
	],
	[DECKLINK_INC=${PARENT}/DeckLink/include
	])

AC_CHECK_FILE(${DECKLINK_INC}/DeckLinkAPI.h, FOUND_DECKLINK=yes, FOUND_DECKLINK=no)
AC_CHECK_FILE(${DECKLINK_INC}/DeckLinkAPIDispatch.cpp, FOUND_DECKLINK=yes, FOUND_DECKLINK=no)
AC_CHECK_LIB(DeckLinkAPI, main, FOUND_DECKLINK=yes, FOUND_DECKLINK=no)


if test $FOUND_DECKLINK = yes
then
    DECKLINK_PATH=${DECKLINK_INC}
    DECKLINK_INC=-I${DECKLINK_INC}
    DECKLINK_OBJ="src/video_capture/DeckLinkAPIDispatch.o src/video_capture/decklink.o src/video_display/decklink.o"
    AC_DEFINE([HAVE_DECKLINK], [1], [Build with DeckLink support])
    LINKER=$CXX
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

case $host in
    *-linux-*)

        AC_ARG_ENABLE(quad, 
            [  --enable-quad           enable HD-SDI Master Quad support],
            [if test $enableval = yes
            then
                AC_DEFINE([HAVE_QUAD], [1], [Build with Quad support])
            fi])

        AC_ARG_WITH(decklink,
            [  --with-quad=DIR         specify location of QUAD],
            [QUAD_PATH=$withval
            ],
            [QUAD_PATH=${PARENT}/quad
            ])

        QUAD_INC=${QUAD_PATH}/include

        AC_CHECK_FILE(${QUAD_INC}/master.h, FOUND_QUAD=yes, FOUND_QUAD=no)
        AC_CHECK_FILE(${QUAD_INC}/sdi.h, FOUND_QUAD=yes, FOUND_QUAD=no)

        if test $FOUND_QUAD = yes
        then
            QUAD_INC="-I$QUAD_INC -I$QUAD_PATH/Examples"
            QUAD_OBJ="$QUAD_PATH/Examples/util.o src/video_capture/quad.o"
            AC_DEFINE([HAVE_QUAD], [1], [Build with Quad support])
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

case $host_os in
	*darwin*)
		AC_CHECK_HEADERS([QuickTime/QuickTime.h], [FOUND_QUICKTIME_H=yes], [echo "QuickTime headers not found."])
		AC_CHECK_HEADERS([Carbon/Carbon.h], [FOUND_CARBON_H=yes], [echo "Carbon headers not found."])
	
		if test $FOUND_QUICKTIME_H = yes -a $FOUND_QUICKTIME_H = yes
		then
			QUICKTIME_INC=
			QUICKTIME_LIB="-framework QuickTime -framework Carbon -framework AppKit"	
			QUICKTIME_OBJ="src/video_capture/quicktime.o src/video_display/quicktime.o"
		fi
    ;;
esac

AC_SUBST(QUICKTIME_INC)
AC_SUBST(QUICKTIME_LIB)
AC_SUBST(QUICKTIME_OBJ)

# -------------------------------------------------------------------------------------------------
# SDL Stuff
SDL_OBJ=

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
        AC_CHECK_HEADERS([SDL/SDL.h],
                         [AC_CHECK_LIB(SDL, main, FOUND_SDL_L=yes, echo "SDL library not found or not usable.", -L/opt/local/lib)],
                         [echo "SDL.h not found."]
                        )
            if test $FOUND_SDL_L = yes
            then
                CFLAGS="$CFLAGS -D_GNU_SOURCE=1 -D_THREAD_SAFE -I/opt/local/include/SDL"
                LIBS="$LIBS -L/opt/local/lib -lSDLmain -lSDL -Wl,-framework,Cocoa"
                SDL_OBJ="$SDL_OBJ src/video_display/sdl.o"
                AC_DEFINE([HAVE_SDL], [1],  [Build with SDL support])
                SDL_VERSION_REQUIRED=1.2.10
                AC_CHECK_FILE(/opt/local/bin/sdl-config, FOUND_SDL_CONFIG=yes, echo "sdl-config not found.")
                if test $FOUND_SDL_CONFIG = yes
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
            fi
        ;;
        *)
        AC_CHECK_HEADERS([SDL/SDL.h],
                         [AC_CHECK_LIB(SDL, main, FOUND_SDL_L=yes, echo "SDL library not found or not usable.")],
                         [echo "SDL.h not found."]
                        )
            if test $FOUND_SDL_L = yes
            then
                LIBS="$LIBS -lSDL"
                SDL_OBJ="$SDL_OBJ src/video_display/sdl.o"
                AC_DEFINE([HAVE_SDL], [1],  [Build with SDL support])
                SDL_VERSION=1.2.10
                AM_PATH_SDL($SDL_VERSION,
                        AC_DEFINE([HAVE_SDL_1210], [1], [We have SDL version >= 1.21]),
                        :)
            fi
        ;;
esac

AC_SUBST(SDL_OBJ)

# -------------------------------------------------------------------------------------------------
# OpenGL stuff

GL_INC=
GL_OBJ=

case "$host_os" in
	*darwin*)
	AC_CHECK_HEADERS([OpenGL/gl.h OpenGL/glu.h],
			 [FOUND_GL_H=yes],
			 [FOUND_GL_H=no],
			)

	AC_CHECK_HEADERS([OpenGL/glext.h],
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

		if test $FOUND_GL_L = yes
		then
			case "$host_os" in
			*darwin8*)
				LIBS="$LIBS -framework GLUT -framework OpenGL -framework Cocoa"
			;;
			*darwin9*)
				LIBS="$LIBS -framework GLUT -framework OpenGL -framework Cocoa -dylib_file /System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib:/System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib"
			;;
			esac

                        FOUND_GLUT=no
			FOUND_GLEW_L=no
			AC_CHECK_HEADERS([/opt/local/include/GL/glew.h],
					 [AC_CHECK_LIB(GLEW, main, FOUND_GLEW_L=yes , echo "GLEW library not found or not usable.", -L/opt/local/lib)],
					 [echo "GL/glew.h not found."]
					)
			AC_CHECK_HEADERS([GLUT/glut.h],
					 [AC_CHECK_FILE(/System/Library/Frameworks/GLUT.framework/GLUT, FOUND_GLUT=yes, echo "GLUT framework not found.")],
					 [echo "GLUT/glut.h not found."]
					)
			if test $FOUND_GLEW_L = yes -a $FOUND_GLUT = yes
			then
				LIBS="$LIBS /opt/local/lib/libGLEW.a"
				GL_INC="$GL_INC -I/opt/local/include"
				GL_OBJ="$GL_OBJ src/video_display/gl.o"
				AC_DEFINE([HAVE_GL], [1], [Build with OpenGL output])
			else
				echo "OpenGL (GLEW/GLUT) libraries not found or not usable."
			fi
		else
			echo "OpenGL libraries not found or not usable."
		fi

	else
		echo "OpenGL headers not found or not usable."
	fi
	;;
	*) 
	AC_CHECK_HEADERS([GL/gl.h GL/glu.h],
			 [FOUND_GL_H=yes],
			 [FOUND_GL_H=no],
			)

	AC_CHECK_HEADERS([GL/glext.h],
			 [FOUND_GL_H=yes],
			 [FOUND_GL_H=no], 
			 [
		 	  #include <GL/gl.h>
			  #include <GL/glext.h>
			 ]
			)

	if test $FOUND_GL_H = yes
	then
		AC_CHECK_LIB(GL, glXChooseVisual, FOUND_GL_L=yes, FOUND_GL_L=no)
		AC_CHECK_LIB(GLU, main, FOUND_GL_L=yes, FOUND_GL_L=no)

		if test $FOUND_GL_L = yes
		then
			LIBS="$LIBS -lGL -lGLU"
	
			FOUND_GLEW_L=no
			FOUND_GLUT_L=yes
			AC_CHECK_HEADERS([GL/glew.h],
					 [AC_CHECK_LIB(GLEW, main, FOUND_GLEW_L=yes , echo "GLEW library not found or not usable.")],
					 [echo "GL/glew.h not found."]
					)
			AC_CHECK_HEADERS([GL/glut.h],
					 [AC_CHECK_LIB(glut, glutInit, FOUND_GLUT_L=yes , echo "GLUT library not found or not usable.")],
					 [echo "GL/glut.h not found."]
					)
			if test $FOUND_GLEW_L = yes -a $FOUND_GLUT_L = yes
			then
				LIBS="$LIBS -lGLEW -lglut"
				GL_OBJ="$GL_OBJ src/video_display/gl.o"
				AC_DEFINE([HAVE_GL], [1], [Build with OpenGL output])
			else
				echo "OpenGL (GLEW/GLUT) libraries not found or not usable."
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

DXT_OBJ=
COMPRESS_OBJ=

AC_ARG_ENABLE(fastdxt, 
	[  --enable-fastdxt        enable FastDXT],
	[if test $enableval = yes
	then
		AC_DEFINE([HAVE_FASTDXT], [1], [Build with support for FastDXT])
	fi])

AC_ARG_WITH(fastdxt,
	[  --with-fastdxt=DIR      specify location of FastDXT],
	[DXT_INC=$withval
	 DXT_LIB=$withval],
	[DXT_INC=${PARENT}/FastDXT
	 DXT_LIB=${PARENT}/FastDXT])

AC_CHECK_HEADER(${DXT_INC}/libdxt.h, FOUND_DXT=yes, FOUND_DXT=no)
AC_CHECK_LIB(dxt, main, FOUND_DXT=yes, FOUND_DXT=no, -I${DXT_INC} -L${DXT_LIB})

if test $FOUND_DXT = yes
then
	DXT_INC=-I${DXT_INC}
	DXT_LIB="-L${DXT_LIB} -ldxt"
	DXT_OBJ="$DXT_OBJ"
	COMPRESS_OBJ="$COMPRESS_OBJ src/video_compress.o"
        AC_DEFINE([HAVE_FASTDXT], [1], [Build with support for FastDXT])
else
	DXT_INC=
	DXT_LIB=
fi

AC_SUBST(DXT_INC)
AC_SUBST(DXT_LIB)
AC_SUBST(DXT_OBJ)
AC_SUBST(COMPRESS_OBJ)

# -------------------------------------------------------------------------------------------------
# SAGE stuff

SAGE_OBJ=
SAGE_INC=
SAGE_LIB=

AC_ARG_ENABLE(sage, 
	[  --enable-sage           enable SAGE display support],
	[if test $enableval = yes
	then
		AC_DEFINE([HAVE_SAGE], [1], [Build with SAGE support])
	fi])

AC_ARG_WITH(sage,
	[  --with-sage=DIR         specify location of SAGE],
	[SAGE_INC=$withval/include
	 SAGE_LIB=$withval/lib],
	[SAGE_INC=${PARENT}/sage3.0/include
	 SAGE_LIB=${PARENT}/sage3.0/lib])

AC_CHECK_FILE(${SAGE_INC}/sail.h, FOUND_SAGE=yes, FOUND_SAGE=no)
AC_CHECK_LIB(sail, main, FOUND_SAGE=yes, FOUND_SAGE=no, -I${SAGE_INC} -L${SAGE_LIB})

if test $FOUND_SAGE = yes
then
    SAGE_INC=-I${SAGE_INC}
    SAGE_LIB=-"L${SAGE_LIB} -lsail"
    SAGE_OBJ="$SAGE_OBJ src/video_display/sage.o src/video_display/sage_wrapper.o"
    AC_DEFINE([HAVE_SAGE], [1], [Build with SAGE support])
    LINKER=$CXX
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

AUDIO_INC=
AUDIO_LIB=
AUDIO_OBJ=

AC_ARG_ENABLE(portaudio, 
	[  --enable-portaudio      enable Portaudio framework support],
	[if test $enableval = yes
	then
                AC_DEFINE([HAVE_AUDIO], [1], [Build with Portaudio support])
	fi])

AC_ARG_WITH(portaudio,
	[  --with-portaudio=DIR    specify location of Portaudio],
	[AUDIO_INC=$withval/include
	 AUDIO_LIB=$withval/lib],
	[AUDIO_INC=${PARENT}/portaudio/include
	 AUDIO_LIB=${PARENT}/portaudio/lib/.libs])

AC_CHECK_FILE(${AUDIO_INC}/portaudio.h, FOUND_AUDIO=yes, FOUND_AUDIO=no)
AC_CHECK_LIB(portaudio, main, FOUND_AUDIO=yes, FOUND_AUDIO=no, -I${AUDIO_INC} -L${AUDIO_LIB})

if test $FOUND_AUDIO = yes
then
    AUDIO_INC=-I${AUDIO_INC}
case "$host_os" in
        *darwin*)
                AUDIO_LIB="${AUDIO_LIB}/libportaudio.a -framework CoreAudio -framework AudioToolbox -framework AudioUnit -framework CoreServices"
        ;;
        *)
                AUDIO_LIB="${AUDIO_LIB}/libportaudio.a"
        ;;
esac
    AUDIO_OBJ="$AUDIO_OBJ src/audio/audio.o"
    AC_DEFINE([HAVE_AUDIO], [1], [Build with Portaudio support])
else
    AUDIO_INC=
    AUDIO_LIB=
    AUDIO_OBJ=
fi

AC_SUBST(AUDIO_INC)
AC_SUBST(AUDIO_LIB)
AC_SUBST(AUDIO_OBJ)

# -------------------------------------------------------------------------------------------------
# Finally, substitute things into the Makefile and config.h
AC_CONFIG_HEADER(src/config.h)
AC_CONFIG_FILES([Makefile])
AC_OUTPUT

