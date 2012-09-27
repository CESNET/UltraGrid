AC_PREREQ([2.65])
AC_INIT([libgpujpeg], [0.0.1], [martin.srom@mail.muni.cz], [libgpujpeg], [https://sourceforge.net/p/gpujpeg/])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_SRCDIR([src/main.c])
AM_INIT_AUTOMAKE

AC_CONFIG_HEADERS([config.h])

GPUJPEG_MAJOR_VERSION=0
GPUJPEG_MINOR_VERSION=0
GPUJPEG_MICRO_VERSION=1

GPUJPEG_LIBRARY_VERSION=0:1:0

GPUJPEG_API_VERSION=0.1
AC_SUBST(GPUJPEG_API_VERSION)
AC_SUBST(GPUJPEG_LIBRARY_VERSION)

LT_PREREQ([2.2])
LT_INIT

AC_PROG_LIBTOOL
AM_PROG_LIBTOOL

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_INSTALL
AM_PROG_CC_C_O

# debug
AC_ARG_ENABLE(debug, [  --enable-debug             compile with debug info] )
if test x$enable_debug = x ; then
	enable_debug=no
fi

if test x$enable_debug = xyes ; then
	COMMON_CFLAGS="$COMMON_CFLAGS -g -D_DEBUG -O0"
	NVCCFLAGS="$NVCCFLAGS -G"
else
	COMMON_CFLAGS="$COMMON_CFLAGS -O2"
fi

# huffmann
AC_ARG_ENABLE(constant-tables, [  --disable-constant-tables  disable huffman coder tables in constant memory] )
if test x$enable_constant_tables = x ; then
	enable_constant_tables=yes
fi

if test x$enable_constant_tables = xyes ; then
	AC_DEFINE([GPUJPEG_HUFFMAN_CODER_TABLES_IN_CONSTANT], [1], [Build with huffmann coder tables in constant memory])
fi

# opengl
AC_ARG_ENABLE(opengl, [  --enable-opengl            enable opengl support] )
if test x$enable_opengl = x ; then
	enable_opengl=no
fi

if test $enable_opengl = yes ; then
        AC_CHECK_HEADER(GL/glew.h, FOUND_GLEW_H=yes)
        AC_CHECK_LIB(GLEW, glewIsSupported, FOUND_GLEW_L=yes)
        AC_CHECK_LIB(GL, glBindTexture, FOUND_GL_L=yes)

        if test "$FOUND_GLEW_L" = yes -a "$FOUND_GLEW_H" = yes -a "$FOUND_GL_L" = yes; then
                AC_DEFINE([GPUJPEG_USE_OPENGL], [1], [Build with OpenGL support])
                GPUJPEG_LIBS="$GPUJPEG_LIBS -lGLEW -lGL"
        else
                enable_opengl=no
        fi
fi


# CUDA
AC_ARG_WITH(cuda,
        [  --with-cuda=DIR         specify cuda root],
        [CUDA_INSTALL_PATH=$withval]
)

AC_PATH_PROG(CUDA, nvcc, nvcc_not_found, [$PATH$PATH_SEPARATOR$CUDA_INSTALL_PATH/bin]dnl
	[$PATH_SEPARATOR/opt/cuda/bin$PATH_SEPARATOR/usr/local/cuda/bin])

if test x$CUDA = xnvcc_not_found ; then
	AC_MSG_FAILURE([CUDA is required to build the gpujpeg library!])
fi

CUDA=`dirname $CUDA`
CUDA=`dirname $CUDA`
CUDA_INSTALL_PATH=$CUDA

LBITS=`getconf LONG_BIT`
if test $LBITS = 64 -a `uname -s` != 'Darwin'; then
    CUDA_INSTALL_LIB="${CUDA_INSTALL_PATH}/lib64"
else
    CUDA_INSTALL_LIB="${CUDA_INSTALL_PATH}/lib"
fi

NVCC=$CUDA_INSTALL_PATH/bin/nvcc
CUDA_MAJOR=`$NVCC --version |grep release|sed  's/^.*release \(@<:@0-9@:>@@<:@0-9@:>@*\).*$/\1/'`
CUDA_MINOR=`$NVCC --version |grep release|sed  's/^.*release @<:@^.@:>@\.\(@<:@0-9@:>@@<:@0-9@:>@*\).*$/\1/'`

if test $CUDA_MAJOR -gt 4 -o \( $CUDA_MAJOR -ge 4 -a $CUDA_MINOR -ge 2 \)
then
        CUDA_EXTRA_ARCH="$CUDA_EXTRA_ARCH -gencode arch=compute_30,code=sm_30"
fi

AM_CONDITIONAL([DARWIN], [test `uname -s` = Darwin])

if test `uname -s` = 'Darwin'; then
    CFLAGS="$CFLAGS -arch x86_64 -arch i386"
    LIBGPUJPEG_CUDA_OBJS=" \
	build/universal/gpujpeg_huffman_gpu_encoder.o \
        build/universal/gpujpeg_dct_gpu.o \
        build/universal/gpujpeg_preprocessor.o \
        build/universal/gpujpeg_huffman_gpu_decoder.o"
else
    LIBGPUJPEG_CUDA_OBJS=" \
        src/gpujpeg_huffman_gpu_encoder.cu.o \
        src/gpujpeg_dct_gpu.cu.o \
        src/gpujpeg_preprocessor.cu.o \
        src/gpujpeg_huffman_gpu_decoder.cu.o"
fi

GPUJPEG_LDFLAGS="$GPUJPEG_LDFLAGS -L${CUDA_INSTALL_LIB} -lcudart"
COMMON_CFLAGS="$COMMON_CFLAGS -I. -I${CUDA_INSTALL_PATH}/include"

AC_SUBST(CUDA_EXTRA_ARCH)
AC_SUBST(CUDA_INSTALL_PATH)
AC_SUBST(CUDA_INSTALL_LIB)
AC_SUBST(COMMON_CFLAGS)
AC_SUBST(GPUJPEG_CFLAGS)
AC_SUBST(GPUJPEG_LDFLAGS)
AC_SUBST(GPUJPEG_LIBS)
AC_SUBST(LIBGPUJPEG_CUDA_OBJS)

AC_CONFIG_FILES([Makefile libgpujpeg.pc test/memcheck/Makefile test/opengl_interop/Makefile ])
AC_OUTPUT

AC_MSG_RESULT([
Configuration summary:

  Target ...................... $host
  Prefix ...................... $prefix
  Debug ....................... $debug

  Constant tables.............. $enable_constant_tables
  OpenGL ...................... $enable_opengl

  CUDA root ................... $CUDA_INSTALL_PATH
  

])


