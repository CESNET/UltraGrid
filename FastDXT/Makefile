CC=g++
CXX=g++
LDFLAGS=-O3
LDLIBS=-lpthread
CXXFLAGS=-O3 -funroll-loops -fomit-frame-pointer -fPIC

GLEWLIB=-lGLEW
SDLLIBS=`sdl-config --libs`

OPTIM=1

MACHINE=$(shell uname -s)
ARCHITECTURE=$(shell uname -m)

ifeq ($(MACHINE), Darwin)

CXXFLAGS+=-I/opt/local/include -arch i386 -arch x86_64
LDLIBS+=-framework OpenGL -arch i386 -arch x86_64
GLEWLIB=/opt/local/lib/libGLEW.a
SDLLIBS=-framework SDL -framework CoreFoundation -framework Carbon

ifeq ($(ARCHITECTURE), i386)
	# Intel Mac

ifeq ($(OPTIM), 1)
# MacPro
CXXFLAGS+= -msse3 -DDXT_INTR
# *.a 
# MacbookPro
# CXXFLAGS+=-march=i686 -msse2 -DDXT_INTR

else

#
CXXFLAGS+=
CXXFLAGS=-g
endif

endif

ifeq ($(ARCHITECTURE), powerpc)
	# PowerPC Mac
echo "Disabling SSE instructions on PowerPC"
CXXFLAGS+=

endif

else

LDLIBS+=-lGL

ifeq ($(ARCHITECTURE), x86_64)
	# Opteron
CXXFLAGS+=-march=opteron -msse2 

else

	# everything considered i386/linux
CXXFLAGS+=-msse2 

endif

endif

CFLAGS=$(CXXFLAGS) `sdl-config --cflags` -I/opt/local/include


# We need just libdxt for ultragrid. Also, there are still some unresolved
# issues building viewdxt and playdxt on some systems
#
# default: main example 2dxt lib viewdxt playdxt
default: main example 2dxt lib

main: dxt.o main.o util.o intrinsic.o
example: example.o libdxt.o dxt.o util.o intrinsic.o
lib: libdxt.o dxt.o intrinsic.o
	ar rcu libdxt.a libdxt.o dxt.o intrinsic.o
	ranlib libdxt.a
2dxt: dxt.o 2dxt.o libdxt.o util.o intrinsic.o


viewdxt:viewdxt.o glsl.o dxt.o
	${CC} -o viewdxt viewdxt.o glsl.o dxt.o intrinsic.o $(SDLLIBS) $(LDLIBS) $(GLEWLIB)

playdxt:playdxt.o glsl.o dxt.o
	${CC} -o playdxt playdxt.o glsl.o dxt.o intrinsic.o $(SDLLIBS) $(LDLIBS) $(GLEWLIB)

install: default
	/bin/cp -f main example 2dxt playdxt viewdxt bin

clean:
	/bin/rm -f *.a *.o *~ main example 2dxt playdxt viewdxt *.rgba *.dxt out.*

distclean: clean
