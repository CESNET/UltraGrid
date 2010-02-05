#
# check for HOSTTYPE and set HOSTTARGET
#
HOSTTARGET =
ifeq ($(HOSTTYPE),i386)
  HOSTTARGET = linux-x86
endif
ifeq ($(HOSTTYPE),i586)
  HOSTTARGET = linux-x86
endif
ifeq ($(HOSTTYPE),i686)
  HOSTTARGET = linux-x86
endif
ifeq ($(HOSTTYPE),i386-linux)
  HOSTTARGET = linux-x86
endif
ifeq ($(HOSTTYPE),i486-linux)
  HOSTTARGET = linux-x86
endif
ifeq ($(HOSTTYPE),i586-linux)
  HOSTTARGET = linux-x86
endif
ifeq ($(HOSTTYPE),i686-linux)
  HOSTTARGET = linux-x86
endif
ifeq ($(OSTYPE),linux)
  ifeq ($(HOSTTYPE),ia64)
    HOSTTARGET = linux-ia64
  endif
  ifeq ($(HOSTTYPE),x86_64)
    HOSTTARGET = linux-x86_64
  endif
  ifeq ($(HOSTTYPE),x86_64-linux)
    HOSTTARGET = linux-x86_64
  endif
endif

#
# set ROOTCOMPILE directory
#
ifndef ROOTCOMPILE
  ROOTCOMPILE = ../../../$(HOSTTARGET)
endif

#
# set DEBUG/RELEASE flags
#
ifdef RELEASE
  SDKDIR        = release
  SDKLIB        = lib
  DEBUG_FLAGS   =
  RELEASE_FLAGS = -O2
else
  SDKDIR        = debug
  SDKLIB        = lib
  DEBUG_FLAGS   = -ggdb -DDEBUG
  RELEASE_FLAGS =
endif

SDKCOMPILE = $(ROOTCOMPILE)/$(SDKDIR)
DVSLIB     = $(ROOTCOMPILE)/$(SDKLIB)

DVS_AR = ar
DVS_RANLIB = ranlib 

#
# set BASE_CFLAGS
#
DVS_CC       = gcc
BASE_CFLAGS  = $(DEBUG_FLAGS) $(RELEASE_FLAGS)
BASE_LDFLAGS =

LIBDSO = -ldl

CC     = $(DVS_CC)
AR     = $(DVS_AR)
RANLIB = $(DVS_RANLIB) $(BASE_LDFLAGS)


COMPILEFILE = $(CC) $(BASE_CFLAGS) $(CFLAGS) $(DEFS) -c -o $@ $<
ARFILE      = $(AR) -rc
RANLIBFILE  = $(RANLIB)

DVSHEADER =	../../header/dvs_clib.h		\
		../../header/dvs_fm.h		\
		../../header/dvs_errors.h	\
		../../header/dvs_setup.h	\
		../../header/dvs_version.h


all:	$(ROOTCOMPILE) $(ROOTCOMPILE)/$(SDKDIR) $(DESTDIR) compile

$(ROOTCOMPILE):
	mkdir $(ROOTCOMPILE)

$(ROOTCOMPILE)/$(SDKDIR):
	mkdir $(ROOTCOMPILE)/$(SDKDIR)

$(DESTDIR):
	mkdir $(DESTDIR)

todos::
	todos	*.c *.cpp *.h *.mak *.bat *.dsw *.dsp *.rc

fromdos::
	fromdos	*.c *.cpp *.h *.mak *.bat *.dsw *.dsp *.rc
	@if test -d cvs ; then 			\
		mv cvs CVS			; \
		mv CVS/entries CVS/Entries	; \
		mv CVS/repository CVS/Repository; \
		mv CVS/root CVS/Root		; \
	fi
