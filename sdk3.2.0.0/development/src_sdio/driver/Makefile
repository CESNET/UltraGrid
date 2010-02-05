#	Copyright (c) 1996-2008 DVS Digital Video Systems AG
#
#	DVSDriver:  Linux post-compile step makefile
#
#	usage: make		- builds both, release and debug target
#	       make RELEASE=1	- builds release target
#	       make DEBUG=1	- builds debug target
#

ifndef ROOTCOMPILE
  ROOTCOMPILE	= /tmp/compile
endif

ifndef UNAME
  UNAME = `uname -r`
endif

ifndef OSTYPE
  OSTYPE = $(shell uname -o)
endif

ifndef HOSTTYPE
  HOSTTYPE = $(shell uname -m)
endif

#
# check for HOSTTYPE and set HOSTTARGET
#
HOSTTARGET=linux-i686
ifeq ($(HOSTTYPE),i386)
  HOSTTARGET=linux-i686
endif
ifeq ($(HOSTTYPE),i586)
  HOSTTARGET=linux-i686
endif
ifeq ($(HOSTTYPE),i686)
  HOSTTARGET=linux-i686
endif
ifeq ($(HOSTTYPE),i386-linux)
  HOSTTARGET=linux-i686
endif
ifeq ($(HOSTTYPE),i486-linux)
  HOSTTARGET=linux-i686
endif
ifeq ($(HOSTTYPE),i586-linux)
  HOSTTARGET=linux-i686
endif
ifeq ($(HOSTTYPE),i686-linux)
  HOSTTARGET=linux-i686
endif
ifeq ($(HOSTTYPE),alpha)
  HOSTTARGET=linux-alpha
endif
ifeq ($(OSTYPE),linux)
  ifeq ($(HOSTTYPE),ia64)
    HOSTTARGET=linux-ia64
  endif
  ifeq ($(HOSTTYPE),x86_64)
    HOSTTARGET=linux-x86_64
  endif
  ifeq ($(HOSTTYPE),x86_64-linux)
    HOSTTARGET=linux-x86_64
  endif
endif
ifeq ($(OSTYPE),linux-gnu)
  ifeq ($(HOSTTYPE),ia64)
    HOSTTARGET=linux-ia64
  endif
  ifeq ($(HOSTTYPE),x86_64)
    HOSTTARGET=linux-x86_64
  endif
  ifeq ($(HOSTTYPE),x86_64-linux)
    HOSTTARGET=linux-x86_64
  endif
endif
ifeq ($(OSTYPE),GNU/Linux)
  ifeq ($(HOSTTYPE),ia64)
    HOSTTARGET=linux-ia64
  endif
  ifeq ($(HOSTTYPE),x86_64)
    HOSTTARGET=linux-x86_64
  endif
  ifeq ($(HOSTTYPE),x86_64-linux)
    HOSTTARGET=linux-x86_64
  endif
endif
ifeq ($(HOSTTYPE),sun4)
  HOSTTARGET=solaris
endif
ifeq ($(HOSTTYPE),i86pc)
  HOSTTARGET=solaris
endif
ifeq ($(HOSTTYPE),sunos7)
  HOSTTARGET=solaris
endif
ifeq ($(HOSTTYPE),sunos8)
  HOSTTARGET=solaris
endif
ifeq ($(HOSTTYPE),irix)
  HOSTTARGET=irix
endif
ifeq ($(HOSTTYPE),iris4d)
  HOSTTARGET=irix
endif
ifeq ($(HOSTTYPE),macintosh)
  HOSTTARGET=macintosh
endif

ifeq ($(HOSTTARGET),linux-ia64)
BASE_CFLAGS     = -D__KERNEL__ -DHAVE_WORKING_TEXT_ALIGN -Wall -Wstrict-prototypes -Wno-trigraphs -fno-strict-aliasing -fno-common -pipe -ffixed-r13 -mfixed-range=f12-f15,f32-f127 -falign-functions=32 -frename-registers -O2 -fomit-frame-pointer -DDRIVER -DDVS_LINUX_POSTCOMPILE -I/lib/modules/$(UNAME)/build/include -I/lib/modules/$(UNAME)/source/include -DMODULE

CFLAGS		= $(BASE_CFLAGS) -DCOMPILE_SDIO
LDFLAGS         = -r -T /lib/modules/$(UNAME)/source/arch/ia64/module.lds
else
  ifeq ($(HOSTTARGET),linux-x86_64)
    BASE_CFLAGS = -O -Wall -Wstrict-prototypes -Wno-trigraphs -fno-strict-aliasing -fno-common  -mno-red-zone -mcmodel=kernel -pipe -fno-reorder-blocks       -Wno-sign-compare  -fomit-frame-pointer -g -D__KERNEL__ -DDRIVER -DDVS_LINUX_POSTCOMPILE -I/lib/modules/$(UNAME)/build/include -I/lib/modules/$(UNAME)/source/include -DMODULE

    CFLAGS	= $(BASE_CFLAGS) -DCOMPILE_SDIO
    LDFLAGS     = -r -m elf_x86_64
  else
    BASE_CFLAGS	= -O -Wall -D__KERNEL__ -DDRIVER -DDVS_LINUX_POSTCOMPILE -I/lib/modules/$(UNAME)/build/include -I/lib/modules/$(UNAME)/source/include -I/lib/modules/$(UNAME)/build/include/asm-i386/mach-default -I/lib/modules/$(UNAME)/source/include/asm-i386/mach-default -DMODULE

    ifeq ($(COMPILE_REGPARM),1)
        CFLAGS      = -mpreferred-stack-boundary=2 -march=i686 -mregparm=3 -DCOMPILE_REGPARM $(BASE_CFLAGS) -DCOMPILE_SDIO
    else
        CFLAGS	= -march=i686 $(BASE_CFLAGS) -DCOMPILE_SDIO
    endif
    LDFLAGS     = -r -m elf_i386
  endif
endif

CC		= gcc

ifndef RELEASE
  ifndef DEBUG
    SDKBUILD	= 1
  endif
endif

SUBDIR =
ifdef RELEASE
SUBDIR = release
endif
ifdef DEBUG
SUBDIR = debug
endif

ifeq ($(SDKBUILD), 1)
  ifeq ($(HOSTTARGET),linux-i686)
    ROOTCOMPILE	= ../../../linux-x86/driver
  endif  
  ifeq ($(HOSTTARGET),linux-x86_64)
    ROOTCOMPILE	= ../../../linux-x86_64/driver
  endif  
  ifeq ($(HOSTTARGET),linux-ia64)
    ROOTCOMPILE	= ../../../linux-ia64/driver
  endif  
  TARGETDIR = $(ROOTCOMPILE)
  DESTDIR	= .
  TARGET	= dvsdriver_sdio.ko dvsdebug_sdio.ko
  OBJS		= $(DESTDIR)/linux.o
  OBJS_D	= $(DESTDIR)/linux_d.o
  SOURCEBIN	= $(TARGETDIR)/dvsdriver_sdio.pre
  SOURCEBIN_D	= $(TARGETDIR)/dvsdebug_sdio.pre
  TARGETFILE	= $(TARGETDIR)/dvsdriver_sdio.post
  TARGETFILE_D	= $(TARGETDIR)/dvsdebug_sdio.post
  COMPILEFILE	= $(CC) $(CFLAGS) -c -o $@ $<
  COMPILEFILE_D	= $(CC) $(CFLAGS) -D_DEBUG -DCONFIG_MODULE_FORCE_UNLOAD -c -o $@ $<
  LINKTARGET	= $(CC) -r -d -nostdlib -o $(TARGETFILE) $(OBJS) $(SOURCEBIN)
  LINKTARGET_D	= $(CC) -r -d -nostdlib -o $(TARGETFILE_D) $(OBJS_D) $(SOURCEBIN_D)
else
  TARGETDIR = $(ROOTCOMPILE)/$(HOSTTARGET)/$(SUBDIR)
  ifdef RELEASE
    COMPILEFILE	= $(CC) $(CFLAGS) -c -o $@ $<
  endif
  ifdef DEBUG
    COMPILEFILE	= $(CC) $(CFLAGS) -D_DEBUG -DCONFIG_MODULE_FORCE_UNLOAD -c -o $@ $<
  endif
  DESTDIR	= $(TARGETDIR)/.objdir.dvsdriver_sdio
  TARGET	= dvsdriver_sdio.ko
  OBJS		= $(DESTDIR)/linux.o
  SOURCEBIN	= $(TARGETDIR)/dvsdriver_sdio.pre
  TARGETFILE	= $(TARGETDIR)/dvsdriver_sdio.post
  LINKTARGET	= $(CC) -r -d -nostdlib -o $(TARGETFILE) $(OBJS) $(SOURCEBIN)
endif

HEADER		= ../../header/dvs_errors.h	\
		  ../../header/dvs_setup.h	\
		  ps_common.h

all::	$(TARGET)

$(DESTDIR)/linux.o:	linux.c $(HEADER)
	(test -f /lib/modules/$(UNAME)/build/include/linux/rhconfig.h) && $(COMPILEFILE) -DDVS_RH_COMPILE || true
	(test -f /lib/modules/$(UNAME)/build/include/linux/rhconfig.h) || $(COMPILEFILE)

$(DESTDIR)/linux_d.o:		linux.c $(HEADER)
	(test -f /lib/modules/$(UNAME)/build/include/linux/rhconfig.h) && $(COMPILEFILE_D) -DDVS_RH_COMPILE || true
	(test -f /lib/modules/$(UNAME)/build/include/linux/rhconfig.h) || $(COMPILEFILE_D)

$(TARGETFILE):		$(TARGETDIR) $(OBJS) $(SOURCEBIN)
	$(LINKTARGET)

$(TARGETFILE_D):	$(TARGETDIR) $(OBJS_D) $(SOURCEBIN_D)
	$(LINKTARGET_D)

$(TARGETDIR)/dvsdriver_sdio.post.mod.o:	$(TARGETFILE)
	@if test -f /lib/modules/$(UNAME)/build/scripts/modpost ; then \
		/lib/modules/$(UNAME)/build/scripts/modpost $(TARGETFILE) ; \
	fi
	@if test -f /lib/modules/$(UNAME)/build/scripts/mod/modpost.c -a ! -f /lib/modules/$(UNAME)/build/scripts/mod/modpost ; then \
		echo "ERROR: /lib/modules/$(UNAME)/build/scripts/mod/modpost is not compiled." ; \
		false ; \
	fi
	@if test -f /lib/modules/$(UNAME)/source/scripts/mod/modpost.c -a ! -f /lib/modules/$(UNAME)/source/scripts/mod/modpost ; then \
		echo "ERROR: /lib/modules/$(UNAME)/source/scripts/mod/modpost is not compiled." ; \
		false ; \
	fi
	@if test -f /lib/modules/$(UNAME)/build/scripts/mod/modpost ; then \
		/lib/modules/$(UNAME)/build/scripts/mod/modpost $(TARGETFILE) ; \
	fi
	@if test -f /lib/modules/$(UNAME)/source/scripts/mod/modpost ; then \
		/lib/modules/$(UNAME)/source/scripts/mod/modpost $(TARGETFILE) ; \
	fi
	@touch $(TARGETDIR)/dvsdriver_sdio.post.mod.c
	@rm -f $(TARGETDIR)/dvsdriver_sdio.post.mod.c.tmp
	@echo "#include \"linux.h\"" > $(TARGETDIR)/dvsdriver_sdio.post.mod.c.tmp
	@cat $(TARGETDIR)/dvsdriver_sdio.post.mod.c >> $(TARGETDIR)/dvsdriver_sdio.post.mod.c.tmp
	@cp $(TARGETDIR)/dvsdriver_sdio.post.mod.c.tmp $(TARGETDIR)/dvsdriver_sdio.post.mod.c
	@rm -f $(TARGETDIR)/dvsdriver_sdio.post.mod.c.tmp
	@$(CC) $(CFLAGS) -I. -c -o $(TARGETDIR)/dvsdriver_sdio.post.mod.o $(TARGETDIR)/dvsdriver_sdio.post.mod.c

$(TARGETDIR)/dvsdebug_sdio.post.mod.o:	$(TARGETFILE_D)
	@if test -f /lib/modules/$(UNAME)/build/scripts/modpost ; then \
		/lib/modules/$(UNAME)/build/scripts/modpost $(TARGETFILE_D) ; \
	fi
	@if test -f /lib/modules/$(UNAME)/build/scripts/mod/modpost.c -a ! -f /lib/modules/$(UNAME)/build/scripts/mod/modpost ; then \
		echo "ERROR: /lib/modules/$(UNAME)/build/scripts/mod/modpost is not compiled." ; \
		false ; \
	fi
	@if test -f /lib/modules/$(UNAME)/source/scripts/mod/modpost.c -a ! -f /lib/modules/$(UNAME)/source/scripts/mod/modpost ; then \
		echo "ERROR: /lib/modules/$(UNAME)/source/scripts/mod/modpost is not compiled." ; \
		false ; \
	fi
	@if test -f /lib/modules/$(UNAME)/build/scripts/mod/modpost ; then \
		/lib/modules/$(UNAME)/build/scripts/mod/modpost $(TARGETFILE_D) ; \
	fi
	@if test -f /lib/modules/$(UNAME)/source/scripts/mod/modpost ; then \
		/lib/modules/$(UNAME)/source/scripts/mod/modpost $(TARGETFILE_D) ; \
	fi
	@touch $(TARGETDIR)/dvsdebug_sdio.post.mod.c
	@rm -f $(TARGETDIR)/dvsdebug_sdio.post.mod.c.tmp
	@echo "#include \"linux.h\"" > $(TARGETDIR)/dvsdebug_sdio.post.mod.c.tmp
	@cat $(TARGETDIR)/dvsdebug_sdio.post.mod.c >> $(TARGETDIR)/dvsdebug_sdio.post.mod.c.tmp
	@cp $(TARGETDIR)/dvsdebug_sdio.post.mod.c.tmp $(TARGETDIR)/dvsdebug_sdio.post.mod.c
	@rm -f $(TARGETDIR)/dvsdebug_sdio.post.mod.c.tmp
	@$(CC) $(CFLAGS) -I. -c -o $(TARGETDIR)/dvsdebug_sdio.post.mod.o $(TARGETDIR)/dvsdebug_sdio.post.mod.c

dvsdriver_sdio.ko:	$(TARGETFILE) $(TARGETDIR)/dvsdriver_sdio.post.mod.o
	ld $(LDFLAGS) -o $(TARGETDIR)/dvsdriver_sdio.ko $(TARGETFILE) $(TARGETDIR)/dvsdriver_sdio.post.mod.o
	@rm -f $(TARGETFILE) $(TARGETDIR)/dvsdriver_sdio.post.mod.o $(TARGETDIR)/dvsdriver_sdio.post.mod.c

dvsdebug_sdio.ko:	$(TARGETFILE_D) $(TARGETDIR)/dvsdebug_sdio.post.mod.o
	ld $(LDFLAGS) -o $(TARGETDIR)/dvsdebug_sdio.ko $(TARGETFILE_D) $(TARGETDIR)/dvsdebug_sdio.post.mod.o
	@rm -f $(TARGETFILE_D) $(TARGETDIR)/dvsdebug_sdio.post.mod.o $(TARGETDIR)/dvsdebug_sdio.post.mod.c

$(TARGETDIR):
	mkdir -p $@

clean:
	@rm -f $(TARGETFILE) $(TARGETFILE_D)
	@rm -f $(TARGETDIR)/dvsdriver_sdio.post.mod.* $(TARGETDIR)/dvsdebug_sdio.post.mod.*
	@rm -f $(OBJS) $(OBJS_D)
	rm -f $(TARGETDIR)/dvsdriver_sdio.ko $(TARGETDIR)/dvsdebug_sdio.ko
