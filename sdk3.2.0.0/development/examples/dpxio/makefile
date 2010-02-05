#
#	Makefile for sdk example dpxio
#

DIR		= dpxio
DESTDIR		= $(SDKCOMPILE)/.objdir.dpxio
TARGETDIR	= $(SDKCOMPILE)
CFLAGS    =
LDFLAGS   = $(BASE_LDFLAGS)

include ../../makefile.inc

HEADER		= dpxio.h tags.h
OBJS		= $(DESTDIR)/dpxio.o		\
		  $(DESTDIR)/dpxio_audio.o	\
		  $(DESTDIR)/dpxio_anc.o	\
		  $(DESTDIR)/dpxio_dpx.o	\
		  $(DESTDIR)/dpxio_lut.o	\
		  $(DESTDIR)/common_dpxformat.o \
		  $(DESTDIR)/common_fileops.o

ifeq ($(HOSTTARGET),linux)
DVSLIB_EXTRA = -lpthread
endif
ifeq ($(HOSTTARGET),irix)
DVSLIB_EXTRA = -lds -lpthread
endif

compile::	$(TARGETDIR)/dpxio

$(DESTDIR)/dpxio.o:		dpxio.c $(HEADER)
	$(COMPILEFILE)

$(DESTDIR)/dpxio_audio.o:	dpxio_audio.c $(HEADER)
	$(COMPILEFILE)

$(DESTDIR)/dpxio_anc.o:		dpxio_anc.c $(HEADER)
	$(COMPILEFILE)

$(DESTDIR)/dpxio_dpx.o:		dpxio_dpx.c $(HEADER)
	$(COMPILEFILE)

$(DESTDIR)/dpxio_lut.o:		dpxio_lut.c $(HEADER)
	$(COMPILEFILE)

$(DESTDIR)/common_dpxformat.o:	../common/dpxformat.c $(HEADER)
	$(COMPILEFILE)

$(DESTDIR)/common_fileops.o:	../common/fileops.c $(HEADER)
	$(COMPILEFILE)

$(TARGETDIR)/dpxio:		$(OBJS) 
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $(OBJS)  -L$(DVSLIB) -ldvsoem $(DVSLIB_EXTRA)

clean::
	rm -rf $(DESTDIR)
	rm -f $(TARGETDIR)/dpxio

tags:   tags_gen

tags_gen::
	ctags --c-types=-cdegmnpstuvxAC+f *.c
	grep -v donttag tags | grep -v static | sort -d +1 | tags2h >tags.h
	rm -f tags

