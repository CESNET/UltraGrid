#
#	Makefile for sdk example dpxrender
#

DIR       = dpxrender
DESTDIR   = $(SDKCOMPILE)/.objdir.dpxrender
TARGETDIR = $(SDKCOMPILE)
CFLAGS    =
LDFLAGS   = $(BASE_LDFLAGS)

HEADER		= dpxrender.h \
            tags.h

OBJS      = $(DESTDIR)/dpxrender.o \
            $(DESTDIR)/dpxrender_lut.o \
            $(DESTDIR)/common_dpxformat.o \
            $(DESTDIR)/common_fileops.o \
            $(DESTDIR)/common_dvs_thread.o

include ../../makefile.inc

compile::	$(TARGETDIR)/dpxrender

$(DESTDIR)/dpxrender.o:         dpxrender.c $(HEADER)
	$(COMPILEFILE)

$(DESTDIR)/dpxrender_lut.o:     dpxrender_lut.c $(HEADER)
	$(COMPILEFILE)
 
$(DESTDIR)/common_dpxformat.o:  ../common/dpxformat.c $(HEADER)
	$(COMPILEFILE)

$(DESTDIR)/common_fileops.o:    ../common/fileops.c $(HEADER)
	$(COMPILEFILE)

$(DESTDIR)/common_dvs_thread.o: ../common/dvs_thread.c $(HEADER)
	$(COMPILEFILE)

$(TARGETDIR)/dpxrender:	$(OBJS) 
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $(OBJS) -L$(DVSLIB) -ldvsoem -lpthread

clean::
	rm -rf $(DESTDIR)
	rm -f $(TARGETDIR)/dpxrender

tags:   tags_gen

tags_gen::
	ctags --c-types=-cdegmnpstuvx+f *.c
	grep -v donttag tags | grep -v static | sort -k 2 -d | ../../platform/linux/tags2h >tags.h
	rm -f tags

