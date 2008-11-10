#
#	Makefile for sv
#


DESTDIR   = $(SDKCOMPILE)/.objdir.sv
TARGETDIR = $(SDKCOMPILE)
DIR       = svram
HEADER    = $(DVSHEADER) ../common/dvs_support.h svprogram.h svtags.h
DEFS      = -DHAVE_USLEEP=1 -DCLIPSTATION -DCONFIG_TIFF
CFLAGS    =      
LDFLAGS	  = $(BASE_LDFLAGS) -L$(DVSLIB)


OBJS =		$(DESTDIR)/sv_clib.o		\
		$(DESTDIR)/sv_compare.o		\
		$(DESTDIR)/sv_fm.o		\
		$(DESTDIR)/sv_main.o		\
		$(DESTDIR)/sv_sequence.o	\
		$(DESTDIR)/dvs_support.o

include ../../makefile.inc

compile::	svram

dist::	svram

svram:		$(TARGETDIR)/svram

$(DESTDIR)/sv_clib.o:		sv_clib.c $(HEADER)
	$(COMPILEFILE)

$(DESTDIR)/sv_compare.o:	sv_compare.c $(HEADER)
	$(COMPILEFILE)

$(DESTDIR)/sv_fm.o:		sv_fm.c $(HEADER)
	$(COMPILEFILE)

$(DESTDIR)/sv_main.o:		sv_main.c $(HEADER)
	$(COMPILEFILE)

$(DESTDIR)/sv_sequence.o:	sv_sequence.c $(HEADER)
	$(COMPILEFILE)

$(DESTDIR)/dvs_support.o:	../common/dvs_support.c $(HEADER)
	$(COMPILEFILE)

$(TARGETDIR)/sv:		$(OBJS) $(DVSLIB)/libdvsoem.a
	$(CC) $(LDFLAGS) $(OBJS) -o $@ -ldvsoem -ldvsfm -lm $(LIBDSO) $(DVS_SOCKETLIB)

$(TARGETDIR)/svram:		$(OBJS) $(DVSLIB)/libdvsoem.a
	$(CC) $(LDFLAGS) $(OBJS) -o $@ -ldvsoem -ldvsfm -lm $(LIBDSO) $(DVS_SOCKETLIB)

clean::
	rm -rf $(DESTDIR)
	rm -f $(TARGETDIR)/sv $(TARGETDIR)/svram

tags::
	ctags --c-types=-cdegmnpstuvx+f *.c
	cat tags | grep -v donttag | grep -v static | grep -v export | sort | tags2h > svtags.hx
	sed -e s/TAGS/SVTAGS/ < svtags.hx > svtags.h
	rm -f svtags.hx
	rm -f tags

amd64:	sv.amd64 svram.amd64

ia64:	sv.ia64 svram.ia64

fromdos::
	fromdos makefile.sdk

todos::
	todos makefile.sdk
