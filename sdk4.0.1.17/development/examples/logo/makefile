#
#	Makefile for sdk example logo
#

DIR		= logo
DESTDIR		= $(SDKCOMPILE)/.objdir.logo
TARGETDIR	= $(SDKCOMPILE)
CFLAGS    =
LDFLAGS   = $(BASE_LDFLAGS)

include ../../makefile.inc

compile::	$(TARGETDIR)/logo $(TARGETDIR)/dvs.bmp

$(TARGETDIR)/dvs.bmp:	dvs.bmp
	cp dvs.bmp $(TARGETDIR)/dvs.bmp

$(DESTDIR)/logo.o:		logo.c
	$(COMPILEFILE)

$(TARGETDIR)/logo:		$(DESTDIR)/logo.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $?  -L$(DVSLIB) -ldvsoem

clean::
	rm -rf $(DESTDIR)
	rm -f $(TARGETDIR)/logo $(TARGETDIR)/dvs.bmp
