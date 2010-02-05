#
#	Makefile for sdk example dmaspeed
#

DIR		= dmaspeed
DESTDIR		= $(SDKCOMPILE)/.objdir.dmaspeed
TARGETDIR	= $(SDKCOMPILE)
CFLAGS    =
LDFLAGS   = $(BASE_LDFLAGS)

include ../../makefile.inc

compile::	$(TARGETDIR)/dmaspeed

$(DESTDIR)/dmaspeed.o:	dmaspeed.c
	$(COMPILEFILE)

$(TARGETDIR)/dmaspeed:	$(DESTDIR)/dmaspeed.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $<  -L$(DVSLIB) -ldvsoem $(DVS_SOCKETLIB)

clean::
	rm -rf $(DESTDIR)
	rm -f $(TARGETDIR)/dmaspeed
