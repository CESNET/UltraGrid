#
#	Makefile for sdk example counter
#

DIR		= counter
DESTDIR         = $(SDKCOMPILE)/.objdir.counter
TARGETDIR       = $(SDKCOMPILE)
CFLAGS    =
LDFLAGS   = $(BASE_LDFLAGS)

include ../../makefile.inc

compile:	$(TARGETDIR)/counter $(TARGETDIR)/csp.fnt

$(TARGETDIR)/csp.fnt:	csp.fnt
	cp csp.fnt $(TARGETDIR)

$(DESTDIR)/counter.o:	counter.c
	$(COMPILEFILE)

$(TARGETDIR)/counter:	$(DESTDIR)/counter.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $< -L$(DVSLIB) -ldvsoem

clean::
	rm -rf $(DESTDIR)
	rm -f $(TARGETDIR)/counter $(TARGETDIR)/csp.fnt
