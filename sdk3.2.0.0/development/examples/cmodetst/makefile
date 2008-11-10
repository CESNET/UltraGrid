#
#	Makefile for sdk example cmodetst
#

DIR 		= cmodetst
DESTDIR		= $(SDKCOMPILE)/.objdir.cmodetst
TARGETDIR	= $(SDKCOMPILE)
CFLAGS		=
LDFLAGS		= $(BASE_LDFLAGS)

include ../../makefile.inc

compile::	$(TARGETDIR)/cmodetst

$(DESTDIR)/cmodetst.o:   cmodetst.c
	$(COMPILEFILE)

$(TARGETDIR)/cmodetst:	$(DESTDIR)/cmodetst.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $? -L$(DVSLIB) -ldvsoem

clean::
	rm -rf $(DESTDIR)
	rm -f $(TARGETDIR)/cmodetst
