#
#	Makefile for sdk example dmaloop
#

DIR		= dmaloop
DESTDIR		= $(SDKCOMPILE)/.objdir.dmaloop
TARGETDIR	= $(SDKCOMPILE)
CFLAGS		=
LDFLAGS		= $(BASE_LDFLAGS)
OBJS            = $(DESTDIR)/dmaloop.o \
                  $(DESTDIR)/dvs_thread.o

include ../../makefile.inc

compile::	$(TARGETDIR)/dmaloop

$(DESTDIR)/dmaloop.o:		dmaloop.c
	$(COMPILEFILE)

$(DESTDIR)/dvs_thread.o:	../common/dvs_thread.c
	$(COMPILEFILE)

$(TARGETDIR)/dmaloop:	$(OBJS)
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $(OBJS) -L$(DVSLIB) -ldvsoem -lpthread

clean::
	rm -rf $(DESTDIR)
	rm -f $(TARGETDIR)/dmaloop
