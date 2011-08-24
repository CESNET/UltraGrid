#
#	Makefile for sdk example overlay
#

DIR		= overlay
DESTDIR		= $(SDKCOMPILE)/.objdir.overlay
TARGETDIR	= $(SDKCOMPILE)
CFLAGS		=
LDFLAGS		= $(BASE_LDFLAGS)

include ../../makefile.inc

OBJS		= $(DESTDIR)/overlay.o		\
		  $(DESTDIR)/overlay_dvs.o

compile::	$(TARGETDIR)/overlay

$(DESTDIR)/overlay.o:		overlay.c
	$(COMPILEFILE)

$(DESTDIR)/overlay_dvs.o:	overlay_dvs.c
	$(COMPILEFILE)

$(TARGETDIR)/overlay:	$(OBJS)
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $(OBJS) -L$(DVSLIB) -ldvsoem

clean::
	rm -rf $(DESTDIR)
	rm -f $(TARGETDIR)/overlay
