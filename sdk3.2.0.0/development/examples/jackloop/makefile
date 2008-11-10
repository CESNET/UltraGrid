#
#	$Id: makefile,v 1.1 2008/09/12 08:45:00 xliska Exp $
#

DIR		= jackloop
DESTDIR         = $(SDKCOMPILE)/.objdir.jackloop
TARGETDIR       = $(SDKCOMPILE)
CFLAGS    =
LDFLAGS   = $(BASE_LDFLAGS)

include ../../makefile.inc

HEADER = abstractthread.h \
	       buffer.h \
	       defines.h \
	       inputthread.h \
	       log.h \
	       outputthread.h \
	       support.h \
	       threadmanager.h \
         ../common/dvs_support.h


OBJS = $(DESTDIR)/abstractthread.o \
       $(DESTDIR)/buffer.o \
       $(DESTDIR)/inputthread.o \
       $(DESTDIR)/log.o \
       $(DESTDIR)/main.o \
       $(DESTDIR)/outputthread.o \
       $(DESTDIR)/support.o \
       $(DESTDIR)/threadmanager.o \
       $(DESTDIR)/dvs_thread.o \
       $(DESTDIR)/dvs_support.o

compile:	$(TARGETDIR)/jackloop

$(DESTDIR)/main.o:	main.cpp
	$(COMPILEFILE)

$(DESTDIR)/abstractthread.o:	abstractthread.cpp
	$(COMPILEFILE)

$(DESTDIR)/buffer.o:	buffer.cpp
	$(COMPILEFILE)

$(DESTDIR)/inputthread.o:	inputthread.cpp
	$(COMPILEFILE)

$(DESTDIR)/log.o:	log.cpp
	$(COMPILEFILE)

$(DESTDIR)/outputthread.o:	outputthread.cpp
	$(COMPILEFILE)

$(DESTDIR)/support.o:	support.cpp
	$(COMPILEFILE)

$(DESTDIR)/threadmanager.o:	threadmanager.cpp
	$(COMPILEFILE)

$(DESTDIR)/dvs_thread.o:	../common/dvs_thread.c $(HEADERS)
	$(COMPILEFILE) -D_GNU_SOURCE

$(DESTDIR)/dvs_support.o:	../common/dvs_support.c $(HEADERS)
	$(COMPILEFILE) -D_GNU_SOURCE

$(TARGETDIR)/jackloop:	$(OBJS)
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $(OBJS) -L$(DVSLIB) -ldvsoem -lpthread -lstdc++

clean::
	rm -rf $(DESTDIR)
	rm -f $(TARGETDIR)/jackloop
