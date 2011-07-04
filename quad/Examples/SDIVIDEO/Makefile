# Makefile for the SMPTE 259M-C and SMPTE 292M video example programs

SHELL = /bin/sh
INSTALL = install

INCLUDEDIR = ../../include
prefix = /usr/local
exec_prefix = $(prefix)
bindir = $(exec_prefix)/bin

.SUFFIXES:
.SUFFIXES: .c .o

.PHONY: all clean install uninstall

CFLAGS = -O2 -Wall -W -I$(INCLUDEDIR) -D_LARGEFILE_SOURCE \
	$(shell getconf LFS_CFLAGS; getconf LFS_LDFLAGS; getconf LFS_LIBS)
TARGETS = bt801video eg1video mvideocapture mvideoplayout rp219 \
	sdivideocfg sdivideorxcfg sdivideotxcfg \
	videocapture videoplayout

.c:
	$(CC) $(CFLAGS) -o $@ $@.c

all: $(TARGETS)

bt801video: bt801video.c $(INCLUDEDIR)/master.h
eg1video: eg1video.c $(INCLUDEDIR)/master.h
mvideocapture: mvideocapture.c $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
mvideoplayout: mvideoplayout.c $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
rp219: rp219.c $(INCLUDEDIR)/master.h
sdivideocfg: sdivideocfg.c $(INCLUDEDIR)/sdivideo.h $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
sdivideorxcfg: sdivideorxcfg.c $(INCLUDEDIR)/sdivideo.h $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
sdivideotxcfg: sdivideotxcfg.c $(INCLUDEDIR)/sdivideo.h $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
videocapture: videocapture.c $(INCLUDEDIR)/sdivideo.h $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
videoplayout: videoplayout.c $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o

clean:
	$(RM) *.o *~ core $(TARGETS)

install: all
	$(INSTALL) $(TARGETS) $(bindir)

uninstall:
	$(RM) $(foreach prog,$(TARGETS),$(bindir)/$(prog))

