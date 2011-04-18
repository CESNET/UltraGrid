# Makefile for the SMPTE 259M-C example programs

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
TARGETS = bt801 capture eg1 mcapture mplayout playout sdicfg sdirxcfg sditxcfg

.c:
	$(CC) $(CFLAGS) -o $@ $@.c

all: $(TARGETS)

bt801: bt801.c $(INCLUDEDIR)/master.h
capture: capture.c $(INCLUDEDIR)/sdi.h $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
eg1: eg1.c $(INCLUDEDIR)/master.h
mcapture: mcapture.c $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
mplayout: mplayout.c $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
playout: playout.c $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
sdicfg: sdicfg.c $(INCLUDEDIR)/sdi.h $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
sdirxcfg: sdirxcfg.c $(INCLUDEDIR)/sdi.h $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
sditxcfg: sditxcfg.c $(INCLUDEDIR)/sdi.h $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o

clean:
	$(RM) *.o *~ core $(TARGETS)

install: all
	$(INSTALL) $(TARGETS) $(bindir)

uninstall:
	$(RM) $(foreach prog,$(TARGETS),$(bindir)/$(prog))

