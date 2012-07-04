# Makefile for the DVB ASI example programs

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
TARGETS = asicfg txcfg rxcfg txmon rxmon txtest rxtest \
	calcstuff mknull cknull stampinfo

.c:
	$(CC) $(CFLAGS) -o $@ $@.c

all: $(TARGETS)

asicfg: asicfg.c $(INCLUDEDIR)/asi.h $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
txcfg: txcfg.c $(INCLUDEDIR)/asi.h $(INCLUDEDIR)/master.h ../pci_ids.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
rxcfg: rxcfg.c $(INCLUDEDIR)/asi.h $(INCLUDEDIR)/master.h ../pci_ids.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
txmon: txmon.c $(INCLUDEDIR)/asi.h $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
rxmon: rxmon.c $(INCLUDEDIR)/asi.h $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
txtest: txtest.c $(INCLUDEDIR)/asi.h $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
rxtest: rxtest.c $(INCLUDEDIR)/asi.h $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
calcstuff: calcstuff.c $(INCLUDEDIR)/master.h
mknull: mknull.c $(INCLUDEDIR)/master.h
cknull: cknull.c $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
stampinfo: stampinfo.c $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o

clean:
	$(RM) *.o *~ core $(TARGETS)

install: all
	$(INSTALL) $(TARGETS) $(bindir)

uninstall:
	$(RM) $(foreach prog,$(TARGETS),$(bindir)/$(prog))

