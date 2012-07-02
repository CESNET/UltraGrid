# Makefile for the SMPTE 259M-C and SMPTE 292M audio example programs

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
TARGETS = rp155 ac3aes3 aes3data audiocapture audioplayout \
	sdiaudiocfg sdiaudiorxcfg sdiaudiotxcfg

.c:
	$(CC) $(CFLAGS) -o $@ $@.c

all: $(TARGETS)

rp155: rp155.c $(INCLUDEDIR)/master.h
	$(CC) $(CFLAGS) -lm -o $@ $@.c
ac3aes3: ac3aes3.c $(INCLUDEDIR)/master.h
aes3data: aes3data.c $(INCLUDEDIR)/master.h
audiocapture: audiocapture.c $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
audioplayout: audioplayout.c $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
sdiaudiocfg: sdiaudiocfg.c $(INCLUDEDIR)/sdiaudio.h $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
sdiaudiorxcfg: sdiaudiorxcfg.c $(INCLUDEDIR)/sdiaudio.h $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o
sdiaudiotxcfg: sdiaudiotxcfg.c $(INCLUDEDIR)/sdiaudio.h $(INCLUDEDIR)/master.h ../util.o
	$(CC) $(CFLAGS) -o $@ $@.c ../util.o

clean:
	$(RM) *.o *~ core $(TARGETS)

install: all
	$(INSTALL) $(TARGETS) $(bindir)

uninstall:
	$(RM) $(foreach prog,$(TARGETS),$(bindir)/$(prog))

