# Makefile for the Master Linux example programs

SHELL = /bin/sh
INSTALL = install

INCLUDEDIR = ../include
prefix = /usr/local
exec_prefix = $(prefix)
bindir = $(exec_prefix)/bin

.SUFFIXES:
.SUFFIXES: .c .o

.PHONY: all clean install uninstall

CFLAGS = -O2 -Wall -W -I$(INCLUDEDIR) -D_LARGEFILE_SOURCE \
	$(shell getconf LFS_CFLAGS; getconf LFS_LDFLAGS; getconf LFS_LIBS)
TARGETS = mastercfg
SUBDIRS = ASI SDI SDIAUDIO SDIVIDEO

.c:
	$(CC) $(CFLAGS) -o $@ $@.c

all: $(TARGETS)
	for n in $(SUBDIRS); do $(MAKE) -C $$n || exit 1; done

mastercfg: mastercfg.c $(INCLUDEDIR)/master.h util.o
	$(CC) $(CFLAGS) -o $@ $@.c util.o

util.o: util.c pci_ids.h util.h

clean:
	$(RM) *.o *~ core $(TARGETS)
	for n in $(SUBDIRS); do $(MAKE) -C $$n clean; done

install: all
	$(INSTALL) $(TARGETS) $(bindir)
	for n in $(SUBDIRS); do $(MAKE) -C $$n install || exit 1; done

uninstall:
	$(RM) $(foreach prog,$(TARGETS),$(bindir)/$(prog))
	for n in $(SUBDIRS); do $(MAKE) -C $$n uninstall || exit 1; done

