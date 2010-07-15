# Makefile for the Master Linux Software Development Kit

SHELL = /bin/sh

SUBDIRS = src Examples

.PHONY: all clean install uninstall

all:
	for n in $(SUBDIRS); do $(MAKE) -C $$n || exit 1; done

clean:
	for n in $(SUBDIRS); do $(MAKE) -C $$n clean; done

install:
	for n in $(SUBDIRS); do $(MAKE) -C $$n install || exit 1; done

uninstall:
	for n in $(SUBDIRS); do $(MAKE) -C $$n uninstall || exit 1; done

