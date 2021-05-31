SRCDIR ?= .

all:
	gcc -g -Wall -pthread -o hd-rum $(SRCDIR)/hd-rum.c

install:
	install -m 755 hd-rum /usr/local/bin
	install -m 644 hd-rum.1 /usr/local/man/man1

uninstall:
	$(RM) /usr/local/bin/hd-rum
	$(RM) /usr/local/man/man1/hd-rum.1

