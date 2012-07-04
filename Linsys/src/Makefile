# Makefile for the Master Linux drivers and APIs

ifneq ($(KERNELRELEASE),)
obj-m := asi.o dvbm.o sdi.o sdiaudio.o sdivideo.o \
	sdim.o sdilpm.o mmas.o mmsa.o sdiqoe.o \
	sdiqie.o ls_as.o ls_jtag.o hdsdim.o
asi-objs := asicore.o miface.o mdma.o
dvbm-objs := dvbmaster.o mdev.o \
	lsdma.o plx9080.o gt64131.o \
	dvbm_fd.o dvbm_fdu.o dvbm_rx.o dvbm_rxu.o dvbm_tx.o dvbm_txu.o \
	dvbm_lpfd.o dvbm_qlf.o dvbm_qi.o \
	dvbm_qio.o dvbm_qo.o dvbm_qdual.o dvbm_q3ioe.o dvbm_q3inoe.o \
	dvbm_lptxe.o dvbm_lprxe.o dvbm_lpqo.o
ls_as-objs := asmi.o eeprom.o
ls_jtag-objs := jtag.o eeprom.o
sdi-objs := sdicore.o miface.o mdma.o
sdiaudio-objs := sdiaudiocore.o miface.o mdma.o
sdivideo-objs := sdivideocore.o miface.o mdma.o
sdim-objs := sdimaster.o mdev.o plx9080.o
sdilpm-objs := sdim_lpfd.o mdev.o lsdma.o
sdiqoe-objs := sdim_qoe.o mdev.o lsdma.o
sdiqie-objs := sdim_qie.o mdev.o plx9080.o lsdma.o
hdsdim-objs := hdsdimaster.o mdev.o lsdma.o \
	hdsdim_qie.o hdsdim_txe.o hdsdim_rxe.o
mmas-objs := as.o mdev.o plx9080.o
mmsa-objs := sa.o mdev.o plx9080.o

else

SHELL = /bin/sh
INSTALL = install
DEPMOD = /sbin/depmod

.SUFFIXES:
.SUFFIXES: .c .o

.PHONY: all clean install uninstall

KERNELPTR := /lib/modules/$(shell uname -r)/build
KERNELSRC := $(shell if [ -e $(KERNELPTR) ]; then echo $(KERNELPTR); fi)
ifeq ($(KERNELSRC),)
$(error $(KERNELPTR) not found)
endif
PWD := $(shell pwd)
INSTALLDIR := /lib/modules/$(shell uname -r)/master
TARGETS = asi.ko dvbm.ko sdi.ko sdiaudio.ko sdivideo.ko \
	sdim.ko sdilpm.ko sdiqoe.ko sdiqie.ko mmas.ko mmsa.ko hdsdim.ko

all:
	$(MAKE) -C $(KERNELSRC) M=$(PWD) modules

clean:
	$(MAKE) -C $(KERNELSRC) M=$(PWD) clean

install: all
# Don't use kbuild to install unless you blacklist the ASMI and JTAG modules,
# because they silently conflict with the other modules.
#	$(MAKE) -C $(KERNELSRC) M=$(PWD) modules_install
	$(INSTALL) -d $(INSTALLDIR)
	$(INSTALL) -m 0644 $(TARGETS) $(INSTALLDIR)
	$(DEPMOD) -a

uninstall:
	$(RM) $(foreach module,$(TARGETS),$(INSTALLDIR)/$(module))
	$(DEPMOD) -a
	rmdir $(INSTALLDIR)
endif

