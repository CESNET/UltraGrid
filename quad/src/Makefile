# Makefile for the Master Linux drivers and APIs

ifneq ($(KERNELRELEASE),)
obj-m := asi.o dvbm.o sdi.o sdim.o sdilpm.o mmas.o mmsa.o sdiqoe.o\
	sdiqie.o hdsdiqi.o ls_as.o ls_jtag.o
asi-objs := asicore.o
dvbm-objs := dvbmaster.o gt64131.o \
	dvbm_fd.o dvbm_fdu.o dvbm_rx.o dvbm_rxu.o dvbm_tx.o dvbm_txu.o \
	dvbm_lpfd.o dvbm_qlf.o dvbm_qi.o dvbm_qo.o dvbm_qdual.o dvbm_q3ioe.o \
	dvbm_q3inoe.o
ls_as-objs := asmi.o
ls_jtag-objs := jtag.o
sdi-objs := sdicore.o
sdim-objs := sdimaster.o
sdilpm-objs := sdim_lpfd.o
sdiqoe-objs := sdim_qoe.o
sdiqie-objs := sdim_qie.o
hdsdiqi-objs := hdsdi_qie.o
mmas-objs := as.o
mmsa-objs := sa.o

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
TARGETS = asi.ko dvbm.ko sdi.ko sdim.ko sdilpm.ko sdiqoe.ko sdiqie.ko hdsdiqi.ko mmas.ko mmsa.ko 
	
all:
	$(MAKE) -C $(KERNELSRC) SUBDIRS=$(PWD) modules

clean:
	$(RM) *.o *.ko *.mod.c .*.cmd Module.symvers
	$(RM) -r .tmp_versions

install: all
	$(INSTALL) -d $(INSTALLDIR)
	$(INSTALL) -m 0744 $(TARGETS) $(INSTALLDIR)
	$(DEPMOD) -a

uninstall:
	$(RM) $(foreach module,$(TARGETS),$(INSTALLDIR)/$(module))
	$(DEPMOD) -a
	rmdir $(INSTALLDIR)
endif

