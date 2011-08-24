# Parameter list
# DRIVERNAME - name of the driver e.g. lucy
# PATH_SRC   - Path to for linux.c (src)
# PATH_BIN   - Path to binary blob

ifneq ($(KERNELRELEASE),)
# kbuild part of makefile

# Target object
obj-m := $(DRIVERNAME).o

# Object list for target object
$(DRIVERNAME)-y := $(PATH_SRC)/linux.o $(PATH_BIN)/$(DRIVERNAME).pre.o

# Set special c flags
ccflags-$(CONFIG_REGPARM) += -DCOMPILE_REGPARM 

else
# normal makefile

KERNELDIR := /lib/modules/`uname -r`/build

all::
	$(MAKE) -C $(KERNELDIR) M=`pwd` 

clean::
	$(MAKE) -C $(KERNELDIR) M=`pwd` clean

endif
