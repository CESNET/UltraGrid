/* dvbm_fdu.c
 *
 * Linux driver functions for Linear Systems Ltd.
 * DVB Master FD-U, DVB Master FD-B, and DVB Master II FD.
 *
 * Copyright (C) 2003-2010 Linear Systems Ltd.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either Version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public Licence for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 *
 * Linear Systems can be contacted at <http://www.linsys.ca/>.
 *
 */

#include <linux/kernel.h> /* KERN_INFO */
#include <linux/module.h> /* THIS_MODULE */

#include <linux/fs.h> /* inode, file, file_operations */
#include <linux/sched.h> /* pt_regs */
#include <linux/pci.h> /* pci_resource_start () */
#include <linux/slab.h> /* kzalloc () */
#include <linux/list.h> /* INIT_LIST_HEAD () */
#include <linux/spinlock.h> /* spin_lock_init () */
#include <linux/init.h> /* __devinit */
#include <linux/errno.h> /* error codes */
#include <linux/interrupt.h> /* irqreturn_t */
#include <linux/device.h> /* device_create_file () */
#include <linux/mutex.h> /* mutex_init () */

#include <asm/uaccess.h> /* put_user () */
#include <asm/bitops.h> /* set_bit () */

#include "asicore.h"
#include "../include/master.h"
#include "mdev.h"
#include "mdma.h"
#include "dvbm.h"
#include "plx9080.h"
#include "dvbm_fdu.h"

static const char dvbm_fdu_name[] = DVBM_NAME_FDU;
static const char dvbm_fdur_name[] = DVBM_NAME_FDU_R;
static const char dvbm_fdb_name[] = DVBM_NAME_FDB;
static const char dvbm_fdbr_name[] = DVBM_NAME_FDB_R;
static const char dvbm_2fd_name[] = DVBM_NAME_2FD;
static const char dvbm_2fdr_name[] = DVBM_NAME_2FD_R;
static const char dvbm_2fdrs_name[] = DVBM_NAME_2FD_RS;
static const char atscm_2fd_name[] = ATSCM_NAME_2FD;
static const char atscm_2fdr_name[] = ATSCM_NAME_2FD_R;
static const char atscm_2fdrs_name[] = ATSCM_NAME_2FD_RS;
static const char dvbm_pci_2fde_name[] = DVBM_NAME_2FDE;
static const char dvbm_pci_2fder_name[] = DVBM_NAME_2FDE_R;
static const char dvbm_pci_2fders_name[] = DVBM_NAME_2FDE_RS;
static const char dvbm_pci_fde_name[] = DVBM_NAME_FDE;
static const char dvbm_pci_fder_name[] = DVBM_NAME_FDE_R;
static const char dvbm_pci_fdeb_name[] = DVBM_NAME_FDEB;
static const char dvbm_pci_fdebr_name[] = DVBM_NAME_FDEB_R;
static const char atscm_pci_2fde_name[] = ATSCM_NAME_2FDE;
static const char atscm_pci_2fder_name[] = ATSCM_NAME_2FDE_R;

/* Static function prototypes */
static ssize_t dvbm_fdu_show_bypass_mode (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static ssize_t dvbm_fdu_store_bypass_mode (struct device *dev,
	struct device_attribute *attr,
	const char *buf,
	size_t count);
static ssize_t dvbm_fdu_show_bypass_status (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static ssize_t dvbm_fdu_show_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static ssize_t dvbm_fdu_store_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	const char *buf,
	size_t count);
static ssize_t dvbm_fdu_show_gpi (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static ssize_t dvbm_fdu_show_gpo (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static ssize_t dvbm_fdu_store_gpo (struct device *dev,
	struct device_attribute *attr,
	const char *buf,
	size_t count);
static ssize_t dvbm_fdu_show_watchdog (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static ssize_t dvbm_fdu_store_watchdog (struct device *dev,
	struct device_attribute *attr,
	const char *buf,
	size_t count);
static irqreturn_t IRQ_HANDLER(dvbm_fdu_irq_handler,irq,dev_id,regs);
static void dvbm_fdu_txinit (struct master_iface *iface);
static void dvbm_fdu_txstart (struct master_iface *iface);
static void dvbm_fdu_txstop (struct master_iface *iface);
static void dvbm_fdu_txexit (struct master_iface *iface);
static void dvbm_fdu_start_tx_dma (struct master_iface *iface);
static long dvbm_fdu_txunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int FSYNC_HANDLER(dvbm_fdu_txfsync,filp,datasync);
static void dvbm_fdu_rxinit (struct master_iface *iface);
static void dvbm_fdu_rxstart (struct master_iface *iface);
static void dvbm_fdu_rxstop (struct master_iface *iface);
static void dvbm_fdu_rxexit (struct master_iface *iface);
static long dvbm_fdu_rxunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int FSYNC_HANDLER(dvbm_fdu_rxfsync,filp,datasync);

struct file_operations dvbm_fdu_txfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.write = asi_write,
	.poll = asi_txpoll,
	.unlocked_ioctl = dvbm_fdu_txunlocked_ioctl,
	.compat_ioctl = dvbm_fdu_txunlocked_ioctl,
	.open = asi_open,
	.release = asi_release,
	.fsync = dvbm_fdu_txfsync,
	.fasync = NULL
};

struct file_operations dvbm_fdu_rxfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.read = asi_read,
	.poll = asi_rxpoll,
	.unlocked_ioctl = dvbm_fdu_rxunlocked_ioctl,
	.compat_ioctl = dvbm_fdu_rxunlocked_ioctl,
	.open = asi_open,
	.release = asi_release,
	.fsync = dvbm_fdu_rxfsync,
	.fasync = NULL
};

struct master_iface_operations dvbm_fdu_txops = {
	.init = dvbm_fdu_txinit,
	.start = dvbm_fdu_txstart,
	.stop = dvbm_fdu_txstop,
	.exit = dvbm_fdu_txexit,
	.start_tx_dma = dvbm_fdu_start_tx_dma
};

struct master_iface_operations dvbm_fdu_rxops = {
	.init = dvbm_fdu_rxinit,
	.start = dvbm_fdu_rxstart,
	.stop = dvbm_fdu_rxstop,
	.exit = dvbm_fdu_rxexit
};

/**
 * dvbm_fdu_show_bypass_mode - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
static ssize_t
dvbm_fdu_show_bypass_mode (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	/* Atomic read of CSR, so we don't lock */
	return snprintf (buf, PAGE_SIZE, "%u\n",
		master_inl (card, DVBM_FDU_CSR) & DVBM_FDU_CSR_BYPASS_MASK);
}

/**
 * dvbm_fdu_store_bypass_mode - interface attribute write handler
 * @dev: device being written
 * @attr: device attribute
 * @buf: input buffer
 * @count:
 **/
static ssize_t
dvbm_fdu_store_bypass_mode (struct device *dev,
	struct device_attribute *attr,
	const char *buf,
	size_t count)
{
	struct master_dev *card = dev_get_drvdata(dev);
	char *endp;
	unsigned long val = simple_strtoul (buf, &endp, 0);
	const unsigned long max = (card->capabilities & MASTER_CAP_WATCHDOG) ?
		MASTER_CTL_BYPASS_WATCHDOG : MASTER_CTL_BYPASS_DISABLE;
	int retcode = count;
	unsigned int reg;

	if ((endp == buf) || (val > max)) {
		return -EINVAL;
	}
	spin_lock (&card->reg_lock);
	reg = master_inl (card, DVBM_FDU_CSR) & ~DVBM_FDU_CSR_BYPASS_MASK;
	master_outl (card, DVBM_FDU_CSR, reg | val);
	spin_unlock (&card->reg_lock);
	return retcode;
}

/**
 * dvbm_fdu_show_bypass_status - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
static ssize_t
dvbm_fdu_show_bypass_status (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	return snprintf (buf, PAGE_SIZE, "%u\n",
		(master_inl (card, DVBM_FDU_ICSR) &
		DVBM_FDU_ICSR_BYPASS) >> 15);
}

/**
 * dvbm_fdu_show_blackburst_type - interface attribute read handler
 * @dev: device being read
 * @attr: device_attribute
 * @buf: output buffer
 **/
static ssize_t
dvbm_fdu_show_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	return snprintf (buf, PAGE_SIZE, "%u\n",
		(master_inl (card, DVBM_FDU_TCSR) & DVBM_FDU_TCSR_PAL) >> 13);
}

/**
 * dvbm_fdu_store_blackburst_type - interface attribute write handler
 * @dev: device being written
 * @attr: device attribute
 * @buf: input buffer
 * @count:
 **/
static ssize_t
dvbm_fdu_store_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	const char *buf,
	size_t count)
{
	struct master_dev *card = dev_get_drvdata(dev);
	char *endp;
	unsigned long val = simple_strtoul (buf, &endp, 0);
	unsigned int reg;
	const unsigned long max = MASTER_CTL_BLACKBURST_PAL;
	int retcode = count;
	struct master_iface *txiface = list_entry (card->iface_list.next,
		struct master_iface, list);

	if ((endp == buf) || (val > max)) {
		return -EINVAL;
	}
	mutex_lock (&card->users_mutex);
	if (txiface->users) {
		retcode = -EBUSY;
		goto OUT;
	}
	spin_lock (&card->reg_lock);
	reg = master_inl (card, DVBM_FDU_TCSR) & ~DVBM_FDU_TCSR_PAL;
	master_outl (card, DVBM_FDU_TCSR, reg | (val << 13));
	spin_unlock (&card->reg_lock);
OUT:
	mutex_unlock (&card->users_mutex);
	return retcode;
}

/**
 * dvbm_fdu_show_gpi - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
static ssize_t
dvbm_fdu_show_gpi (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	/* Atomic read of CSR, so we don't lock */
	return snprintf (buf, PAGE_SIZE, "0x%X\n",
		(master_inl (card, DVBM_FDU_CSR) &
		DVBM_FDU_CSR_GPI_MASK) >> DVBM_FDU_CSR_GPI_SHIFT);
}

/**
 * dvbm_fdu_show_gpo - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
static ssize_t
dvbm_fdu_show_gpo (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	/* Atomic read of CSR, so we don't lock */
	return snprintf (buf, PAGE_SIZE, "0x%X\n",
		(master_inl (card, DVBM_FDU_CSR) &
		DVBM_FDU_CSR_GPO_MASK) >> DVBM_FDU_CSR_GPO_SHIFT);
}

/**
 * dvbm_fdu_store_gpo - interface attribute write handler
 * @dev: device being written
 * @attr: device attribute
 * @buf: input buffer
 * @count:
 **/
static ssize_t
dvbm_fdu_store_gpo (struct device *dev,
	struct device_attribute *attr,
	const char *buf,
	size_t count)
{
	struct master_dev *card = dev_get_drvdata(dev);
	char *endp;
	unsigned long val = simple_strtoul (buf, &endp, 0);
	int retcode = count;
	unsigned int reg;

	if ((endp == buf) || (val > 1)) {
		return -EINVAL;
	}
	spin_lock (&card->reg_lock);
	reg = master_inl (card, DVBM_FDU_CSR) & ~DVBM_FDU_CSR_GPO_MASK;
	master_outl (card, DVBM_FDU_CSR,
		reg | ((val << DVBM_FDU_CSR_GPO_SHIFT) & DVBM_FDU_CSR_GPO_MASK));
	spin_unlock (&card->reg_lock);
	return retcode;
}

/**
 * dvbm_fdu_show_uid - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
ssize_t
dvbm_fdu_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	return snprintf (buf, PAGE_SIZE, "0x%08X%08X\n",
		master_inl (card, DVBM_FDU_UIDR_HI),
		master_inl (card, DVBM_FDU_UIDR_LO));
}

/**
 * dvbm_fdu_show_watchdog - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
static ssize_t
dvbm_fdu_show_watchdog (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	/* convert 40Mhz ticks to milliseconds */
	return snprintf (buf, PAGE_SIZE, "%u\n",
		(master_inl (card, DVBM_FDU_WDTLR) / 40000));
}

/**
 * dvbm_fdu_store_watchdog - interface attribute write handler
 * @dev: device being written
 * @attr: device attribute
 * @buf: input buffer
 * @count:
 **/
static ssize_t
dvbm_fdu_store_watchdog (struct device *dev,
	struct device_attribute *attr,
	const char *buf,
	size_t count)
{
	struct master_dev *card = dev_get_drvdata(dev);
	char *endp;
	unsigned long val = simple_strtoul (buf, &endp, 0);
	const unsigned long max = MASTER_WATCHDOG_MAX;
	int retcode = count;

	if ((endp == buf) || (val > max)){
		return -EINVAL;
	}

	/* Convert milliseconds to 40Mhz ticks */
	val = val * 40000;

	master_outl (card, DVBM_FDU_WDTLR, val);
	return retcode;
}

static DEVICE_ATTR(bypass_mode,S_IRUGO|S_IWUSR,
	dvbm_fdu_show_bypass_mode,dvbm_fdu_store_bypass_mode);
static DEVICE_ATTR(bypass_status,S_IRUGO,
	dvbm_fdu_show_bypass_status,NULL);
static DEVICE_ATTR(blackburst_type,S_IRUGO|S_IWUSR,
	dvbm_fdu_show_blackburst_type,dvbm_fdu_store_blackburst_type);
static DEVICE_ATTR(gpi,S_IRUGO,
	dvbm_fdu_show_gpi,NULL);
static DEVICE_ATTR(gpo,S_IRUGO|S_IWUSR,
	dvbm_fdu_show_gpo,dvbm_fdu_store_gpo);
static DEVICE_ATTR(uid,S_IRUGO,
	dvbm_fdu_show_uid,NULL);
static DEVICE_ATTR(watchdog,S_IRUGO|S_IWUSR,
	dvbm_fdu_show_watchdog,dvbm_fdu_store_watchdog);

/**
 * dvbm_fdu_pci_probe - PCI insertion handler for a DVB Master FD-U
 * @pdev: PCI device
 *
 * Handle the insertion of a DVB Master FD-U.
 * Returns a negative error code on failure and 0 on success.
 **/
int __devinit
dvbm_fdu_pci_probe (struct pci_dev *pdev)
{
	int err;
	unsigned int cap, transport;
	struct master_dev *card;

	err = dvbm_pci_probe_generic (pdev);
	if (err < 0) {
		goto NO_PCI;
	}

	/* Initialize the driver_data pointer so that dvbm_fdu_pci_remove()
	 * doesn't try to free it if an error occurs */
	pci_set_drvdata (pdev, NULL);

	/* Allocate and initialize a board info structure */
	if ((card = (struct master_dev *)
		kzalloc (sizeof (*card), GFP_KERNEL)) == NULL) {
		err = -ENOMEM;
		goto NO_MEM;
	}

	card->bridge_addr = ioremap_nocache (pci_resource_start (pdev, 0),
		pci_resource_len (pdev, 0));
	card->core.port = pci_resource_start (pdev, 2);
	card->version = master_inl (card, DVBM_FDU_CSR) >> 16;
	switch (pdev->device) {
	default:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDU:
		card->name = dvbm_fdu_name;
		card->capabilities = 0;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDU_R:
		card->name = dvbm_fdur_name;
		card->capabilities = 0;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDB:
		card->name = dvbm_fdb_name;
		card->capabilities = MASTER_CAP_BLACKBURST;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDB_R:
		card->name = dvbm_fdbr_name;
		card->capabilities = MASTER_CAP_BYPASS |
			MASTER_CAP_WATCHDOG |
			MASTER_CAP_BLACKBURST;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD:
		card->name = dvbm_2fd_name;
		card->capabilities = MASTER_CAP_BYPASS |
			MASTER_CAP_BLACKBURST | MASTER_CAP_GPI | MASTER_CAP_GPO;
		if (card->version >= 0x1106) {
			card->capabilities |= MASTER_CAP_UID;
		}
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD_R:
		card->name = dvbm_2fdr_name;
		card->capabilities = MASTER_CAP_BYPASS | MASTER_CAP_BLACKBURST |
			MASTER_CAP_WATCHDOG |
			MASTER_CAP_GPI;
		if (card->version >= 0x1106) {
			card->capabilities |= MASTER_CAP_UID;
		}
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD_RS:
		card->name = dvbm_2fdrs_name;
		card->capabilities = MASTER_CAP_BYPASS | MASTER_CAP_BLACKBURST |
			MASTER_CAP_WATCHDOG |
			MASTER_CAP_GPI;
		if (card->version >= 0x1106) {
			card->capabilities |= MASTER_CAP_UID;
		}
		break;
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FD:
		card->name = atscm_2fd_name;
		card->capabilities = MASTER_CAP_BYPASS |
			MASTER_CAP_GPI | MASTER_CAP_GPO;
		if (card->version >= 0x0403) {
			card->capabilities |= MASTER_CAP_UID;
		}
		break;
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FD_R:
		card->name = atscm_2fdr_name;
		card->capabilities = MASTER_CAP_BYPASS |
			MASTER_CAP_WATCHDOG |
			MASTER_CAP_GPI;
		if (card->version >= 0x0403) {
			card->capabilities |= MASTER_CAP_UID;
		}
		break;
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FD_RS:
		card->name = atscm_2fdrs_name;
		card->capabilities = MASTER_CAP_BYPASS |
			MASTER_CAP_WATCHDOG |
			MASTER_CAP_GPI;
		if (card->version >= 0x0403) {
			card->capabilities |= MASTER_CAP_UID;
		}
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE:
		card->name = dvbm_pci_2fde_name;
		card->capabilities = MASTER_CAP_BYPASS |
			MASTER_CAP_BLACKBURST | MASTER_CAP_GPI |
			MASTER_CAP_GPO | MASTER_CAP_UID;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE_R:
		card->name = dvbm_pci_2fder_name;
		card->capabilities = MASTER_CAP_BYPASS | MASTER_CAP_BLACKBURST |
			MASTER_CAP_WATCHDOG |
			MASTER_CAP_GPI |
			MASTER_CAP_UID;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE_RS:
		card->name = dvbm_pci_2fders_name;
		card->capabilities = MASTER_CAP_BYPASS | MASTER_CAP_BLACKBURST |
			MASTER_CAP_WATCHDOG |
			MASTER_CAP_GPI |
			MASTER_CAP_UID;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDE:
		card->name = dvbm_pci_fde_name;
		card->capabilities = 0;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDE_R:
		card->name = dvbm_pci_fder_name;
		card->capabilities = MASTER_CAP_UID;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDEB:
		card->name = dvbm_pci_fdeb_name;
		card->capabilities = MASTER_CAP_BLACKBURST | MASTER_CAP_UID;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDEB_R:
		card->name = dvbm_pci_fdebr_name;
		card->capabilities = MASTER_CAP_BYPASS |
			MASTER_CAP_WATCHDOG |
			MASTER_CAP_BLACKBURST |
			MASTER_CAP_UID;
		break;
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FDE:
		card->name = atscm_pci_2fde_name;
		card->capabilities = MASTER_CAP_BYPASS |
			MASTER_CAP_GPI | MASTER_CAP_GPO |
			MASTER_CAP_UID;
		break;
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FDE_R:
		card->name = atscm_pci_2fder_name;
		card->capabilities = MASTER_CAP_BYPASS |
			MASTER_CAP_WATCHDOG |
			MASTER_CAP_GPI |
			MASTER_CAP_UID;
		break;
	}
	card->id = pdev->device;
	card->irq = pdev->irq;
	card->irq_handler = dvbm_fdu_irq_handler;
	INIT_LIST_HEAD(&card->iface_list);
	/* Lock for ICSR, DMACSR1 */
	spin_lock_init (&card->irq_lock);
	/* Lock for CSR, IBSTR, IPSTR, FTR, PFLUT, TCSR, RCSR, TPCR, TSTAMP */
	spin_lock_init (&card->reg_lock);
	mutex_init (&card->users_mutex);
	card->parent = &pdev->dev;

	/* Print the firmware version */
	printk (KERN_INFO "%s: %s detected, firmware version %u.%u (0x%04X)\n",
		dvbm_driver_name, card->name,
		card->version >> 8, card->version & 0x00ff, card->version);

	/* Store the pointer to the board info structure
	 * in the PCI info structure */
	pci_set_drvdata (pdev, card);

	/* Reset the PCI 9056 */
	plx_reset_bridge (card->bridge_addr);

	/* Setup the PCI 9056 */
	writel (PLX_INTCSR_PCIINT_ENABLE |
		PLX_INTCSR_PCILOCINT_ENABLE |
		PLX_INTCSR_DMA0INT_ENABLE |
		PLX_INTCSR_DMA1INT_ENABLE,
		card->bridge_addr + PLX_INTCSR);
	writel (PLX_DMAMODE_32BIT | PLX_DMAMODE_READY |
		PLX_DMAMODE_LOCALBURST | PLX_DMAMODE_CHAINED |
		PLX_DMAMODE_INT | PLX_DMAMODE_CLOC |
		PLX_DMAMODE_DEMAND | PLX_DMAMODE_INTPCI,
		card->bridge_addr + PLX_DMAMODE0);
	writel (PLX_DMAMODE_32BIT | PLX_DMAMODE_READY |
		PLX_DMAMODE_LOCALBURST | PLX_DMAMODE_CHAINED |
		PLX_DMAMODE_INT | PLX_DMAMODE_CLOC |
		PLX_DMAMODE_DEMAND | PLX_DMAMODE_INTPCI,
		card->bridge_addr + PLX_DMAMODE1);
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + PLX_INTCSR);

	/* Reset the FPGA */
	master_outl (card, DVBM_FDU_TCSR, DVBM_FDU_TCSR_RST);
	master_outl (card, DVBM_FDU_RCSR, DVBM_FDU_RCSR_RST);

	/* Register a DVB Master device */
	if ((err = dvbm_register (card)) < 0) {
		goto NO_DEV;
	}

	/* Add device attributes */
	if (card->capabilities & MASTER_CAP_BYPASS) {
		if ((err = device_create_file (card->dev,
			&dev_attr_bypass_mode)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'bypass_mode'\n",
				dvbm_driver_name);
		}

		if ((err = device_create_file (card->dev,
			&dev_attr_bypass_status)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'bypass_status'\n",
				dvbm_driver_name);
		}
	}
	if (card->capabilities & MASTER_CAP_BLACKBURST) {
		if ((err = device_create_file (card->dev,
			&dev_attr_blackburst_type)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'blackburst_type'\n",
				dvbm_driver_name);
		}
	}
	if (card->capabilities & MASTER_CAP_GPI) {
		if ((err = device_create_file (card->dev,
			&dev_attr_gpi)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'gpi'\n",
				dvbm_driver_name);
		}
	}
	if (card->capabilities & MASTER_CAP_GPO) {
		if ((err = device_create_file (card->dev,
			&dev_attr_gpo)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'gpo'\n",
				dvbm_driver_name);
		}
	}
	if (card->capabilities & MASTER_CAP_UID) {
		if ((err = device_create_file (card->dev,
			&dev_attr_uid)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'uid'\n",
				dvbm_driver_name);
		}
	}
	if (card->capabilities & MASTER_CAP_WATCHDOG) {
		if ((err = device_create_file (card->dev,
			&dev_attr_watchdog)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'watchdog'\n",
				dvbm_driver_name);
		}
	}

	/* Register a transmit interface */
	cap = ASI_CAP_TX_SETCLKSRC | ASI_CAP_TX_FIFOUNDERRUN |
		ASI_CAP_TX_DATA | ASI_CAP_TX_RXCLKSRC;
	switch (pdev->device) {
	default:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDU:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDU_R:
		cap |= ASI_CAP_TX_MAKE204 | ASI_CAP_TX_FINETUNING |
			ASI_CAP_TX_BYTECOUNTER |
			ASI_CAP_TX_LARGEIB |
			ASI_CAP_TX_INTERLEAVING |
			ASI_CAP_TX_27COUNTER |
			ASI_CAP_TX_TIMESTAMPS |
			ASI_CAP_TX_NULLPACKETS;
		if (card->version >= 0x0e07) {
			cap |= ASI_CAP_TX_PTIMESTAMPS;
		}
		transport = ASI_CTL_TRANSPORT_DVB_ASI;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDE_R:
		cap |= ASI_CAP_TX_MAKE204 | ASI_CAP_TX_FINETUNING |
			ASI_CAP_TX_BYTECOUNTER |
			ASI_CAP_TX_LARGEIB |
			ASI_CAP_TX_INTERLEAVING |
			ASI_CAP_TX_CHANGENEXTIP |
			ASI_CAP_TX_27COUNTER |
			ASI_CAP_TX_TIMESTAMPS |
			ASI_CAP_TX_PTIMESTAMPS |
			ASI_CAP_TX_NULLPACKETS;
		if (pci_resource_len (pdev, 2) == 256) {
			cap |= ASI_CAP_TX_PCRSTAMP;
		}
		transport = ASI_CTL_TRANSPORT_DVB_ASI;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDB:
		cap |= ASI_CAP_TX_MAKE204 | ASI_CAP_TX_FINETUNING |
			ASI_CAP_TX_BYTECOUNTER |
			ASI_CAP_TX_LARGEIB |
			ASI_CAP_TX_INTERLEAVING |
			ASI_CAP_TX_27COUNTER |
			ASI_CAP_TX_TIMESTAMPS |
			ASI_CAP_TX_NULLPACKETS;
		if (card->version >= 0x0e00) {
			cap |= ASI_CAP_TX_PTIMESTAMPS;
		}
		if (card->version >= 0x0f11) {
			cap |= ASI_CAP_TX_CHANGENEXTIP;
			if (pci_resource_len (pdev, 2) == 256) {
				cap |= ASI_CAP_TX_PCRSTAMP;
			}
		}
		transport = ASI_CTL_TRANSPORT_DVB_ASI;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDEB:
		cap |= ASI_CAP_TX_MAKE204 | ASI_CAP_TX_FINETUNING |
			ASI_CAP_TX_BYTECOUNTER |
			ASI_CAP_TX_LARGEIB |
			ASI_CAP_TX_INTERLEAVING |
			ASI_CAP_TX_CHANGENEXTIP |
			ASI_CAP_TX_27COUNTER |
			ASI_CAP_TX_TIMESTAMPS |
			ASI_CAP_TX_PTIMESTAMPS |
			ASI_CAP_TX_NULLPACKETS;
		if (pci_resource_len (pdev, 2) == 256) {
			cap |= ASI_CAP_TX_PCRSTAMP;
		}
		transport = ASI_CTL_TRANSPORT_DVB_ASI;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDB_R:
		cap |= ASI_CAP_TX_MAKE204 | ASI_CAP_TX_FINETUNING |
			ASI_CAP_TX_BYTECOUNTER |
			ASI_CAP_TX_LARGEIB |
			ASI_CAP_TX_INTERLEAVING |
			ASI_CAP_TX_27COUNTER |
			ASI_CAP_TX_TIMESTAMPS |
			ASI_CAP_TX_NULLPACKETS;
		if (card->version >= 0x0e00) {
			cap |= ASI_CAP_TX_PTIMESTAMPS;
		}
		transport = ASI_CTL_TRANSPORT_DVB_ASI;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDEB_R:
		cap |= ASI_CAP_TX_MAKE204 | ASI_CAP_TX_FINETUNING |
			ASI_CAP_TX_BYTECOUNTER |
			ASI_CAP_TX_LARGEIB |
			ASI_CAP_TX_INTERLEAVING |
			ASI_CAP_TX_CHANGENEXTIP |
			ASI_CAP_TX_27COUNTER |
			ASI_CAP_TX_TIMESTAMPS |
			ASI_CAP_TX_PTIMESTAMPS |
			ASI_CAP_TX_NULLPACKETS;
		if (pci_resource_len (pdev, 2) == 256) {
			cap |= ASI_CAP_TX_PCRSTAMP;
		}
		transport = ASI_CTL_TRANSPORT_DVB_ASI;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD_R:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD_RS:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE_R:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE_RS:
		cap |= ASI_CAP_TX_MAKE204 | ASI_CAP_TX_FINETUNING |
			ASI_CAP_TX_BYTECOUNTER |
			ASI_CAP_TX_LARGEIB |
			ASI_CAP_TX_INTERLEAVING |
			ASI_CAP_TX_27COUNTER |
			ASI_CAP_TX_TIMESTAMPS |
			ASI_CAP_TX_PTIMESTAMPS |
			ASI_CAP_TX_NULLPACKETS;
		if (card->version >= 0x1200) {
			cap |= ASI_CAP_TX_CHANGENEXTIP;
			if (pci_resource_len (pdev, 2) == 256) {
				cap |= ASI_CAP_TX_PCRSTAMP;
			}
		}
		transport = ASI_CTL_TRANSPORT_DVB_ASI;
		break;
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FD:
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FD_R:
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FD_RS:
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FDE:
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FDE_R:
		transport = ASI_CTL_TRANSPORT_SMPTE_310M;
		break;
	}
	if ((err = asi_register_iface (card,
		&plx_dma_ops,
		DVBM_FDU_FIFO,
		MASTER_DIRECTION_TX,
		&dvbm_fdu_txfops,
		&dvbm_fdu_txops,
		cap,
		4,
		transport)) < 0) {
		goto NO_IFACE;
	}

	/* Register a receive interface */
	cap = ASI_CAP_RX_SYNC | ASI_CAP_RX_INVSYNC | ASI_CAP_RX_CD;
	switch (pdev->device) {
	default:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDU:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDU_R:
		cap |= ASI_CAP_RX_MAKE188 |
			ASI_CAP_RX_BYTECOUNTER |
			ASI_CAP_RX_DATA |
			ASI_CAP_RX_PIDFILTER |
			ASI_CAP_RX_27COUNTER |
			ASI_CAP_RX_TIMESTAMPS | ASI_CAP_RX_PTIMESTAMPS |
			ASI_CAP_RX_NULLPACKETS;
		if (1) { /* XXX fix this */
			cap |= ASI_CAP_RX_PIDCOUNTER |
				ASI_CAP_RX_4PIDCOUNTER;
		}
		transport = ASI_CTL_TRANSPORT_DVB_ASI;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDE_R:
		cap |= ASI_CAP_RX_MAKE188 |
			ASI_CAP_RX_BYTECOUNTER |
			ASI_CAP_RX_DATA |
			ASI_CAP_RX_PIDFILTER |
			ASI_CAP_RX_PIDCOUNTER |
			ASI_CAP_RX_4PIDCOUNTER |
			ASI_CAP_RX_27COUNTER |
			ASI_CAP_RX_TIMESTAMPS |
			ASI_CAP_RX_PTIMESTAMPS |
			ASI_CAP_RX_NULLPACKETS;
		transport = ASI_CTL_TRANSPORT_DVB_ASI;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDB:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDB_R:
		cap |= ASI_CAP_RX_MAKE188 |
			ASI_CAP_RX_BYTECOUNTER |
			ASI_CAP_RX_DATA |
			ASI_CAP_RX_PIDFILTER |
			ASI_CAP_RX_TIMESTAMPS | ASI_CAP_RX_PTIMESTAMPS |
			ASI_CAP_RX_NULLPACKETS;
		if (card->version < 0x0e03) {
			cap |= ASI_CAP_RX_27COUNTER;
		}
		transport = ASI_CTL_TRANSPORT_DVB_ASI;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDEB:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDEB_R:
		cap |= ASI_CAP_RX_MAKE188 |
			ASI_CAP_RX_BYTECOUNTER |
			ASI_CAP_RX_DATA |
			ASI_CAP_RX_PIDFILTER |
			ASI_CAP_RX_27COUNTER |
			ASI_CAP_RX_TIMESTAMPS |
			ASI_CAP_RX_PTIMESTAMPS |
			ASI_CAP_RX_NULLPACKETS;
		transport = ASI_CTL_TRANSPORT_DVB_ASI;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD_R:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE_R:
		cap |= ASI_CAP_RX_MAKE188 |
			ASI_CAP_RX_BYTECOUNTER |
			ASI_CAP_RX_DATA |
			ASI_CAP_RX_PIDFILTER |
			ASI_CAP_RX_27COUNTER |
			ASI_CAP_RX_TIMESTAMPS |
			ASI_CAP_RX_PTIMESTAMPS |
			ASI_CAP_RX_NULLPACKETS;
		transport = ASI_CTL_TRANSPORT_DVB_ASI;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD_RS:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE_RS:
		cap |= ASI_CAP_RX_MAKE188 |
			ASI_CAP_RX_BYTECOUNTER |
			ASI_CAP_RX_DATA |
			ASI_CAP_RX_PIDFILTER |
			ASI_CAP_RX_TIMESTAMPS | ASI_CAP_RX_PTIMESTAMPS |
			ASI_CAP_RX_NULLPACKETS | ASI_CAP_RX_REDUNDANT;
		if (card->version >= 0x1201) {
			cap |= ASI_CAP_RX_PIDCOUNTER |
				ASI_CAP_RX_4PIDCOUNTER;
		}
		if (card->version >= 0x1202) {
			cap |= ASI_CAP_RX_DATA2;
		}
		transport = ASI_CTL_TRANSPORT_DVB_ASI;
		break;
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FD:
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FD_R:
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FDE:
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FDE_R:
		transport = ASI_CTL_TRANSPORT_SMPTE_310M;
		break;
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FD_RS:
		cap |= ASI_CAP_RX_REDUNDANT | ASI_CAP_RX_PIDCOUNTER |
			ASI_CAP_RX_4PIDCOUNTER;
		transport = ASI_CTL_TRANSPORT_SMPTE_310M;
		break;
	}
	if ((err = asi_register_iface (card,
		&plx_dma_ops,
		DVBM_FDU_FIFO,
		MASTER_DIRECTION_RX,
		&dvbm_fdu_rxfops,
		&dvbm_fdu_rxops,
		cap,
		4,
		transport)) < 0) {
		goto NO_IFACE;
	}

	return 0;

NO_IFACE:
NO_DEV:
NO_MEM:
	dvbm_fdu_pci_remove (pdev);
NO_PCI:
	return err;
}

/**
 * dvbm_fdu_pci_remove - PCI removal handler for a DVB Master FD-U
 * @pdev: PCI device
 *
 * Handle the removal of a DVB Master FD-U.
 * This function may be called during PCI probe error handling,
 * so don't mark it as __devexit.
 **/
void
dvbm_fdu_pci_remove (struct pci_dev *pdev)
{
	struct master_dev *card = pci_get_drvdata (pdev);

	if (card) {
		/* Unregister the device and all interfaces */
		dvbm_unregister_all (card);

		if (card->capabilities & MASTER_CAP_BYPASS) {
			master_outl (card, DVBM_FDU_CSR, 0);
		}
		iounmap (card->bridge_addr);
		kfree (card);
	}
	dvbm_pci_remove_generic (pdev);
	return;
}

/**
 * dvbm_fdu_irq_handler - DVB Master FD-U interrupt service routine
 * @irq: interrupt number
 * @dev_id: pointer to the device data structure
 * @regs: processor context
 **/
static irqreturn_t
IRQ_HANDLER(dvbm_fdu_irq_handler,irq,dev_id,regs)
{
	struct master_dev *card = dev_id;
	unsigned int intcsr = readl (card->bridge_addr + PLX_INTCSR);
	unsigned int status, interrupting_iface = 0;
	struct master_iface *txiface = list_entry (card->iface_list.next,
		struct master_iface, list);
	struct master_iface *rxiface = list_entry (card->iface_list.prev,
		struct master_iface, list);

	if (intcsr & PLX_INTCSR_DMA0INT_ACTIVE) {
		/* Read the interrupt type and clear it */
		status = readb (card->bridge_addr + PLX_DMACSR0);
		writeb (PLX_DMACSR_ENABLE | PLX_DMACSR_CLINT,
			card->bridge_addr + PLX_DMACSR0);

		/* Increment the buffer pointer */
		mdma_advance (txiface->dma);

		/* Flag end-of-chain */
		if (status & PLX_DMACSR_DONE) {
			set_bit (ASI_EVENT_TX_BUFFER_ORDER, &txiface->events);
			set_bit (0, &txiface->dma_done);
		}

		interrupting_iface |= 0x1;
	}
	if (intcsr & PLX_INTCSR_DMA1INT_ACTIVE) {
		struct master_dma *dma = rxiface->dma;

		/* Read the interrupt type and clear it */
		spin_lock (&card->irq_lock);
		status = readb (card->bridge_addr + PLX_DMACSR1);
		writeb (status | PLX_DMACSR_CLINT,
			card->bridge_addr + PLX_DMACSR1);
		spin_unlock (&card->irq_lock);

		/* Increment the buffer pointer */
		mdma_advance (dma);

		if (mdma_rx_isempty (dma)) {
			set_bit (ASI_EVENT_RX_BUFFER_ORDER, &rxiface->events);
		}

		/* Flag end-of-chain */
		if (status & PLX_DMACSR_DONE) {
			set_bit (0, &rxiface->dma_done);
		}

		interrupting_iface |= 0x2;
	}
	if (intcsr & PLX_INTCSR_PCILOCINT_ACTIVE) {
		/* Clear the source of the interrupt */
		spin_lock (&card->irq_lock);
		status = master_inl (card, DVBM_FDU_ICSR);
		master_outl (card, DVBM_FDU_ICSR, status);
		spin_unlock (&card->irq_lock);

		if (status & DVBM_FDU_ICSR_TXUIS) {
			set_bit (ASI_EVENT_TX_FIFO_ORDER,
				&txiface->events);
			interrupting_iface |= 0x1;
		}
		if (status & DVBM_FDU_ICSR_TXDIS) {
			set_bit (ASI_EVENT_TX_DATA_ORDER,
				&txiface->events);
			interrupting_iface |= 0x1;
		}
		if (status & DVBM_FDU_ICSR_RXCDIS) {
			set_bit (ASI_EVENT_RX_CARRIER_ORDER,
				&rxiface->events);
			interrupting_iface |= 0x2;
		}
		if (status & DVBM_FDU_ICSR_RXAOSIS) {
			set_bit (ASI_EVENT_RX_AOS_ORDER,
				&rxiface->events);
			interrupting_iface |= 0x2;
		}
		if (status & DVBM_FDU_ICSR_RXLOSIS) {
			set_bit (ASI_EVENT_RX_LOS_ORDER,
				&rxiface->events);
			interrupting_iface |= 0x2;
		}
		if (status & DVBM_FDU_ICSR_RXOIS) {
			set_bit (ASI_EVENT_RX_FIFO_ORDER,
				&rxiface->events);
			interrupting_iface |= 0x2;
		}
		if (status & DVBM_FDU_ICSR_RXDIS) {
			set_bit (ASI_EVENT_RX_DATA_ORDER,
				&rxiface->events);
			interrupting_iface |= 0x2;
		}
	}

	if (interrupting_iface) {
		/* Dummy read to flush PCI posted writes */
		readb (card->bridge_addr + PLX_DMACSR1);

		if (interrupting_iface & 0x1) {
			wake_up (&txiface->queue);
		}
		if (interrupting_iface & 0x2) {
			wake_up (&rxiface->queue);
		}
		return IRQ_HANDLED;
	}
	return IRQ_NONE;
}

/**
 * dvbm_fdu_txinit - Initialize the DVB Master FD-U transmitter
 * @iface: interface
 **/
static void
dvbm_fdu_txinit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg = iface->null_packets ? DVBM_FDU_TCSR_NP : 0;

	switch (iface->timestamps) {
	default:
	case ASI_CTL_TSTAMP_NONE:
		reg |= 0;
		break;
	case ASI_CTL_TSTAMP_APPEND:
		reg |= DVBM_FDU_TCSR_TSS;
		break;
	case ASI_CTL_TSTAMP_PREPEND:
		reg |= DVBM_FDU_TCSR_PRC;
		break;
	}
	switch (iface->mode) {
	default:
	case ASI_CTL_TX_MODE_188:
		reg |= 0;
		break;
	case ASI_CTL_TX_MODE_204:
		reg |= DVBM_FDU_TCSR_204;
		break;
	case ASI_CTL_TX_MODE_MAKE204:
		reg |= DVBM_FDU_TCSR_MAKE204;
		break;
	}
	switch (iface->clksrc) {
	default:
	case ASI_CTL_TX_CLKSRC_ONBOARD:
		reg |= 0;
		break;
	case ASI_CTL_TX_CLKSRC_EXT:
		reg |= DVBM_FDU_TCSR_EXTCLK;
		break;
	case ASI_CTL_TX_CLKSRC_RX:
		reg |= DVBM_FDU_TCSR_RXCLK;
		break;
	}
	spin_lock (&card->reg_lock);
	reg |= master_inl (card, DVBM_FDU_TCSR) & DVBM_FDU_TCSR_PAL;
	master_outl (card, DVBM_FDU_TCSR, reg | DVBM_FDU_TCSR_RST);
	master_outl (card, DVBM_FDU_TCSR, reg);
	spin_unlock (&card->reg_lock);
	master_outl (card, DVBM_FDU_TFCR,
		(DVBM_FDU_TFSL << 16) | DVBM_FDU_TDMATL);
	/* There will be no races on IBSTR, IPSTR, and FTR
	 * until this code returns, so we don't need to lock them */
	master_outl (card, DVBM_FDU_IBSTR, 0);
	master_outl (card, DVBM_FDU_IPSTR, 0);
	master_outl (card, DVBM_FDU_FTR, 0);

	/* Reset byte counter */
	master_inl (card, DVBM_FDU_TXBCOUNTR);

	/* Do not touch registers above 128 bytes
	 * unless we know they exist */
	if (iface->capabilities & ASI_CAP_TX_PCRSTAMP) {
		master_outl (card, DVBM_FDU_TPIDR, 0);
		master_outl (card, DVBM_FDU_TSTAMPR_HI, 0);
	}
	return;
}

/**
 * dvbm_fdu_txstart - Activate the DVB Master FD-U transmitter
 * @iface: interface
 **/
static void
dvbm_fdu_txstart (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;

	/* Enable DMA */
	writeb (PLX_DMACSR_ENABLE, card->bridge_addr + PLX_DMACSR0);

	/* Enable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = master_inl (card, DVBM_FDU_ICSR) &
		DVBM_FDU_ICSR_RXCTRL_MASK;
	reg |= DVBM_FDU_ICSR_TXUIE | DVBM_FDU_ICSR_TXDIE;
	master_outl (card, DVBM_FDU_ICSR, reg);
	spin_unlock_irq (&card->irq_lock);

	/* Enable the transmitter */
	spin_lock (&card->reg_lock);
	reg = master_inl (card, DVBM_FDU_TCSR);
	master_outl (card, DVBM_FDU_TCSR, reg | DVBM_FDU_TCSR_EN);
	spin_unlock (&card->reg_lock);

	return;
}

/**
 * dvbm_fdu_txstop - Deactivate the DVB Master FD-U transmitter
 * @iface: interface
 **/
static void
dvbm_fdu_txstop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	struct master_dma *dma = iface->dma;
	unsigned int reg;

	plx_tx_link_all (dma);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	plx_reset (dma);

	if (!iface->null_packets) {
		/* Wait for the onboard FIFOs to empty */
		/* Atomic read of ICSR, so we don't need to lock */
		wait_event (iface->queue,
			!(master_inl (card, DVBM_FDU_ICSR) & DVBM_FDU_ICSR_TXD));
	}

	/* Disable the transmitter */
	spin_lock (&card->reg_lock);
	reg = master_inl (card, DVBM_FDU_TCSR);
	master_outl (card, DVBM_FDU_TCSR, reg & ~DVBM_FDU_TCSR_EN);
	spin_unlock (&card->reg_lock);

	/* Disable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = master_inl (card, DVBM_FDU_ICSR) &
		DVBM_FDU_ICSR_RXCTRL_MASK;
	reg |= DVBM_FDU_ICSR_TXUIS | DVBM_FDU_ICSR_TXDIS;
	master_outl (card, DVBM_FDU_ICSR, reg);
	spin_unlock_irq (&card->irq_lock);

	/* Disable DMA */
	writeb (0, card->bridge_addr + PLX_DMACSR0);

	return;
}

/**
 * dvbm_fdu_txexit - Clean up the DVB Master FD-U transmitter
 * @iface: interface
 **/
static void
dvbm_fdu_txexit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;

	/* Reset the transmitter */
	spin_lock (&card->reg_lock);
	reg = master_inl (card, DVBM_FDU_TCSR);
	master_outl (card, DVBM_FDU_TCSR, reg | DVBM_FDU_TCSR_RST);
	spin_unlock (&card->reg_lock);

	return;
}

/**
 * dvbm_fdu_start_tx_dma - start transmit DMA
 * @iface: interface
 **/
static void
dvbm_fdu_start_tx_dma (struct master_iface *iface)
{
	struct master_dev *card = iface->card;

	writel (plx_head_desc_bus_addr (iface->dma) |
		PLX_DMADPR_DLOC_PCI,
		card->bridge_addr + PLX_DMADPR0);
	clear_bit (0, &iface->dma_done);
	wmb ();
	writeb (PLX_DMACSR_ENABLE | PLX_DMACSR_START,
		card->bridge_addr + PLX_DMACSR0);
	/* Dummy read to flush PCI posted writes */
	readb (card->bridge_addr + PLX_DMACSR0);
	return;
}

/**
 * dvbm_fdu_txunlocked_ioctl - DVB Master FD-U transmitter unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
dvbm_fdu_txunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	struct asi_txstuffing stuffing;
	unsigned int val;
	struct asi_pcrstamp pcrstamp;
	int i;

	switch (cmd) {
	case ASI_IOC_TXSETSTUFFING:
		if (iface->transport != ASI_CTL_TRANSPORT_DVB_ASI) {
			return -ENOTTY;
		}
		if (copy_from_user (&stuffing,
			(struct asi_txstuffing __user *)arg,
			sizeof (stuffing))) {
			return -EFAULT;
		}
		if ((stuffing.ib > 0xffff) ||
			(stuffing.ip > 0xffffff) ||
			(stuffing.normal_ip > 0xff) ||
			(stuffing.big_ip > 0xff) ||
			((stuffing.il_normal + stuffing.il_big) > 0xf) ||
			(stuffing.il_normal > stuffing.normal_ip) ||
			(stuffing.il_big > stuffing.big_ip)) {
			return -EINVAL;
		}
		spin_lock (&card->reg_lock);
		master_outl (card, DVBM_FDU_IBSTR, stuffing.ib);
		master_outl (card, DVBM_FDU_IPSTR, stuffing.ip);
		master_outl (card, DVBM_FDU_FTR,
			(stuffing.il_big << DVBM_FDU_FTR_ILBIG_SHIFT) |
			(stuffing.big_ip << DVBM_FDU_FTR_BIGIP_SHIFT) |
			(stuffing.il_normal << DVBM_FDU_FTR_ILNORMAL_SHIFT) |
			stuffing.normal_ip);
		spin_unlock (&card->reg_lock);
		break;
	case ASI_IOC_TXGETBYTECOUNT:
		if (!(iface->capabilities & ASI_CAP_TX_BYTECOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (master_inl (card, DVBM_FDU_TXBCOUNTR),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_TXGETTXD:
		/* Atomic read of ICSR, so we don't need to lock */
		if (put_user ((master_inl (card, DVBM_FDU_ICSR) &
			DVBM_FDU_ICSR_TXD) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_TXGET27COUNT:
		if (!(iface->capabilities & ASI_CAP_TX_27COUNTER)) {
			return -ENOTTY;
		}
		if (put_user (master_inl (card, DVBM_FDU_27COUNTR),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_TXSETPID_DEPRECATED:
	case ASI_IOC_TXSETPID:
		if (!(iface->capabilities & ASI_CAP_TX_PCRSTAMP)) {
			return -ENOTTY;
		}
		if (get_user (val, (unsigned int __user *)arg)) {
			return -EFAULT;
		}
		if (val > 0x1fff) {
			return -EINVAL;
		}
		master_outl (card, DVBM_FDU_TPIDR, val);
		break;
	case ASI_IOC_TXGETPCRSTAMP:
		if (!(iface->capabilities & ASI_CAP_TX_PCRSTAMP)) {
			return -ENOTTY;
		}
		spin_lock (&card->reg_lock);
		master_outl (card, DVBM_FDU_TSTAMPR_HI, DVBM_FDU_TSTAMPR_LOCK);
		val = master_inl (card, DVBM_FDU_TPCRR_HI);
		pcrstamp.adaptation_field_length = val >> 24;
		pcrstamp.adaptation_field_flags = (val >> 16) & 0x000000ff;
		pcrstamp.PCR[0] = (val >> 8) & 0x000000ff;
		pcrstamp.PCR[1] = val & 0x000000ff;
		val = master_inl (card, DVBM_FDU_TPCRR_LO);
		pcrstamp.PCR[2] = (val >> 24) & 0x000000ff;
		pcrstamp.PCR[3] = (val >> 16) & 0x000000ff;
		pcrstamp.PCR[4] = (val >> 8) & 0x000000ff;
		pcrstamp.PCR[5] = val & 0x000000ff;
		pcrstamp.count = master_inl (card, DVBM_FDU_TSTAMPR_HI) &
			~DVBM_FDU_TSTAMPR_LOCK;
		pcrstamp.count <<= 32;
		pcrstamp.count |= master_inl (card, DVBM_FDU_TSTAMPR_LO);
		mb ();
		master_outl (card, DVBM_FDU_TSTAMPR_HI, 0);
		spin_unlock (&card->reg_lock);
		if (copy_to_user ((struct asi_pcrstamp __user *)arg, &pcrstamp,
			sizeof (pcrstamp))) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_TXCHANGENEXTIP_DEPRECATED:
	case ASI_IOC_TXCHANGENEXTIP:
		if (!(iface->capabilities & ASI_CAP_TX_CHANGENEXTIP)) {
			return -ENOTTY;
		}
		if (get_user (i, (int __user *)arg)) {
			return -EFAULT;
		}
		switch (i) {
		case -1:
			val = DVBM_FDU_IPSTR_DELETE |
				DVBM_FDU_IPSTR_CHANGENEXT;
			break;
		case 0:
			return 0;
		case 1:
			val = DVBM_FDU_IPSTR_CHANGENEXT;
			break;
		default:
			return -EINVAL;
		}
		spin_lock (&card->reg_lock);
		master_outl (card, DVBM_FDU_IPSTR,
			master_inl (card, DVBM_FDU_IPSTR) | val);
		spin_unlock (&card->reg_lock);
		break;
	default:
		return asi_txioctl (filp, cmd, arg);
	}
	return 0;
}

/**
 * dvbm_fdu_txfsync - DVB Master FD-U transmitter fsync() method
 * @filp: file to flush
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
FSYNC_HANDLER(dvbm_fdu_txfsync,filp,datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dma *dma = iface->dma;

	mutex_lock (&iface->buf_mutex);
	plx_tx_link_all (dma);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	plx_reset (dma);

	if (!iface->null_packets) {
		struct master_dev *card = iface->card;

		/* Wait for the onboard FIFOs to empty */
		/* Atomic read of ICSR, so we don't need to lock */
		wait_event (iface->queue,
			!(master_inl (card, DVBM_FDU_ICSR) &
			DVBM_FDU_ICSR_TXD));
	}

	mutex_unlock (&iface->buf_mutex);
	return 0;
}

/**
 * dvbm_fdu_rxinit - Initialize the DVB Master FD-U receiver
 * @iface: interface
 **/
static void
dvbm_fdu_rxinit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int i, reg = DVBM_FDU_RCSR_RF |
		(iface->null_packets ? DVBM_FDU_RCSR_NP : 0);

	switch (iface->timestamps) {
	default:
	case ASI_CTL_TSTAMP_NONE:
		reg |= 0;
		break;
	case ASI_CTL_TSTAMP_APPEND:
		reg |= DVBM_FDU_RCSR_TSE;
		break;
	case ASI_CTL_TSTAMP_PREPEND:
		reg |= DVBM_FDU_RCSR_PTSE;
		break;
	}
	switch (iface->mode) {
	default:
	case ASI_CTL_RX_MODE_RAW:
		reg |= 0;
		break;
	case ASI_CTL_RX_MODE_188:
		reg |= DVBM_FDU_RCSR_188 | DVBM_FDU_RCSR_PFE;
		break;
	case ASI_CTL_RX_MODE_204:
		reg |= DVBM_FDU_RCSR_204 | DVBM_FDU_RCSR_PFE;
		break;
	case ASI_CTL_RX_MODE_AUTO:
		reg |= DVBM_FDU_RCSR_AUTO | DVBM_FDU_RCSR_PFE;
		break;
	case ASI_CTL_RX_MODE_AUTOMAKE188:
		reg |= DVBM_FDU_RCSR_AUTO | DVBM_FDU_RCSR_RSS |
			DVBM_FDU_RCSR_PFE;
		break;
	case ASI_CTL_RX_MODE_204MAKE188:
		reg |= DVBM_FDU_RCSR_204 | DVBM_FDU_RCSR_RSS |
			DVBM_FDU_RCSR_PFE;
		break;
	}
	/* There will be no races on RCSR
	 * until this code returns, so we don't need to lock it */
	master_outl (card, DVBM_FDU_RCSR, reg | DVBM_FDU_RCSR_RST);
	wmb ();
	master_outl (card, DVBM_FDU_RCSR, reg);
	wmb ();
	master_outl (card, DVBM_FDU_RFCR, DVBM_FDU_RDMATL);

	/* Reset byte counter */
	master_inl (card, DVBM_FDU_RXBCOUNTR);

	/* Reset PID filter.
	 * There will be no races on PFLUT
	 * until this code returns, so we don't need to lock it */
	for (i = 0; i < 256; i++) {
		master_outl (card, DVBM_FDU_PFLUTAR, i);
		wmb ();
		master_outl (card, DVBM_FDU_PFLUTR, 0xffffffff);
		wmb ();
	}

	/* Clear PID registers */
	master_outl (card, DVBM_FDU_PIDR0, 0);
	master_outl (card, DVBM_FDU_PIDR1, 0);
	master_outl (card, DVBM_FDU_PIDR2, 0);
	master_outl (card, DVBM_FDU_PIDR3, 0);

	/* Reset PID counters */
	master_inl (card, DVBM_FDU_PIDCOUNTR0);
	master_inl (card, DVBM_FDU_PIDCOUNTR1);
	master_inl (card, DVBM_FDU_PIDCOUNTR2);
	master_inl (card, DVBM_FDU_PIDCOUNTR3);

	return;
}

/**
 * dvbm_fdu_rxstart - Activate the DVB Master FD-U receiver
 * @iface: interface
 **/
static void
dvbm_fdu_rxstart (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;

	/* Enable and start DMA */
	writel (plx_head_desc_bus_addr (iface->dma) |
		PLX_DMADPR_DLOC_PCI | PLX_DMADPR_LB2PCI,
		card->bridge_addr + PLX_DMADPR1);
	writeb (PLX_DMACSR_ENABLE,
		card->bridge_addr + PLX_DMACSR1);
	clear_bit (0, &iface->dma_done);
	wmb ();
	writeb (PLX_DMACSR_ENABLE | PLX_DMACSR_START,
		card->bridge_addr + PLX_DMACSR1);
	/* Dummy read to flush PCI posted writes */
	readb (card->bridge_addr + PLX_DMACSR1);

	/* Enable receiver interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = master_inl (card, DVBM_FDU_ICSR) &
		DVBM_FDU_ICSR_TXCTRL_MASK;
	reg |= DVBM_FDU_ICSR_RXCDIE | DVBM_FDU_ICSR_RXAOSIE |
		DVBM_FDU_ICSR_RXLOSIE | DVBM_FDU_ICSR_RXOIE |
		DVBM_FDU_ICSR_RXDIE;
	master_outl (card, DVBM_FDU_ICSR, reg);
	spin_unlock_irq (&card->irq_lock);

	/* Enable the receiver */
	spin_lock (&card->reg_lock);
	reg = master_inl (card, DVBM_FDU_RCSR);
	master_outl (card, DVBM_FDU_RCSR, reg | DVBM_FDU_RCSR_EN);
	spin_unlock (&card->reg_lock);

	return;
}

/**
 * dvbm_fdu_rxstop - Deactivate the DVB Master FD-U receiver
 * @iface: interface
 **/
static void
dvbm_fdu_rxstop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;

	/* Disable the receiver */
	spin_lock (&card->reg_lock);
	reg = master_inl (card, DVBM_FDU_RCSR);
	master_outl (card, DVBM_FDU_RCSR, reg & ~DVBM_FDU_RCSR_EN);
	spin_unlock (&card->reg_lock);

	/* Disable receiver interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = master_inl (card, DVBM_FDU_ICSR) &
		DVBM_FDU_ICSR_TXCTRL_MASK;
	reg |= DVBM_FDU_ICSR_RXCDIS | DVBM_FDU_ICSR_RXAOSIS |
		DVBM_FDU_ICSR_RXLOSIS | DVBM_FDU_ICSR_RXOIS |
		DVBM_FDU_ICSR_RXDIS;
	master_outl (card, DVBM_FDU_ICSR, reg);

	/* Disable and abort DMA */
	writeb (0, card->bridge_addr + PLX_DMACSR1);
	wmb ();
	writeb (PLX_DMACSR_START | PLX_DMACSR_ABORT,
		card->bridge_addr + PLX_DMACSR1);
	/* Dummy read to flush PCI posted writes */
	readb (card->bridge_addr + PLX_DMACSR1);
	spin_unlock_irq (&card->irq_lock);
	wait_event_timeout (iface->queue, test_bit (0, &iface->dma_done), HZ);

	return;
}

/**
 * dvbm_fdu_rxexit - Clean up the DVB Master FD-U receiver
 * @iface: interface
 **/
static void
dvbm_fdu_rxexit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;

	/* Reset the receiver.
	 * There will be no races on RCSR here,
	 * so we don't need to lock it */
	master_outl (card, DVBM_FDU_RCSR, DVBM_FDU_RCSR_RST);

	return;
}

/**
 * dvbm_fdu_rxunlocked_ioctl - DVB Master FD-U receiver unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
dvbm_fdu_rxunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	int val;
	unsigned int reg = 0, *pflut, i;

	switch (cmd) {
	case ASI_IOC_RXGETSTATUS:
		/* Atomic reads of ICSR and RCSR, so we don't need to lock */
		reg = master_inl (card, DVBM_FDU_ICSR);
		switch (master_inl (card, DVBM_FDU_RCSR) &
			DVBM_FDU_RCSR_SYNC_MASK) {
		case 0:
			val = 1;
			break;
		case DVBM_FDU_RCSR_188:
			val = (reg & DVBM_FDU_ICSR_RXPASSING) ? 188 : 0;
			break;
		case DVBM_FDU_RCSR_204:
			val = (reg & DVBM_FDU_ICSR_RXPASSING) ? 204 : 0;
			break;
		case DVBM_FDU_RCSR_AUTO:
			if (reg & DVBM_FDU_ICSR_RXPASSING) {
				val = (reg & DVBM_FDU_ICSR_RX204) ? 204 : 188;
			} else {
				val = 0;
			}
			break;
		default:
			return -EIO;
		}
		if (put_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXGETBYTECOUNT:
		if (!(iface->capabilities & ASI_CAP_RX_BYTECOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (master_inl (card, DVBM_FDU_RXBCOUNTR),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETINVSYNC:
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		switch (val) {
		case 0:
			reg |= 0;
			break;
		case 1:
			reg |= DVBM_FDU_RCSR_INVSYNC;
			break;
		default:
			return -EINVAL;
		}
		spin_lock (&card->reg_lock);
		master_outl (card, DVBM_FDU_RCSR,
			(master_inl (card, DVBM_FDU_RCSR) &
			~DVBM_FDU_RCSR_INVSYNC) | reg);
		spin_unlock (&card->reg_lock);
		break;
	case ASI_IOC_RXGETCARRIER:
		/* Atomic read of ICSR, so we don't need to lock */
		if (put_user ((master_inl (card, DVBM_FDU_ICSR) &
			DVBM_FDU_ICSR_RXCD) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETDSYNC:
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if (val) {
			return -EINVAL;
		}
		break;
	case ASI_IOC_RXGETRXD:
		/* Atomic read of ICSR, so we don't need to lock */
		if (put_user ((master_inl (card, DVBM_FDU_ICSR) &
			DVBM_FDU_ICSR_RXD) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPF:
		if (!(iface->capabilities & ASI_CAP_RX_PIDFILTER)) {
			return -ENOTTY;
		}
		pflut = (unsigned int *)
			kmalloc (sizeof (unsigned int [256]), GFP_KERNEL);
		if (pflut == NULL) {
			return -ENOMEM;
		}
		if (copy_from_user (pflut, (unsigned int __user *)arg,
			sizeof (unsigned int [256]))) {
			kfree (pflut);
			return -EFAULT;
		}
		spin_lock (&card->reg_lock);
		for (i = 0; i < 256; i++) {
			master_outl (card, DVBM_FDU_PFLUTAR, i);
			wmb ();
			master_outl (card, DVBM_FDU_PFLUTR, pflut[i]);
			wmb ();
		}
		spin_unlock (&card->reg_lock);
		kfree (pflut);
		break;
	case ASI_IOC_RXSETPID0:
		if (!(iface->capabilities & ASI_CAP_RX_PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		master_outl (card, DVBM_FDU_PIDR0, val);
		/* Reset PID count */
		master_inl (card, DVBM_FDU_PIDCOUNTR0);
		break;
	case ASI_IOC_RXGETPID0COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (master_inl (card, DVBM_FDU_PIDCOUNTR0),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPID1:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		master_outl (card, DVBM_FDU_PIDR1, val);
		/* Reset PID count */
		master_inl (card, DVBM_FDU_PIDCOUNTR1);
		break;
	case ASI_IOC_RXGETPID1COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (master_inl (card, DVBM_FDU_PIDCOUNTR1),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPID2:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		master_outl (card, DVBM_FDU_PIDR2, val);
		/* Reset PID count */
		master_inl (card, DVBM_FDU_PIDCOUNTR2);
		break;
	case ASI_IOC_RXGETPID2COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (master_inl (card, DVBM_FDU_PIDCOUNTR2),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPID3:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		master_outl (card, DVBM_FDU_PIDR3, val);
		/* Reset PID count */
		master_inl (card, DVBM_FDU_PIDCOUNTR3);
		break;
	case ASI_IOC_RXGETPID3COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (master_inl (card, DVBM_FDU_PIDCOUNTR3),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXGET27COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_27COUNTER)) {
			return -ENOTTY;
		}
		if (put_user (master_inl (card, DVBM_FDU_27COUNTR),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXGETSTATUS2:
		if (!(iface->capabilities & ASI_CAP_RX_REDUNDANT)) {
			return -ENOTTY;
		}
		/* Atomic reads of ICSR and RCSR, so we don't need to lock */
		val = ((master_inl (card, DVBM_FDU_ICSR) &
			DVBM_FDU_ICSR_RXRS) ||
			((master_inl (card, DVBM_FDU_RCSR) &
			DVBM_FDU_RCSR_SYNC_MASK) == 0)) ? 1 : 0;
		if (put_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETINPUT_DEPRECATED:
	case ASI_IOC_RXSETINPUT:
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if (!(iface->capabilities & ASI_CAP_RX_REDUNDANT)) {
			if (val) {
				return -EINVAL;
			}
		} else {
			switch (val) {
			case 0:
				reg |= 0;
				break;
			case 1:
				reg |= DVBM_FDU_RCSR_SEL;
				break;
			default:
				return -EINVAL;
			}
			spin_lock (&card->reg_lock);
			master_outl (card, DVBM_FDU_RCSR,
				(master_inl (card, DVBM_FDU_RCSR) &
				~DVBM_FDU_RCSR_SEL) | reg);
			spin_unlock (&card->reg_lock);
		}
		break;
	case ASI_IOC_RXGETRXD2:
		/* Atomic read of ICSR, so we don't need to lock */
		if (put_user ((master_inl (card, DVBM_FDU_ICSR) &
			DVBM_FDU_ICSR_RXD2) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return asi_rxioctl (filp, cmd, arg);
	}
	return 0;
}

/**
 * dvbm_fdu_rxfsync - DVB Master FD-U receiver fsync() method
 * @filp: file to flush
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
FSYNC_HANDLER(dvbm_fdu_rxfsync,filp,datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	unsigned int reg;

	mutex_lock (&iface->buf_mutex);

	/* Stop the receiver */
	dvbm_fdu_rxstop (iface);

	/* Reset the onboard FIFO and driver buffers */
	spin_lock (&card->reg_lock);
	reg = master_inl (card, DVBM_FDU_RCSR);
	master_outl (card, DVBM_FDU_RCSR, reg | DVBM_FDU_RCSR_RST);
	wmb ();
	master_outl (card, DVBM_FDU_RCSR, reg);
	spin_unlock (&card->reg_lock);
	iface->events = 0;
	plx_reset (iface->dma);

	/* Start the receiver */
	dvbm_fdu_rxstart (iface);

	mutex_unlock (&iface->buf_mutex);
	return 0;
}

