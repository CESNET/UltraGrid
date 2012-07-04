/* jtag.c
 *
 * Linux driver for the JTAG interface on
 * some Linear Systems Ltd. boards.
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

#include <linux/version.h> /* LINUX_VERSION_CODE */
#include <linux/kernel.h> /* KERN_INFO */
#include <linux/module.h> /* MODULE_LICENSE */
#include <linux/moduleparam.h> /* module_param () */

#include <linux/init.h> /* module_init () */
#include <linux/fs.h> /* register_chrdev_region () */
#include <linux/pci.h> /* pci_register_driver () */
#include <linux/errno.h> /* error codes */
#include <linux/list.h> /* list_add_tail () */
#include <linux/slab.h> /* kzalloc () */
#include <linux/types.h> /* u32 */
#include <linux/cdev.h> /* cdev_init () */
#include <linux/device.h> /* class_create () */
#include <linux/mutex.h> /* mutex_init () */

#include <asm/io.h> /* readl () */
#include <asm/uaccess.h> /* put_user () */

#include "../include/master.h"
#include "plx9080.h"
#include "dvbm_fdu.h"
#include "dvbm_lpfd.h"
#include "mmas.h"
#include "mmsa.h"
#include "sdim.h"
#include "sdim_qoe.h"
#include "dvbm_lpqo.h"
#include "dvbm_qlf.h"
#include "dvbm_qdual.h"
#include "eeprom.h"
#include "hdsdim_txe.h"
#include "hdsdim_rxe.h"
#include "hdsdim_qie.h"

#ifndef DEFINE_PCI_DEVICE_TABLE
#define DEFINE_PCI_DEVICE_TABLE(_table) \
	const struct pci_device_id _table[]
#endif

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,27))
#define device_create(cls,parent,devt,drvdata,fmt,...) \
	device_create(cls,parent,devt,fmt,##__VA_ARGS__)
#endif

static struct class *lsj_class;

/**
 * struct lsj - Linear Systems Ltd. device
 * @list: device list pointers
 * @core_addr: address of memory region
 * @data_addr: JTAG port address
 * @bridge_addr: address of bridge memory region
 * @lock: mutex
 * @pdev: PCI device
 * @dev: pointer to device
 * @cdev: character device structure
 * @users: usage count
 **/
struct lsj {
	struct list_head list;
	void __iomem *core_addr;
	void __iomem *data_addr;
	void __iomem *bridge_addr;
	struct mutex lock;
	struct pci_dev *pdev;
	struct device *dev;
	struct cdev cdev;
	int users;
};

/* Static function prototypes */
static int lsj_open (struct inode *inode, struct file *filp);
static int lsj_release (struct inode *inode, struct file *filp);
static ssize_t lsj_read (struct file *filp,
	char __user *data,
	size_t length,
	loff_t *offset);
static ssize_t lsj_write (struct file *filp,
	const char __user *data,
	size_t length,
	loff_t *offset);
static int lsj_register (struct lsj *card);
static void lsj_unregister (struct lsj *card);
static ssize_t lsj_show_range (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static ssize_t lsj_store_range (struct device *dev,
	struct device_attribute *attr,
	const char *buf,
	size_t count);
static int lsj_pci_probe (struct pci_dev *pdev,
	const struct pci_device_id *id) __devinit;
static void lsj_pci_remove (struct pci_dev *pdev);
static int lsj_init_module (void) __init;
static void lsj_cleanup_module (void) __exit;

static char lsj_module_name[] = "ls_jtag";

/* The major number can be overridden from the command line
 * when the module is loaded. */
static unsigned int major = 122;
module_param(major,uint,S_IRUGO);
MODULE_PARM_DESC(major,"Major number of the first device, or zero for dynamic");

static const unsigned int count = 256;

MODULE_AUTHOR("Linear Systems Ltd.");
MODULE_DESCRIPTION("JTAG driver");
MODULE_LICENSE("GPL");
MODULE_VERSION(MASTER_DRIVER_VERSION);

static DEFINE_PCI_DEVICE_TABLE(lsj_pci_id_table) = {
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBFDU)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBFDU_R)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBTXU)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBRXU)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			MMSA_PCI_DEVICE_ID_LINSYS)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			SDIM_PCI_DEVICE_ID_LINSYS)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			MMAS_PCI_DEVICE_ID_LINSYS)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBFDB)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBFDB_R)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBLPFD)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBLPFDE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQOE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQOE_MINIBNC)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			SDIM_PCI_DEVICE_ID_LINSYS_SDILPFD)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			SDIM_PCI_DEVICE_ID_LINSYS_SDILPFDE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			SDIM_PCI_DEVICE_ID_LINSYS_SDIQOE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQDUALE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQDUALE_MINIBNC)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF_MINIBNC)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF_HEADER)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBLPTXE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBLPRXE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			HDSDIM_PCI_DEVICE_ID_LINSYS_HDSDITXE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			HDSDIM_PCI_DEVICE_ID_LINSYS_HDSDIRXE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			HDSDIM_PCI_DEVICE_ID_LINSYS_HDSDIQIE)
	},
	{0, }
};

static struct pci_driver lsj_pci_driver = {
	.name = lsj_module_name,
	.id_table = lsj_pci_id_table,
	.probe = lsj_pci_probe,
	.remove = lsj_pci_remove
};

MODULE_DEVICE_TABLE(pci,lsj_pci_id_table);

static struct file_operations lsj_fops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.read = lsj_read,
	.write = lsj_write,
	.poll = NULL,
	.unlocked_ioctl = NULL,
	.compat_ioctl = NULL,
	.open = lsj_open,
	.release = lsj_release,
	.fsync = NULL,
	.fasync = NULL
};

static LIST_HEAD(lsj_list);

/**
 * lsj_open - Linear Systems Ltd. device open() method
 * @inode: inode being opened
 * @filp: file being opened
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
lsj_open (struct inode *inode, struct file *filp)
{
	struct lsj *card = container_of(inode->i_cdev,struct lsj,cdev);

	filp->private_data = card;
	mutex_lock (&card->lock);
	if (card->users) {
		mutex_unlock (&card->lock);
		return -EBUSY;
	}
	card->users++;
	mutex_unlock (&card->lock);
	return nonseekable_open (inode, filp);
}

/**
 * lsj_release - Linear Systems Ltd. device release() method
 * @inode: inode being released
 * @filp: file being released
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
lsj_release (struct inode *inode, struct file *filp)
{
	struct lsj *card = filp->private_data;

	mutex_lock (&card->lock);
	card->users--;
	mutex_unlock (&card->lock);
	return 0;
}

/**
 * lsj_read - Linear Systems Ltd. device read() method
 * @filp: file being read
 * @data: read buffer
 * @length: size of data being read
 * @offset:
 *
 * Returns a negative error code on failure and
 * the number of bytes read on success.
 **/
static ssize_t
lsj_read (struct file *filp,
	char __user *data,
	size_t length,
	loff_t *offset)
{
	struct lsj *card = filp->private_data;
	ssize_t retcode = 1;

	if (length == 0) {
		return 0;
	}
	mutex_lock (&card->lock);
	if (put_user (readl (card->data_addr), data)) {
		retcode = -EFAULT;
	}
	mutex_unlock (&card->lock);
	return retcode;
}

/**
 * lsj_write - Linear Systems Ltd. device write() method
 * @filp: file being written
 * @data: data being written
 * @length: size of data being written
 * @offset:
 *
 * Returns a negative error code on failure and
 * the number of bytes written on success.
 **/
static ssize_t
lsj_write (struct file *filp,
	const char __user *data,
	size_t length,
	loff_t *offset)
{
	struct lsj *card = filp->private_data;
	u32 reg;

	if (length == 0) {
		return 0;
	}
	mutex_lock (&card->lock);
	if (get_user (reg, data)) {
		mutex_unlock (&card->lock);
		return -EFAULT;
	}
	writel (reg, card->data_addr);

	/* Dummy read to flush PCI posted writes */
	readl (card->data_addr);

	mutex_unlock (&card->lock);
	return 1;
}

/**
 * lsj_register - map a minor number to a Linear Systems Ltd. device
 * @card: Linear Systems Ltd. device to map
 *
 * Assign the lowest unused minor number to this device
 * and add it to the list of devices.
 * Returns a negative error code on failure and 0 on success.
 **/
static int
lsj_register (struct lsj *card)
{
	struct list_head *p;
	struct lsj *entry;
	unsigned int minor = 0, found;
	unsigned int maxminor = count - 1;
	int err;

	/* Assign the lowest unused minor number to this device */
	while (minor <= maxminor) {
		found = 0;
		list_for_each (p, &lsj_list) {
			entry = list_entry (p, struct lsj, list);
			if (MINOR(entry->cdev.dev) == minor) {
				found = 1;
				break;
			}
		}
		if (!found) {
			break;
		}
		minor++;
	}
	if (minor > maxminor) {
		printk (KERN_WARNING "%s: unable to register board\n",
			lsj_module_name);
		err = -ENOSPC;
		goto NO_MINOR;
	}

	/* Add this device to the list */
	list_add_tail (&card->list, &lsj_list);

	/* Register the device */
	card->dev = device_create (lsj_class,
		NULL,
		MKDEV(major,minor),
		card,
		"lsj%u", minor);
	if (IS_ERR(card->dev)) {
		err = PTR_ERR(card->dev);
		goto NO_DEV;
	}
	dev_set_drvdata (card->dev, card);

	/* Activate the cdev */
	if ((err = cdev_add (&card->cdev, MKDEV(major,minor), 1)) < 0) {
		printk (KERN_WARNING "%s: unable to add character device\n",
			lsj_module_name);
		goto NO_CDEV;
	}

	printk (KERN_INFO "%s: registered board %u\n",
		lsj_module_name, minor);
	return 0;

NO_CDEV:
	device_destroy (lsj_class, MKDEV(major,minor));
NO_DEV:
	list_del (&card->list);
NO_MINOR:
	return err;
}

/**
 * lsj_unregister - remove a Linear Systems Ltd. device from the list
 * @card: Linear Systems Ltd. device to remove
 *
 * Unlinks the Linear Systems Ltd. device from the driver's list.
 **/
static void
lsj_unregister (struct lsj *card)
{
	cdev_del (&card->cdev);
	device_destroy (lsj_class, card->cdev.dev);
	list_del (&card->list);
	return;
}

/**
 * lsj_show_range - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
static ssize_t
lsj_show_range (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct lsj *card = dev_get_drvdata (dev);
	int retcode;

	mutex_lock (&card->lock);
	retcode = snprintf (buf, PAGE_SIZE, "%u\n",
		~(ee_read (card->bridge_addr + PLX_CNTRL, 0x14 >> 1) & ~0x3) + 1);
	mutex_unlock (&card->lock);

	return retcode;
}

/**
 * lsj_store_range - interface attribute write handler
 * @dev: device being written
 * @attr: device attribute
 * @buf: input buffer
 * @count:
 **/
static ssize_t
lsj_store_range (struct device *dev,
	struct device_attribute *attr,
	const char *buf,
	size_t count)
{
	struct lsj *card = dev_get_drvdata (dev);
	char *endp;
	unsigned long val = simple_strtoul (buf, &endp, 0), goodval = 4;
	int retcode = count;

	if ((endp == buf) || (val < 4) || (val > 256)) {
		return -EINVAL;
	}
	while (goodval != val) {
		if (goodval > val) {
			return -EINVAL;
		}
		goodval <<= 1;
	}
	val = (~val + 1) | PLX_LASRR_IO;
	mutex_lock (&card->lock);
	ee_ewen (card->bridge_addr + PLX_CNTRL);
	ee_write (card->bridge_addr + PLX_CNTRL, val >> 16, 0x14 >> 1);
	ee_write (card->bridge_addr + PLX_CNTRL, val & 0xffff, 0x16 >> 1);
	ee_ewds (card->bridge_addr + PLX_CNTRL);
	mutex_unlock (&card->lock);
	return retcode;
}

static DEVICE_ATTR(range,S_IRUGO|S_IWUSR,
	lsj_show_range,lsj_store_range);

/**
 * lsj_pci_probe - PCI insertion handler
 * @pdev: PCI device
 * @id: PCI ID
 *
 * Checks if a PCI device should be handled by this driver.
 * Returns a negative error code on failure and 0 on success.
 **/
static int __devinit
lsj_pci_probe (struct pci_dev *pdev,
	const struct pci_device_id *id)
{
	int err;
	unsigned int core_bar, jtag_addr, bridge_bar;
	struct lsj *card;
	const char *board_name;

	/* Initialize the driver_data pointer so that lsj_pci_remove ()
	 * doesn't try to free it if an error occurs */
	pci_set_drvdata (pdev, NULL);

	/* Wake a sleeping device */
	if ((err = pci_enable_device (pdev)) < 0) {
		printk (KERN_WARNING "%s: unable to enable device\n",
			lsj_module_name);
		goto NO_PCI;
	}

	/* Request the PCI I/O resources */
	if ((err = pci_request_regions (pdev, lsj_module_name)) < 0) {
		goto NO_PCI;
	}

	/* Allocate a board info structure */
	if (!(card = (struct lsj *)kzalloc (sizeof (*card), GFP_KERNEL))) {
		err = -ENOMEM;
		goto NO_MEM;
	}

	/* Initialize the board info structure and select the BAR */
	switch (id->device) {
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDU:
		board_name = DVBM_NAME_FDU;
		core_bar = 3;
		jtag_addr = DVBM_FDU_JTAGR;
		bridge_bar = 0;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDU_R:
		board_name = DVBM_NAME_FDU_R;
		core_bar = 3;
		jtag_addr = DVBM_FDU_JTAGR;
		bridge_bar = 0;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBTXU:
		board_name = DVBM_NAME_TXU;
		core_bar = 3;
		jtag_addr = DVBM_FDU_JTAGR;
		bridge_bar = 0;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBRXU:
		board_name = DVBM_NAME_RXU;
		core_bar = 3;
		jtag_addr = DVBM_FDU_JTAGR;
		bridge_bar = 0;
		break;
	case MMSA_PCI_DEVICE_ID_LINSYS:
		board_name = MMSA_NAME;
		core_bar = 3;
		jtag_addr = DVBM_FDU_JTAGR;
		bridge_bar = 0;
		break;
	case SDIM_PCI_DEVICE_ID_LINSYS:
		board_name = SDIM_NAME;
		core_bar = 3;
		jtag_addr = DVBM_FDU_JTAGR;
		bridge_bar = 0;
		break;
	case MMAS_PCI_DEVICE_ID_LINSYS:
		board_name = MMAS_NAME;
		core_bar = 3;
		jtag_addr = DVBM_FDU_JTAGR;
		bridge_bar = 0;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDB:
		board_name = DVBM_NAME_FDB;
		core_bar = 3;
		jtag_addr = DVBM_FDU_JTAGR;
		bridge_bar = 0;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDB_R:
		board_name = DVBM_NAME_FDB_R;
		core_bar = 3;
		jtag_addr = DVBM_FDU_JTAGR;
		bridge_bar = 0;
		break;
	case SDIM_PCI_DEVICE_ID_LINSYS_SDILPFD:
		board_name = SDIM_NAME_LPFD;
		core_bar = 0;
		jtag_addr = SDIM_JTAGR;
		bridge_bar = 1;
		break;
	case SDIM_PCI_DEVICE_ID_LINSYS_SDILPFDE:
		board_name = SDIM_NAME_LPFDE;
		core_bar = 0;
		jtag_addr = SDIM_JTAGR;
		bridge_bar = 2;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPFD:
		board_name = DVBM_NAME_LPFD;
		core_bar = 0;
		jtag_addr = DVBM_LPFD_JTAGR;
		bridge_bar = 1;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPFDE:
		board_name = DVBM_NAME_LPFDE;
		core_bar = 0;
		jtag_addr = DVBM_LPFD_JTAGR;
		bridge_bar = 2;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPTXE:
		board_name = DVBM_NAME_LPTXE;
		core_bar = 0;
		jtag_addr = DVBM_LPFD_JTAGR;
		bridge_bar = 2;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPRXE:
		board_name = DVBM_NAME_LPRXE;
		core_bar = 0;
		jtag_addr = DVBM_LPFD_JTAGR;
		bridge_bar = 2;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQOE:
		board_name = DVBM_NAME_LPQOE;
		core_bar = 0;
		jtag_addr = DVBM_LPQO_JTAGR;
		bridge_bar = 2;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQOE_MINIBNC:
		board_name = DVBM_NAME_LPQOE_MINIBNC;
		core_bar = 0;
		jtag_addr = DVBM_LPQO_JTAGR;
		bridge_bar = 2;
		break;
	case SDIM_PCI_DEVICE_ID_LINSYS_SDIQOE:
		board_name = SDIM_NAME_QOE;
		core_bar = 0;
		jtag_addr = SDIM_QOE_JTAGR;
		bridge_bar = 2;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQDUALE:
		board_name = DVBM_NAME_LPQDUALE;
		core_bar = 0;
		jtag_addr = DVBM_QDUAL_JTAGR;
		bridge_bar = 2;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQDUALE_MINIBNC:
		board_name = DVBM_NAME_LPQDUALE_MINIBNC;
		core_bar = 0;
		jtag_addr = DVBM_QDUAL_JTAGR;
		bridge_bar = 2;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF_MINIBNC:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF_HEADER:
		board_name = DVBM_NAME_LPQLF;
		core_bar = 0;
		jtag_addr = DVBM_QLF_JTAGR;
		bridge_bar = 2;
		break;
	case HDSDIM_PCI_DEVICE_ID_LINSYS_HDSDITXE:
		board_name = HDSDIM_NAME_TXE;
		core_bar = 0;
		jtag_addr = HDSDIM_TXE_JTAGR;
		bridge_bar = DEVICE_COUNT_RESOURCE;
		break;
	case HDSDIM_PCI_DEVICE_ID_LINSYS_HDSDIRXE:
		board_name = HDSDIM_NAME_RXE;
		core_bar = 0;
		jtag_addr = HDSDIM_RXE_JTAGR;
		bridge_bar = DEVICE_COUNT_RESOURCE;
		break;
	case HDSDIM_PCI_DEVICE_ID_LINSYS_HDSDIQIE:
		board_name = HDSDIM_NAME_QIE;
		core_bar = 0;
		jtag_addr = HDSDIM_QIE_JTAGR;
		bridge_bar = 2;
		break;
	default:
		board_name = "Unknown device";
		core_bar = 0;
		jtag_addr = 0;
		bridge_bar = DEVICE_COUNT_RESOURCE;
		break;
	}
	card->core_addr = ioremap_nocache (pci_resource_start (pdev, core_bar),
		pci_resource_len (pdev, core_bar));
	card->data_addr = card->core_addr + jtag_addr;
	if (bridge_bar < DEVICE_COUNT_RESOURCE) {
		card->bridge_addr = ioremap_nocache (pci_resource_start (pdev, bridge_bar),
			pci_resource_len (pdev, bridge_bar));
	}
	mutex_init (&card->lock);
	card->pdev = pdev;
	cdev_init (&card->cdev, &lsj_fops);
	card->cdev.owner = THIS_MODULE;
	printk (KERN_INFO "%s: %s detected\n",
		lsj_module_name, board_name);

	/* Store the pointer to the board info structure
	 * in the PCI info structure */
	pci_set_drvdata (pdev, card);

	/* Register the card */
	if ((err = lsj_register (card)) < 0) {
		goto NO_DEV;
	}

	/* Add device attributes */
	switch (id->device) {
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDU:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDU_R:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBTXU:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBRXU:
	case MMSA_PCI_DEVICE_ID_LINSYS:
	case SDIM_PCI_DEVICE_ID_LINSYS:
	case MMAS_PCI_DEVICE_ID_LINSYS:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDB:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDB_R:
		/* Allow the PCI I/O address range to be changed
		 * on boards which have new firmware which
		 * exceeds the original address range */
		if ((err = device_create_file (card->dev,
			&dev_attr_range)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'range'\n",
				lsj_module_name);
		}
		break;
	default:
		break;
	}


	return 0;

NO_DEV:
NO_MEM:
	lsj_pci_remove (pdev);
NO_PCI:
	return err;
}

/**
 * lsj_pci_remove - PCI removal handler
 * @pdev: PCI device
 **/
static void
lsj_pci_remove (struct pci_dev *pdev)
{
	struct lsj *card = pci_get_drvdata (pdev);

	if (card) {
		struct list_head *p;

		list_for_each (p, &lsj_list) {
			if (p == &card->list) {
				lsj_unregister (card);
				break;
			}
		}
		if (card->bridge_addr) {
			iounmap (card->bridge_addr);
		}
		iounmap (card->core_addr);
		kfree (card);
		pci_set_drvdata (pdev, NULL);
	}
	pci_release_regions (pdev);
	pci_disable_device (pdev);
	return;
}

/**
 * lsj_init_module - initialize the module
 *
 * Register the module as a character PCI driver.
 * Returns a negative error code on failure and 0 on success.
 **/
static int __init
lsj_init_module (void)
{
	int err;
	dev_t dev = MKDEV(major,0);

	printk (KERN_INFO "%s: Linear Systems Ltd. "
		"JTAG driver from master-%s (%s)\n",
		lsj_module_name, MASTER_DRIVER_VERSION, MASTER_DRIVER_DATE);

	/* Create a device class */
	lsj_class = class_create (THIS_MODULE, lsj_module_name);
	if (IS_ERR(lsj_class)) {
		err = PTR_ERR(lsj_class);
		goto NO_CLASS;
	}

	/* Reserve a range of device numbers */
	if (major) {
		err = register_chrdev_region (dev,
			count,
			lsj_module_name);
	} else {
		err = alloc_chrdev_region (&dev,
			0,
			count,
			lsj_module_name);
	}
	if (err < 0) {
		printk (KERN_WARNING
			"%s: unable to reserve device number range\n",
			lsj_module_name);
		goto NO_RANGE;
	}
	major = MAJOR(dev);

	/* Register with the PCI subsystem */
	if ((err = pci_register_driver (&lsj_pci_driver)) < 0) {
		printk (KERN_WARNING
			"%s: unable to register with PCI subsystem\n",
			lsj_module_name);
		goto NO_PCI;
	}

	return 0;

NO_PCI:
	unregister_chrdev_region (MKDEV(major,0), count);
NO_RANGE:
	class_destroy (lsj_class);
NO_CLASS:
	return err;
}

/**
 * lsj_cleanup_module - clean up at module removal
 *
 * Unregister the module as a character PCI driver.
 **/
static void __exit
lsj_cleanup_module (void)
{
	pci_unregister_driver (&lsj_pci_driver);
	unregister_chrdev_region (MKDEV(major,0), count);
	class_destroy (lsj_class);
	return;
}

module_init (lsj_init_module);
module_exit (lsj_cleanup_module);

