/* asmi.c
 *
 * Linux driver for the Active Serial interface on
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
#include "gt64131.h"
#include "dvbm_qi.h"
#include "dvbm_fdu.h"
#include "dvbm_qlf.h"
#include "dvbm_qio.h"
#include "eeprom.h"
#include "dvbm_qdual.h"
#include "dvbm_qlf.h"
#include "mmas.h"
#include "mmsa.h"
#include "sdim.h"
#include "sdim_qie.h"

#ifndef DEFINE_PCI_DEVICE_TABLE
#define DEFINE_PCI_DEVICE_TABLE(_table) \
	const struct pci_device_id _table[]
#endif

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,27))
#define device_create(cls,parent,devt,drvdata,fmt,...) \
	device_create(cls,parent,devt,fmt,##__VA_ARGS__)
#endif

static struct class *lsa_class;

/**
 * struct lsa - Linear Systems Ltd. device
 * @list: device list pointers
 * @flash_addr: address of memory region
 * @data_addr: Active Serial port address
 * @bridge_addr: address of bridge memory region
 * @lock: mutex
 * @pdev: PCI device
 * @dev: pointer to device
 * @cdev: character device structure
 * @users: usage count
 **/
struct lsa {
	struct list_head list;
	void __iomem *flash_addr;
	void __iomem *data_addr;
	void __iomem *bridge_addr;
	struct mutex lock;
	struct pci_dev *pdev;
	struct device *dev;
	struct cdev cdev;
	int users;
};

/* Static function prototypes */
static int lsa_open (struct inode *inode, struct file *filp);
static int lsa_release (struct inode *inode, struct file *filp);
static ssize_t lsa_read (struct file *filp,
	char __user *data,
	size_t length,
	loff_t *offset);
static ssize_t lsa_write (struct file *filp,
	const char __user *data,
	size_t length,
	loff_t *offset);
static int lsa_register (struct lsa *card);
static void lsa_unregister (struct lsa *card);
static ssize_t lsa_show_range (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static ssize_t lsa_store_range (struct device *dev,
	struct device_attribute *attr,
	const char *buf,
	size_t count);
static int lsa_pci_probe (struct pci_dev *pdev,
	const struct pci_device_id *id) __devinit;
static void lsa_pci_remove (struct pci_dev *pdev);
static int lsa_init_module (void) __init;
static void lsa_cleanup_module (void) __exit;

static char lsa_module_name[] = "ls_as";

/* The major number can be overridden from the command line
 * when the module is loaded. */
static unsigned int major = 123;
module_param(major,uint,S_IRUGO);
MODULE_PARM_DESC(major,"Major number of the first device, or zero for dynamic");

static const unsigned int count = 256;

MODULE_AUTHOR("Linear Systems Ltd.");
MODULE_DESCRIPTION("Active Serial driver");
MODULE_LICENSE("GPL");
MODULE_VERSION(MASTER_DRIVER_VERSION);

static DEFINE_PCI_DEVICE_TABLE(lsa_pci_id_table) = {
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBQI)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBQLF)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBQLF4)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD_R)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			ATSCM_PCI_DEVICE_ID_LINSYS_2FD)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			ATSCM_PCI_DEVICE_ID_LINSYS_2FD_R)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD_RS)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBQO)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBQOE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBQDUAL)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBQDUALE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE_R)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE_RS)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			ATSCM_PCI_DEVICE_ID_LINSYS_2FDE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			ATSCM_PCI_DEVICE_ID_LINSYS_2FDE_R)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			ATSCM_PCI_DEVICE_ID_LINSYS_2FD_RS)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBFDE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBFDE_R)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBFDEB)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBFDEB_R)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBTXE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBRXE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			SDIME_PCI_DEVICE_ID_LINSYS)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			MMSAE_PCI_DEVICE_ID_LINSYS)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			MMASE_PCI_DEVICE_ID_LINSYS)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBQIE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			SDIM_PCI_DEVICE_ID_LINSYS_SDIQIE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBQ3IOE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBQ3INOE)
	},
	{0, }
};

static struct pci_driver lsa_pci_driver = {
	.name = lsa_module_name,
	.id_table = lsa_pci_id_table,
	.probe = lsa_pci_probe,
	.remove = lsa_pci_remove
};

MODULE_DEVICE_TABLE(pci,lsa_pci_id_table);

static struct file_operations lsa_fops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.read = lsa_read,
	.write = lsa_write,
	.poll = NULL,
	.unlocked_ioctl = NULL,
	.compat_ioctl = NULL,
	.open = lsa_open,
	.release = lsa_release,
	.fsync = NULL,
	.fasync = NULL
};

static LIST_HEAD(lsa_list);

/**
 * lsa_open - Linear Systems Ltd. device open() method
 * @inode: inode being opened
 * @filp: file being opened
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
lsa_open (struct inode *inode, struct file *filp)
{
	struct lsa *card = container_of(inode->i_cdev,struct lsa,cdev);

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
 * lsa_release - Linear Systems Ltd. device release() method
 * @inode: inode being released
 * @filp: file being released
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
lsa_release (struct inode *inode, struct file *filp)
{
	struct lsa *card = filp->private_data;

	mutex_lock (&card->lock);
	card->users--;
	mutex_unlock (&card->lock);
	return 0;
}

/**
 * lsa_read - Linear Systems Ltd. read() method
 * @filp: file being read
 * @data: read buffer
 * @length: size of data being read
 * @offset:
 *
 * Returns a negative error code on failure and
 * the number of bytes read on success.
 **/
static ssize_t
lsa_read (struct file *filp,
	char __user *data,
	size_t length,
	loff_t *offset)
{
	struct lsa *card = filp->private_data;
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
 * lsa_write - Linear Systems Ltd. device write() method
 * @filp: file being written
 * @data: data being written
 * @length: size of data being written
 * @offset:
 *
 * Returns a negative error code on failure and
 * the number of bytes written on success.
 **/
static ssize_t
lsa_write (struct file *filp,
	const char __user *data,
	size_t length,
	loff_t *offset)
{
	struct lsa *card = filp->private_data;
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
 * lsa_register - map a minor number to a Linear Systems Ltd. device
 * @card: Linear Systems Ltd. device to map
 *
 * Assign the lowest unused minor number to this device
 * and add it to the list of devices.
 * Returns a negative error code on failure and 0 on success.
 **/
static int
lsa_register (struct lsa *card)
{
	struct list_head *p;
	struct lsa *entry;
	unsigned int minor = 0, found;
	unsigned int maxminor = count - 1;
	int err;

	/* Assign the lowest unused minor number to this device */
	while (minor <= maxminor) {
		found = 0;
		list_for_each (p, &lsa_list) {
			entry = list_entry (p, struct lsa, list);
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
			lsa_module_name);
		err = -ENOSPC;
		goto NO_MINOR;
	}

	/* Add this device to the list */
	list_add_tail (&card->list, &lsa_list);

	/* Register the device */
	card->dev = device_create (lsa_class,
		NULL,
		MKDEV(major,minor),
		card,
		"lsa%u", minor);
	if (IS_ERR(card->dev)) {
		err = PTR_ERR(card->dev);
		goto NO_DEV;
	}
	dev_set_drvdata (card->dev, card);

	/* Activate the cdev */
	if ((err = cdev_add (&card->cdev, MKDEV(major,minor), 1)) < 0) {
		printk (KERN_WARNING "%s: unable to add character device\n",
			lsa_module_name);
		goto NO_CDEV;
	}

	printk (KERN_INFO "%s: registered board %u\n",
		lsa_module_name, minor);
	return 0;

NO_CDEV:
	device_destroy (lsa_class, MKDEV(major,minor));
NO_DEV:
	list_del (&card->list);
NO_MINOR:
	return err;
}

/**
 * lsa_unregister - remove a Linear Systems Ltd. device from the list
 * @card: Linear Systems Ltd. device to remove
 *
 * Unlinks the Linear Systems Ltd. device from the driver's list.
 **/
static void
lsa_unregister (struct lsa *card)
{
	cdev_del (&card->cdev);
	device_destroy (lsa_class, card->cdev.dev);
	list_del (&card->list);
	return;
}

/**
 * lsa_show_range - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
static ssize_t
lsa_show_range (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct lsa *card = dev_get_drvdata (dev);
	int retcode;

	mutex_lock (&card->lock);
	retcode = snprintf (buf, PAGE_SIZE, "%u\n",
		~(ee_read (card->bridge_addr + PLX_CNTRL, 0x14 >> 1) & ~0x3) + 1);
	mutex_unlock (&card->lock);

	return retcode;
}

/**
 * lsa_store_range - interface attribute write handler
 * @dev: device being written
 * @attr: device attribute
 * @buf: input buffer
 * @count:
 **/
static ssize_t
lsa_store_range (struct device *dev,
	struct device_attribute *attr,
	const char *buf,
	size_t count)
{
	struct lsa *card = dev_get_drvdata (dev);
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
	lsa_show_range,lsa_store_range);

/**
 * lsa_pci_probe - PCI insertion handler
 * @pdev: PCI device
 * @id: PCI ID
 *
 * Checks if a PCI device should be handled by this driver.
 * Returns a negative error code on failure and 0 on success.
 **/
static int __devinit
lsa_pci_probe (struct pci_dev *pdev,
	const struct pci_device_id *id)
{
	int err;
	struct lsa *card;
	const char *name;

	/* Initialize the driver_data pointer so that lsa_pci_remove ()
	 * doesn't try to free it if an error occurs */
	pci_set_drvdata (pdev, NULL);

	/* Wake a sleeping device */
	if ((err = pci_enable_device (pdev)) < 0) {
		printk (KERN_WARNING "%s: unable to enable device\n",
			lsa_module_name);
		goto NO_PCI;
	}

	/* Request the PCI I/O resources */
	if ((err = pci_request_regions (pdev, lsa_module_name)) < 0) {
		goto NO_PCI;
	}

	/* Allocate a board info structure */
	if (!(card = (struct lsa *)kzalloc (sizeof (*card), GFP_KERNEL))) {
		err = -ENOMEM;
		goto NO_MEM;
	}

	/* Initialize the board info structure */
	switch (id->device) {
	case SDIM_PCI_DEVICE_ID_LINSYS_SDIQIE:
		name = SDIM_NAME_QIE;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 2),
			pci_resource_len (pdev, 2));
		card->data_addr = card->flash_addr + SDIM_QIE_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case SDIM_PCI_DEVICE_ID_LINSYS_SDIQI:
		name = SDIM_NAME_QIE;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 2),
			pci_resource_len (pdev, 2));
		card->data_addr = card->flash_addr + SDIM_QIE_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQDUAL:
		name = DVBM_NAME_QDUAL;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 2),
			pci_resource_len (pdev, 2));
		card->data_addr = card->flash_addr + DVBM_QDUAL_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start(pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQDUALE:
		name = DVBM_NAME_QDUALE;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 2),
			pci_resource_len (pdev, 2));
		card->data_addr = card->flash_addr + DVBM_QDUAL_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start(pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQO:
		name = DVBM_NAME_QO;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 2),
			pci_resource_len (pdev, 2));
		card->data_addr = card->flash_addr + DVBM_QIO_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start(pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQOE:
		name = DVBM_NAME_QOE;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 2),
			pci_resource_len (pdev, 2));
		card->data_addr = card->flash_addr + DVBM_QIO_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start(pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQI:
		name = DVBM_NAME_DVBQI;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		card->data_addr = card->flash_addr;
		/* Setup the bridge */
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 4),
			pci_resource_len (pdev, 4));
		writel (DVBM_QI_TURNOFF |
			DVBM_QI_ACCTOFIRST << 3 |
			DVBM_QI_ACCTONEXT << 7 |
			DVBM_QI_ALETOWR << 11 |
			DVBM_QI_WRACTIVE << 14 |
			DVBM_QI_WRHIGH << 17 |
			DVBM_QI_DEVWIDTH << 20 |
			DVBM_QI_BOOTDMAFLYBY << 22 |
			DVBM_QI_DEVLOC << 23 |
			DVBM_QI_DMAFLYBY << 26,
			card->bridge_addr + GT64_DEVICE(0));
		writel (DVBM_QI_TIMEOUT0 |
			DVBM_QI_TIMEOUT1 << 8 |
			DVBM_QI_RETRYCTR << 16,
			card->bridge_addr + GT64_PCITOR);
		writel (0x000000ff, card->bridge_addr + GT64_SCS0LDA);
		writel (0x00000000, card->bridge_addr + GT64_SCS0HDA);
		writel (0x000000ff, card->bridge_addr + GT64_SCS1LDA);
		writel (0x00000000, card->bridge_addr + GT64_SCS1HDA);
		writel (0x000000ff, card->bridge_addr + GT64_SCS2LDA);
		writel (0x00000000, card->bridge_addr + GT64_SCS2HDA);
		writel (0x000000ff, card->bridge_addr + GT64_SCS3LDA);
		writel (0x00000000, card->bridge_addr + GT64_SCS3HDA);
		writel (0x000000ff, card->bridge_addr + GT64_CS0LDA);
		writel (0x00000000, card->bridge_addr + GT64_CS0HDA);
		writel (0x000000ff, card->bridge_addr + GT64_CS1LDA);
		writel (0x00000000, card->bridge_addr + GT64_CS1HDA);
		writel (0x000000ff, card->bridge_addr + GT64_CS2LDA);
		writel (0x00000000, card->bridge_addr + GT64_CS2HDA);
		writel (0x000000ff, card->bridge_addr + GT64_CS3LDA);
		writel (0x00000000, card->bridge_addr + GT64_CS3HDA);
		writel (0x00000000, card->bridge_addr + GT64_BOOTCSLDA);
		writel (0x000000ff, card->bridge_addr + GT64_BOOTCSHDA);
		writel (0x00000015, card->bridge_addr + GT64_SDRAM(1));
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQLF:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQLF4:
		name = DVBM_NAME_QLF;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 2),
			pci_resource_len (pdev, 2));
		card->data_addr = card->flash_addr + DVBM_QLF_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD:
		name = DVBM_NAME_2FD;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		card->data_addr = card->flash_addr + DVBM_FDU_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD_R:
		name = DVBM_NAME_2FD_R;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		card->data_addr = card->flash_addr + DVBM_FDU_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD_RS:
		name = DVBM_NAME_2FD_RS;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		card->data_addr = card->flash_addr + DVBM_FDU_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FD:
		name = ATSCM_NAME_2FD;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		card->data_addr = card->flash_addr + DVBM_FDU_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FD_R:
		name = ATSCM_NAME_2FD_R;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		card->data_addr = card->flash_addr + DVBM_FDU_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FD_RS:
		name = ATSCM_NAME_2FD_RS;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		card->data_addr = card->flash_addr + DVBM_FDU_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE:
		name = DVBM_NAME_2FDE;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		card->data_addr = card->flash_addr + DVBM_FDU_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE_R:
		name = DVBM_NAME_2FDE_R;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		card->data_addr = card->flash_addr + DVBM_FDU_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE_RS:
		name = DVBM_NAME_2FDE_RS;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		card->data_addr = card->flash_addr + DVBM_FDU_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FDE:
		name = ATSCM_NAME_2FDE;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		card->data_addr = card->flash_addr + DVBM_FDU_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FDE_R:
		name = ATSCM_NAME_2FDE_R;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		card->data_addr = card->flash_addr + DVBM_FDU_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDE:
		name = DVBM_NAME_FDE;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		card->data_addr = card->flash_addr + DVBM_FDU_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDE_R:
		name = DVBM_NAME_FDE_R;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		card->data_addr = card->flash_addr + DVBM_FDU_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDEB:
		name = DVBM_NAME_FDEB;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		card->data_addr = card->flash_addr + DVBM_FDU_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDEB_R:
		name = DVBM_NAME_FDEB_R;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		card->data_addr = card->flash_addr + DVBM_FDU_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBTXE:
		name = DVBM_NAME_TXE;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		card->data_addr = card->flash_addr + DVBM_FDU_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBRXE:
		name = DVBM_NAME_RXE;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		card->data_addr = card->flash_addr + DVBM_FDU_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case SDIME_PCI_DEVICE_ID_LINSYS:
		name = SDIME_NAME;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		card->data_addr = card->flash_addr + SDIM_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case MMSAE_PCI_DEVICE_ID_LINSYS:
		name = MMSAE_NAME;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		card->data_addr = card->flash_addr + MMSA_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case MMASE_PCI_DEVICE_ID_LINSYS:
		name = MMASE_NAME;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		card->data_addr = card->flash_addr + MMAS_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQIE:
		name = DVBM_NAME_QIE;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 2),
			pci_resource_len (pdev, 2));
		card->data_addr = card->flash_addr + DVBM_QLF_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQ3IOE:
		name = DVBM_NAME_Q3IOE;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 2),
			pci_resource_len (pdev, 2));
		card->data_addr = card->flash_addr + DVBM_QIO_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start(pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQ3INOE:
		name = DVBM_NAME_Q3INOE;
		card->flash_addr = ioremap_nocache (pci_resource_start (pdev, 2),
			pci_resource_len (pdev, 2));
		card->data_addr = card->flash_addr + DVBM_QIO_ASMIR;
		card->bridge_addr = ioremap_nocache (
			pci_resource_start(pdev, 0),
			pci_resource_len (pdev, 0));
		break;
	default:
		name = "";
		break;
	}
	mutex_init (&card->lock);
	card->pdev = pdev;
	cdev_init (&card->cdev, &lsa_fops);
	card->cdev.owner = THIS_MODULE;
	printk (KERN_INFO "%s: %s detected\n",
		lsa_module_name, name);

	/* Store the pointer to the board info structure
	 * in the PCI info structure */
	pci_set_drvdata (pdev, card);

	/* Register the card */
	if ((err = lsa_register (card)) < 0) {
		goto NO_DEV;
	}

	/* Add device attributes */
	switch (id->device) {
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD_R:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD_RS:
		/* Allow the PCI I/O address range to be changed
		 * on boards which have new firmware which
		 * exceeds the original address range */
		if ((err = device_create_file (card->dev,
			&dev_attr_range)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'range'\n",
				lsa_module_name);
		}
		break;
	default:
		break;
	}

	return 0;

NO_DEV:
NO_MEM:
	lsa_pci_remove (pdev);
NO_PCI:
	return err;
}

/**
 * lsa_pci_remove - PCI removal handler
 * @pdev: PCI device
 **/
static void
lsa_pci_remove (struct pci_dev *pdev)
{
	struct lsa *card = pci_get_drvdata (pdev);

	if (card) {
		struct list_head *p;

		list_for_each (p, &lsa_list) {
			if (p == &card->list) {
				lsa_unregister (card);
				break;
			}
		}
		iounmap (card->bridge_addr);
		iounmap (card->flash_addr);
		kfree (card);
		pci_set_drvdata (pdev, NULL);
	}
	pci_release_regions (pdev);
	pci_disable_device (pdev);
	return;
}

/**
 * lsa_init_module - initialize the module
 *
 * Register the module as a character PCI driver.
 * Returns a negative error code on failure and 0 on success.
 **/
static int __init
lsa_init_module (void)
{
	int err;
	dev_t dev = MKDEV(major,0);

	printk (KERN_INFO "%s: Linear Systems Ltd. "
		"Active Serial driver from master-%s (%s)\n",
		lsa_module_name, MASTER_DRIVER_VERSION, MASTER_DRIVER_DATE);

	/* Create a device class */
	lsa_class = class_create (THIS_MODULE, lsa_module_name);
	if (IS_ERR(lsa_class)) {
		err = PTR_ERR(lsa_class);
		goto NO_CLASS;
	}

	/* Reserve a range of device numbers */
	if (major) {
		err = register_chrdev_region (dev,
			count,
			lsa_module_name);
	} else {
		err = alloc_chrdev_region (&dev,
			0,
			count,
			lsa_module_name);
	}
	if (err < 0) {
		printk (KERN_WARNING
			"%s: unable to reserve device number range\n",
			lsa_module_name);
		goto NO_RANGE;
	}
	major = MAJOR(dev);

	/* Register with the PCI subsystem */
	if ((err = pci_register_driver (&lsa_pci_driver)) < 0) {
		printk (KERN_WARNING
			"%s: unable to register with PCI subsystem\n",
			lsa_module_name);
		goto NO_PCI;
	}

	return 0;

NO_PCI:
	unregister_chrdev_region (MKDEV(major,0), count);
NO_RANGE:
	class_destroy (lsa_class);
NO_CLASS:
	return err;
}

/**
 * lsa_cleanup_module - clean up at module removal
 *
 * Unregister the module as a character PCI driver.
 **/
static void __exit
lsa_cleanup_module (void)
{
	pci_unregister_driver (&lsa_pci_driver);
	unregister_chrdev_region (MKDEV(major,0), count);
	class_destroy (lsa_class);
	return;
}

module_init (lsa_init_module);
module_exit (lsa_cleanup_module);

