/* sdicore.c
 *
 * Linear Systems Ltd. SMPTE 259M-C API.
 *
 * Copyright (C) 2004-2010 Linear Systems Ltd.
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
#include <linux/kernel.h> /* snprintf () */
#include <linux/module.h> /* MODULE_LICENSE */
#include <linux/moduleparam.h> /* module_param () */

#include <linux/init.h> /* module_init () */
#include <linux/fs.h> /* register_chrdev_region () */
#include <linux/slab.h> /* kzalloc () */
#include <linux/errno.h> /* error codes */
#include <linux/list.h> /* list_add_tail () */
#include <linux/spinlock.h> /* spin_lock_init () */
#include <linux/wait.h> /* init_waitqueue_head () */
#include <linux/device.h> /* class_create () */
#include <linux/cdev.h> /* cdev_init () */
#include <linux/mutex.h> /* mutex_init () */

#include <asm/uaccess.h> /* put_user () */
#include <asm/bitops.h> /* test_and_clear_bit () */

#include "sdicore.h"
#include "../include/master.h"
#include "miface.h"
#include "mdma.h"

/* Static function prototypes */
static int sdi_ioctl (struct master_iface *iface,
	unsigned int cmd,
	unsigned long arg);
static int sdi_validate_buffers (struct master_iface *iface,
	unsigned long val);
static int sdi_validate_bufsize (struct master_iface *iface,
	unsigned long val);
static int sdi_validate_clksrc (struct master_iface *iface,
	unsigned long val);
static int sdi_validate_mode (struct master_iface *iface,
	unsigned long val);
static ssize_t sdi_store_buffers (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static ssize_t sdi_store_bufsize (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static ssize_t sdi_store_clksrc (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static ssize_t sdi_store_mode (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static int sdi_init_module (void) __init;
static void sdi_cleanup_module (void) __exit;

/* The major number can be overridden from the command line
 * when the module is loaded. */
static unsigned int major = SDI_MAJOR;
module_param(major,uint,S_IRUGO);
MODULE_PARM_DESC(major,"Major number of the first device, or zero for dynamic");

static const unsigned int count = 1 << MASTER_MINORBITS;

MODULE_AUTHOR("Linear Systems Ltd.");
MODULE_DESCRIPTION("SMPTE 259M-C module");
MODULE_LICENSE("GPL");
MODULE_VERSION(MASTER_DRIVER_VERSION);

EXPORT_SYMBOL(sdi_open);
EXPORT_SYMBOL(sdi_write);
EXPORT_SYMBOL(sdi_read);
EXPORT_SYMBOL(sdi_txpoll);
EXPORT_SYMBOL(sdi_rxpoll);
EXPORT_SYMBOL(sdi_txioctl);
EXPORT_SYMBOL(sdi_rxioctl);
EXPORT_SYMBOL(sdi_mmap);
EXPORT_SYMBOL(sdi_release);
EXPORT_SYMBOL(sdi_register_iface);
EXPORT_SYMBOL(sdi_unregister_iface);

static char sdi_driver_name[] = SDI_DRIVER_NAME;

static LIST_HEAD(sdi_iface_list);

static spinlock_t sdi_iface_lock;

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,27))
#define device_create(cls,parent,devt,drvdata,fmt,...) \
	device_create(cls,parent,devt,fmt,##__VA_ARGS__)
#endif

static struct class *sdi_class;
static CLASS_ATTR(version,S_IRUGO,
	miface_show_version,NULL);

/**
 * sdi_open - SDI interface open() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
int
sdi_open (struct inode *inode, struct file *filp)
{
	return miface_open (inode, filp);
}

/**
 * sdi_write - SDI interface write() method
 * @filp: file
 * @data: data
 * @length: size of data
 * @offset:
 *
 * Returns a negative error code on failure and
 * the number of bytes written on success.
 **/
ssize_t
sdi_write (struct file *filp,
	const char __user *data,
	size_t length,
	loff_t *offset)
{
	return miface_write (filp, data, length, offset);
}

/**
 * sdi_read - SDI interface read() method
 * @filp: file
 * @data: read buffer
 * @length: size of data
 * @offset:
 *
 * Returns a negative error code on failure and
 * the number of bytes read on success.
 **/
ssize_t
sdi_read (struct file *filp,
	char __user *data,
	size_t length,
	loff_t *offset)
{
	return miface_read (filp, data, length, offset);
}

/**
 * sdi_txpoll - transmitter poll() method
 * @filp: file
 * @wait:
 **/
unsigned int
sdi_txpoll (struct file *filp, poll_table *wait)
{
	return miface_txpoll (filp, wait);
}

/**
 * sdi_rxpoll - receiver poll() method
 * @filp: file
 * @wait:
 **/
unsigned int
sdi_rxpoll (struct file *filp, poll_table *wait)
{
	return miface_rxpoll (filp, wait);
}

/**
 * sdi_ioctl - generic ioctl() method
 * @iface: interface
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
sdi_ioctl (struct master_iface *iface,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_dev *card = iface->card;

	switch (cmd) {
	case SDI_IOC_GETID:
		if (put_user (card->id,
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_GETVERSION:
		if (put_user (card->version, (unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return -ENOTTY;
	}
	return 0;
}

/**
 * sdi_txioctl - SMPTE 259M-C transmitter interface ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
long
sdi_txioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	unsigned int reg = 0, i;

	switch (cmd) {
	case SDI_IOC_TXGETCAP:
		if (put_user (iface->capabilities,
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_TXGETEVENTS:
		for (i = 0; i < 8 * sizeof (reg); i++) {
			if (test_and_clear_bit (i, &iface->events)) {
				reg |= (1 << i);
			}
		}
		if (put_user (reg, (unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_TXGETBUFLEVEL:
		if (put_user (mdma_tx_buflevel (iface->dma),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_QBUF_DEPRECATED:
	case SDI_IOC_QBUF_DEPRECATED2:
	case SDI_IOC_QBUF:
		return miface_txqbuf (filp, arg);
	case SDI_IOC_DQBUF_DEPRECATED:
	case SDI_IOC_DQBUF_DEPRECATED2:
	case SDI_IOC_DQBUF:
		return miface_txdqbuf (filp, arg);
	default:
		return sdi_ioctl (iface, cmd, arg);
	}
	return 0;
}

/**
 * sdi_rxioctl - SMPTE 259M-C receiver interface ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
long
sdi_rxioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	unsigned int reg = 0, i;

	switch (cmd) {
	case SDI_IOC_RXGETCAP:
		if (put_user (iface->capabilities,
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_RXGETEVENTS:
		for (i = 0; i < 8 * sizeof (reg); i++) {
			if (test_and_clear_bit (i, &iface->events)) {
				reg |= (1 << i);
			}
		}
		if (put_user (reg, (unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_RXGETBUFLEVEL:
		if (put_user (mdma_rx_buflevel (iface->dma),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_QBUF_DEPRECATED:
	case SDI_IOC_QBUF_DEPRECATED2:
	case SDI_IOC_QBUF:
		return miface_rxqbuf (filp, arg);
	case SDI_IOC_DQBUF_DEPRECATED:
	case SDI_IOC_DQBUF_DEPRECATED2:
	case SDI_IOC_DQBUF:
		return miface_rxdqbuf (filp, arg);
	default:
		return sdi_ioctl (iface, cmd, arg);
	}
	return 0;
}

/**
 * sdi_mmap - SDI interface mmap() method
 * @filp: file
 * @vma: VMA
 **/
int
sdi_mmap (struct file *filp, struct vm_area_struct *vma)
{
	return miface_mmap (filp, vma);
}

/**
 * sdi_release - SDI interface release() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
int
sdi_release (struct inode *inode, struct file *filp)
{
	return miface_release (inode, filp);
}

/**
 * sdi_validate_buffers - validate a buffers attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
sdi_validate_buffers (struct master_iface *iface,
	unsigned long val)
{
	const unsigned int min = (iface->direction == MASTER_DIRECTION_TX) ?
		SDI_TX_BUFFERS_MIN : SDI_RX_BUFFERS_MIN;
	const unsigned int max = SDI_BUFFERS_MAX;

	if ((val < min) || (val > max)) {
		return -EINVAL;
	}
	return 0;
}

/**
 * sdi_validate_bufsize - validate a bufsize attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
sdi_validate_bufsize (struct master_iface *iface,
	unsigned long val)
{
	const unsigned int min = (iface->direction == MASTER_DIRECTION_TX) ?
		SDI_TX_BUFSIZE_MIN : SDI_RX_BUFSIZE_MIN;
	const unsigned int max = SDI_BUFSIZE_MAX;
	const unsigned int mult = iface->granularity;

	if ((val < min) || (val > max) || (val % mult)) {
		return -EINVAL;
	}
	return 0;
}

/**
 * sdi_validate_clksrc - validate a clksrc attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
sdi_validate_clksrc (struct master_iface *iface,
	unsigned long val)
{
	const unsigned int min = SDI_CTL_TX_CLKSRC_ONBOARD;
	const unsigned int max = SDI_CTL_TX_CLKSRC_RX;

	if ((val < min) || (val > max)) {
		return -EINVAL;
	}
	return 0;
}

/**
 * sdi_validate_mode - validate a mode attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
sdi_validate_mode (struct master_iface *iface,
	unsigned long val)
{
	const unsigned int min = SDI_CTL_MODE_8BIT;
	const unsigned int max = SDI_CTL_MODE_10BIT;

	if ((val < min) || (val > max)) {
		return -EINVAL;
	}
	return 0;
}

/**
 * sdi_store_* - SMPTE 259M-C interface attribute write handler
 * @dev: device being written
 * @attr: device attribute
 * @buf: input buffer
 * @count:
 **/
#define SDI_STORE(var) \
	static ssize_t sdi_store_##var (struct device *dev, \
		struct device_attribute *attr, \
		const char *buf, \
		size_t count) \
	{ \
		struct master_iface *iface = dev_get_drvdata (dev); \
		char *endp; \
		unsigned long val = simple_strtoul (buf, &endp, 0); \
		ssize_t err; \
		if ((endp == buf) || \
			sdi_validate_##var (iface, val)) { \
			return -EINVAL; \
		} \
		err = miface_store (iface, &iface->var, val); \
		if (err) { \
			return err; \
		} \
		return count; \
	}
SDI_STORE(buffers)
SDI_STORE(bufsize)
SDI_STORE(clksrc)
SDI_STORE(mode)

static DEVICE_ATTR(buffers,S_IRUGO|S_IWUSR,
	miface_show_buffers,sdi_store_buffers);
static DEVICE_ATTR(bufsize,S_IRUGO|S_IWUSR,
	miface_show_bufsize,sdi_store_bufsize);
static DEVICE_ATTR(clock_source,S_IRUGO|S_IWUSR,
	miface_show_clksrc,sdi_store_clksrc);
static DEVICE_ATTR(mode,S_IRUGO|S_IWUSR,
	miface_show_mode,sdi_store_mode);

/**
 * sdi_register_iface - register an interface
 * @card: pointer to the board info structure
 * @dma_ops: pointer to DMA helper functions
 * @data_addr: local bus address of the FIFO
 * @direction: direction of data flow
 * @fops: file operations structure
 * @iface_ops: pointer to Master interface helper functions
 * @cap: capabilities flags
 * @granularity: buffer size granularity in bytes
 *
 * Allocate and initialize an interface information structure.
 * Assign the lowest unused minor number to this interface
 * and add it to the list of interfaces for this device
 * and the list of all interfaces.
 * Also initialize the device parameters.
 * Returns a negative error code on failure and 0 on success.
 **/
int
sdi_register_iface (struct master_dev *card,
	struct master_dma_operations *dma_ops,
	u32 data_addr,
	unsigned int direction,
	struct file_operations *fops,
	struct master_iface_operations *iface_ops,
	unsigned int cap,
	unsigned int granularity)
{
	struct master_iface *iface, *entry;
	const char *type;
	struct list_head *p;
	unsigned int minminor, minor, maxminor, found;
	int err;

	/* Allocate an interface info structure */
	iface = (struct master_iface *)kzalloc (sizeof (*iface), GFP_KERNEL);
	if (iface == NULL) {
		err = -ENOMEM;
		goto NO_IFACE;
	}

	/* Initialize an interface info structure */
	iface->direction = direction;
	cdev_init (&iface->cdev, fops);
	iface->cdev.owner = THIS_MODULE;
	iface->capabilities = cap;
	if (iface->direction == MASTER_DIRECTION_TX) {
		iface->buffers = SDI_TX_BUFFERS;
		iface->bufsize = SDI_TX_BUFSIZE;
		iface->clksrc = SDI_CTL_TX_CLKSRC_ONBOARD;
	} else {
		iface->buffers = SDI_RX_BUFFERS;
		iface->bufsize = SDI_RX_BUFSIZE;
	}
	iface->granularity = granularity;
	iface->mode = SDI_CTL_MODE_10BIT;
	iface->ops = iface_ops;
	iface->dma_ops = dma_ops;
	iface->data_addr = data_addr;
	iface->dma_flags = MDMA_MMAP;
	init_waitqueue_head (&iface->queue);
	mutex_init (&iface->buf_mutex);
	iface->card = card;

	/* Assign the lowest unused minor number to this interface */
	switch (iface->direction) {
	case MASTER_DIRECTION_TX:
		type = "transmitter";
		minminor = minor = 0;
		break;
	case MASTER_DIRECTION_RX:
		type = "receiver";
		minminor = minor = 1 << (MASTER_MINORBITS - 1);
		break;
	default:
		err = -EINVAL;
		goto NO_MINOR;
	}
	maxminor = minor + (1 << (MASTER_MINORBITS - 1)) - 1;
	spin_lock (&sdi_iface_lock);
	while (minor <= maxminor) {
		found = 0;
		list_for_each (p, &sdi_iface_list) {
			entry = list_entry (p, struct master_iface, list_all);
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
	spin_unlock (&sdi_iface_lock);
	if (minor > maxminor) {
		err = -ENOSPC;
		goto NO_MINOR;
	}

	/* Add this interface to the list of all interfaces */
	spin_lock (&sdi_iface_lock);
	list_add_tail (&iface->list_all, &sdi_iface_list);
	spin_unlock (&sdi_iface_lock);

	/* Add this interface to the list for this device */
	list_add_tail (&iface->list, &card->iface_list);

	/* Create the device */
	iface->dev = device_create (sdi_class,
		card->dev,
		MKDEV(major, minor),
		iface,
		"sdi%cx%i",
		type[0],
		minor & ((1 << (MASTER_MINORBITS - 1)) - 1));
	if (IS_ERR(iface->dev)) {
		printk (KERN_WARNING "%s: unable to register device\n",
			sdi_driver_name);
		err = PTR_ERR(iface->dev);
		goto NO_DEV;
	}
	dev_set_drvdata (iface->dev, iface);

	/* Add device attributes */
	if ((err = device_create_file (iface->dev,
		&dev_attr_buffers)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create file 'buffers'\n",
			sdi_driver_name);
	}
	if ((err = device_create_file (iface->dev,
		&dev_attr_bufsize)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create file 'bufsize'\n",
			sdi_driver_name);
	}
	if (iface->direction == MASTER_DIRECTION_TX) {
		if (iface->capabilities & SDI_CAP_TX_RXCLKSRC) {
			if ((err = device_create_file (iface->dev,
				&dev_attr_clock_source)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'clock_source'\n",
					sdi_driver_name);
			}
		}
	}
	if ((err = device_create_file (iface->dev,
		&dev_attr_mode)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create file 'mode'\n",
			sdi_driver_name);
	}

	/* Activate the cdev */
	if ((err = cdev_add (&iface->cdev, MKDEV(major,minor), 1)) < 0) {
		printk (KERN_WARNING "%s: unable to add character device\n",
			sdi_driver_name);
		goto NO_CDEV;
	}

	printk (KERN_INFO "%s: registered %s %u\n",
		sdi_driver_name, type, minor - minminor);
	return 0;

NO_CDEV:
	device_destroy (sdi_class, MKDEV(major,minor));
NO_DEV:
	list_del (&iface->list);
	spin_lock (&sdi_iface_lock);
	list_del (&iface->list_all);
	spin_unlock (&sdi_iface_lock);
NO_MINOR:
	kfree (iface);
NO_IFACE:
	return err;
}

/**
 * sdi_unregister_iface - remove an interface from the list
 * @iface: interface
 **/
void
sdi_unregister_iface (struct master_iface *iface)
{
	cdev_del (&iface->cdev);
	device_destroy (sdi_class, iface->cdev.dev);
	list_del (&iface->list);
	spin_lock (&sdi_iface_lock);
	list_del (&iface->list_all);
	spin_unlock (&sdi_iface_lock);
	kfree (iface);
	return;
}

/**
 * sdi_init_module - initialize the module
 *
 * Register the module as a character PCI driver.
 * Returns a negative error code on failure and 0 on success.
 **/
static int __init
sdi_init_module (void)
{
	int err;
	dev_t dev = MKDEV(major,0);

	spin_lock_init (&sdi_iface_lock);

	/* Create a device class */
	sdi_class = class_create (THIS_MODULE, sdi_driver_name);
	if (IS_ERR(sdi_class)) {
		printk (KERN_WARNING "%s: unable to register device class\n",
			sdi_driver_name);
		err = PTR_ERR(sdi_class);
		goto NO_CLASS;
	}

	/* Add class attributes */
	if ((err = class_create_file (sdi_class, &class_attr_version)) < 0) {
		printk (KERN_WARNING "%s: unable to create file 'version' \n",
			sdi_driver_name);
	}

	/* Reserve a range of device numbers */
	if (major) {
		err = register_chrdev_region (dev,
			count,
			sdi_driver_name);
	} else {
		err = alloc_chrdev_region (&dev,
			0,
			count,
			sdi_driver_name);
	}
	if (err < 0) {
		printk (KERN_WARNING
			"%s: unable to reserve device number range\n",
			sdi_driver_name);
		goto NO_RANGE;
	}
	major = MAJOR(dev);

	return 0;

NO_RANGE:
	class_destroy (sdi_class);
NO_CLASS:
	return err;
}

/**
 * sdi_cleanup_module - cleanup the module
 *
 * Unregister the module as a character PCI driver.
 **/
static void __exit
sdi_cleanup_module (void)
{
	unregister_chrdev_region (MKDEV(major,0), count);
	class_destroy (sdi_class);
	return;
}

module_init (sdi_init_module);
module_exit (sdi_cleanup_module);

