/* asicore.c
 *
 * Linear Systems Ltd. DVB ASI API.
 *
 * Copyright (C) 1999 Tony Bolger <d7v@indigo.ie>
 * Copyright (C) 2000-2010 Linear Systems Ltd.
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

#include "asicore.h"
#include "../include/master.h"
#include "miface.h"

/* Static function prototypes */
static int asi_ioctl (struct master_iface *iface,
	unsigned int cmd,
	unsigned long arg);
static int asi_validate_buffers (struct master_iface *iface,
	unsigned long val);
static int asi_validate_bufsize (struct master_iface *iface,
	unsigned long val);
static int asi_validate_clksrc (struct master_iface *iface,
	unsigned long val);
static int asi_validate_mode (struct master_iface *iface,
	unsigned long val);
static int asi_validate_timestamps (struct master_iface *iface,
	unsigned long val);
static ssize_t asi_store_buffers (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static ssize_t asi_store_bufsize (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static ssize_t asi_store_clksrc (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static ssize_t asi_store_count27 (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static ssize_t asi_store_mode (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static ssize_t asi_store_null_packets (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static ssize_t asi_store_timestamps (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static int asi_init_module (void) __init;
static void asi_cleanup_module (void) __exit;

/* The major number can be overridden from the command line
 * when the module is loaded. */
static unsigned int major = ASI_MAJOR;
module_param(major,uint,S_IRUGO);
MODULE_PARM_DESC(major,"Major number of the first device, or zero for dynamic");

static const unsigned int count = 1 << MASTER_MINORBITS;

MODULE_AUTHOR("Linear Systems Ltd.");
MODULE_DESCRIPTION("DVB ASI module");
MODULE_LICENSE("GPL");
MODULE_VERSION(MASTER_DRIVER_VERSION);

EXPORT_SYMBOL(asi_open);
EXPORT_SYMBOL(asi_write);
EXPORT_SYMBOL(asi_read);
EXPORT_SYMBOL(asi_txpoll);
EXPORT_SYMBOL(asi_rxpoll);
EXPORT_SYMBOL(asi_txioctl);
EXPORT_SYMBOL(asi_rxioctl);
EXPORT_SYMBOL(asi_mmap);
EXPORT_SYMBOL(asi_release);
EXPORT_SYMBOL(asi_register_iface);
EXPORT_SYMBOL(asi_unregister_iface);

static char asi_driver_name[] = ASI_DRIVER_NAME;

static LIST_HEAD(asi_iface_list);

static spinlock_t asi_iface_lock;

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,27))
#define device_create(cls,parent,devt,drvdata,fmt,...) \
	device_create(cls,parent,devt,fmt,##__VA_ARGS__)
#endif

static struct class *asi_class;
static CLASS_ATTR(version,S_IRUGO,
	miface_show_version,NULL);

/**
 * asi_open - ASI interface open() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
int
asi_open (struct inode *inode, struct file *filp)
{
	return miface_open (inode, filp);
}

/**
 * asi_write - ASI interface write() method
 * @filp: file
 * @data: data
 * @length: size of data
 * @offset:
 *
 * Returns a negative error code on failure and
 * the number of bytes written on success.
 **/
ssize_t
asi_write (struct file *filp,
	const char __user *data,
	size_t length,
	loff_t *offset)
{
	return miface_write (filp, data, length, offset);
}

/**
 * asi_read - ASI interface read() method
 * @filp: file
 * @data: read buffer
 * @length: size of data
 * @offset:
 *
 * Returns a negative error code on failure and
 * the number of bytes read on success.
 **/
ssize_t
asi_read (struct file *filp,
	char __user *data,
	size_t length,
	loff_t *offset)
{
	return miface_read (filp, data, length, offset);
}

/**
 * asi_txpoll - transmitter poll() method
 * @filp: file
 * @wait:
 **/
unsigned int
asi_txpoll (struct file *filp, poll_table *wait)
{
	return miface_txpoll (filp, wait);
}

/**
 * asi_rxpoll - receiver poll() method
 * @filp: file
 * @wait:
 **/
unsigned int
asi_rxpoll (struct file *filp, poll_table *wait)
{
	return miface_rxpoll (filp, wait);
}

/**
 * asi_ioctl - generic ioctl() method
 * @iface: interface
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
asi_ioctl (struct master_iface *iface,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_dev *card = iface->card;

	switch (cmd) {
	case ASI_IOC_GETID:
		if (put_user (card->id,
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_GETVERSION:
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
 * asi_txioctl - ASI transmitter interface ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
long
asi_txioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	unsigned int reg = 0, i;

	switch (cmd) {
	case ASI_IOC_TXGETCAP:
		if (put_user (iface->capabilities,
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_TXGETEVENTS:
		for (i = 0; i < 8 * sizeof (reg); i++) {
			if (test_and_clear_bit (i, &iface->events)) {
				reg |= (1 << i);
			}
		}
		if (put_user (reg, (unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_TXGETBUFLEVEL:
		if (put_user (mdma_tx_buflevel (iface->dma),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return asi_ioctl (iface, cmd, arg);
	}
	return 0;
}

/**
 * asi_rxioctl - ASI receiver interface ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
long
asi_rxioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	unsigned int reg = 0, i;

	switch (cmd) {
	case ASI_IOC_RXGETCAP:
		if (put_user (iface->capabilities,
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXGETEVENTS:
		for (i = 0; i < 8 * sizeof (reg); i++) {
			if (test_and_clear_bit (i, &iface->events)) {
				reg |= (1 << i);
			}
		}
		if (put_user (reg, (unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXGETBUFLEVEL:
		if (put_user (mdma_rx_buflevel (iface->dma),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return asi_ioctl (iface, cmd, arg);
	}
	return 0;
}

/**
 * asi_mmap - ASI interface mmap() method
 * @filp: file
 * @vma: VMA
 **/
int
asi_mmap (struct file *filp, struct vm_area_struct *vma)
{
	return miface_mmap (filp, vma);
}

/**
 * asi_release - ASI interface release() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
int
asi_release (struct inode *inode, struct file *filp)
{
	return miface_release (inode, filp);
}

/**
 * asi_validate_buffers - validate a buffers attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
asi_validate_buffers (struct master_iface *iface,
	unsigned long val)
{
	const unsigned int min = (iface->direction == MASTER_DIRECTION_TX) ?
		ASI_TX_BUFFERS_MIN : ASI_RX_BUFFERS_MIN;
	const unsigned int max = ASI_BUFFERS_MAX;

	if ((val < min) || (val > max)) {
		return -EINVAL;
	}
	return 0;
}

/**
 * asi_validate_bufsize - validate a bufsize attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
asi_validate_bufsize (struct master_iface *iface,
	unsigned long val)
{
	const unsigned int min = (iface->direction == MASTER_DIRECTION_TX) ?
		ASI_TX_BUFSIZE_MIN : ASI_RX_BUFSIZE_MIN;
	const unsigned int max = ASI_BUFSIZE_MAX;
	const unsigned int mult = iface->granularity;

	if ((val < min) || (val > max) || (val % mult)) {
		return -EINVAL;
	}
	return 0;
}

/**
 * asi_validate_clksrc - validate a clksrc attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
asi_validate_clksrc (struct master_iface *iface,
	unsigned long val)
{
	switch (val) {
	case ASI_CTL_TX_CLKSRC_ONBOARD:
		break;
	case ASI_CTL_TX_CLKSRC_EXT:
		if (!(iface->capabilities & ASI_CAP_TX_SETCLKSRC)) {
			return -EINVAL;
		}
		break;
	case ASI_CTL_TX_CLKSRC_RX:
		if (!(iface->capabilities & ASI_CAP_TX_RXCLKSRC)) {
			return -EINVAL;
		}
		break;
	case ASI_CTL_TX_CLKSRC_EXT2:
		if (!(iface->capabilities & ASI_CAP_TX_EXTCLKSRC2)) {
			return -EINVAL;
		}
		break;
	default:
		return -EINVAL;
	}
	return 0;
}

#define asi_validate_count27(iface,val) 0

/**
 * asi_validate_mode - validate a mode attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
asi_validate_mode (struct master_iface *iface,
	unsigned long val)
{
	unsigned int min, max;

	if (iface->direction == MASTER_DIRECTION_TX) {
		min = ASI_CTL_TX_MODE_188;
		max = (iface->capabilities & ASI_CAP_TX_MAKE204) ?
			ASI_CTL_TX_MODE_MAKE204 : ASI_CTL_TX_MODE_204;
	} else {
		min = ASI_CTL_RX_MODE_RAW;
		max = (iface->capabilities & ASI_CAP_RX_MAKE188) ?
			ASI_CTL_RX_MODE_204MAKE188 : ASI_CTL_RX_MODE_AUTO;
	}
	if ((val < min) || (val > max)) {
		return -EINVAL;
	}
	return 0;
}

#define asi_validate_null_packets(iface,val) 0

/**
 * asi_validate_timestamps - validate a timestamps attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
asi_validate_timestamps (struct master_iface *iface,
	unsigned long val)
{
	const unsigned int min = ASI_CTL_TSTAMP_NONE;
	unsigned int max;

	if (iface->direction == MASTER_DIRECTION_TX) {
		max = (iface->capabilities & ASI_CAP_TX_PTIMESTAMPS) ?
			ASI_CTL_TSTAMP_PREPEND : ASI_CTL_TSTAMP_APPEND;
	} else {
		max = (iface->capabilities & ASI_CAP_RX_PTIMESTAMPS) ?
			ASI_CTL_TSTAMP_PREPEND : ASI_CTL_TSTAMP_APPEND;
	}
	if ((val < min) || (val > max)) {
		return -EINVAL;
	}
	return 0;
}

/**
 * asi_store_* - ASI interface attribute write handler
 * @dev: device being written
 * @attr: device attribute
 * @buf: input buffer
 * @count: buffer size
 **/
#define ASI_STORE(var) \
	static ssize_t asi_store_##var (struct device *dev, \
		struct device_attribute *attr, \
		const char *buf, \
		size_t count) \
	{ \
		struct master_iface *iface = dev_get_drvdata (dev); \
		char *endp; \
		unsigned long val = simple_strtoul (buf, &endp, 0); \
		ssize_t err; \
		if ((endp == buf) || \
			asi_validate_##var (iface, val)) { \
			return -EINVAL; \
		} \
		err = miface_store (iface, &iface->var, val); \
		if (err) { \
			return err; \
		} \
		return count; \
	}
ASI_STORE(buffers)
ASI_STORE(bufsize)
ASI_STORE(clksrc)
ASI_STORE(count27)
ASI_STORE(mode)
ASI_STORE(null_packets)
ASI_STORE(timestamps)

static DEVICE_ATTR(buffers,S_IRUGO|S_IWUSR,
	miface_show_buffers,asi_store_buffers);
static DEVICE_ATTR(bufsize,S_IRUGO|S_IWUSR,
	miface_show_bufsize,asi_store_bufsize);
static DEVICE_ATTR(clock_source,S_IRUGO|S_IWUSR,
	miface_show_clksrc,asi_store_clksrc);
static DEVICE_ATTR(count27,S_IRUGO|S_IWUSR,
	miface_show_count27,asi_store_count27);
static DEVICE_ATTR(granularity,S_IRUGO,
	miface_show_granularity,NULL);
static DEVICE_ATTR(mode,S_IRUGO|S_IWUSR,
	miface_show_mode,asi_store_mode);
static DEVICE_ATTR(null_packets,S_IRUGO|S_IWUSR,
	miface_show_null_packets,asi_store_null_packets);
static DEVICE_ATTR(timestamps,S_IRUGO|S_IWUSR,
	miface_show_timestamps,asi_store_timestamps);
static DEVICE_ATTR(transport,S_IRUGO,
	miface_show_transport,NULL);

/**
 * asi_register_iface - register an interface
 * @card: pointer to the board info structure
 * @dma_ops: pointer to DMA helper functions
 * @data_addr: local bus address of the FIFO
 * @direction: direction of data flow
 * @fops: file operations structure
 * @iface_ops: pointer to Master interface helper functions
 * @cap: capabilities flags
 * @granularity: buffer size granularity in bytes
 * @transport: transport type
 *
 * Allocate and initialize an interface information structure.
 * Assign the lowest unused minor number to this interface
 * and add it to the list of interfaces for this device
 * and the list of all interfaces.
 * Also initialize the device parameters.
 * Returns a negative error code on failure and 0 on success.
 **/
int
asi_register_iface (struct master_dev *card,
	struct master_dma_operations *dma_ops,
	u32 data_addr,
	unsigned int direction,
	struct file_operations *fops,
	struct master_iface_operations *iface_ops,
	unsigned int cap,
	unsigned int granularity,
	unsigned int transport)
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
		iface->buffers = ASI_TX_BUFFERS;
		iface->bufsize = ASI_TX_BUFSIZE;
		iface->clksrc = ASI_CTL_TX_CLKSRC_ONBOARD;
		iface->mode = ASI_CTL_TX_MODE_188;
		if (iface->capabilities & ASI_CAP_TX_27COUNTER) {
			iface->count27 = 1;
		}
	} else {
		iface->buffers = ASI_RX_BUFFERS;
		iface->bufsize = ASI_RX_BUFSIZE;
		iface->mode = ASI_CTL_RX_MODE_RAW;
		if (iface->capabilities & ASI_CAP_RX_SYNC) {
			iface->mode = ASI_CTL_RX_MODE_188;
		}
		if (iface->capabilities & ASI_CAP_RX_27COUNTER) {
			iface->count27 = 1;
		}
	}
	iface->granularity = granularity;
	iface->transport = transport;
	iface->ops = iface_ops;
	iface->dma_ops = dma_ops;
	iface->data_addr = data_addr;
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
	spin_lock (&asi_iface_lock);
	while (minor <= maxminor) {
		found = 0;
		list_for_each (p, &asi_iface_list) {
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
	spin_unlock (&asi_iface_lock);
	if (minor > maxminor) {
		err = -ENOSPC;
		goto NO_MINOR;
	}

	/* Add this interface to the list of all interfaces */
	spin_lock (&asi_iface_lock);
	list_add_tail (&iface->list_all, &asi_iface_list);
	spin_unlock (&asi_iface_lock);

	/* Add this interface to the list for this device */
	list_add_tail (&iface->list, &card->iface_list);

	/* Create the device */
	iface->dev = device_create (asi_class,
		card->dev,
		MKDEV(major, minor),
		iface,
		"asi%cx%u",
		type[0],
		minor & ((1 << (MASTER_MINORBITS - 1)) - 1));
	if (IS_ERR(iface->dev)) {
		printk (KERN_WARNING "%s: unable to create device\n",
			asi_driver_name);
		err = PTR_ERR(iface->dev);
		goto NO_DEV;
	}
	dev_set_drvdata (iface->dev, iface);

	/* Add device attributes */
	if ((err = device_create_file (iface->dev,
		&dev_attr_buffers)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create file 'buffers'\n",
			asi_driver_name);
	}
	if ((err = device_create_file (iface->dev,
		&dev_attr_bufsize)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create file 'bufsize'\n",
			asi_driver_name);
	}
	if ((err = device_create_file (iface->dev,
		&dev_attr_granularity)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create file 'granularity'\n",
			asi_driver_name);
	}
	if ((err = device_create_file (iface->dev,
		&dev_attr_transport)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create file 'transport'\n",
			asi_driver_name);
	}
	if (iface->direction == MASTER_DIRECTION_TX) {
		if (iface->capabilities & ASI_CAP_TX_SETCLKSRC) {
			if ((err = device_create_file (iface->dev,
				&dev_attr_clock_source)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'clock_source'\n",
					asi_driver_name);
			}
		}
		if (iface->capabilities & ASI_CAP_TX_BYTESOR27) {
			if ((err = device_create_file (iface->dev,
				&dev_attr_count27)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'count27'\n",
					asi_driver_name);
			}
		}
		if ((err = device_create_file (iface->dev,
			&dev_attr_mode)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'mode'\n",
				asi_driver_name);
		}
		if (iface->capabilities & ASI_CAP_TX_NULLPACKETS) {
			if ((err = device_create_file (iface->dev,
				&dev_attr_null_packets)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'null_packets'\n",
					asi_driver_name);
			}
		}
		if (iface->capabilities & ASI_CAP_TX_TIMESTAMPS) {
			if ((err = device_create_file (iface->dev,
				&dev_attr_timestamps)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'timestamps'\n",
					asi_driver_name);
			}
		}
	} else {
		if (iface->capabilities & ASI_CAP_RX_BYTESOR27) {
			if ((err = device_create_file (iface->dev,
				&dev_attr_count27)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'count27'\n",
					asi_driver_name);
			}
		}
		if (iface->capabilities & ASI_CAP_RX_SYNC) {
			if ((err = device_create_file (iface->dev,
				&dev_attr_mode)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'mode'\n",
					asi_driver_name);
			}
		}
		if (iface->capabilities & ASI_CAP_RX_NULLPACKETS) {
			if ((err = device_create_file (iface->dev,
				&dev_attr_null_packets)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'null_packets'\n",
					asi_driver_name);
			}
		}
		if (iface->capabilities & ASI_CAP_RX_TIMESTAMPS) {
			if ((err = device_create_file (iface->dev,
				&dev_attr_timestamps)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'timestamps'\n",
					asi_driver_name);
			}
		}
	}

	/* Activate the cdev */
	if ((err = cdev_add (&iface->cdev, MKDEV(major,minor), 1)) < 0) {
		printk (KERN_WARNING "%s: unable to add character device\n",
			asi_driver_name);
		goto NO_CDEV;
	}

	printk (KERN_INFO "%s: registered %s %u\n",
		asi_driver_name, type, minor - minminor);
	return 0;

NO_CDEV:
	device_destroy (asi_class, MKDEV(major,minor));
NO_DEV:
	list_del (&iface->list);
	spin_lock (&asi_iface_lock);
	list_del (&iface->list_all);
	spin_unlock (&asi_iface_lock);
NO_MINOR:
	kfree (iface);
NO_IFACE:
	return err;
}

/**
 * asi_unregister_iface - remove an interface from the list
 * @iface: interface
 **/
void
asi_unregister_iface (struct master_iface *iface)
{
	cdev_del (&iface->cdev);
	device_destroy (asi_class, iface->cdev.dev);
	list_del (&iface->list);
	spin_lock (&asi_iface_lock);
	list_del (&iface->list_all);
	spin_unlock (&asi_iface_lock);
	kfree (iface);
	return;
}

/**
 * asi_init_module - initialize the module
 *
 * Register the module as a character driver and create a device class.
 * Returns a negative error code on failure and 0 on success.
 **/
static int __init
asi_init_module (void)
{
	int err;
	dev_t dev = MKDEV(major,0);

	spin_lock_init (&asi_iface_lock);

	/* Create a device class */
	asi_class = class_create (THIS_MODULE, asi_driver_name);
	if (IS_ERR(asi_class)) {
		printk (KERN_WARNING "%s: unable to create device class\n",
			asi_driver_name);
		err = PTR_ERR(asi_class);
		goto NO_CLASS;
	}

	/* Add class attributes */
	if ((err = class_create_file (asi_class, &class_attr_version)) < 0) {
		printk (KERN_WARNING "%s: unable to create file 'version' \n",
			asi_driver_name);
	}

	/* Reserve a range of device numbers */
	if (major) {
		err = register_chrdev_region (dev,
			count,
			asi_driver_name);
	} else {
		err = alloc_chrdev_region (&dev,
			0,
			count,
			asi_driver_name);
	}
	if (err < 0) {
		printk (KERN_WARNING
			"%s: unable to reserve device number range\n",
			asi_driver_name);
		goto NO_RANGE;
	}
	major = MAJOR(dev);

	return 0;

NO_RANGE:
	class_destroy (asi_class);
NO_CLASS:
	return err;
}

/**
 * asi_cleanup_module - cleanup the module
 *
 * Unregister the module as a character driver.
 **/
static void __exit
asi_cleanup_module (void)
{
	unregister_chrdev_region (MKDEV(major,0), count);
	class_destroy (asi_class);
	return;
}

module_init (asi_init_module);
module_exit (asi_cleanup_module);

