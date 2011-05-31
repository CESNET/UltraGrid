/* sdiaudiocore.c
 *
 * Linear Systems Ltd. SDI audio API.
 *
 * Copyright (C) 2009-2010 Linear Systems Ltd.
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

#include "sdiaudiocore.h"
#include "../include/master.h"
#include "miface.h"
#include "mdma.h"

/* Static function prototypes */
static int sdiaudio_ioctl (struct master_iface *iface,
	unsigned int cmd,
	unsigned long arg);
static int sdiaudio_validate_buffers (struct master_iface *iface,
	unsigned long val);
static int sdiaudio_validate_bufsize (struct master_iface *iface,
	unsigned long val);
static int sdiaudio_validate_channels (struct master_iface *iface,
	unsigned long val);
static int sdiaudio_validate_nonaudio (struct master_iface *iface,
	unsigned long val);
static int sdiaudio_validate_sample_rate (struct master_iface *iface,
	unsigned long val);
static int sdiaudio_validate_sample_size (struct master_iface *iface,
	unsigned long val);
static ssize_t sdiaudio_store_buffers (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static ssize_t sdiaudio_store_bufsize (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static ssize_t sdiaudio_store_channels (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static ssize_t sdiaudio_store_nonaudio (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static ssize_t sdiaudio_store_sample_rate (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static ssize_t sdiaudio_store_sample_size (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static int sdiaudio_init_module (void) __init;
static void sdiaudio_cleanup_module (void) __exit;

/* The major number can be overridden from the command line
 * when the module is loaded. */
static unsigned int major = SDIAUDIO_MAJOR;
module_param(major,uint,S_IRUGO);
MODULE_PARM_DESC(major,"Major number of the first device, or zero for dynamic");

static const unsigned int count = 1 << MASTER_MINORBITS;

MODULE_AUTHOR("Linear Systems Ltd.");
MODULE_DESCRIPTION("SMPTE 292M and SMPTE 259M-C audio module");
MODULE_LICENSE("GPL");
MODULE_VERSION(MASTER_DRIVER_VERSION);

EXPORT_SYMBOL(sdiaudio_open);
EXPORT_SYMBOL(sdiaudio_write);
EXPORT_SYMBOL(sdiaudio_read);
EXPORT_SYMBOL(sdiaudio_txpoll);
EXPORT_SYMBOL(sdiaudio_rxpoll);
EXPORT_SYMBOL(sdiaudio_txioctl);
EXPORT_SYMBOL(sdiaudio_rxioctl);
EXPORT_SYMBOL(sdiaudio_mmap);
EXPORT_SYMBOL(sdiaudio_release);
EXPORT_SYMBOL(sdiaudio_register_iface);
EXPORT_SYMBOL(sdiaudio_unregister_iface);

static char sdiaudio_driver_name[] = SDIAUDIO_DRIVER_NAME;

static LIST_HEAD(sdiaudio_iface_list);

static spinlock_t sdiaudio_iface_lock;

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,27))
#define device_create(cls,parent,devt,drvdata,fmt,...) \
	device_create(cls,parent,devt,fmt,##__VA_ARGS__)
#endif

static struct class *sdiaudio_class;
static CLASS_ATTR(version,S_IRUGO,
	miface_show_version,NULL);

/**
 * sdiaudio_open - SMPTE 292M and SMPTE 259M-C audio interface open() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
int
sdiaudio_open (struct inode *inode, struct file *filp)
{
	return miface_open (inode, filp);
}

/**
 * sdiaudio_write - SMPTE 292M and SMPTE 259M-C audio interface write() method
 * @filp: file
 * @data: data
 * @length: size of data
 * @offset:
 *
 * Returns a negative error code on failure and
 * the number of bytes written on success.
 **/
ssize_t
sdiaudio_write (struct file *filp,
	const char __user *data,
	size_t length,
	loff_t *offset)
{
	return miface_write (filp, data, length, offset);
}

/**
 * sdiaudio_read - SMPTE 292M and SMPTE 259M-C audio interface read() method
 * @filp: file
 * @data: read buffer
 * @length: size of data
 * @offset:
 *
 * Returns a negative error code on failure and
 * the number of bytes read on success.
 **/
ssize_t
sdiaudio_read (struct file *filp,
	char __user *data,
	size_t length,
	loff_t *offset)
{
	return miface_read (filp, data, length, offset);
}

/**
 * sdiaudio_txpoll - transmitter poll() method
 * @filp: file
 * @wait:
 **/
unsigned int
sdiaudio_txpoll (struct file *filp, poll_table *wait)
{
	return miface_txpoll (filp, wait);
}

/**
 * sdiaudio_rxpoll - receiver poll() method
 * @filp: file
 * @wait:
 **/
unsigned int
sdiaudio_rxpoll (struct file *filp, poll_table *wait)
{
	return miface_rxpoll (filp, wait);
}

/**
 * sdiaudio_ioctl - generic ioctl() method
 * @iface: interface
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
sdiaudio_ioctl (struct master_iface *iface,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_dev *card = iface->card;

	switch (cmd) {
	case SDIAUDIO_IOC_GETID:
		if (put_user (card->id,
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIAUDIO_IOC_GETVERSION:
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
 * sdiaudio_txioctl - SMPTE 292M and SMPTE 259M-C audio transmitter interface ioctl() method
 * @iface: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
long
sdiaudio_txioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	unsigned int reg = 0, i;

	switch (cmd) {
	case SDIAUDIO_IOC_TXGETCAP:
		if (put_user (iface->capabilities,
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIAUDIO_IOC_TXGETEVENTS:
		for (i = 0; i < 8 * sizeof (reg); i++) {
			if (test_and_clear_bit (i, &iface->events)) {
				reg |= (1 << i);
			}
		}
		if (put_user (reg, (unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIAUDIO_IOC_TXGETBUFLEVEL:
		if (put_user (mdma_tx_buflevel (iface->dma),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIAUDIO_IOC_QBUF_DEPRECATED:
	case SDIAUDIO_IOC_QBUF:
		return miface_txqbuf (filp, arg);
	case SDIAUDIO_IOC_DQBUF_DEPRECATED:
	case SDIAUDIO_IOC_DQBUF:
		return miface_txdqbuf (filp, arg);
	default:
		return sdiaudio_ioctl (iface, cmd, arg);
	}
	return 0;
}

/**
 * sdiaudio_rxioctl - SMPTE 292M and SMPTE 259M-C audio receiver interface ioctl() method
 * @iface: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
long
sdiaudio_rxioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	unsigned int reg = 0, i;

	switch (cmd) {
	case SDIAUDIO_IOC_RXGETCAP:
		if (put_user (iface->capabilities,
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIAUDIO_IOC_RXGETEVENTS:
		for (i = 0; i < 8 * sizeof (reg); i++) {
			if (test_and_clear_bit (i, &iface->events)) {
				reg |= (1 << i);
			}
		}
		if (put_user (reg, (unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIAUDIO_IOC_RXGETBUFLEVEL:
		if (put_user (mdma_rx_buflevel (iface->dma),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIAUDIO_IOC_QBUF_DEPRECATED:
	case SDIAUDIO_IOC_QBUF:
		return miface_rxqbuf (filp, arg);
	case SDIAUDIO_IOC_DQBUF_DEPRECATED:
	case SDIAUDIO_IOC_DQBUF:
		return miface_rxdqbuf (filp, arg);
	default:
		return sdiaudio_ioctl (iface, cmd, arg);
	}
	return 0;
}

/**
 * sdiaudio_mmap - SMPTE 292M and SMPTE 259M-C audio interface mmap() method
 * @filp: file
 * @vma: VMA
 **/
int
sdiaudio_mmap (struct file *filp, struct vm_area_struct *vma)
{
	return miface_mmap (filp, vma);
}

/**
 * sdiaudio_release - SMPTE 292M and SMPTE 259M-C audio interface release() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
int
sdiaudio_release (struct inode *inode, struct file *filp)
{
	return miface_release (inode, filp);
}

/**
 * sdiaudio_validate_buffers - validate a buffers attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
sdiaudio_validate_buffers (struct master_iface *iface,
	unsigned long val)
{
	const unsigned int min = (iface->direction == MASTER_DIRECTION_TX) ?
		SDIAUDIO_TX_BUFFERS_MIN : SDIAUDIO_RX_BUFFERS_MIN;
	const unsigned int max = SDIAUDIO_BUFFERS_MAX;

	if ((val < min) || (val > max)) {
		return -EINVAL;
	}
	return 0;
}

/**
 * sdiaudio_validate_bufsize - validate a bufsize attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
sdiaudio_validate_bufsize (struct master_iface *iface,
	unsigned long val)
{
	const unsigned int min = (iface->direction == MASTER_DIRECTION_TX) ?
		SDIAUDIO_TX_BUFSIZE_MIN : SDIAUDIO_RX_BUFSIZE_MIN;
	const unsigned int max = SDIAUDIO_BUFSIZE_MAX;
	const unsigned int mult = iface->granularity;

	if ((val < min) || (val > max) || (val % mult)) {
		return -EINVAL;
	}
	return 0;
}

/**
 * sdiaudio_validate_channels - validate a channels attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
sdiaudio_validate_channels (struct master_iface *iface,
	unsigned long val)
{
	const unsigned int min = SDIAUDIO_CTL_AUDCH_EN_2;
	const unsigned int max = SDIAUDIO_CTL_AUDCH_EN_8;
	const unsigned int mult = 2;

	if ((val < min) || (val > max) || (val % mult)) {
		return -EINVAL;
	}
	return 0;
}

/**
 * sdiaudio_validate_nonaudio - validate a nonaudio attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
sdiaudio_validate_nonaudio (struct master_iface *iface,
	unsigned long val)
{
	const unsigned int min = SDIAUDIO_CTL_PCM_ALLCHANNEL;
	const unsigned int max = SDIAUDIO_CTL_NONAUDIO_ALLCHANNEL;

	if ((val < min) || (val > max)) {
		return -EINVAL;
	}
	return 0;
}

/**
 * sdiaudio_validate_sample_rate - validate a sample_rate attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
sdiaudio_validate_sample_rate (struct master_iface *iface,
	unsigned long val)
{
	switch (val) {
	case 32000:
	case 44100:
	case 48000:
		break;
	default:
		return -EINVAL;
	}
	return 0;
}

/**
 * sdiaudio_validate_sample_size - validate a sample_size attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
sdiaudio_validate_sample_size (struct master_iface *iface,
	unsigned long val)
{
	const unsigned int min = SDIAUDIO_CTL_AUDSAMP_SZ_16;
	const unsigned int max = SDIAUDIO_CTL_AUDSAMP_SZ_32;
	const unsigned int mult = ((iface->direction == MASTER_DIRECTION_RX) &&
		(iface->capabilities & SDIAUDIO_CAP_RX_24BIT)) ? 8 : 16;

	if ((val < min) || (val > max) || (val % mult)) {
		return -EINVAL;
	}
	return 0;
}

/**
 * sdiaudio_store_* - BT601 576i, SMPTE 125M, SMPTE 274M, SMPTE 260M,
 * SMPTE 296M interface attribute write handler
 * @dev: device being written
 * @attr: device attribute
 * @buf: input buffer
 * @count:
 **/
#define SDIAUDIO_STORE(var) \
	static ssize_t sdiaudio_store_##var (struct device *dev, \
		struct device_attribute *attr, \
		const char *buf, \
		size_t count) \
	{ \
		struct master_iface *iface = dev_get_drvdata (dev); \
		char *endp; \
		unsigned long val = simple_strtoul (buf, &endp, 0); \
		ssize_t err; \
		if ((endp == buf) || \
			sdiaudio_validate_##var (iface, val)) { \
			return -EINVAL; \
		} \
		err = miface_store (iface, &iface->var, val); \
		if (err) { \
			return err; \
		} \
		return count; \
	}
SDIAUDIO_STORE(buffers)
SDIAUDIO_STORE(bufsize)
SDIAUDIO_STORE(channels)
SDIAUDIO_STORE(nonaudio)
SDIAUDIO_STORE(sample_rate)
SDIAUDIO_STORE(sample_size)

static DEVICE_ATTR(buffers,S_IRUGO|S_IWUSR,
	miface_show_buffers,sdiaudio_store_buffers);
static DEVICE_ATTR(bufsize,S_IRUGO|S_IWUSR,
	miface_show_bufsize,sdiaudio_store_bufsize);
static DEVICE_ATTR(channels,S_IRUGO|S_IWUSR,
	miface_show_channels,sdiaudio_store_channels);
static DEVICE_ATTR(non_audio,S_IRUGO|S_IWUSR,
	miface_show_nonaudio,sdiaudio_store_nonaudio);
static DEVICE_ATTR(sample_rate,S_IRUGO|S_IWUSR,
	miface_show_sample_rate,sdiaudio_store_sample_rate);
static DEVICE_ATTR(sample_size,S_IRUGO|S_IWUSR,
	miface_show_sample_size,sdiaudio_store_sample_size);

/**
 * sdiaudio_register_iface - register an interface
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
sdiaudio_register_iface (struct master_dev *card,
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
		iface->buffers = SDIAUDIO_TX_BUFFERS;
		iface->bufsize = SDIAUDIO_TX_BUFSIZE;
		iface->sample_rate = 48000;
		iface->nonaudio = SDIAUDIO_CTL_PCM_ALLCHANNEL;
	} else {
		iface->buffers = SDIAUDIO_RX_BUFFERS;
		iface->bufsize = SDIAUDIO_RX_BUFSIZE;
	}
	iface->granularity = granularity;
	iface->sample_size = SDIAUDIO_CTL_AUDSAMP_SZ_16;
	iface->channels = SDIAUDIO_CTL_AUDCH_EN_2;
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
	spin_lock (&sdiaudio_iface_lock);
	while (minor <= maxminor) {
		found = 0;
		list_for_each (p, &sdiaudio_iface_list) {
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
	spin_unlock (&sdiaudio_iface_lock);
	if (minor > maxminor) {
		err = -ENOSPC;
		goto NO_MINOR;
	}

	/* Add this interface to the list of all interfaces */
	spin_lock (&sdiaudio_iface_lock);
	list_add_tail (&iface->list_all, &sdiaudio_iface_list);
	spin_unlock (&sdiaudio_iface_lock);

	/* Add this interface to the list for this device */
	list_add_tail (&iface->list, &card->iface_list);

	/* Create the device */
	iface->dev = device_create (sdiaudio_class,
		card->dev,
		MKDEV(major, minor),
		iface,
		"sdiaudio%cx%i",
		type[0],
		minor & ((1 << (MASTER_MINORBITS - 1)) - 1));
	if (IS_ERR(iface->dev)) {
		printk (KERN_WARNING "%s: unable to register device\n",
			sdiaudio_driver_name);
		err = PTR_ERR(iface->dev);
		goto NO_DEV;
	}
	dev_set_drvdata (iface->dev, iface);

	/* Add device attributes */
	if ((err = device_create_file (iface->dev,
		&dev_attr_buffers)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create file 'buffers'\n",
			sdiaudio_driver_name);
	}
	if ((err = device_create_file (iface->dev,
		&dev_attr_bufsize)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create file 'bufsize'\n",
			sdiaudio_driver_name);
	}
	if ((err = device_create_file (iface->dev,
		&dev_attr_channels)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create file 'channels'\n",
			sdiaudio_driver_name);
	}
	if (iface->direction == MASTER_DIRECTION_TX) {
		if ((err = device_create_file (iface->dev,
			&dev_attr_non_audio)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'non_audio'\n",
				sdiaudio_driver_name);
		}
		if ((err = device_create_file (iface->dev,
			&dev_attr_sample_rate)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'sample_rate'\n",
				sdiaudio_driver_name);
		}
	}
	if ((err = device_create_file (iface->dev,
		&dev_attr_sample_size)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create file 'sample_size'\n",
			sdiaudio_driver_name);
	}

	/* Activate the cdev */
	if ((err = cdev_add (&iface->cdev, MKDEV(major,minor), 1)) < 0) {
		printk (KERN_WARNING "%s: unable to add character device\n",
			sdiaudio_driver_name);
		goto NO_CDEV;
	}

	printk (KERN_INFO "%s: registered %s %u\n",
		sdiaudio_driver_name, type, minor - minminor);
	return 0;

NO_CDEV:
	device_destroy (sdiaudio_class, MKDEV(major,minor));
NO_DEV:
	list_del (&iface->list);
	spin_lock (&sdiaudio_iface_lock);
	list_del (&iface->list_all);
	spin_unlock (&sdiaudio_iface_lock);
NO_MINOR:
	kfree (iface);
NO_IFACE:
	return err;
}

/**
 * sdiaudio_unregister_iface - remove an interface from the list
 * @iface: interface
 **/
void
sdiaudio_unregister_iface (struct master_iface *iface)
{
	cdev_del (&iface->cdev);
	device_destroy (sdiaudio_class, iface->cdev.dev);
	list_del (&iface->list);
	spin_lock (&sdiaudio_iface_lock);
	list_del (&iface->list_all);
	spin_unlock (&sdiaudio_iface_lock);
	kfree (iface);
	return;
}

/**
 * sdiaudio_init_module - initialize the module
 *
 * Register the module as a character PCI driver.
 * Returns a negative error code on failure and 0 on success.
 **/
static int __init
sdiaudio_init_module (void)
{
	int err;
	dev_t dev = MKDEV(major,0);

	spin_lock_init (&sdiaudio_iface_lock);

	/* Create a device class */
	sdiaudio_class = class_create (THIS_MODULE, sdiaudio_driver_name);
	if (IS_ERR(sdiaudio_class)) {
		printk (KERN_WARNING "%s: unable to register device class\n",
			sdiaudio_driver_name);
		err = PTR_ERR(sdiaudio_class);
		goto NO_CLASS;
	}

	/* Add class attributes */
	if ((err = class_create_file (sdiaudio_class, &class_attr_version)) < 0) {
		printk (KERN_WARNING "%s: unable to create file 'version' \n",
			sdiaudio_driver_name);
	}

	/* Reserve a range of device numbers */
	if (major) {
		err = register_chrdev_region (dev,
			count,
			sdiaudio_driver_name);
	} else {
		err = alloc_chrdev_region (&dev,
			0,
			count,
			sdiaudio_driver_name);
	}
	if (err < 0) {
		printk (KERN_WARNING
			"%s: unable to reserve device number range\n",
			sdiaudio_driver_name);
		goto NO_RANGE;
	}
	major = MAJOR(dev);

	return 0;

NO_RANGE:
	class_destroy (sdiaudio_class);
NO_CLASS:
	return err;
}

/**
 * sdiaudio_cleanup_module - cleanup the module
 *
 * Unregister the module as a character PCI driver.
 **/
static void __exit
sdiaudio_cleanup_module (void)
{
	unregister_chrdev_region (MKDEV(major,0), count);
	class_destroy (sdiaudio_class);
	return;
}

module_init (sdiaudio_init_module);
module_exit (sdiaudio_cleanup_module);

