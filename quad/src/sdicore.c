/* sdicore.c
 *
 * Linear Systems Ltd. SMPTE 259M-C API.
 *
 * Copyright (C) 2004-2007 Linear Systems Ltd.
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
#include <linux/slab.h> /* kmalloc () */
#include <linux/errno.h> /* error codes */
#include <linux/list.h> /* list_add_tail () */
#include <linux/spinlock.h> /* spin_lock_init () */
#include <linux/wait.h> /* init_waitqueue_head () */
#include <linux/device.h> /* class_create () */
#include <linux/cdev.h> /* cdev_init () */

#if defined(__x86_64__) && !defined(HAVE_COMPAT_IOCTL)
#include <linux/ioctl32.h> /* register_ioctl32_conversion () */
#endif

#include <asm/uaccess.h> /* put_user () */
#include <asm/semaphore.h> /* sema_init () */
#include <asm/bitops.h> /* test_and_clear_bit () */

#include "sdicore.h"
#include "../include/master.h"
// Temporary fix for Linux kernel 2.6.21
#include "miface.c"

/* Static function prototypes */
static int sdi_ioctl (struct master_iface *iface,
	unsigned int cmd,
	unsigned long arg);
static void sdi_set_buffers_minmaxmult (struct master_iface *iface,
	unsigned long *min, unsigned long *max, unsigned long *mult);
static void sdi_set_bufsize_minmaxmult (struct master_iface *iface,
	unsigned long *min, unsigned long *max, unsigned long *mult);
static void sdi_set_clksrc_minmaxmult (struct master_iface *iface,
	unsigned long *min, unsigned long *max, unsigned long *mult);
static void sdi_set_mode_minmaxmult (struct master_iface *iface,
	unsigned long *min, unsigned long *max, unsigned long *mult);
static ssize_t sdi_store_buffers (struct class_device *cd,
	const char *buf, size_t count);
static ssize_t sdi_store_bufsize (struct class_device *cd,
	const char *buf, size_t count);
static ssize_t sdi_store_clksrc (struct class_device *cd,
	const char *buf, size_t count);
static ssize_t sdi_store_mode (struct class_device *cd,
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

#ifdef MODULE_VERSION
MODULE_VERSION(MASTER_DRIVER_VERSION);
#endif

EXPORT_SYMBOL(sdi_txioctl);
EXPORT_SYMBOL(sdi_rxioctl);
EXPORT_SYMBOL(sdi_compat_ioctl);
EXPORT_SYMBOL(sdi_register_iface);
EXPORT_SYMBOL(sdi_unregister_iface);

static char sdi_driver_name[] = SDI_DRIVER_NAME;

static LIST_HEAD(sdi_iface_list);

static spinlock_t sdi_iface_lock;

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,13))
#define class_create_file(cls,attr) 0
#define class_create class_simple_create
#define class_destroy class_simple_destroy
#define class_device_destroy(cls,devt) class_simple_device_remove(devt)
static struct class_simple *sdi_class;
#else
static struct class *sdi_class;
static CLASS_ATTR(version,S_IRUGO,
	miface_show_version,NULL);
#endif

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,13))
#define class_device_create(cls,parent,devt,fmt,...) \
	class_simple_device_add(cls,devt,fmt,##__VA_ARGS__)
#elif (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,15))
#define class_device_create(cls,parent,devt,fmt,...) \
	class_device_create(cls,devt,fmt,##__VA_ARGS__)
#endif

static char sdi_parent_link[] = "parent";

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
		if (put_user (card->pdev->device,
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_GETVERSION:
		if (put_user (card->version, (unsigned int *)arg)) {
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
 * @iface: interface
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
int
sdi_txioctl (struct master_iface *iface,
	unsigned int cmd,
	unsigned long arg)
{
	unsigned int reg = 0, i;

	switch (cmd) {
	case SDI_IOC_TXGETCAP:
		if (put_user (iface->capabilities, (unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_TXGETEVENTS:
		for (i = 0; i < 8 * sizeof (reg); i++) {
			if (test_and_clear_bit (i, &iface->events)) {
				reg |= (1 << i);
			}
		}
		if (put_user (reg, (unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return sdi_ioctl (iface, cmd, arg);
	}
	return 0;
}

/**
 * sdi_rxioctl - SMPTE 259M-C receiver interface ioctl() method
 * @iface: interface
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
int
sdi_rxioctl (struct master_iface *iface,
	unsigned int cmd,
	unsigned long arg)
{
	unsigned int reg = 0, i;

	switch (cmd) {
	case SDI_IOC_RXGETCAP:
		if (put_user (iface->capabilities, (unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_RXGETEVENTS:
		for (i = 0; i < 8 * sizeof (reg); i++) {
			if (test_and_clear_bit (i, &iface->events)) {
				reg |= (1 << i);
			}
		}
		if (put_user (reg, (unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return sdi_ioctl (iface, cmd, arg);
	}
	return 0;
}

/**
 * sdi_compat_ioctl - 32-bit ioctl handler
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
long
sdi_compat_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct inode *inode = filp->f_dentry->d_inode;

	return filp->f_op->ioctl (inode, filp, cmd, arg);
}

/**
 * sdi_set_buffers_minmaxmult - return the desired buffers attribute properties
 * @iface: interface being written
 * @min: pointer to the minimum value
 * @max: pointer to the maximum value
 * @mult: pointer to the granularity
 **/
static void
sdi_set_buffers_minmaxmult (struct master_iface *iface,
	unsigned long *min, unsigned long *max, unsigned long *mult)
{
	*min = (iface->direction == MASTER_DIRECTION_TX) ?
		SDI_TX_BUFFERS_MIN : SDI_RX_BUFFERS_MIN;
	*max = SDI_BUFFERS_MAX;
	*mult = 1;
	return;
}

/**
 * sdi_set_bufsize_minmaxmult - return the desired bufsize attribute properties
 * @iface: interface being written
 * @min: pointer to the minimum value
 * @max: pointer to the maximum value
 * @mult: pointer to the granularity
 **/
static void
sdi_set_bufsize_minmaxmult (struct master_iface *iface,
	unsigned long *min, unsigned long *max, unsigned long *mult)
{
	*min = (iface->direction == MASTER_DIRECTION_TX) ?
		SDI_TX_BUFSIZE_MIN : SDI_RX_BUFSIZE_MIN;
	*max = SDI_BUFSIZE_MAX;
	*mult = 4;
	return;
}

/**
 * sdi_set_clksrc_minmaxmult - return the desired clksrc attribute properties
 * @iface: interface being written
 * @min: pointer to the minimum value
 * @max: pointer to the maximum value
 * @mult: pointer to the granularity
 **/
static void
sdi_set_clksrc_minmaxmult (struct master_iface *iface,
	unsigned long *min, unsigned long *max, unsigned long *mult)
{
	*min = SDI_CTL_TX_CLKSRC_ONBOARD;
	*max = SDI_CTL_TX_CLKSRC_RX;
	*mult = 1;
	return;
}

/**
 * sdi_set_mode_minmaxmult - return the desired mode attribute properties
 * @iface: interface being written
 * @min: pointer to the minimum value
 * @max: pointer to the maximum value
 * @mult: pointer to the granularity
 **/
static void
sdi_set_mode_minmaxmult (struct master_iface *iface,
	unsigned long *min, unsigned long *max, unsigned long *mult)
{
	/* For regular SDI */
	/*
	*min = SDI_CTL_MODE_8BIT;
	*max = SDI_CTL_MODE_10BIT;
	*/
	/* Specific for HD-SDI QIe */
	/* debug: Temporary solution, fix it */
	*min = HDSDI_CTL_MODE_RAW;
	*max = HDSDI_CTL_MODE_DEINTERLACE;
	*mult = 1;
	return;
}

/**
 * sdi_store_* - SMPTE 259M-C interface attribute write handler
 * @cd: class_device being written
 * @buf: input buffer
 * @count:
 **/
#define SDI_STORE(var) \
	static ssize_t sdi_store_##var (struct class_device *cd, \
		const char *buf, \
		size_t count) \
	{ \
		struct master_iface *iface = class_get_devdata (cd); \
		unsigned long min, max, mult; \
		sdi_set_##var##_minmaxmult (iface, &min, &max, &mult); \
		return miface_store (iface, \
			&iface->var, \
			buf, \
			count, \
			min, \
			max, \
			mult); \
	}
SDI_STORE(buffers)
SDI_STORE(bufsize)
SDI_STORE(clksrc)
SDI_STORE(mode)

static CLASS_DEVICE_ATTR(buffers,S_IRUGO|S_IWUSR,
	miface_show_buffers,sdi_store_buffers);
static CLASS_DEVICE_ATTR(bufsize,S_IRUGO|S_IWUSR,
	miface_show_bufsize,sdi_store_bufsize);
static CLASS_DEVICE_ATTR(clock_source,S_IRUGO|S_IWUSR,
	miface_show_clksrc,sdi_store_clksrc);
static CLASS_DEVICE_ATTR(mode,S_IRUGO|S_IWUSR,
	miface_show_mode,sdi_store_mode);

/**
 * sdi_register_iface - register an interface
 * @card: pointer to the board info structure
 * @direction: direction of data flow
 * @fops: file operations structure
 * @cap: capabilities flags
 * @granularity: buffer size granularity in bytes
 *
 * Allocate and initialize an interface information structure.
 * Assign the lowest unused minor number to this interface
 * and add it to the list of interfaces for this device
 * and the list of all interfaces.
 * Also initialize the class_device parameters.
 * Returns a negative error code on failure and 0 on success.
 **/
int
sdi_register_iface (struct master_dev *card,
	unsigned int direction,
	struct file_operations *fops,
	unsigned int cap,
	unsigned int granularity)
{
	struct master_iface *iface, *entry;
	const char *type;
	struct list_head *p;
	unsigned int minminor, minor, maxminor, found, id;
	int err;
	char name[BUS_ID_SIZE];

	/* Allocate an interface info structure */
	iface = (struct master_iface *)kmalloc (sizeof (*iface), GFP_KERNEL);
	if (iface == NULL) {
		err = -ENOMEM;
		goto NO_IFACE;
	}

	/* Initialize an interface info structure */
	memset (iface, 0, sizeof (*iface));
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
	iface->mode = SDI_CTL_MODE_10BIT;
	init_waitqueue_head (&iface->queue);
	sema_init (&iface->buf_sem, 1);
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

	/* Create the class_device */
	iface->class_dev = class_device_create (sdi_class,
		NULL,
		MKDEV(major, minor),
		&card->pdev->dev,
		"sdi%cx%i",
		type[0],
		minor & ((1 << (MASTER_MINORBITS - 1)) - 1));
	if (IS_ERR(iface->class_dev)) {
		printk (KERN_WARNING "%s: unable to register class_device\n",
			sdi_driver_name);
		err = PTR_ERR(iface->class_dev);
		goto NO_CLASSDEV;
	}
	class_set_devdata (iface->class_dev, iface);

	/* Add class_device attributes */
	if ((err = class_device_create_file (iface->class_dev,
		&class_device_attr_buffers)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create file 'buffers'\n",
			sdi_driver_name);
	}
	if ((err = class_device_create_file (iface->class_dev,
		&class_device_attr_bufsize)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create file 'bufsize'\n",
			sdi_driver_name);
	}
	if (iface->direction == MASTER_DIRECTION_TX) {
		if (iface->capabilities & SDI_CAP_TX_RXCLKSRC) {
			if ((err = class_device_create_file (iface->class_dev,
				&class_device_attr_clock_source)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'clock_source'\n",
					sdi_driver_name);
			}
		}
	}
	if ((err = class_device_create_file (iface->class_dev,
		&class_device_attr_mode)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create file 'mode'\n",
			sdi_driver_name);
	}
	if ((err = sysfs_create_link (&iface->class_dev->kobj,
		&card->class_dev.kobj,
		sdi_parent_link)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create symbolic link\n",
			sdi_driver_name);
	}
	id = 0;
	list_for_each (p, &card->iface_list) {
		if (p == &iface->list) {
			break;
		}
		id++;
	}
	snprintf (name, sizeof (name), "%u", id);
	if ((err = sysfs_create_link (&card->class_dev.kobj,
		&iface->class_dev->kobj,
		name)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create symbolic link\n",
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
	sysfs_remove_link (&card->class_dev.kobj, name);
	sysfs_remove_link (&iface->class_dev->kobj, sdi_parent_link);
	class_device_destroy (sdi_class, iface->cdev.dev);
NO_CLASSDEV:
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
	unsigned int id;
	struct list_head *p;
	char name[BUS_ID_SIZE];

	cdev_del (&iface->cdev);
	sysfs_remove_link (&iface->card->class_dev.kobj, name);
	sysfs_remove_link (&iface->class_dev->kobj, sdi_parent_link);
	class_device_destroy (sdi_class, iface->cdev.dev);
	id = 0;
	list_for_each (p, &iface->card->iface_list) {
		if (p == &iface->list) {
			break;
		}
		id++;
	}
	snprintf (name, sizeof (name), "%u", id);
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

#if defined(__x86_64__) && !defined(HAVE_COMPAT_IOCTL)
	register_ioctl32_conversion (SDI_IOC_TXGETCAP, NULL);
	register_ioctl32_conversion (SDI_IOC_TXGETEVENTS, NULL);
	register_ioctl32_conversion (SDI_IOC_TXGETBUFLEVEL, NULL);
	register_ioctl32_conversion (SDI_IOC_TXGETTXD, NULL);
	register_ioctl32_conversion (SDI_IOC_RXGETCAP, NULL);
	register_ioctl32_conversion (SDI_IOC_RXGETEVENTS, NULL);
	register_ioctl32_conversion (SDI_IOC_RXGETBUFLEVEL, NULL);
	register_ioctl32_conversion (SDI_IOC_RXGETCARRIER, NULL);
	register_ioctl32_conversion (SDI_IOC_RXGETSTATUS, NULL);
	register_ioctl32_conversion (SDI_IOC_RXGET27COUNT, NULL);
	register_ioctl32_conversion (SDI_IOC_RXGETTIMESTAMP, NULL);
	register_ioctl32_conversion (SDI_IOC_GETID, NULL);
	register_ioctl32_conversion (SDI_IOC_GETVERSION, NULL);
	register_ioctl32_conversion (SDI_IOC_QBUF, NULL);
	register_ioctl32_conversion (SDI_IOC_DQBUF, NULL);
#endif

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
#if defined(__x86_64__) && !defined(HAVE_COMPAT_IOCTL)
	unregister_ioctl32_conversion (SDI_IOC_TXGETCAP);
	unregister_ioctl32_conversion (SDI_IOC_TXGETEVENTS);
	unregister_ioctl32_conversion (SDI_IOC_TXGETBUFLEVEL);
	unregister_ioctl32_conversion (SDI_IOC_RXGETCAP);
	unregister_ioctl32_conversion (SDI_IOC_RXGETEVENTS);
	unregister_ioctl32_conversion (SDI_IOC_RXGETBUFLEVEL);
	unregister_ioctl32_conversion (SDI_IOC_RXGETCARRIER);
	unregister_ioctl32_conversion (SDI_IOC_RXGETSTATUS);	
	unregister_ioctl32_conversion (SDI_IOC_RXGET27COUNT, NULL);
	unregister_ioctl32_conversion (SDI_IOC_RXGETTIMESTAMP, NULL);
	unregister_ioctl32_conversion (SDI_IOC_GETID);
	unregister_ioctl32_conversion (SDI_IOC_GETVERSION);
	unregister_ioctl32_conversion (SDI_IOC_QBUF);
	unregister_ioctl32_conversion (SDI_IOC_DQBUF);
#endif
	unregister_chrdev_region (MKDEV(major,0), count);
	class_destroy (sdi_class);
	return;
}

module_init (sdi_init_module);
module_exit (sdi_cleanup_module);

