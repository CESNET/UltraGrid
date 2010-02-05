/* asicore.c
 *
 * Linear Systems Ltd. DVB ASI API.
 *
 * Copyright (C) 1999 Tony Bolger <d7v@indigo.ie>
 * Copyright (C) 2000-2007 Linear Systems Ltd.
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

#include "asicore.h"
#include "../include/master.h"
// Temporary fix for Linux kernel 2.6.21
#include "miface.c"

/* Static function prototypes */
static int asi_ioctl (struct master_iface *iface,
	unsigned int cmd,
	unsigned long arg);
static void asi_set_buffers_minmaxmult (struct master_iface *iface,
	unsigned long *min, unsigned long *max, unsigned long *mult);
static void asi_set_bufsize_minmaxmult (struct master_iface *iface,
	unsigned long *min, unsigned long *max, unsigned long *mult);
static void asi_set_clksrc_minmaxmult (struct master_iface *iface,
	unsigned long *min, unsigned long *max, unsigned long *mult);
static void asi_set_mode_minmaxmult (struct master_iface *iface,
	unsigned long *min, unsigned long *max, unsigned long *mult);
static void asi_set_timestamps_minmaxmult (struct master_iface *iface,
	unsigned long *min, unsigned long *max, unsigned long *mult);
static ssize_t asi_store_buffers (struct class_device *cd,
	const char *buf, size_t count);
static ssize_t asi_store_bufsize (struct class_device *cd,
	const char *buf, size_t count);
static ssize_t asi_store_clksrc (struct class_device *cd,
	const char *buf, size_t count);
static ssize_t asi_store_count27 (struct class_device *cd,
	const char *buf, size_t count);
static ssize_t asi_store_mode (struct class_device *cd,
	const char *buf, size_t count);
static ssize_t asi_store_null_packets (struct class_device *cd,
	const char *buf, size_t count);
static ssize_t asi_store_timestamps (struct class_device *cd,
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

#ifdef MODULE_VERSION
MODULE_VERSION(MASTER_DRIVER_VERSION);
#endif

EXPORT_SYMBOL(asi_txioctl);
EXPORT_SYMBOL(asi_rxioctl);
EXPORT_SYMBOL(asi_compat_ioctl);
EXPORT_SYMBOL(asi_register_iface);
EXPORT_SYMBOL(asi_unregister_iface);

static char asi_driver_name[] = ASI_DRIVER_NAME;

static LIST_HEAD(asi_iface_list);

static spinlock_t asi_iface_lock;

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,13))
#define class_create_file(cls,attr) 0
#define class_create class_simple_create
#define class_destroy class_simple_destroy
#define class_device_destroy(cls,devt) class_simple_device_remove(devt)
static struct class_simple *asi_class;
#else
static struct class *asi_class;
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

static char asi_parent_link[] = "parent";

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
		if (put_user (card->pdev->device,
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_GETVERSION:
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
 * asi_txioctl - ASI transmitter interface ioctl() method
 * @iface: ASI interface
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
int
asi_txioctl (struct master_iface *iface,
	unsigned int cmd,
	unsigned long arg)
{
	unsigned int reg = 0, i;

	switch (cmd) {
	case ASI_IOC_TXGETCAP:
		if (put_user (iface->capabilities, (unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_TXGETEVENTS:
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
		return asi_ioctl (iface, cmd, arg);
	}
	return 0;
}

/**
 * asi_rxioctl - ASI receiver interface ioctl() method
 * @iface: ASI interface
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
int
asi_rxioctl (struct master_iface *iface,
	unsigned int cmd,
	unsigned long arg)
{
	unsigned int reg = 0, i;

	switch (cmd) {
	case ASI_IOC_RXGETCAP:
		if (put_user (iface->capabilities, (unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXGETEVENTS:
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
		return asi_ioctl (iface, cmd, arg);
	}
	return 0;
}

/**
 * asi_compat_ioctl - 32-bit ioctl handler
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
long
asi_compat_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct inode *inode = filp->f_dentry->d_inode;

	return filp->f_op->ioctl (inode, filp, cmd, arg);
}

/**
 * asi_set_buffers_minmaxmult - return the desired buffers attribute properties
 * @iface: interface being written
 * @min: pointer to the minimum value
 * @max: pointer to the maximum value
 * @mult: pointer to the granularity
 **/
static void
asi_set_buffers_minmaxmult (struct master_iface *iface,
	unsigned long *min, unsigned long *max, unsigned long *mult)
{
	*min = (iface->direction == MASTER_DIRECTION_TX) ?
		ASI_TX_BUFFERS_MIN : ASI_RX_BUFFERS_MIN;
	*max = ASI_BUFFERS_MAX;
	*mult = 1;
	return;
}

/**
 * asi_set_bufsize_minmaxmult - return the desired bufsize attribute properties
 * @iface: interface being written
 * @min: pointer to the minimum value
 * @max: pointer to the maximum value
 * @mult: pointer to the granularity
 **/
static void
asi_set_bufsize_minmaxmult (struct master_iface *iface,
	unsigned long *min, unsigned long *max, unsigned long *mult)
{
	*min = (iface->direction == MASTER_DIRECTION_TX) ?
		ASI_TX_BUFSIZE_MIN : ASI_RX_BUFSIZE_MIN;
	*max = ASI_BUFSIZE_MAX;
	*mult = iface->granularity;
	return;
}

/**
 * asi_set_clksrc_minmaxmult - return the desired clksrc attribute properties
 * @iface: interface being written
 * @min: pointer to the minimum value
 * @max: pointer to the maximum value
 * @mult: pointer to the granularity
 **/
static void
asi_set_clksrc_minmaxmult (struct master_iface *iface,
	unsigned long *min, unsigned long *max, unsigned long *mult)
{
	*min = ASI_CTL_TX_CLKSRC_ONBOARD;
	*max = (iface->capabilities & ASI_CAP_TX_RXCLKSRC) ?
		ASI_CTL_TX_CLKSRC_RX : ASI_CTL_TX_CLKSRC_EXT;
	*mult = 1;
	return;
}

#define asi_set_count27_minmaxmult miface_set_boolean_minmaxmult
#define asi_set_null_packets_minmaxmult miface_set_boolean_minmaxmult

/**
 * asi_set_mode_minmaxmult - return the desired mode attribute properties
 * @iface: interface being written
 * @min: pointer to the minimum value
 * @max: pointer to the maximum value
 * @mult: pointer to the granularity
 **/
static void
asi_set_mode_minmaxmult (struct master_iface *iface,
	unsigned long *min, unsigned long *max, unsigned long *mult)
{
	if (iface->direction == MASTER_DIRECTION_TX) {
		*min = ASI_CTL_TX_MODE_188;
		*max = (iface->capabilities & ASI_CAP_TX_MAKE204) ?
			ASI_CTL_TX_MODE_MAKE204 : ASI_CTL_TX_MODE_204;
	} else {
		*min = ASI_CTL_RX_MODE_RAW;
		*max = (iface->capabilities & ASI_CAP_RX_MAKE188) ?
			ASI_CTL_RX_MODE_204MAKE188 : ASI_CTL_RX_MODE_AUTO;
	}
	*mult = 1;
	return;
}

/**
 * asi_set_timestamps_minmaxmult - return the desired timestamps attribute properties
 * @iface: interface being written
 * @min: pointer to the minimum value
 * @max: pointer to the maximum value
 * @mult: pointer to the granularity
 **/
static void
asi_set_timestamps_minmaxmult (struct master_iface *iface,
	unsigned long *min, unsigned long *max, unsigned long *mult)
{
	*min = 0;
	if (iface->direction == MASTER_DIRECTION_TX) {
		*max = (iface->capabilities & ASI_CAP_TX_PTIMESTAMPS) ?
			2 : 1;
	} else {
		*max = (iface->capabilities & ASI_CAP_RX_PTIMESTAMPS) ?
			2 : 1;
	}
	*mult = 1;
	return;
}

/**
 * asi_store_* - ASI interface attribute write handler
 * @cd: class_device being written
 * @buf: input buffer
 * @count: buffer size
 **/
#define ASI_STORE(var) \
	static ssize_t asi_store_##var (struct class_device *cd, \
		const char *buf, \
		size_t count) \
	{ \
		struct master_iface *iface = class_get_devdata (cd); \
		unsigned long min, max, mult; \
		asi_set_##var##_minmaxmult (iface, &min, &max, &mult); \
		return miface_store (iface, \
			&iface->var, \
			buf, \
			count, \
			min, \
			max, \
			mult); \
	}
ASI_STORE(buffers)
ASI_STORE(bufsize)
ASI_STORE(clksrc)
ASI_STORE(count27)
ASI_STORE(mode)
ASI_STORE(null_packets)
ASI_STORE(timestamps)

static CLASS_DEVICE_ATTR(buffers,S_IRUGO|S_IWUSR,
	miface_show_buffers,asi_store_buffers);
static CLASS_DEVICE_ATTR(bufsize,S_IRUGO|S_IWUSR,
	miface_show_bufsize,asi_store_bufsize);
static CLASS_DEVICE_ATTR(clock_source,S_IRUGO|S_IWUSR,
	miface_show_clksrc,asi_store_clksrc);
static CLASS_DEVICE_ATTR(count27,S_IRUGO|S_IWUSR,
	miface_show_count27,asi_store_count27);
static CLASS_DEVICE_ATTR(granularity,S_IRUGO,
	miface_show_granularity,NULL);
static CLASS_DEVICE_ATTR(mode,S_IRUGO|S_IWUSR,
	miface_show_mode,asi_store_mode);
static CLASS_DEVICE_ATTR(null_packets,S_IRUGO|S_IWUSR,
	miface_show_null_packets,asi_store_null_packets);
static CLASS_DEVICE_ATTR(timestamps,S_IRUGO|S_IWUSR,
	miface_show_timestamps,asi_store_timestamps);
static CLASS_DEVICE_ATTR(transport,S_IRUGO,
	miface_show_transport,NULL);

/**
 * asi_register_iface - register an interface
 * @card: pointer to the board info structure
 * @direction: direction of data flow
 * @fops: file operations structure
 * @cap: capabilities flags
 * @granularity: buffer size granularity in bytes
 * @transport: transport type
 *
 * Allocate and initialize an interface information structure.
 * Assign the lowest unused minor number to this interface
 * and add it to the list of interfaces for this device
 * and the list of all interfaces.
 * Also initialize the class_device parameters.
 * Returns a negative error code on failure and 0 on success.
 **/
int
asi_register_iface (struct master_dev *card,
	unsigned int direction,
	struct file_operations *fops,
	unsigned int cap,
	unsigned int granularity,
	unsigned int transport)
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

	/* Create the class_device */
	iface->class_dev = class_device_create (asi_class,
		NULL,
		MKDEV(major, minor),
		&card->pdev->dev,
		"asi%cx%u",
		type[0],
		minor & ((1 << (MASTER_MINORBITS - 1)) - 1));
	if (IS_ERR(iface->class_dev)) {
		printk (KERN_WARNING "%s: unable to create class_device\n",
			asi_driver_name);
		err = PTR_ERR(iface->class_dev);
		goto NO_CLASSDEV;
	}
	class_set_devdata (iface->class_dev, iface);

	/* Add class_device attributes */
	if ((err = class_device_create_file (iface->class_dev,
		&class_device_attr_buffers)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create file 'buffers'\n",
			asi_driver_name);
	}
	if ((err = class_device_create_file (iface->class_dev,
		&class_device_attr_bufsize)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create file 'bufsize'\n",
			asi_driver_name);
	}
	if ((err = class_device_create_file (iface->class_dev,
		&class_device_attr_granularity)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create file 'granularity'\n",
			asi_driver_name);
	}
	if ((err = class_device_create_file (iface->class_dev,
		&class_device_attr_transport)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create file 'transport'\n",
			asi_driver_name);
	}
	if (iface->direction == MASTER_DIRECTION_TX) {
		if (iface->capabilities & ASI_CAP_TX_SETCLKSRC) {
			if ((err = class_device_create_file (iface->class_dev,
				&class_device_attr_clock_source)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'clock_source'\n",
					asi_driver_name);
			}
		}
		if (iface->capabilities & ASI_CAP_TX_BYTESOR27) {
			if ((err = class_device_create_file (iface->class_dev,
				&class_device_attr_count27)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'count27'\n",
					asi_driver_name);
			}
		}
		if ((err = class_device_create_file (iface->class_dev,
			&class_device_attr_mode)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'mode'\n",
				asi_driver_name);
		}
		if (iface->capabilities & ASI_CAP_TX_NULLPACKETS) {
			if ((err = class_device_create_file (iface->class_dev,
				&class_device_attr_null_packets)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'null_packets'\n",
					asi_driver_name);
			}
		}
		if (iface->capabilities & ASI_CAP_TX_TIMESTAMPS) {
			if ((err = class_device_create_file (iface->class_dev,
				&class_device_attr_timestamps)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'timestamps'\n",
					asi_driver_name);
			}
		}
	} else {
		if (iface->capabilities & ASI_CAP_RX_BYTESOR27) {
			if ((err = class_device_create_file (iface->class_dev,
				&class_device_attr_count27)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'count27'\n",
					asi_driver_name);
			}
		}
		if (iface->capabilities & ASI_CAP_RX_SYNC) {
			if ((err = class_device_create_file (iface->class_dev,
				&class_device_attr_mode)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'mode'\n",
					asi_driver_name);
			}
		}
		if (iface->capabilities & ASI_CAP_RX_NULLPACKETS) {
			if ((err = class_device_create_file (iface->class_dev,
				&class_device_attr_null_packets)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'null_packets'\n",
					asi_driver_name);
			}
		}
		if (iface->capabilities & ASI_CAP_RX_TIMESTAMPS) {
			if ((err = class_device_create_file (iface->class_dev,
				&class_device_attr_timestamps)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'timestamps'\n",
					asi_driver_name);
			}
		}
	}
	if ((err = sysfs_create_link (&iface->class_dev->kobj,
		&card->class_dev.kobj,
		asi_parent_link)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create symbolic link\n",
			asi_driver_name);
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
			asi_driver_name);
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
	sysfs_remove_link (&card->class_dev.kobj, name);
	sysfs_remove_link (&iface->class_dev->kobj, asi_parent_link);
	class_device_destroy (asi_class, iface->cdev.dev);
NO_CLASSDEV:
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
	unsigned int id;
	struct list_head *p;
	char name[BUS_ID_SIZE];

	cdev_del (&iface->cdev);
	sysfs_remove_link (&iface->card->class_dev.kobj, name);
	sysfs_remove_link (&iface->class_dev->kobj, asi_parent_link);
	class_device_destroy (asi_class, iface->cdev.dev);
	id = 0;
	list_for_each (p, &iface->card->iface_list) {
		if (p == &iface->list) {
			break;
		}
		id++;
	}
	snprintf (name, sizeof (name), "%u", id);
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

#if defined(__x86_64__) && !defined(HAVE_COMPAT_IOCTL)
	register_ioctl32_conversion (ASI_IOC_TXGETCAP, NULL);
	register_ioctl32_conversion (ASI_IOC_TXGETEVENTS, NULL);
	register_ioctl32_conversion (ASI_IOC_TXGETBUFLEVEL, NULL);
	register_ioctl32_conversion (ASI_IOC_TXSETSTUFFING, NULL);
	register_ioctl32_conversion (ASI_IOC_TXGETBYTECOUNT, NULL);
	register_ioctl32_conversion (ASI_IOC_TXGETTXD, NULL);
	register_ioctl32_conversion (ASI_IOC_TXGET27COUNT, NULL);
	register_ioctl32_conversion (ASI_IOC_RXGETCAP, NULL);
	register_ioctl32_conversion (ASI_IOC_RXGETEVENTS, NULL);
	register_ioctl32_conversion (ASI_IOC_RXGETBUFLEVEL, NULL);
	register_ioctl32_conversion (ASI_IOC_RXGETSTATUS, NULL);
	register_ioctl32_conversion (ASI_IOC_RXGETBYTECOUNT, NULL);
	register_ioctl32_conversion (ASI_IOC_RXSETINVSYNC, NULL);
	register_ioctl32_conversion (ASI_IOC_RXGETCARRIER, NULL);
	register_ioctl32_conversion (ASI_IOC_RXSETDSYNC, NULL);
	register_ioctl32_conversion (ASI_IOC_RXGETRXD, NULL);
	register_ioctl32_conversion (ASI_IOC_RXSETPF, NULL);
	register_ioctl32_conversion (ASI_IOC_RXSETPID0, NULL);
	register_ioctl32_conversion (ASI_IOC_RXGETPID0COUNT, NULL);
	register_ioctl32_conversion (ASI_IOC_RXSETPID1, NULL);
	register_ioctl32_conversion (ASI_IOC_RXGETPID1COUNT, NULL);
	register_ioctl32_conversion (ASI_IOC_RXSETPID2, NULL);
	register_ioctl32_conversion (ASI_IOC_RXGETPID2COUNT, NULL);
	register_ioctl32_conversion (ASI_IOC_RXSETPID3, NULL);
	register_ioctl32_conversion (ASI_IOC_RXGETPID3COUNT, NULL);
	register_ioctl32_conversion (ASI_IOC_RXGET27COUNT, NULL);
	register_ioctl32_conversion (ASI_IOC_RXGETSTATUS2, NULL);
	register_ioctl32_conversion (ASI_IOC_RXSETINPUT, NULL);
	register_ioctl32_conversion (ASI_IOC_GETID, NULL);
	register_ioctl32_conversion (ASI_IOC_GETVERSION, NULL);
#endif

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
#if defined(__x86_64__) && !defined(HAVE_COMPAT_IOCTL)
	unregister_ioctl32_conversion (ASI_IOC_TXGETCAP);
	unregister_ioctl32_conversion (ASI_IOC_TXGETEVENTS);
	unregister_ioctl32_conversion (ASI_IOC_TXGETBUFLEVEL);
	unregister_ioctl32_conversion (ASI_IOC_TXSETSTUFFING);
	unregister_ioctl32_conversion (ASI_IOC_TXGETBYTECOUNT);
	unregister_ioctl32_conversion (ASI_IOC_TXGETTXD);
	unregister_ioctl32_conversion (ASI_IOC_TXGET27COUNT);
	unregister_ioctl32_conversion (ASI_IOC_RXGETCAP);
	unregister_ioctl32_conversion (ASI_IOC_RXGETEVENTS);
	unregister_ioctl32_conversion (ASI_IOC_RXGETBUFLEVEL);
	unregister_ioctl32_conversion (ASI_IOC_RXGETSTATUS);
	unregister_ioctl32_conversion (ASI_IOC_RXGETBYTECOUNT);
	unregister_ioctl32_conversion (ASI_IOC_RXSETINVSYNC);
	unregister_ioctl32_conversion (ASI_IOC_RXGETCARRIER);
	unregister_ioctl32_conversion (ASI_IOC_RXSETDSYNC);
	unregister_ioctl32_conversion (ASI_IOC_RXGETRXD);
	unregister_ioctl32_conversion (ASI_IOC_RXSETPF);
	unregister_ioctl32_conversion (ASI_IOC_RXSETPID0);
	unregister_ioctl32_conversion (ASI_IOC_RXGETPID0COUNT);
	unregister_ioctl32_conversion (ASI_IOC_RXSETPID1);
	unregister_ioctl32_conversion (ASI_IOC_RXGETPID1COUNT);
	unregister_ioctl32_conversion (ASI_IOC_RXSETPID2);
	unregister_ioctl32_conversion (ASI_IOC_RXGETPID2COUNT);
	unregister_ioctl32_conversion (ASI_IOC_RXSETPID3);
	unregister_ioctl32_conversion (ASI_IOC_RXGETPID3COUNT);
	unregister_ioctl32_conversion (ASI_IOC_RXGET27COUNT);
	unregister_ioctl32_conversion (ASI_IOC_RXGETSTATUS2);
	unregister_ioctl32_conversion (ASI_IOC_RXSETINPUT);
	unregister_ioctl32_conversion (ASI_IOC_GETID);
	unregister_ioctl32_conversion (ASI_IOC_GETVERSION);
#endif
	unregister_chrdev_region (MKDEV(major,0), count);
	class_destroy (asi_class);
	return;
}

module_init (asi_init_module);
module_exit (asi_cleanup_module);

