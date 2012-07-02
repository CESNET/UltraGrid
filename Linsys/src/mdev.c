/* mdev.c
 *
 * Support functions for Linear Systems Ltd. Master devices.
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

#include <linux/slab.h> /* kfree () */
#include <linux/list.h> /* list_del () */
#include <linux/device.h> /* device_create () */
#include <linux/fs.h> /* MKDEV () */

#include "../include/master.h"
#include "mdev.h"

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,26)
static inline const char *
dev_name(struct device *dev)
{
	return dev->bus_id;
}
#endif

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,27))
#define device_create(cls,parent,devt,drvdata,fmt,...) \
	device_create(cls,parent,devt,fmt,##__VA_ARGS__)
#endif

/* Static function prototypes */
static ssize_t mdev_show_fw_version (struct device *dev,
	struct device_attribute *attr,
	char *buf);
#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,34))
#define mdev_show_version(cls,attr,buf) mdev_show_version(cls,buf)
#endif
static ssize_t mdev_show_version (struct class *cls,
	struct class_attribute *attr,
	char *buf);

/**
 * mdev_index - return the index of an interface
 * @card: Master device
 * @list: pointer to an interface linked list structure
 **/
unsigned int
mdev_index (struct master_dev *card, struct list_head *list)
{
	struct list_head *p;
	unsigned int i = 0;

	list_for_each (p, &card->iface_list) {
		if (p == list) {
			break;
		}
		i++;
	}
	return i;
}

/**
 * mdev_show_fw_version - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
static ssize_t
mdev_show_fw_version (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	return snprintf (buf, PAGE_SIZE, "0x%04X\n", card->version);
}

static DEVICE_ATTR(fw_version,S_IRUGO,
	mdev_show_fw_version,NULL);

/**
 * mdev_register - add this device to the list for this driver
 * @card: Master device
 * @devlist: pointer to linked list of devices
 * @driver_name: driver name
 * @cls: device class
 *
 * Returns a negative error code on failure and 0 on success.
 **/
int
mdev_register (struct master_dev *card,
	struct list_head *devlist,
	char *driver_name,
	struct class *cls)
{
	const unsigned int maxnum = 99;
	unsigned int num = 0, found;
	struct list_head *p;
	struct master_dev *entry;
	char name[3];
	int err;

	/* Find an unused name for this device */
	while (num <= maxnum) {
		found = 0;
		snprintf (name, 3, "%u", num);
		list_for_each (p, devlist) {
			entry = list_entry (p, struct master_dev, list);
			if (!strcmp (dev_name(entry->dev),
				name)) {
				found = 1;
				break;
			}
		}
		if (!found) {
			break;
		}
		num++;
	}

	/* Add this device to the list for this driver */
	list_add_tail (&card->list, devlist);

	/* Create the device */
	card->dev = device_create (cls,
		card->parent,
		MKDEV(0,0),
		card,
		name);
	if (IS_ERR(card->dev)) {
		printk (KERN_WARNING "%s: unable to create device\n",
			driver_name);
		err = PTR_ERR(card->dev);
		goto NO_DEV;
	}
	dev_set_drvdata (card->dev, card);

	/* Add device attributes */
	if ((err = device_create_file (card->dev,
		&dev_attr_fw_version)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create file 'fw_version'\n",
			driver_name);
	}
	printk (KERN_INFO "%s: registered board %s\n",
		driver_name, dev_name(card->dev));
	return 0;

NO_DEV:
	list_del (&card->list);
	return err;
}

/**
 * mdev_unregister - remove this device from the list for this driver
 * @card: Master device
 * @cls: device class
 **/
void
mdev_unregister (struct master_dev *card, struct class *cls)
{
	list_del (&card->list);
	device_destroy (cls, MKDEV(0,0));
	return;
}

/**
 * mdev_show_version - class attribute read handler
 * @cls: class being read
 * @attr: class attribute
 * @buf: output buffer
 **/
static ssize_t
mdev_show_version (struct class *cls,
	struct class_attribute *attr,
	char *buf)
{
	return snprintf (buf, PAGE_SIZE, "%s\n", MASTER_DRIVER_VERSION);
}

static CLASS_ATTR(version,S_IRUGO,
	mdev_show_version,NULL);

/**
 * mdev_init - create the device class
 * @name: class name
 *
 * Returns a negative error code on failure and a pointer to the device class on success.
 **/
struct class *
mdev_init (char *name)
{
	struct class *cls;
	int err;

	/* Create a device class */
	cls = class_create (THIS_MODULE, name);
	if (IS_ERR(cls)) {
		printk (KERN_WARNING "%s: unable to create device class\n",
			name);
		err = PTR_ERR(cls);
		goto NO_CLASS;
	}

	/* Add class attributes */
	if ((err = class_create_file (cls, &class_attr_version)) < 0) {
		printk (KERN_WARNING "%s: unable to create file 'version'\n",
			name);
		goto NO_ATTR;
	}

	return cls;

NO_ATTR:
	class_destroy (cls);
NO_CLASS:
	return ERR_PTR(err);
}

/**
 * mdev_cleanup - destroy the device class
 * @cls: pointer to the device class
 **/
void
mdev_cleanup (struct class *cls)
{
	class_destroy (cls);
	return;
}

