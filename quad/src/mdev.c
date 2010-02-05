/* mdev.c
 *
 * Support functions for Linear Systems Ltd. Master devices.
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

#include <linux/kernel.h> /* snprintf () */

#include <linux/slab.h> /* kfree () */
#include <linux/list.h> /* list_del () */
#include <linux/pci.h> /* pci_register_driver () */
#include <linux/device.h> /* class_device_unregister () */

#include "../include/master.h"
#include "mdev.h"
#include "miface.h"

/* Static function prototypes */
static ssize_t mdev_show_fw_version (struct class_device *cd, char *buf);
static ssize_t mdev_show_version (struct class *cls, char *buf);

/**
 * mdev_users - return the total usage count for a device
 * @card: Master device
 *
 * Call this with card->users_sem held!
 **/
unsigned int
mdev_users (struct master_dev *card)
{
	struct list_head *p;
	struct master_iface *iface;
	unsigned int total_users = 0;

	list_for_each (p, &card->iface_list) {
		iface = list_entry (p, struct master_iface, list);
		total_users += iface->users;
	}
	return total_users;
}

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
 * mdev_class_device_release - release a Master class_device
 * @cd: class_device being released
 **/
void
mdev_class_device_release (struct class_device *cd)
{
	struct master_dev *card = to_master_dev(cd);

	kfree (card);
	return;
}

/**
 * mdev_show_fw_version - interface attribute read handler
 * @cd: class_device being read
 * @buf: output buffer
 **/
static ssize_t
mdev_show_fw_version (struct class_device *cd, char *buf)
{
	struct master_dev *card = to_master_dev(cd);

	return snprintf (buf, PAGE_SIZE, "0x%04X\n", card->version);
}

static CLASS_DEVICE_ATTR(fw_version,S_IRUGO,
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
	unsigned int num = 0, found;
	struct list_head *p;
	struct master_dev *entry;
	int err;

	/* Find an unused name for this device */
	while (1) {
		found = 0;
		snprintf (card->class_dev.class_id,
			sizeof (card->class_dev.class_id),
			"%u",
			num);
		list_for_each (p, devlist) {
			entry = list_entry (p, struct master_dev, list);
			if (!strcmp (entry->class_dev.class_id,
				card->class_dev.class_id)) {
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

	/* Initialize the class_device parameters */
	card->class_dev.class = cls;
	card->class_dev.dev = &card->pdev->dev;

	/* Register the class_device */
	if ((err = class_device_register (&card->class_dev)) < 0) {
		printk (KERN_WARNING "%s: unable to register class_device\n",
			driver_name);
		goto NO_CLASSDEV;
	}

	/* Add class_device attributes */
	if ((err = class_device_create_file (&card->class_dev,
		&class_device_attr_fw_version)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create file 'fw_version'\n",
			driver_name);
	}
	printk (KERN_INFO "%s: registered board %s\n",
		driver_name, card->class_dev.class_id);
	return 0;

NO_CLASSDEV:
	list_del (&card->list);
	return err;
}

/**
 * mdev_unregister - remove this device from the list for this driver
 * @card: Master device
 **/
void
mdev_unregister (struct master_dev *card)
{
	list_del (&card->list);
	class_device_unregister (&card->class_dev);
	return;
}

/**
 * mdev_show_version - class attribute read handler
 * @cls: class being read
 * @buf: output buffer
 **/
static ssize_t
mdev_show_version (struct class *cls, char *buf)
{
	return snprintf (buf, PAGE_SIZE, "%s\n", MASTER_DRIVER_VERSION);
}

static CLASS_ATTR(version,S_IRUGO,
	mdev_show_version,NULL);

/**
 * mdev_init_module - register the module as a PCI driver
 * @pci_drv:
 * @cls:
 * @name:
 *
 * Returns a negative error code on failure and 0 on success.
 **/
int __init
mdev_init_module (struct pci_driver *pci_drv,
	struct class *cls,
	char *name)
{
	int err;

	/* Create a device class */
	if ((err = class_register (cls)) < 0) {
		printk (KERN_WARNING "%s: unable to register device class\n",
			name);
		goto NO_CLASS;
	}

	/* Add class attributes */
	if ((err = class_create_file (cls, &class_attr_version)) < 0) {
		printk (KERN_WARNING "%s: unable to create file 'version'\n",
			name);
	}
	/* Register with the PCI subsystem */
	if ((err = pci_register_driver (pci_drv)) < 0) {
		printk (KERN_WARNING
			"%s: unable to register with PCI subsystem\n",
			name);
		goto NO_PCI;
	}

	return 0;

NO_PCI:
	class_unregister (cls);
NO_CLASS:
	return err;
}

/**
 * mdev_cleanup_module - unregister the module as a PCI driver
 * @pci_drv:
 * @cls:
 **/
void __exit
mdev_cleanup_module (struct pci_driver *pci_drv,
	struct class *cls)
{
	pci_unregister_driver (pci_drv);
	class_unregister (cls);
	return;
}

