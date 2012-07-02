/* hdsdimaster.c
 *
 * Linux driver for Linear Systems Ltd. VidPort SMPTE 292M and SMPTE 259M-C interface boards.
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
#include <linux/kernel.h> /* KERN_INFO */
#include <linux/module.h> /* MODULE_LICENSE */

#include <linux/types.h> /* u32 */
#include <linux/pci.h> /* pci_enable_device () */
#include <linux/dma-mapping.h> /* DMA_BIT_MASK */
#include <linux/init.h> /* module_init () */
#include <linux/list.h> /* list_for_each () */
#include <linux/errno.h> /* error codes */

#include "sdivideocore.h"
#include "sdiaudiocore.h"
#include "../include/master.h"
#include "mdev.h"
#include "hdsdim.h"
#include "hdsdim_qie.h"
#include "hdsdim_txe.h"
#include "hdsdim_rxe.h"

#ifndef DEFINE_PCI_DEVICE_TABLE
#define DEFINE_PCI_DEVICE_TABLE(_table) \
	const struct pci_device_id _table[]
#endif

#ifndef DMA_BIT_MASK
#define DMA_BIT_MASK(n)	(((n) == 64) ? ~0ULL : ((1ULL<<(n))-1))
#endif

/* Static function prototypes */
static int hdsdim_pci_probe (struct pci_dev *pdev,
	const struct pci_device_id *id) __devinit;
static void hdsdim_pci_remove (struct pci_dev *pdev) __devexit;
static int hdsdim_init_module (void) __init;
static void hdsdim_cleanup_module (void) __exit;

MODULE_AUTHOR("Linear Systems Ltd.");
MODULE_DESCRIPTION("VidPort driver");
MODULE_LICENSE("GPL");
MODULE_VERSION(MASTER_DRIVER_VERSION);

char hdsdim_driver_name[] = "hdsdim";

static DEFINE_PCI_DEVICE_TABLE(hdsdim_pci_id_table) = {
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			HDSDIM_PCI_DEVICE_ID_LINSYS_HDSDIQIE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			HDSDIM_PCI_DEVICE_ID_LINSYS_HDSDIRXE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			HDSDIM_PCI_DEVICE_ID_LINSYS_HDSDITXE)
	},
	{0,
	}
};

static struct pci_driver hdsdim_pci_driver = {
	.name = hdsdim_driver_name,
	.id_table = hdsdim_pci_id_table,
	.probe = hdsdim_pci_probe,
	.remove = hdsdim_pci_remove
};

MODULE_DEVICE_TABLE(pci,hdsdim_pci_id_table);

static LIST_HEAD(hdsdim_card_list);

static struct class *hdsdim_class;

/**
 * hdsdim_pci_probe_generic - generic PCI insertion handler
 * @pdev: PCI device
 *
 * Perform generic PCI device initialization.
 * Returns a negative error code on failure and 0 on success.
 **/
int __devinit
hdsdim_pci_probe_generic (struct pci_dev *pdev)
{
	int err;

	/* Wake a sleeping device.
	 * This is done before pci_request_regions ()
	 * as described in Documentation/PCI/pci.txt. */
	if ((err = pci_enable_device (pdev)) < 0) {
		printk (KERN_WARNING "%s: unable to enable device\n",
			hdsdim_driver_name);
		return err;
	}

	/* Enable bus mastering */
	pci_set_master (pdev);

	/* Request I/O resources */
	if ((err = pci_request_regions (pdev, hdsdim_driver_name)) < 0) {
		printk (KERN_WARNING "%s: unable to get I/O resources\n",
			hdsdim_driver_name);
		pci_disable_device (pdev);
		return err;
	}

	/* Set PCI DMA addressing limitations */
	if (!(err = pci_set_dma_mask (pdev, DMA_BIT_MASK(64)))) {
		pci_set_consistent_dma_mask (pdev, DMA_BIT_MASK(64));
	} else if (!(err = pci_set_dma_mask (pdev, DMA_BIT_MASK(32)))) {
		pci_set_consistent_dma_mask (pdev, DMA_BIT_MASK(32));
	} else {
		printk (KERN_WARNING "%s: unable to set PCI DMA mask\n",
			hdsdim_driver_name);
		pci_disable_device (pdev);
		pci_release_regions (pdev);
		return err;
	}

	return 0;
}

/**
 * hdsdim_pci_probe - PCI insertion handler
 * @pdev: PCI device
 * @id: PCI ID
 *
 * Call the appropriate PCI insertion handler.
 * Returns a negative error code on failure and 0 on success.
 **/
static int __devinit
hdsdim_pci_probe (struct pci_dev *pdev,
	const struct pci_device_id *id)
{
	switch (id->device) {
	case HDSDIM_PCI_DEVICE_ID_LINSYS_HDSDIQIE:
		return hdsdim_qie_pci_probe (pdev);
	case HDSDIM_PCI_DEVICE_ID_LINSYS_HDSDITXE:
		return hdsdim_txe_pci_probe (pdev);
	case HDSDIM_PCI_DEVICE_ID_LINSYS_HDSDIRXE:
		return hdsdim_rxe_pci_probe (pdev);
	default:
		break;
	}
	return -ENODEV;
}

/**
 * hdsdim_pci_remove_generic - generic PCI removal handler
 * @pdev: PCI device
 *
 * Perform generic PCI device shutdown.
 * This function may be called during PCI probe error handling,
 * so don't mark it as __devexit.
 **/
void
hdsdim_pci_remove_generic (struct pci_dev *pdev)
{
	pci_disable_device (pdev);
	pci_release_regions (pdev);
	return;
}

/**
 * hdsdim_pci_remove - PCI removal handler
 * @pdev: PCI device
 *
 * Call the appropriate PCI removal handler.
 **/
static void __devexit
hdsdim_pci_remove (struct pci_dev *pdev)
{
	switch (pdev->device) {
	case HDSDIM_PCI_DEVICE_ID_LINSYS_HDSDIQIE:
		hdsdim_qie_pci_remove (pdev);
		break;
	case HDSDIM_PCI_DEVICE_ID_LINSYS_HDSDITXE:
		hdsdim_txe_pci_remove (pdev);
		break;
	case HDSDIM_PCI_DEVICE_ID_LINSYS_HDSDIRXE:
		hdsdim_rxe_pci_remove (pdev);
		break;
	default:
		break;
	}
	return;
}

/**
 * hdsdim_register - register a VidPort device
 * @card: pointer to the board info structure
 **/
int
hdsdim_register (struct master_dev *card)
{
	return mdev_register (card,
		&hdsdim_card_list,
		hdsdim_driver_name,
		hdsdim_class);
}

/**
 * hdsdim_unregister_all - unregister a VidPort device and all interfaces
 * @card: pointer to the board info structure
 **/
void
hdsdim_unregister_all (struct master_dev *card)
{
	struct master_iface *iface;
	struct list_head *p;

	while (!list_empty (&card->iface_list)) {
		/* Unregister the video interface if one was registered */
		iface = list_entry (card->iface_list.next,
			struct master_iface, list);
		sdivideo_unregister_iface (iface);

		/* Unregister the audio interface if one was registered */
		if (!list_empty (&card->iface_list)) {
			iface = list_entry (card->iface_list.next,
				struct master_iface, list);
			sdiaudio_unregister_iface (iface);
		}
	}

	/* Unregister the device if it was registered */
	list_for_each (p, &hdsdim_card_list) {
		if (p == &card->list) {
			mdev_unregister (card, hdsdim_class);
			break;
		}
	}
	return;
}

/**
 * hdsdim_init_module - register the module as a Master and PCI driver
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int __init
hdsdim_init_module (void)
{
	int err;

	printk (KERN_INFO "%s: Linear Systems Ltd. "
		"VidPort driver from master-%s (%s)\n",
		hdsdim_driver_name, MASTER_DRIVER_VERSION, MASTER_DRIVER_DATE);

	/* Create a device class */
	hdsdim_class = mdev_init (hdsdim_driver_name);
	if (IS_ERR(hdsdim_class)) {
		err = PTR_ERR(hdsdim_class);
		goto NO_CLASS;
	}

	/* Register with the PCI subsystem */
	if ((err = pci_register_driver (&hdsdim_pci_driver)) < 0) {
		printk (KERN_WARNING
			"%s: unable to register with PCI subsystem\n",
			hdsdim_driver_name);
		goto NO_PCI;
	}

	return 0;

NO_PCI:
	mdev_cleanup (hdsdim_class);
NO_CLASS:
	return err;
}

/**
 * hdsdim_cleanup_module - unregister the module as a Master and PCI driver
 **/
static void __exit
hdsdim_cleanup_module (void)
{
	pci_unregister_driver (&hdsdim_pci_driver);
	mdev_cleanup (hdsdim_class);
	return;
}

module_init (hdsdim_init_module);
module_exit (hdsdim_cleanup_module);

