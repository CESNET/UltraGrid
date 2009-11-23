/* dvbmaster.c
 *
 * Linux driver for Linear Systems Ltd. DVB Master ASI interface boards.
 *
 * Copyright (C) 2004-2008 Linear Systems Ltd.
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

#include <linux/kernel.h> /* KERN_INFO */
#include <linux/module.h> /* MODULE_LICENSE */

#include <linux/types.h> /* u32 */
#include <linux/pci.h> /* pci_enable_device () */
#include <linux/dma-mapping.h> /* DMA_32BIT_MASK */
#include <linux/init.h> /* module_init () */
#include <linux/list.h> /* list_for_each () */
#include <linux/errno.h> /* error codes */

#include "asicore.h"
#include "../include/master.h"
// Temporary fix for Linux kernel 2.6.21
#include "mdev.c"
#include "masterlsdma.c"
#include "masterplx.c"
#include "miface.h"
#include "dvbm.h"
#include "dvbm_fd.h"
#include "dvbm_fdu.h"
#include "dvbm_rx.h"
#include "dvbm_tx.h"
#include "dvbm_lpfd.h"
#include "dvbm_qlf.h"
#include "dvbm_qi.h"
#include "dvbm_qo.h"
#include "dvbm_qdual.h"
#include "dvbm_q3ioe.h"
#include "dvbm_q3inoe.h"

#ifndef list_for_each_safe
#define list_for_each_safe(pos, n, head) \
	for (pos = (head)->next, n = pos->next; pos != (head); \
		pos = n, n = pos->next)
#endif

/* Static function prototypes */
static int dvbm_pci_probe_valid (struct pci_dev *dev,
	int __devinit (*specific_pci_probe)(struct pci_dev *dev)) __devinit;
static int dvbm_pci_probe (struct pci_dev *dev,
	const struct pci_device_id *id) __devinit;
static void dvbm_pci_remove_valid (struct pci_dev *dev,
	void (*specific_pci_remove)(struct master_dev *card));
static int dvbm_init_module (void) __init;
static void dvbm_cleanup_module (void) __exit;

MODULE_AUTHOR("Linear Systems Ltd.");
MODULE_DESCRIPTION("DVB Master driver");
MODULE_LICENSE("GPL");

#ifdef MODULE_VERSION
MODULE_VERSION(MASTER_DRIVER_VERSION);
#endif

char dvbm_driver_name[] = "dvbm";

static struct pci_device_id dvbm_pci_id_table[] = {
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBFD)
	},
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
			DVBM_PCI_DEVICE_ID_LINSYS_DVBTX)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBTXU)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBRX)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBRXU)
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
			DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD_R)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD_RS)
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
			ATSCM_PCI_DEVICE_ID_LINSYS_2FD_RS)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBLPFD)
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
			DVBM_PCI_DEVICE_ID_LINSYS_DVBQIE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBQI)
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
			DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQOE)
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
			ATSCM_PCI_DEVICE_ID_LINSYS_2FDE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			ATSCM_PCI_DEVICE_ID_LINSYS_2FDE_R)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBLPFDE)
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
			DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQDUALE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBQ3IOE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBQ3INOE)
	},
	{0,
	}
};

static struct pci_driver dvbm_pci_driver = {
	.name = dvbm_driver_name,
	.id_table = dvbm_pci_id_table,
	.probe = dvbm_pci_probe,
	.remove = dvbm_pci_remove
};

MODULE_DEVICE_TABLE(pci,dvbm_pci_id_table);

LIST_HEAD(dvbm_card_list);

struct class dvbm_class = {
	.name = dvbm_driver_name,
	.release = mdev_class_device_release,
	.class_release = NULL
};

/**
 * dvbm_pci_probe_valid - generic DVB Master PCI insertion handler
 * @dev: PCI device
 * @specific_pci_probe: PCI insertion handler
 *
 * Handle the insertion of a DVB Master device.
 * Returns a negative error code on failure and 0 on success.
 **/
static int __devinit
dvbm_pci_probe_valid (struct pci_dev *dev,
	int __devinit (*specific_pci_probe)(struct pci_dev *dev))
{
	int err;

	if ((err = pci_request_regions (dev, dvbm_driver_name)) < 0) {
		return err;
	}
	return specific_pci_probe (dev);
}

/**
 * dvbm_pci_probe - PCI insertion handler
 * @dev: PCI device
 * @id: PCI ID
 *
 * Checks if a PCI device should be handled by this driver.
 * Returns a negative error code on failure and 0 on success.
 **/
static int __devinit
dvbm_pci_probe (struct pci_dev *dev,
	const struct pci_device_id *id)
{
	int err;

	/* Initialize the driver_data pointer so that dvbm_pci_remove()
	 * doesn't try to free it if an error occurs */
	pci_set_drvdata (dev, NULL);

	/* Wake a sleeping device */
	if ((err = pci_enable_device (dev)) < 0) {
		printk (KERN_WARNING "%s: unable to enable device\n",
			dvbm_driver_name);
		return err;
	}

	/* Set PCI DMA addressing limitations */
	if ((err = pci_set_dma_mask (dev, DMA_32BIT_MASK)) < 0) {
		printk (KERN_WARNING "%s: unable to set PCI DMA mask\n",
			dvbm_driver_name);
		return err;
	}

	/* Enable bus mastering */
	pci_set_master (dev);

	/* Validate the device ID before doing more invasive initialization */
	switch (id->device) {
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFD:
		return dvbm_pci_probe_valid (dev, dvbm_fd_pci_probe);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDU:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDU_R:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDB:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDB_R:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD_R:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD_RS:
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FD:
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FD_R:
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FD_RS:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE_R:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE_RS:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDE_R:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDEB:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDEB_R:
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FDE:
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FDE_R:
		return dvbm_pci_probe_valid (dev, dvbm_fdu_pci_probe);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBTX:
		return dvbm_pci_probe_valid (dev, dvbm_tx_pci_probe);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBRX:
		return dvbm_pci_probe_valid (dev, dvbm_rx_pci_probe);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBTXU:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBTXE:
		return dvbm_pci_probe_valid (dev, dvbm_txu_pci_probe);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBRXU:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBRXE:
		return dvbm_pci_probe_valid (dev, dvbm_rxu_pci_probe);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPFD:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPFDE:
		return dvbm_pci_probe_valid (dev, dvbm_lpfd_pci_probe);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQLF:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQIE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQLF4:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF_MINIBNC:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF_HEADER:
		return dvbm_pci_probe_valid (dev, dvbm_qlf_pci_probe);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQI:
		return dvbm_pci_probe_valid (dev, dvbm_qi_pci_probe);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQO:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQOE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQOE:
		return dvbm_pci_probe_valid (dev, dvbm_qo_pci_probe);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQDUAL:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQDUALE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQDUALE:
		return dvbm_pci_probe_valid (dev, dvbm_qdual_pci_probe);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQ3IOE:
		return dvbm_pci_probe_valid (dev, dvbm_q3io_pci_probe);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQ3INOE:
		return dvbm_pci_probe_valid (dev, dvbm_q3ino_pci_probe);
	default:
		break;
	}
	return -ENODEV;
}

/**
 * dvbm_pci_remove_valid - generic DVB Master PCI removal handler
 * @dev: PCI device
 * @specific_pci_remove: PCI removal handler
 **/
static void
dvbm_pci_remove_valid (struct pci_dev *dev,
	void (*specific_pci_remove)(struct master_dev *card))
{
	struct master_dev *card = pci_get_drvdata (dev);

	if (card) {
		struct list_head *p, *n;
		struct master_iface *iface;

		list_for_each_safe (p, n, &card->iface_list) {
			iface = list_entry (p,
				struct master_iface, list);
			asi_unregister_iface (iface);
		}
		if (specific_pci_remove) {
			specific_pci_remove (card);
		}
		iounmap (card->bridge_addr);
		list_for_each (p, &dvbm_card_list) {
			if (p == &card->list) {
				mdev_unregister (card);
				break;
			}
		}
		pci_set_drvdata (dev, NULL);
	}
	pci_release_regions (dev);
	return;
}

/**
 * dvbm_pci_remove - PCI removal handler
 * @dev: PCI device
 **/
void
dvbm_pci_remove (struct pci_dev *dev)
{
	/* Validate the device ID before shutting things down */
	switch (dev->device) {
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFD:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBTX:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBRX:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBTXU:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBRXU:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBTXE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBRXE:
		dvbm_pci_remove_valid (dev, NULL);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDU:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDU_R:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDB:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDB_R:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD_R:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD_RS:
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FD:
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FD_R:
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FD_RS:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE_R:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE_RS:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDE:
 	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDE_R:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDEB:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFDEB_R:
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FDE:
	case ATSCM_PCI_DEVICE_ID_LINSYS_2FDE_R:
		dvbm_pci_remove_valid (dev, dvbm_fdu_pci_remove);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPFD:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPFDE:
		dvbm_pci_remove_valid (dev, dvbm_lpfd_pci_remove);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQLF:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQIE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQLF4:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF_MINIBNC:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF_HEADER:
		dvbm_pci_remove_valid (dev, dvbm_qlf_pci_remove);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQI:
		dvbm_pci_remove_valid (dev, dvbm_qi_pci_remove);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQO:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQOE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQOE:
		dvbm_pci_remove_valid (dev, dvbm_qo_pci_remove);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQDUAL:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQDUALE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQDUALE:
		dvbm_pci_remove_valid (dev, dvbm_qdual_pci_remove);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQ3IOE:
		dvbm_pci_remove_valid (dev, dvbm_q3io_pci_remove);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQ3INOE:
		dvbm_pci_remove_valid (dev, dvbm_q3ino_pci_remove);
		break;
	default:
		break;
	}
	pci_disable_device (dev);
	return;
}

/**
 * dvbm_init_module - register the module as a PCI driver
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int __init
dvbm_init_module (void)
{
	printk (KERN_INFO "%s: Linear Systems Ltd. "
		"DVB Master driver from master-%s (%s)\n",
		dvbm_driver_name, MASTER_DRIVER_VERSION, MASTER_DRIVER_DATE);

	return mdev_init_module (&dvbm_pci_driver,
		&dvbm_class,
		dvbm_driver_name);
}

/**
 * dvbm_cleanup_module - unregister the module as a PCI driver
 **/
static void __exit
dvbm_cleanup_module (void)
{
	mdev_cleanup_module (&dvbm_pci_driver, &dvbm_class);
	return;
}

module_init (dvbm_init_module);
module_exit (dvbm_cleanup_module);

