/* dvbmaster.c
 *
 * Linux driver for Linear Systems Ltd. DVB Master ASI interface boards.
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
#include <linux/kernel.h> /* KERN_INFO */
#include <linux/module.h> /* MODULE_LICENSE */

#include <linux/types.h> /* u32 */
#include <linux/pci.h> /* pci_enable_device () */
#include <linux/dma-mapping.h> /* DMA_BIT_MASK */
#include <linux/init.h> /* module_init () */
#include <linux/list.h> /* list_for_each () */
#include <linux/errno.h> /* error codes */

#include "asicore.h"
#include "../include/master.h"
#include "mdev.h"
#include "dvbm.h"
#include "dvbm_fd.h"
#include "dvbm_fdu.h"
#include "dvbm_rx.h"
#include "dvbm_tx.h"
#include "dvbm_lpfd.h"
#include "dvbm_qlf.h"
#include "dvbm_qi.h"
#include "dvbm_qio.h"
#include "dvbm_lpqo.h"
#include "dvbm_qdual.h"

#ifndef DEFINE_PCI_DEVICE_TABLE
#define DEFINE_PCI_DEVICE_TABLE(_table) \
	const struct pci_device_id _table[]
#endif

#ifndef DMA_BIT_MASK
#define DMA_BIT_MASK(n)	(((n) == 64) ? ~0ULL : ((1ULL<<(n))-1))
#endif

/* Static function prototypes */
static int dvbm_pci_probe (struct pci_dev *pdev,
	const struct pci_device_id *id) __devinit;
static void dvbm_pci_remove (struct pci_dev *pdev) __devexit;
static int dvbm_init_module (void) __init;
static void dvbm_cleanup_module (void) __exit;

MODULE_AUTHOR("Linear Systems Ltd.");
MODULE_DESCRIPTION("DVB Master driver");
MODULE_LICENSE("GPL");
MODULE_VERSION(MASTER_DRIVER_VERSION);

char dvbm_driver_name[] = "dvbm";

static DEFINE_PCI_DEVICE_TABLE(dvbm_pci_id_table) = {
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
			DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQOE_MINIBNC)
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
			DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQDUALE_MINIBNC)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBQ3IOE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBQ3INOE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBLPTXE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			DVBM_PCI_DEVICE_ID_LINSYS_DVBLPRXE)
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

static LIST_HEAD(dvbm_card_list);

static struct class *dvbm_class;

/**
 * dvbm_pci_probe_generic - generic PCI insertion handler
 * @pdev: PCI device
 *
 * Perform generic PCI device initialization.
 * Returns a negative error code on failure and 0 on success.
 **/
int __devinit
dvbm_pci_probe_generic (struct pci_dev *pdev)
{
	int err;

	/* Wake a sleeping device.
	 * This is done before pci_request_regions ()
	 * as described in Documentation/PCI/pci.txt. */
	if ((err = pci_enable_device (pdev)) < 0) {
		printk (KERN_WARNING "%s: unable to enable device\n",
			dvbm_driver_name);
		return err;
	}

	/* Enable bus mastering */
	pci_set_master (pdev);

	/* Request I/O resources */
	if ((err = pci_request_regions (pdev, dvbm_driver_name)) < 0) {
		printk (KERN_WARNING "%s: unable to get I/O resources\n",
			dvbm_driver_name);
		pci_disable_device (pdev);
		return err;
	}

	/* Set PCI DMA addressing limitations */
	if ((err = pci_set_dma_mask (pdev, DMA_BIT_MASK(32))) < 0) {
		printk (KERN_WARNING "%s: unable to set PCI DMA mask\n",
			dvbm_driver_name);
		pci_disable_device (pdev);
		pci_release_regions (pdev);
		return err;
	}

	return 0;
}

/**
 * dvbm_pci_probe - PCI insertion handler
 * @pdev: PCI device
 * @id: PCI ID
 *
 * Call the appropriate PCI insertion handler.
 * Returns a negative error code on failure and 0 on success.
 **/
static int __devinit
dvbm_pci_probe (struct pci_dev *pdev,
	const struct pci_device_id *id)
{
	switch (id->device) {
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFD:
		return dvbm_fd_pci_probe (pdev);
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
		return dvbm_fdu_pci_probe (pdev);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBTX:
		return dvbm_tx_pci_probe (pdev);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBRX:
		return dvbm_rx_pci_probe (pdev);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBTXU:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBTXE:
		return dvbm_txu_pci_probe (pdev);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBRXU:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBRXE:
		return dvbm_rxu_pci_probe (pdev);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPFD:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPFDE:
		return dvbm_lpfd_pci_probe (pdev);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQLF:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQIE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQLF4:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF_MINIBNC:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF_HEADER:
		return dvbm_qlf_pci_probe (pdev);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQI:
		return dvbm_qi_pci_probe (pdev);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQO:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQOE:
		return dvbm_qo_pci_probe (pdev);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQOE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQOE_MINIBNC:
		return dvbm_lpqo_pci_probe (pdev);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQDUAL:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQDUALE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQDUALE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQDUALE_MINIBNC:
		return dvbm_qdual_pci_probe (pdev);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQ3IOE:
		return dvbm_q3io_pci_probe (pdev);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQ3INOE:
		return dvbm_q3ino_pci_probe (pdev);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPTXE:
		return dvbm_lptxe_pci_probe (pdev);
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPRXE:
		return dvbm_lprxe_pci_probe (pdev);
	default:
		break;
	}
	return -ENODEV;
}

/**
 * dvbm_pci_remove_generic - generic PCI removal handler
 * @pdev: PCI device
 *
 * Perform generic PCI device shutdown.
 * This function may be called during PCI probe error handling,
 * so don't mark it as __devexit.
 **/
void
dvbm_pci_remove_generic (struct pci_dev *pdev)
{
	pci_disable_device (pdev);
	pci_release_regions (pdev);
	return;
}

/**
 * dvbm_pci_remove - PCI removal handler
 * @pdev: PCI device
 *
 * Call the appropriate PCI removal handler.
 **/
static void __devexit
dvbm_pci_remove (struct pci_dev *pdev)
{
	switch (pdev->device) {
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBFD:
		dvbm_fd_pci_remove (pdev);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBTX:
		dvbm_tx_pci_remove (pdev);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBRX:
		dvbm_rx_pci_remove (pdev);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBTXU:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBTXE:
		dvbm_txu_pci_remove (pdev);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBRXU:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBRXE:
		dvbm_rxu_pci_remove (pdev);
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
		dvbm_fdu_pci_remove (pdev);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPFD:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPFDE:
		dvbm_lpfd_pci_remove (pdev);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQLF:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQIE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQLF4:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF_MINIBNC:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF_HEADER:
		dvbm_qlf_pci_remove (pdev);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQI:
		dvbm_qi_pci_remove (pdev);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQO:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQOE:
		dvbm_qo_pci_remove (pdev);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQOE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQOE_MINIBNC:
		dvbm_lpqo_pci_remove (pdev);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQDUAL:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQDUALE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQDUALE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQDUALE_MINIBNC:
		dvbm_qdual_pci_remove (pdev);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQ3IOE:
		dvbm_q3io_pci_remove (pdev);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQ3INOE:
		dvbm_q3ino_pci_remove (pdev);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPTXE:
		dvbm_lptxe_pci_remove (pdev);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPRXE:
		dvbm_lprxe_pci_remove (pdev);
		break;
	default:
		break;
	}
	return;
}

/**
 * dvbm_register - register a DVB Master device
 * @card: pointer to the board info structure
 **/
int
dvbm_register (struct master_dev *card)
{
	return mdev_register (card,
		&dvbm_card_list,
		dvbm_driver_name,
		dvbm_class);
}

/**
 * dvbm_unregister_all - unregister a DVB Master device and all interfaces
 * @card: pointer to the board info structure
 **/
void
dvbm_unregister_all (struct master_dev *card)
{
	struct list_head *p, *n;
	struct master_iface *iface;

	/* Unregister all ASI interfaces */
	list_for_each_safe (p, n, &card->iface_list) {
		iface = list_entry (p,
			struct master_iface, list);
		asi_unregister_iface (iface);
	}

	/* Unregister the device if it was registered */
	list_for_each (p, &dvbm_card_list) {
		if (p == &card->list) {
			mdev_unregister (card, dvbm_class);
			break;
		}
	}
	return;
}

/**
 * dvbm_init_module - register the module as a Master and PCI driver
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int __init
dvbm_init_module (void)
{
	int err;

	printk (KERN_INFO "%s: Linear Systems Ltd. "
		"DVB Master driver from master-%s (%s)\n",
		dvbm_driver_name, MASTER_DRIVER_VERSION, MASTER_DRIVER_DATE);

	/* Create a device class */
	dvbm_class = mdev_init (dvbm_driver_name);
	if (IS_ERR(dvbm_class)) {
		err = PTR_ERR(dvbm_class);
		goto NO_CLASS;
	}

	/* Register with the PCI subsystem */
	if ((err = pci_register_driver (&dvbm_pci_driver)) < 0) {
		printk (KERN_WARNING
			"%s: unable to register with PCI subsystem\n",
			dvbm_driver_name);
		goto NO_PCI;
	}

	return 0;

NO_PCI:
	mdev_cleanup (dvbm_class);
NO_CLASS:
	return err;
}

/**
 * dvbm_cleanup_module - unregister the module as a Master and PCI driver
 **/
static void __exit
dvbm_cleanup_module (void)
{
	pci_unregister_driver (&dvbm_pci_driver);
	mdev_cleanup (dvbm_class);
	return;
}

module_init (dvbm_init_module);
module_exit (dvbm_cleanup_module);

