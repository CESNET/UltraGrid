/* dvbm_lptxe.c
 *
 * Linux driver functions for Linear Systems Ltd. DVB Master III Tx LP PCIe.
 *
 * Copyright (C) 2003-2010 Linear Systems Ltd.
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

#include <linux/fs.h> /* inode, file, file_operations */
#include <linux/sched.h> /* pt_regs */
#include <linux/pci.h> /* pci_resource_start () */
#include <linux/slab.h> /* kzalloc () */
#include <linux/list.h> /* INIT_LIST_HEAD () */
#include <linux/spinlock.h> /* spin_lock_init () */
#include <linux/init.h> /* __devinit */
#include <linux/errno.h> /* error codes */
#include <linux/interrupt.h> /* irqreturn_t */
#include <linux/device.h> /* device_create_file () */
#include <linux/mutex.h> /* mutex_init () */

#include <asm/bitops.h> /* set_bit () */

#include "asicore.h"
#include "../include/master.h"
#include "mdev.h"
#include "mdma.h"
#include "dvbm.h"
#include "lsdma.h"
#include "dvbm_lpfd.h"

static const char dvbm_lptxe_name[] = DVBM_NAME_LPTXE;

/* Static function prototypes */
static irqreturn_t IRQ_HANDLER(dvbm_lptxe_irq_handler,irq,dev_id,regs);

static DEVICE_ATTR(blackburst_type,S_IRUGO|S_IWUSR,
	dvbm_lpfd_show_blackburst_type,dvbm_lpfd_store_blackburst_type);
static DEVICE_ATTR(uid,S_IRUGO,
	dvbm_lpfd_show_uid,NULL);

/**
 * dvbm_lptxe_pci_probe - PCI insertion handler for a DVB Master III Tx LP PCIe
 * @pdev: PCI device
 *
 * Handle the insertion of a DVB Master III Tx LP PCIe.
 * Returns a negative error code on failure and 0 on success.
 **/
int __devinit
dvbm_lptxe_pci_probe (struct pci_dev *pdev)
{
	int err;
	unsigned int cap;
	struct master_dev *card;

	err = dvbm_pci_probe_generic (pdev);
	if (err < 0) {
		goto NO_PCI;
	}

	/* Initialize the driver_data pointer so that dvbm_lptxe_pci_remove()
	 * doesn't try to free it if an error occurs */
	pci_set_drvdata (pdev, NULL);

	/* Allocate a board info structure */
	if ((card = (struct master_dev *)
		kzalloc (sizeof (*card), GFP_KERNEL)) == NULL) {
		err = -ENOMEM;
		goto NO_MEM;
	}

	/* Initialize the board info structure */
	/* LS DMA Controller */
	card->bridge_addr = ioremap_nocache (pci_resource_start (pdev, 2),
		pci_resource_len (pdev, 2));
	/* ASI Core */
	card->core.addr = ioremap_nocache (pci_resource_start (pdev, 0),
		pci_resource_len (pdev, 0));
	card->version = readl (card->core.addr + DVBM_LPFD_CSR) >> 16;
	card->name = dvbm_lptxe_name;
	card->id = pdev->device;
	card->irq = pdev->irq;
	card->irq_handler = dvbm_lptxe_irq_handler;
	INIT_LIST_HEAD(&card->iface_list);
	card->capabilities = MASTER_CAP_BLACKBURST | MASTER_CAP_UID;
	/* Lock for ICSR */
	spin_lock_init (&card->irq_lock);
	/* Lock for IBSTR, IPSTR, FTR, PFLUT, TCSR, RCSR */
	spin_lock_init (&card->reg_lock);
	mutex_init (&card->users_mutex);
	card->parent = &pdev->dev;

	/* Print the firmware version */
	printk (KERN_INFO "%s: %s detected, firmware version %u.%u (0x%04X)\n",
		dvbm_driver_name, card->name,
		card->version >> 8, card->version & 0x00ff, card->version);

	/* Store the pointer to the board info structure
	 * in the PCI info structure */
	pci_set_drvdata (pdev, card);

	/* Reset the FPGA */
	writel (DVBM_LPFD_TCSR_RST, card->core.addr + DVBM_LPFD_TCSR);

	/* Setup the LS DMA */
	writel (LSDMA_INTMSK_CH(0) | LSDMA_INTMSK_CH(1),
		card->bridge_addr + LSDMA_INTMSK);
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE,
		card->bridge_addr + LSDMA_CSR(0));
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);

	/* Register a DVB Master device */
	if ((err = dvbm_register (card)) < 0) {
		goto NO_DEV;
	}

	/* Add device attributes */
	if (card->capabilities & MASTER_CAP_BLACKBURST) {
		if ((err = device_create_file (card->dev,
			&dev_attr_blackburst_type)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'blackburst_type'\n",
				dvbm_driver_name);
		}
	}
	if (card->capabilities & MASTER_CAP_UID) {
		if ((err = device_create_file (card->dev,
			&dev_attr_uid)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'uid'\n",
				dvbm_driver_name);
		}
	}

	/* Register a transmit interface */
	cap = ASI_CAP_TX_SETCLKSRC | ASI_CAP_TX_FIFOUNDERRUN |
		ASI_CAP_TX_DATA |
		ASI_CAP_TX_MAKE204 | ASI_CAP_TX_FINETUNING |
		ASI_CAP_TX_LARGEIB |
		ASI_CAP_TX_INTERLEAVING |
		ASI_CAP_TX_TIMESTAMPS |
		ASI_CAP_TX_NULLPACKETS |
		ASI_CAP_TX_PTIMESTAMPS;
	if (card->version >= 0x0101) {
		cap |= ASI_CAP_TX_27COUNTER;
	}
	if ((err = asi_register_iface (card,
		&lsdma_dma_ops,
		DVBM_LPFD_FIFO,
		MASTER_DIRECTION_TX,
		&dvbm_lpfd_txfops,
		&dvbm_lpfd_txops,
		cap,
		4,
		ASI_CTL_TRANSPORT_DVB_ASI)) < 0) {
		goto NO_IFACE;
	}

	return 0;

NO_IFACE:
NO_DEV:
NO_MEM:
	dvbm_lptxe_pci_remove (pdev);
NO_PCI:
	return err;
}

/**
 * dvbm_lptxe_irq_handler - DVB Master III Tx LP PCIe interrupt service routine
 * @irq: interrupt number
 * @dev_id: pointer to the device data structure
 * @regs: processor context
 **/
static irqreturn_t
IRQ_HANDLER(dvbm_lptxe_irq_handler,irq,dev_id,regs)
{
	struct master_dev *card = dev_id;
	unsigned int dmaintsrc = readl (card->bridge_addr + LSDMA_INTSRC);
	unsigned int status, interrupting_iface = 0;
	struct master_iface *txiface = list_entry (card->iface_list.next,
		struct master_iface, list);

	if (dmaintsrc & LSDMA_INTSRC_CH(0)) {
		/* Read the interrupt type and clear it */
		spin_lock (&card->irq_lock);
		status = readl (card->bridge_addr + LSDMA_CSR(0));
		writel (status, card->bridge_addr + LSDMA_CSR(0));
		spin_unlock (&card->irq_lock);

		/* Increment the buffer pointer */
		if (status & LSDMA_CH_CSR_INTSRCBUFFER) {
			mdma_advance (txiface->dma);
		}

		/* Flag end-of-chain */
		if (status & LSDMA_CH_CSR_INTSRCDONE) {
			set_bit (ASI_EVENT_TX_BUFFER_ORDER, &txiface->events);
			set_bit (0, &txiface->dma_done);
		}

		/* Flag DMA abort */
		if (status & LSDMA_CH_CSR_INTSRCSTOP) {
			set_bit (0, &txiface->dma_done);
		}

		interrupting_iface |= 0x1;

	}

	/* Check and clear the source of the interrupt */
	spin_lock (&card->irq_lock);
	status = readl (card->core.addr + DVBM_LPFD_ICSR);
	writel (status, card->core.addr + DVBM_LPFD_ICSR);
	spin_unlock (&card->irq_lock);

	if (status & DVBM_LPFD_ICSR_TXUIS) {
		set_bit (ASI_EVENT_TX_FIFO_ORDER,
			&txiface->events);
		interrupting_iface |= 0x1;
	}
	if (status & DVBM_LPFD_ICSR_TXDIS) {
		set_bit (ASI_EVENT_TX_DATA_ORDER,
			&txiface->events);
		interrupting_iface |= 0x1;
	}

	if (interrupting_iface) {
		/* Dummy read to flush PCI posted writes */
		readl (card->bridge_addr + LSDMA_INTMSK);

		if (interrupting_iface & 0x1) {
			wake_up (&txiface->queue);
		}
		return IRQ_HANDLED;
	}
	return IRQ_NONE;
}

