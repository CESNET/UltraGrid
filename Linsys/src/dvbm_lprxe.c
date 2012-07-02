/* dvbm_lprxe.c
 *
 * Linux driver functions for Linear Systems Ltd. DVB Master III Rx LP PCIe.
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

static const char dvbm_lprxe_name[] = DVBM_NAME_LPRXE;

/* Static function prototypes */
static irqreturn_t IRQ_HANDLER(dvbm_lprxe_irq_handler,irq,dev_id,regs);

static DEVICE_ATTR(uid,S_IRUGO,
	dvbm_lpfd_show_uid,NULL);

/**
 * dvbm_lprxe_pci_probe - PCI insertion handler for a DVB Master III Rx LP PCIe
 * @pdev: PCI device
 *
 * Handle the insertion of a DVB Master III Rx LP PCIe.
 * Returns a negative error code on failure and 0 on success.
 **/
int __devinit
dvbm_lprxe_pci_probe (struct pci_dev *pdev)
{
	int err;
	unsigned int cap;
	struct master_dev *card;

	err = dvbm_pci_probe_generic (pdev);
	if (err < 0) {
		goto NO_PCI;
	}

	/* Initialize the driver_data pointer so that dvbm_lprxe_pci_remove()
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
	card->name = dvbm_lprxe_name;
	card->id = pdev->device;
	card->irq = pdev->irq;
	card->irq_handler = dvbm_lprxe_irq_handler;
	INIT_LIST_HEAD(&card->iface_list);
	card->capabilities = MASTER_CAP_UID;
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
	writel (DVBM_LPFD_RCSR_RST, card->core.addr + DVBM_LPFD_RCSR);

	/* Setup the LS DMA */
	writel (LSDMA_INTMSK_CH(0) | LSDMA_INTMSK_CH(1),
		card->bridge_addr + LSDMA_INTMSK);
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION,
		card->bridge_addr + LSDMA_CSR(1));
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);

	/* Register a DVB Master device */
	if ((err = dvbm_register (card)) < 0) {
		goto NO_DEV;
	}

	/* Add device attributes */
	if (card->capabilities & MASTER_CAP_UID) {
		if ((err = device_create_file (card->dev,
			&dev_attr_uid)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'uid'\n",
				dvbm_driver_name);
		}
	}

	/* Register a receive interface */
	cap = ASI_CAP_RX_SYNC | ASI_CAP_RX_INVSYNC | ASI_CAP_RX_CD |
		ASI_CAP_RX_MAKE188 |
		ASI_CAP_RX_DATA |
		ASI_CAP_RX_PIDFILTER |
		ASI_CAP_RX_TIMESTAMPS |
		ASI_CAP_RX_PTIMESTAMPS |
		ASI_CAP_RX_NULLPACKETS;
	if ((err = asi_register_iface (card,
		&lsdma_dma_ops,
		DVBM_LPFD_FIFO,
		MASTER_DIRECTION_RX,
		&dvbm_lpfd_rxfops,
		&dvbm_lpfd_rxops,
		cap,
		4,
		ASI_CTL_TRANSPORT_DVB_ASI)) < 0) {
		goto NO_IFACE;
	}

	return 0;

NO_IFACE:
NO_DEV:
NO_MEM:
	dvbm_lprxe_pci_remove (pdev);
NO_PCI:
	return err;
}

/**
 * dvbm_lprxe_irq_handler - DVB Master III Rx LP PCIe interrupt service routine
 * @irq: interrupt number
 * @dev_id: pointer to the device data structure
 * @regs: processor context
 **/
static irqreturn_t
IRQ_HANDLER(dvbm_lprxe_irq_handler,irq,dev_id,regs)
{
	struct master_dev *card = dev_id;
	unsigned int dmaintsrc = readl (card->bridge_addr + LSDMA_INTSRC);
	unsigned int status, interrupting_iface = 0;
	struct master_iface *rxiface = list_entry (card->iface_list.prev,
		struct master_iface, list);

	if (dmaintsrc & LSDMA_INTSRC_CH(1)) {
		struct master_dma *dma = rxiface->dma;

		/* Read the interrupt type and clear it */
		spin_lock (&card->irq_lock);
		status = readl (card->bridge_addr + LSDMA_CSR(1));
		writel (status, card->bridge_addr + LSDMA_CSR(1));
		spin_unlock (&card->irq_lock);

		/* Increment the buffer pointer */
		if (status & LSDMA_CH_CSR_INTSRCBUFFER) {
			mdma_advance (dma);
			if (mdma_rx_isempty (dma)) {
				set_bit (ASI_EVENT_RX_BUFFER_ORDER,
					&rxiface->events);
			}
		}

		/* Flag end-of-chain */
		if (status & LSDMA_CH_CSR_INTSRCDONE) {
			set_bit (0, &rxiface->dma_done);
		}

		/* Flag DMA abort */
		if (status & LSDMA_CH_CSR_INTSRCSTOP) {
			set_bit (0, &rxiface->dma_done);
		}

		interrupting_iface |= 0x2;
	}

	/* Check and clear the source of the interrupt */
	spin_lock (&card->irq_lock);
	status = readl (card->core.addr + DVBM_LPFD_ICSR);
	writel (status, card->core.addr + DVBM_LPFD_ICSR);
	spin_unlock (&card->irq_lock);

	if (status & DVBM_LPFD_ICSR_RXCDIS) {
		set_bit (ASI_EVENT_RX_CARRIER_ORDER,
			&rxiface->events);
		interrupting_iface |= 0x2;
	}
	if (status & DVBM_LPFD_ICSR_RXAOSIS) {
		set_bit (ASI_EVENT_RX_AOS_ORDER,
			&rxiface->events);
		interrupting_iface |= 0x2;
	}
	if (status & DVBM_LPFD_ICSR_RXLOSIS) {
		set_bit (ASI_EVENT_RX_LOS_ORDER,
			&rxiface->events);
		interrupting_iface |= 0x2;
	}
	if (status & DVBM_LPFD_ICSR_RXOIS) {
		set_bit (ASI_EVENT_RX_FIFO_ORDER,
			&rxiface->events);
		interrupting_iface |= 0x2;
	}
	if (status & DVBM_LPFD_ICSR_RXDIS) {
		set_bit (ASI_EVENT_RX_DATA_ORDER,
			&rxiface->events);
		interrupting_iface |= 0x2;
	}

	if (interrupting_iface) {
		/* Dummy read to flush PCI posted writes */
		readl (card->bridge_addr + LSDMA_INTMSK);

		if (interrupting_iface & 0x2) {
			wake_up (&rxiface->queue);
		}
		return IRQ_HANDLED;
	}
	return IRQ_NONE;
}

