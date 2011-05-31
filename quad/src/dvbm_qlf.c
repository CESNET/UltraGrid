/* dvbm_qlf.c
 *
 * Linux driver functions for Linear Systems Ltd. DVB Master Q/i RoHS.
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
#include <linux/module.h> /* THIS_MODULE */

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

#include <asm/uaccess.h> /* put_user () */
#include <asm/bitops.h> /* set_bit () */

#include "asicore.h"
#include "../include/master.h"
#include "mdev.h"
#include "mdma.h"
#include "dvbm.h"
#include "plx9080.h"
#include "lsdma.h"
#include "dvbm_qlf.h"

static const char dvbm_qlf_name[] = DVBM_NAME_QLF;
static const char dvbm_qie_name[] = DVBM_NAME_QIE;
static const char dvbm_lpqlf_name[] = DVBM_NAME_LPQLF;

/* Static function prototypes */
static ssize_t dvbm_qlf_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static irqreturn_t IRQ_HANDLER(dvbm_qlf_irq_handler,irq,dev_id,regs);
static void dvbm_qlf_init (struct master_iface *iface);
static void dvbm_qlf_start (struct master_iface *iface);
static void dvbm_qlf_stop (struct master_iface *iface);
static void dvbm_qlf_exit (struct master_iface *iface);
static long dvbm_qlf_unlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int FSYNC_HANDLER(dvbm_qlf_fsync,filp,datasync);

static struct file_operations dvbm_qlf_fops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.read = asi_read,
	.poll = asi_rxpoll,
	.unlocked_ioctl = dvbm_qlf_unlocked_ioctl,
	.compat_ioctl = dvbm_qlf_unlocked_ioctl,
	.open = asi_open,
	.release = asi_release,
	.fsync = dvbm_qlf_fsync,
	.fasync = NULL
};

static struct master_iface_operations dvbm_qlf_ops = {
	.init = dvbm_qlf_init,
	.start = dvbm_qlf_start,
	.stop = dvbm_qlf_stop,
	.exit = dvbm_qlf_exit
};

/**
 * dvbm_qlf_show_uid - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
static ssize_t
dvbm_qlf_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	return snprintf (buf, PAGE_SIZE, "0x%08X%08X\n",
		readl (card->core.addr + DVBM_QLF_UIDR_HI),
		readl (card->core.addr + DVBM_QLF_UIDR_LO));
}

static DEVICE_ATTR(uid,S_IRUGO,
	dvbm_qlf_show_uid,NULL);

/**
 * dvbm_qlf_pci_probe - PCI insertion handler for a DVB Master Q/i RoHS
 * @pdev: PCI device
 *
 * Handle the insertion of a DVB Master Q/i RoHS.
 * Returns a negative error code on failure and 0 on success.
 **/
int __devinit
dvbm_qlf_pci_probe (struct pci_dev *pdev)
{
	int err;
	unsigned int i, cap;
	struct master_dev *card;
	void __iomem *p;

	err = dvbm_pci_probe_generic (pdev);
	if (err < 0) {
		goto NO_PCI;
	}

	/* Initialize the driver_data pointer so that dvbm_qlf_pci_remove()
	 * doesn't try to free it if an error occurs */
	pci_set_drvdata (pdev, NULL);

	/* Allocate a board info structure */
	if ((card = (struct master_dev *)
		kzalloc (sizeof (*card), GFP_KERNEL)) == NULL) {
		err = -ENOMEM;
		goto NO_MEM;
	}

	/* Initialize the board info structure */
	switch (pdev->device) {
	default:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQLF:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQLF4:
		/* DMA Controller */
		card->bridge_addr = ioremap_nocache
			(pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		/* ASI Core */
		card->core.addr = ioremap_nocache
			(pci_resource_start (pdev, 2),
			pci_resource_len (pdev, 2));
		card->name = dvbm_qlf_name;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQIE:
		/* DMA Controller */
		card->bridge_addr = ioremap_nocache
			(pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		/* ASI Core */
		card->core.addr = ioremap_nocache
			(pci_resource_start (pdev, 2),
			pci_resource_len (pdev, 2));
		card->name = dvbm_qie_name;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF_MINIBNC:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF_HEADER:
		/* DMA Controller */
		card->bridge_addr = ioremap_nocache
			(pci_resource_start (pdev, 2),
			pci_resource_len (pdev, 2));
		/* ASI Core */
		card->core.addr = ioremap_nocache
			(pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		card->name = dvbm_lpqlf_name;
		break;
	}
	card->version = readl (card->core.addr + DVBM_QLF_FPGAID) & 0xffff;
	card->id = pdev->device;
	card->irq = pdev->irq;
	card->irq_handler = dvbm_qlf_irq_handler;
	INIT_LIST_HEAD(&card->iface_list);
	card->capabilities = MASTER_CAP_UID;
	/* Lock for LSDMA_CSR, ICSR */
	spin_lock_init (&card->irq_lock);
	/* Lock for PFLUT, RCR */
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

	switch (pdev->device) {
	default:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQLF:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQIE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQLF4:
		/* PCI 9056 */
		p = ioremap_nocache (pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));

		/* Reset the PCI 9056 */
		plx_reset_bridge (p);

		/* Setup the PCI 9056 */
		writel (PLX_INTCSR_PCIINT_ENABLE |
			PLX_INTCSR_PCILOCINT_ENABLE,
			p + PLX_INTCSR);
		/* Dummy read to flush PCI posted writes */
		readl (p + PLX_INTCSR);

		/* Unmap PCI 9056 */
		iounmap (p);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF_MINIBNC:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF_HEADER:
		break;
	}

	/* Reset the FPGA */
	for (i = 0; i < 4; i++) {
		writel (DVBM_QLF_RCR_RST, card->core.addr + DVBM_QLF_RCR(i));
	}

	/* Setup the LS DMA controller */
	writel (LSDMA_INTMSK_CH(0) | LSDMA_INTMSK_CH(1) | LSDMA_INTMSK_CH(2)
		| LSDMA_INTMSK_CH(3),
		card->bridge_addr + LSDMA_INTMSK);
	for (i = 0; i < 4; i++) {
		writel (LSDMA_CH_CSR_INTDONEENABLE |
			LSDMA_CH_CSR_INTSTOPENABLE |
			LSDMA_CH_CSR_DIRECTION,
			card->bridge_addr + LSDMA_CSR(i));
	}

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
				"%s: Unable to create file 'uid'\n",
				dvbm_driver_name);
		}
	}

	/* Register receiver interfaces */
	cap = ASI_CAP_RX_DATA |
		ASI_CAP_RX_SYNC | ASI_CAP_RX_CD | ASI_CAP_RX_MAKE188 |
		ASI_CAP_RX_BYTECOUNTER | ASI_CAP_RX_PIDFILTER |
		ASI_CAP_RX_TIMESTAMPS | ASI_CAP_RX_INVSYNC |
		ASI_CAP_RX_PTIMESTAMPS | ASI_CAP_RX_PIDCOUNTER |
		ASI_CAP_RX_4PIDCOUNTER | ASI_CAP_RX_NULLPACKETS |
		ASI_CAP_RX_27COUNTER;
	for (i = 0; i < 4; i++) {
		if ((err = asi_register_iface (card,
			&lsdma_dma_ops,
			0,
			MASTER_DIRECTION_RX,
			&dvbm_qlf_fops,
			&dvbm_qlf_ops,
			cap,
			4,
			ASI_CTL_TRANSPORT_DVB_ASI)) < 0) {
			goto NO_IFACE;
		}
	}

	return 0;

NO_IFACE:
NO_DEV:
NO_MEM:
	dvbm_qlf_pci_remove (pdev);
NO_PCI:
	return err;
}

/**
 * dvbm_qlf_pci_remove - PCI removal handler for a DVB Master Q/i RoHS
 * @pdev: PCI device
 *
 * Handle the removal of a DVB Master Q/i RoHS.
 * This function may be called during PCI probe error handling,
 * so don't mark it as __devexit.
 **/
void
dvbm_qlf_pci_remove (struct pci_dev *pdev)
{
	struct master_dev *card = pci_get_drvdata (pdev);

	if (card) {
		/* Unregister the device and all interfaces */
		dvbm_unregister_all (card);

		iounmap (card->core.addr);
		iounmap (card->bridge_addr);
		kfree (card);
	}
	dvbm_pci_remove_generic (pdev);
	return;
}

/**
 * dvbm_qlf_irq_handler - DVB Master Q/i RoHS interrupt service routine
 * @irq: interrupt number
 * @dev_id: pointer to the device data structure
 * @regs: processor context
 **/
static irqreturn_t
IRQ_HANDLER(dvbm_qlf_irq_handler,irq,dev_id,regs)
{
	struct master_dev *card = dev_id;
	struct list_head *p = &card->iface_list;
	struct master_iface *iface;
	unsigned int dmaintsrc = readl (card->bridge_addr + LSDMA_INTSRC);
	unsigned int status, interrupting_iface = 0, i;

	for (i = 0; i < 4; i++) {
		p = p->next;
		iface = list_entry (p, struct master_iface, list);

		/* Clear ASI interrupts */
		spin_lock (&card->irq_lock);
		status = readl (card->core.addr + DVBM_QLF_ICSR(i));
		if ((status & DVBM_QLF_ICSR_ISMASK) != 0) {
			writel (status, card->core.addr + DVBM_QLF_ICSR(i));
			spin_unlock (&card->irq_lock);
			if (status & DVBM_QLF_ICSR_RXCDIS) {
				set_bit (ASI_EVENT_RX_CARRIER_ORDER,
					&iface->events);
				interrupting_iface |= (0x1 << i);
			}
			if (status & DVBM_QLF_ICSR_RXAOSIS) {
				set_bit (ASI_EVENT_RX_AOS_ORDER,
					&iface->events);
				interrupting_iface |= (0x1 << i);
			}
			if (status & DVBM_QLF_ICSR_RXLOSIS) {
				set_bit (ASI_EVENT_RX_LOS_ORDER,
					&iface->events);
				interrupting_iface |= (0x1 << i);
			}
			if (status & DVBM_QLF_ICSR_RXOIS) {
				set_bit (ASI_EVENT_RX_FIFO_ORDER,
					&iface->events);
				interrupting_iface |= (0x1 << i);
			}
			if (status & DVBM_QLF_ICSR_RXDIS) {
				set_bit (ASI_EVENT_RX_DATA_ORDER,
					&iface->events);
				interrupting_iface |= (0x1 << i);
			}
		} else {
			spin_unlock (&card->irq_lock);
		}

		/* Clear DMA interrupts */
		if (dmaintsrc & LSDMA_INTSRC_CH(i)) {
			/* Read the interrupt type and clear it */
			spin_lock (&card->irq_lock);
			status = readl (card->bridge_addr + LSDMA_CSR(i));
			writel (status, card->bridge_addr + LSDMA_CSR(i));
			spin_unlock (&card->irq_lock);

			/* Increment the buffer pointer */
			if (status & LSDMA_CH_CSR_INTSRCBUFFER) {
				mdma_advance (iface->dma);
				if (mdma_rx_isempty (iface->dma)) {
					set_bit (ASI_EVENT_RX_BUFFER_ORDER,
						&iface->events);
				}
			}

			/* Flag end-of-chain */
			if (status & LSDMA_CH_CSR_INTSRCDONE) {
				set_bit (0, &iface->dma_done);
			}

			/* Flag DMA abort */
			if (status & LSDMA_CH_CSR_INTSRCSTOP) {
				set_bit (0, &iface->dma_done);
			}

			interrupting_iface |= (0x1 << i);
		}

		if (interrupting_iface & (0x1 << i)) {
			wake_up (&iface->queue);
		}
	}

	if (interrupting_iface) {
		/* Dummy read to flush PCI posted writes */
		readl (card->bridge_addr + LSDMA_INTMSK);
		return IRQ_HANDLED;
	}
	return IRQ_NONE;
}

/**
 * dvbm_qlf_init - Initialize a DVB Master Q/i RoHS receiver
 * @iface: interface
 **/
static void
dvbm_qlf_init (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int i, reg = (iface->null_packets ? DVBM_QLF_RCR_RNP : 0);

	switch (iface->timestamps) {
	default:
	case ASI_CTL_TSTAMP_NONE:
		reg |= 0;
		break;
	case ASI_CTL_TSTAMP_APPEND:
		reg |= DVBM_QLF_RCR_TSE;
		break;
	case ASI_CTL_TSTAMP_PREPEND:
		reg |= DVBM_QLF_RCR_PTSE;
		break;
	}
	switch (iface->mode) {
	default:
	case ASI_CTL_RX_MODE_RAW:
		reg |= 0;
		break;
	case ASI_CTL_RX_MODE_188:
		reg |= DVBM_QLF_RCR_188 | DVBM_QLF_RCR_PFE;
		break;
	case ASI_CTL_RX_MODE_204:
		reg |= DVBM_QLF_RCR_204 | DVBM_QLF_RCR_PFE;
		break;
	case ASI_CTL_RX_MODE_AUTO:
		reg |= DVBM_QLF_RCR_AUTO | DVBM_QLF_RCR_PFE;
		break;
	case ASI_CTL_RX_MODE_AUTOMAKE188:
		reg |= DVBM_QLF_RCR_AUTO | DVBM_QLF_RCR_RSS |
			DVBM_QLF_RCR_PFE;
		break;
	case ASI_CTL_RX_MODE_204MAKE188:
		reg |= DVBM_QLF_RCR_204 | DVBM_QLF_RCR_RSS |
			DVBM_QLF_RCR_PFE;
		break;
	}

	/* There will be no races on RCR
	 * until this code returns, so we don't need to lock it */
	writel (reg | DVBM_QLF_RCR_RST,
		card->core.addr + DVBM_QLF_RCR(channel));
	writel (reg, card->core.addr + DVBM_QLF_RCR(channel));
	writel (DVBM_QLF_RDMATL, card->core.addr + DVBM_QLF_RDMATLR(channel));

	/* Reset the byte counter */
	readl (card->core.addr + DVBM_QLF_RXBCOUNTR(channel));

	/* Reset the PID filter.
	 * There will be no races on PFLUT
	 * until this code returns, so we don't need to lock it */
	for (i = 0; i < 256; i++) {
		writel (i, card->core.addr + DVBM_QLF_PFLUTAR(channel));
		/* Dummy read to flush PCI posted writes */
		readl (card->core.addr + DVBM_QLF_FPGAID);
		writel (0xffffffff, card->core.addr + DVBM_QLF_PFLUTR(channel));
	}

	/* Clear PID registers */
	writel (0, card->core.addr + DVBM_QLF_PIDR0(channel));
	writel (0, card->core.addr + DVBM_QLF_PIDR1(channel));
	writel (0, card->core.addr + DVBM_QLF_PIDR2(channel));
	writel (0, card->core.addr + DVBM_QLF_PIDR3(channel));

	/* Reset PID counters */
	readl (card->core.addr + DVBM_QLF_PIDCOUNTR0(channel));
	readl (card->core.addr + DVBM_QLF_PIDCOUNTR1(channel));
	readl (card->core.addr + DVBM_QLF_PIDCOUNTR2(channel));
	readl (card->core.addr + DVBM_QLF_PIDCOUNTR3(channel));

	return;
}

/**
 * dvbm_qlf_start - Activate the DVB Master Q/i RoHS receiver
 * @iface: interface
 **/
static void
dvbm_qlf_start (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	struct master_dma *dma = iface->dma;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	/* Enable and start DMA */
	writel (mdma_dma_to_desc_low (lsdma_head_desc_bus_addr (dma)),
		card->bridge_addr + LSDMA_DESC(channel));
	writel (mdma_dma_to_desc_high (lsdma_head_desc_bus_addr (dma)),
		card->bridge_addr + LSDMA_DESC_H(channel));
	clear_bit (0, &iface->dma_done);
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION | LSDMA_CH_CSR_ENABLE,
		card->bridge_addr + LSDMA_CSR(channel));
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);

	/* Enable receiver interrupts */
	spin_lock_irq (&card->irq_lock);
	writel (DVBM_QLF_ICSR_RXCDIE | DVBM_QLF_ICSR_RXAOSIE |
		DVBM_QLF_ICSR_RXLOSIE | DVBM_QLF_ICSR_RXOIE |
		DVBM_QLF_ICSR_RXDIE,
		card->core.addr + DVBM_QLF_ICSR(channel));
	spin_unlock_irq (&card->irq_lock);

	/* Enable the receiver */
	reg = readl (card->core.addr + DVBM_QLF_RCR(channel));
	writel (reg | DVBM_QLF_RCR_EN,
		card->core.addr + DVBM_QLF_RCR(channel));

	return;
}

/**
 * dvbm_qlf_stop - Deactivate the DVB Master Q/i RoHS receiver
 * @iface: interface
 **/
static void
dvbm_qlf_stop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	/* Disable the receiver */
	reg = readl (card->core.addr + DVBM_QLF_RCR(channel));
	writel (reg & ~DVBM_QLF_RCR_EN,
		card->core.addr + DVBM_QLF_RCR(channel));

	/* Disable receiver interrupts */
	spin_lock_irq (&card->irq_lock);
	writel (DVBM_QLF_ICSR_RXCDIS | DVBM_QLF_ICSR_RXAOSIS |
		DVBM_QLF_ICSR_RXLOSIS | DVBM_QLF_ICSR_RXOIS |
		DVBM_QLF_ICSR_RXDIS,
		card->core.addr + DVBM_QLF_ICSR(channel));
	spin_unlock_irq (&card->irq_lock);

	/* Disable and abort DMA */
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION,
		card->bridge_addr + LSDMA_CSR(channel));
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION | LSDMA_CH_CSR_STOP,
		card->bridge_addr + LSDMA_CSR(channel));
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);

	wait_event (iface->queue, test_bit (0, &iface->dma_done));

	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION,
		card->bridge_addr + LSDMA_CSR(channel));

	return;
}

/**
 * dvbm_qlf_exit - Clean up the DVB Master Q/i RoHS receiver
 * @iface: interface
 **/
static void
dvbm_qlf_exit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);

	/* Reset the receiver.
	 * There will be no races on RCR here,
	 * so we don't need to lock it */
	writel (DVBM_QLF_RCR_RST, card->core.addr + DVBM_QLF_RCR(channel));

	return;
}

/**
 * dvbm_qlf_unlocked_ioctl - DVB Master Q/i RoHS receiver unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
dvbm_qlf_unlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	int val;
	unsigned int reg = 0, *pflut, i;

	switch (cmd) {
	case ASI_IOC_RXGETSTATUS:
		/* Atomic reads of ICSR and RCR, so we don't need to lock */
		reg = readl (card->core.addr + DVBM_QLF_ICSR(channel));
		switch (readl (card->core.addr + DVBM_QLF_RCR(channel)) &
			DVBM_QLF_RCR_SYNC_MASK) {
		case 0:
			val = 1;
			break;
		case DVBM_QLF_RCR_188:
			val = (reg & DVBM_QLF_ICSR_RXPASSING) ? 188 : 0;
			break;
		case DVBM_QLF_RCR_204:
			val = (reg & DVBM_QLF_ICSR_RXPASSING) ? 204 : 0;
			break;
		case DVBM_QLF_RCR_AUTO:
			if (reg & DVBM_QLF_ICSR_RXPASSING) {
				val = (reg & DVBM_QLF_ICSR_RX204) ? 204 : 188;
			} else {
				val = 0;
			}
			break;
		default:
			return -EIO;
		}
		if (put_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXGETBYTECOUNT:
		if (!(iface->capabilities & ASI_CAP_RX_BYTECOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr +
			DVBM_QLF_RXBCOUNTR(channel)),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETINVSYNC:
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		switch (val) {
		case 0:
			reg |= 0;
			break;
		case 1:
			reg |= DVBM_QLF_RCR_INVSYNC;
			break;
		default:
			return -EINVAL;
		}
		spin_lock (&card->reg_lock);
		writel ((readl (card->core.addr + DVBM_QLF_RCR(channel)) &
			~DVBM_QLF_RCR_INVSYNC) | reg,
			card->core.addr + DVBM_QLF_RCR(channel));
		spin_unlock (&card->reg_lock);
		break;
	case ASI_IOC_RXGETCARRIER:
		/* Atomic read of ICSR, so we don't need to lock */
		if (put_user (
			(readl (card->core.addr + DVBM_QLF_ICSR(channel)) &
			DVBM_QLF_ICSR_RXCD) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETDSYNC:
	case ASI_IOC_RXSETINPUT_DEPRECATED:
	case ASI_IOC_RXSETINPUT:
		/* Dummy ioctl; only zero is valid */
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if (val) {
			return -EINVAL;
		}
		break;
	case ASI_IOC_RXGETRXD:
		/* Atomic read of ICSR, so we don't need to lock */
		if (put_user (
			(readl (card->core.addr + DVBM_QLF_ICSR(channel)) &
			DVBM_QLF_ICSR_RXD) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPF:
		if (!(iface->capabilities & ASI_CAP_RX_PIDFILTER)) {
			return -ENOTTY;
		}
		pflut = (unsigned int *)
			kmalloc (sizeof (unsigned int [256]), GFP_KERNEL);
		if (pflut == NULL) {
			return -ENOMEM;
		}
		if (copy_from_user (pflut, (unsigned int __user *)arg,
			sizeof (unsigned int [256]))) {
			kfree (pflut);
			return -EFAULT;
		}
		spin_lock (&card->reg_lock);
		for (i = 0; i < 256; i++) {
			writel (i, card->core.addr + DVBM_QLF_PFLUTAR(channel));
			/* Dummy read to flush PCI posted writes */
			readl (card->core.addr + DVBM_QLF_FPGAID);
			writel (pflut[i], card->core.addr + DVBM_QLF_PFLUTR(channel));
		}
		spin_unlock (&card->reg_lock);
		kfree (pflut);
		break;
	case ASI_IOC_RXSETPID0:
		if (!(iface->capabilities & ASI_CAP_RX_PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		writel (val, card->core.addr + DVBM_QLF_PIDR0(channel));
		/* Reset PID count */
		readl (card->core.addr + DVBM_QLF_PIDCOUNTR0(channel));
		break;
	case ASI_IOC_RXGETPID0COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr +
			DVBM_QLF_PIDCOUNTR0(channel)),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPID1:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		writel (val, card->core.addr + DVBM_QLF_PIDR1(channel));
		/* Reset PID count */
		readl (card->core.addr + DVBM_QLF_PIDCOUNTR1(channel));
		break;
	case ASI_IOC_RXGETPID1COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr +
			DVBM_QLF_PIDCOUNTR1(channel)),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPID2:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		writel (val, card->core.addr + DVBM_QLF_PIDR2(channel));
		/* Reset PID count */
		readl (card->core.addr + DVBM_QLF_PIDCOUNTR2(channel));
		break;
	case ASI_IOC_RXGETPID2COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr +
			DVBM_QLF_PIDCOUNTR2(channel)),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPID3:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		writel (val, card->core.addr + DVBM_QLF_PIDR3(channel));
		/* Reset PID count */
		readl (card->core.addr + DVBM_QLF_PIDCOUNTR3(channel));
		break;
	case ASI_IOC_RXGETPID3COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr +
			DVBM_QLF_PIDCOUNTR3(channel)),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXGET27COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_27COUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr + DVBM_QLF_27COUNTR),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return asi_rxioctl (filp, cmd, arg);
	}
	return 0;
}

/**
 * dvbm_qlf_fsync - DVB Master Q/i RoHS receiver fsync() method
 * @filp: file to flush
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
FSYNC_HANDLER(dvbm_qlf_fsync,filp,datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	mutex_lock (&iface->buf_mutex);

	/* Stop the receiver */
	dvbm_qlf_stop (iface);

	/* Reset the onboard FIFO and driver buffers */
	spin_lock (&card->reg_lock);
	reg = readl (card->core.addr + DVBM_QLF_RCR(channel));
	writel (reg | DVBM_QLF_RCR_RST,
		card->core.addr + DVBM_QLF_RCR(channel));
	writel (reg, card->core.addr + DVBM_QLF_RCR(channel));
	spin_unlock (&card->reg_lock);
	iface->events = 0;
	lsdma_reset (iface->dma);

	/* Start the receiver */
	dvbm_qlf_start (iface);

	mutex_unlock (&iface->buf_mutex);
	return 0;
}

