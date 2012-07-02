/* dvbm_tx.c
 *
 * Linux driver functions for Linear Systems Ltd. DVB Master Send.
 *
 * Copyright (C) 1999 Tony Bolger <d7v@indigo.ie>
 * Copyright (C) 2000-2010 Linear Systems Ltd.
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
#include <linux/mutex.h> /* mutex_init () */
#include <linux/delay.h> /* msleep () */

#include <asm/uaccess.h> /* put_user () */
#include <asm/bitops.h> /* set_bit () */

#include "asicore.h"
#include "../include/master.h"
#include "mdev.h"
#include "mdma.h"
#include "dvbm.h"
#include "plx9080.h"
#include "dvbm_tx.h"

static const char dvbm_tx_name[] = DVBM_NAME_TX;

/* Static function prototypes */
static irqreturn_t IRQ_HANDLER(dvbm_tx_irq_handler,irq,dev_id,regs);
static void dvbm_tx_init (struct master_iface *iface);
static void dvbm_tx_start (struct master_iface *iface);
static void dvbm_tx_stop (struct master_iface *iface);
static void dvbm_tx_start_tx_dma (struct master_iface *iface);
static long dvbm_tx_unlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int FSYNC_HANDLER(dvbm_tx_fsync,filp,datasync);

static struct file_operations dvbm_tx_fops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.write = asi_write,
	.poll = asi_txpoll,
	.unlocked_ioctl = dvbm_tx_unlocked_ioctl,
	.compat_ioctl = dvbm_tx_unlocked_ioctl,
	.open = asi_open,
	.release = asi_release,
	.fsync = dvbm_tx_fsync,
	.fasync = NULL
};

static struct master_iface_operations dvbm_tx_ops = {
	.init = dvbm_tx_init,
	.start = dvbm_tx_start,
	.stop = dvbm_tx_stop,
	.exit = NULL,
	.start_tx_dma = dvbm_tx_start_tx_dma
};

/**
 * dvbm_tx_pci_probe - PCI insertion handler for a DVB Master Send
 * @pdev: PCI device
 *
 * Handle the insertion of a DVB Master Send.
 * Returns a negative error code on failure and 0 on success.
 **/
int __devinit
dvbm_tx_pci_probe (struct pci_dev *pdev)
{
	int err;
	unsigned int version, cap;
	struct master_dev *card;

	err = dvbm_pci_probe_generic (pdev);
	if (err < 0) {
		goto NO_PCI;
	}

	/* Initialize the driver_data pointer so that dvbm_tx_pci_remove()
	 * doesn't try to free it if an error occurs */
	pci_set_drvdata (pdev, NULL);

	/* Print the firmware version */
	version = inl (pci_resource_start (pdev, 2) + DVBM_TX_STATUS) >> 24;
	printk (KERN_INFO "%s: %s detected, firmware version %u (0x%02X)\n",
		dvbm_driver_name, dvbm_tx_name, version, version);

	/* Allocate and initialize a board info structure */
	if ((card = (struct master_dev *)
		kzalloc (sizeof (*card), GFP_KERNEL)) == NULL) {
		err = -ENOMEM;
		goto NO_MEM;
	}

	card->bridge_addr = ioremap_nocache (pci_resource_start (pdev, 0),
		pci_resource_len (pdev, 0));
	card->core.port = pci_resource_start (pdev, 2);
	card->version = version << 8;
	card->name = dvbm_tx_name;
	card->id = pdev->device;
	card->irq = pdev->irq;
	card->irq_handler = dvbm_tx_irq_handler;
	INIT_LIST_HEAD(&card->iface_list);
	spin_lock_init (&card->irq_lock); /* Unused */
	/* Lock for STUFFING, FINETUNE */
	spin_lock_init (&card->reg_lock);
	mutex_init (&card->users_mutex);
	card->parent = &pdev->dev;

	/* Store the pointer to the board info structure
	 * in the PCI info structure */
	pci_set_drvdata (pdev, card);

	/* Register a DVB Master device */
	if ((err = dvbm_register (card)) < 0) {
		goto NO_DEV;
	}

	/* Register a transmit interface */
	cap = 0;
	if (version >= 4) {
		cap |= ASI_CAP_TX_FINETUNING |
			ASI_CAP_TX_BYTECOUNTER;
	}
	if (version >= 5) {
		cap |= ASI_CAP_TX_27COUNTER |
			ASI_CAP_TX_BYTESOR27;
	}
	if ((err = asi_register_iface (card,
		&plx_dma_ops,
		DVBM_TX_FIFO,
		MASTER_DIRECTION_TX,
		&dvbm_tx_fops,
		&dvbm_tx_ops,
		cap,
		4,
		ASI_CTL_TRANSPORT_DVB_ASI)) < 0) {
		goto NO_IFACE;
	}

	return 0;

NO_IFACE:
NO_DEV:
NO_MEM:
	dvbm_tx_pci_remove (pdev);
NO_PCI:
	return err;
}

/**
 * dvbm_tx_pci_remove - PCI removal handler for a DVB Master Send
 * @pdev: PCI device
 *
 * Handle the removal of a DVB Master Send.
 * This function may be called during PCI probe error handling,
 * so don't mark it as __devexit.
 **/
void
dvbm_tx_pci_remove (struct pci_dev *pdev)
{
	struct master_dev *card = pci_get_drvdata (pdev);

	if (card) {
		/* Unregister the device and all interfaces */
		dvbm_unregister_all (card);

		iounmap (card->bridge_addr);
		kfree (card);
	}
	dvbm_pci_remove_generic (pdev);
	return;
}

/**
 * dvbm_tx_irq_handler - DVB Master Send interrupt service routine
 * @irq: interrupt number
 * @dev_id: pointer to the device data structure
 * @regs: processor context
 **/
static irqreturn_t
IRQ_HANDLER(dvbm_tx_irq_handler,irq,dev_id,regs)
{
	struct master_dev *card = dev_id;
	unsigned int intcsr = readl (card->bridge_addr + PLX_INTCSR);
	struct master_iface *iface = list_entry (card->iface_list.next,
		struct master_iface, list);

	if (intcsr & PLX_INTCSR_DMA0INT_ACTIVE) {
		/* Read the interrupt type and clear it */
		unsigned int status = readb (card->bridge_addr + PLX_DMACSR0);
		writeb (PLX_DMACSR_ENABLE | PLX_DMACSR_CLINT,
			card->bridge_addr + PLX_DMACSR0);

		/* Increment the buffer pointer */
		mdma_advance (iface->dma);

		/* Flag end-of-chain */
		if (status & PLX_DMACSR_DONE) {
			set_bit (ASI_EVENT_TX_BUFFER_ORDER, &iface->events);
			set_bit (0, &iface->dma_done);
		}

		/* Dummy read to flush PCI posted writes */
		readb (card->bridge_addr + PLX_DMACSR1);

		wake_up (&iface->queue);
		return IRQ_HANDLED;
	}
	return IRQ_NONE;
}

/**
 * dvbm_tx_init - Initialize the DVB Master Send
 * @iface: interface
 **/
static void
dvbm_tx_init (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg = 0, temp;

	switch (iface->mode) {
	case ASI_CTL_TX_MODE_188:
		reg |= 0;
		break;
	case ASI_CTL_TX_MODE_204:
		reg |= DVBM_TX_CFG_204;
		break;
	default:
		break;
	}
	if (iface->count27) {
		reg |= DVBM_TX_CFG_27MHZ;
	}
	master_outl (card, DVBM_TX_CFG, reg);

	/* Reset the PCI 9080 */
	plx_reset_bridge (card->bridge_addr);

	/* Setup the PCI 9080 */
	writel (PLX_INTCSR_PCIINT_ENABLE |
		PLX_INTCSR_PCILOCINT_ENABLE |
		PLX_INTCSR_DMA0INT_ENABLE,
		card->bridge_addr + PLX_INTCSR);
	writel (PLX_DMAMODE_32BIT | PLX_DMAMODE_READY |
		PLX_DMAMODE_LOCALBURST | PLX_DMAMODE_CHAINED |
		PLX_DMAMODE_INT | PLX_DMAMODE_CLOC |
		PLX_DMAMODE_DEMAND | PLX_DMAMODE_INTPCI,
		card->bridge_addr + PLX_DMAMODE0);
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + PLX_INTCSR);

	/* Program onboard FIFO almost-full and almost-empty levels */
	master_outl (card, DVBM_TX_CFG, reg | DVBM_TX_CFG_FIFOLEVELS);
	temp = DVBM_TX_PAE & 0xff;
	temp |= temp << 8;
	temp |= temp << 16;
	master_outl (card, DVBM_TX_FIFO, temp);
	temp = DVBM_TX_PAE >> 8;
	temp |= temp << 8;
	temp |= temp << 16;
	master_outl (card, DVBM_TX_FIFO, temp);
	temp = DVBM_TX_PAF & 0xff;
	temp |= temp << 8;
	temp |= temp << 16;
	master_outl (card, DVBM_TX_FIFO, temp);
	temp = DVBM_TX_PAF >> 8;
	temp |= temp << 8;
	temp |= temp << 16;
	master_outl (card, DVBM_TX_FIFO, temp);

	master_outl (card, DVBM_TX_CFG, reg);
	master_outl (card, DVBM_TX_STUFFING, 0);
	if (iface->capabilities & ASI_CAP_TX_FINETUNING) {
		master_outl (card, DVBM_TX_FINETUNE, 0);
	}
	if (iface->capabilities & ASI_CAP_TX_BYTECOUNTER) {
		/* Reset byte counter */
		master_inl (card, DVBM_TX_COUNTR);
	}
	return;
}

/**
 * dvbm_tx_start - Activate the DVB Master Send
 * @iface: interface
 **/
static void
dvbm_tx_start (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;

	/* Enable DMA */
	writeb (PLX_DMACSR_ENABLE, card->bridge_addr + PLX_DMACSR0);

	/* Enable the transmitter */
	reg = DVBM_TX_CFG_ENABLE;
	switch (iface->mode) {
	case ASI_CTL_TX_MODE_188:
		reg |= 0;
		break;
	case ASI_CTL_TX_MODE_204:
		reg |= DVBM_TX_CFG_204;
		break;
	default:
		break;
	}
	if (iface->count27) {
		reg |= DVBM_TX_CFG_27MHZ;
	}
	master_outl (card, DVBM_TX_CFG, reg);

	return;
}

/**
 * dvbm_tx_stop - Deactivate the DVB Master Send
 * @iface: interface
 **/
static void
dvbm_tx_stop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	struct master_dma *dma = iface->dma;

	plx_tx_link_all (dma);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	plx_reset (dma);

	/* Do not disable the transmitter here;
	 * the onboard FIFO may still be emptying.
	 * Check the byte counter if it exists,
	 * otherwise hope that no one re-opens the device
	 * until the FIFO is empty. */
	if (iface->capabilities & ASI_CAP_TX_BYTECOUNTER) {
		master_outl (card, DVBM_TX_CFG,
			(master_inl (card, DVBM_TX_STATUS) &
			~DVBM_TX_STATUS_27MHZ &
			~DVBM_TX_STATUS_VERSIONMASK));

		/* Clear the counter, which will
		 * (hopefully) guarantee that the counter
		 * is not zero because of rollover
		 * the first time it is read in the loop. */
		master_inl (card, DVBM_TX_COUNTR);

		/* Wait until the byte counter doesn't change
		 * over 100 ms. An MPEG-2 transport stream
		 * must contain a PCR every 100 ms,
		 * so the byte counter should change
		 * if data is flowing. */
		do {
			msleep (100);
		} while (master_inl (card, DVBM_TX_COUNTR));
	}

	/* We don't disable the transmitter here
	 * in case the onboard FIFOs are not yet empty */

	/* Disable DMA */
	writeb (0, card->bridge_addr + PLX_DMACSR0);

	return;
}

/**
 * dvbm_tx_start_tx_dma - start transmit DMA
 * @iface: interface
 **/
static void
dvbm_tx_start_tx_dma (struct master_iface *iface)
{
	struct master_dev *card = iface->card;

	writel (plx_head_desc_bus_addr (iface->dma) |
		PLX_DMADPR_DLOC_PCI,
		card->bridge_addr + PLX_DMADPR0);
	clear_bit (0, &iface->dma_done);
	wmb ();
	writeb (PLX_DMACSR_ENABLE | PLX_DMACSR_START,
		card->bridge_addr + PLX_DMACSR0);
	/* Dummy read to flush PCI posted writes */
	readb (card->bridge_addr + PLX_DMACSR0);
	return;
}

/**
 * dvbm_tx_unlocked_ioctl - DVB Master Send unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
dvbm_tx_unlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	struct asi_txstuffing stuffing;

	switch (cmd) {
	case ASI_IOC_TXSETSTUFFING:
		if (copy_from_user (&stuffing,
			(struct asi_txstuffing __user *)arg,
			sizeof (stuffing))) {
			return -EFAULT;
		}
		if ((stuffing.ib > 0x00ff) ||
			(stuffing.ip > 0xffffff)) {
			return -EINVAL;
		}
		if (iface->capabilities & ASI_CAP_TX_FINETUNING) {
			if ((stuffing.normal_ip > 0xff) ||
				(stuffing.big_ip > 0xff)) {
				return -EINVAL;
			}
			spin_lock (&card->reg_lock);
			master_outl (card, DVBM_TX_FINETUNE,
				(stuffing.big_ip << 8) | stuffing.normal_ip);
		} else {
			spin_lock (&card->reg_lock);
		}
		master_outl (card, DVBM_TX_STUFFING,
			(stuffing.ip << 8) | stuffing.ib);
		spin_unlock (&card->reg_lock);
		break;
	case ASI_IOC_TXGETBYTECOUNT:
		if (!(iface->capabilities & ASI_CAP_TX_BYTECOUNTER) ||
			iface->count27) {
			return -ENOTTY;
		}
		if (put_user (master_inl (card, DVBM_TX_COUNTR),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_TXGET27COUNT:
		if (!(iface->count27)) {
			return -ENOTTY;
		}
		if (put_user (master_inl (card, DVBM_TX_COUNTR),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return asi_txioctl (filp, cmd, arg);
	}
	return 0;
}

/**
 * dvbm_tx_fsync - DVB Master Send fsync() method
 * @filp: file to flush
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
FSYNC_HANDLER(dvbm_tx_fsync,filp,datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dma *dma = iface->dma;

	mutex_lock (&iface->buf_mutex);
	plx_tx_link_all (dma);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	plx_reset (dma);

	/* The onboard FIFOs may not be empty yet,
	 * but there's nothing we can do about it */

	mutex_unlock (&iface->buf_mutex);
	return 0;
}

