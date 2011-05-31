/* dvbm_fd.c
 *
 * Linux driver functions for Linear Systems Ltd. DVB Master FD.
 *
 * Copyright (C) 2001-2010 Linear Systems Ltd.
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

#include <asm/uaccess.h> /* put_user () */
#include <asm/bitops.h> /* set_bit () */

#include "asicore.h"
#include "../include/master.h"
#include "mdev.h"
#include "mdma.h"
#include "dvbm.h"
#include "plx9080.h"
#include "dvbm_fd.h"

static const char dvbm_fd_name[] = DVBM_NAME_FD;

/* Static function prototypes */
static irqreturn_t IRQ_HANDLER(dvbm_fd_irq_handler,irq,dev_id,regs);
static void dvbm_fd_txinit (struct master_iface *iface);
static void dvbm_fd_txstart (struct master_iface *iface);
static void dvbm_fd_txstop (struct master_iface *iface);
static void dvbm_fd_start_tx_dma (struct master_iface *iface);
static long dvbm_fd_txunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int FSYNC_HANDLER(dvbm_fd_txfsync,filp,datasync);
static void dvbm_fd_rxinit (struct master_iface *iface);
static void dvbm_fd_rxstart (struct master_iface *iface);
static void dvbm_fd_rxstop (struct master_iface *iface);
static long dvbm_fd_rxunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int FSYNC_HANDLER(dvbm_fd_rxfsync,filp,datasync);

static struct file_operations dvbm_fd_txfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.write = asi_write,
	.poll = asi_txpoll,
	.unlocked_ioctl = dvbm_fd_txunlocked_ioctl,
	.compat_ioctl = dvbm_fd_txunlocked_ioctl,
	.open = asi_open,
	.release = asi_release,
	.fsync = dvbm_fd_txfsync,
	.fasync = NULL
};

static struct file_operations dvbm_fd_rxfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.read = asi_read,
	.poll = asi_rxpoll,
	.unlocked_ioctl = dvbm_fd_rxunlocked_ioctl,
	.compat_ioctl = dvbm_fd_rxunlocked_ioctl,
	.open = asi_open,
	.release = asi_release,
	.fsync = dvbm_fd_rxfsync,
	.fasync = NULL
};

static struct master_iface_operations dvbm_fd_txops = {
	.init = dvbm_fd_txinit,
	.start = dvbm_fd_txstart,
	.stop = dvbm_fd_txstop,
	.exit = NULL,
	.start_tx_dma = dvbm_fd_start_tx_dma
};

static struct master_iface_operations dvbm_fd_rxops = {
	.init = dvbm_fd_rxinit,
	.start = dvbm_fd_rxstart,
	.stop = dvbm_fd_rxstop,
	.exit = NULL
};

/**
 * dvbm_fd_pci_probe - PCI insertion handler for a DVB Master FD
 * @pdev: PCI device
 *
 * Handle the insertion of a DVB Master FD.
 * Returns a negative error code on failure and 0 on success.
 **/
int __devinit
dvbm_fd_pci_probe (struct pci_dev *pdev)
{
	int err;
	unsigned int version, cap;
	struct master_dev *card;

	err = dvbm_pci_probe_generic (pdev);
	if (err < 0) {
		goto NO_PCI;
	}

	/* Initialize the driver_data pointer so that dvbm_fd_pci_remove()
	 * doesn't try to free it if an error occurs */
	pci_set_drvdata (pdev, NULL);

	/* Print the firmware version */
	version = inl (pci_resource_start (pdev, 2) + DVBM_FD_CSR) >> 24;
	printk (KERN_INFO "%s: %s detected, firmware version %u (0x%02X)\n",
		dvbm_driver_name, dvbm_fd_name, version, version);

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
	card->name = dvbm_fd_name;
	card->id = pdev->device;
	card->irq = pdev->irq;
	card->irq_handler = dvbm_fd_irq_handler;
	INIT_LIST_HEAD(&card->iface_list);
	/* Lock for ICSR */
	spin_lock_init (&card->irq_lock);
	/* Lock for CSR, STR, FTR, DMATLR, PFLUT */
	spin_lock_init (&card->reg_lock);
	mutex_init (&card->users_mutex);
	card->parent = &pdev->dev;

	/* Store the pointer to the board info structure
	 * in the PCI info structure */
	pci_set_drvdata (pdev, card);

	/* Reset the FPGA */
	master_outl (card, DVBM_FD_CSR, DVBM_FD_CSR_RXRST | DVBM_FD_CSR_TXRST);

	/* Reset the PCI 9080 */
	plx_reset_bridge (card->bridge_addr);

	/* Setup the PCI 9080 */
	writel (PLX_INTCSR_PCIINT_ENABLE |
		PLX_INTCSR_PCILOCINT_ENABLE |
		PLX_INTCSR_DMA0INT_ENABLE |
		PLX_INTCSR_DMA1INT_ENABLE,
		card->bridge_addr + PLX_INTCSR);
	writel (PLX_DMAMODE_32BIT | PLX_DMAMODE_READY |
		PLX_DMAMODE_LOCALBURST | PLX_DMAMODE_CHAINED |
		PLX_DMAMODE_INT | PLX_DMAMODE_CLOC |
		PLX_DMAMODE_DEMAND | PLX_DMAMODE_INTPCI,
		card->bridge_addr + PLX_DMAMODE0);
	writel (PLX_DMAMODE_32BIT | PLX_DMAMODE_READY |
		PLX_DMAMODE_LOCALBURST | PLX_DMAMODE_CHAINED |
		PLX_DMAMODE_INT | PLX_DMAMODE_CLOC |
		PLX_DMAMODE_DEMAND | PLX_DMAMODE_INTPCI,
		card->bridge_addr + PLX_DMAMODE1);
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + PLX_INTCSR);

	master_outl (card, DVBM_FD_CSR, 0);
	master_outl (card, DVBM_FD_DMATLR,
		(DVBM_FD_RDMATL << 16) | DVBM_FD_TDMATL);
	master_outl (card, DVBM_FD_TFSLR, DVBM_FD_TFSL);

	/* Register a DVB Master device */
	if ((err = dvbm_register (card)) < 0) {
		goto NO_DEV;
	}

	/* Register a transmit interface */
	cap = ASI_CAP_TX_MAKE204 | ASI_CAP_TX_FINETUNING |
		ASI_CAP_TX_BYTECOUNTER |
		ASI_CAP_TX_SETCLKSRC | ASI_CAP_TX_FIFOUNDERRUN |
		ASI_CAP_TX_DATA | ASI_CAP_TX_RXCLKSRC;
	if ((version >> 4) == 0) {
		if ((version & 0x0f) >= 2) {
			cap |= ASI_CAP_TX_LARGEIB |
				ASI_CAP_TX_27COUNTER;
		}
		if ((version & 0x0f) >= 5) {
			cap |= ASI_CAP_TX_INTERLEAVING |
				ASI_CAP_TX_TIMESTAMPS;
		}
	}
	if ((err = asi_register_iface (card,
		&plx_dma_ops,
		DVBM_FD_FIFO,
		MASTER_DIRECTION_TX,
		&dvbm_fd_txfops,
		&dvbm_fd_txops,
		cap,
		4,
		ASI_CTL_TRANSPORT_DVB_ASI)) < 0) {
		goto NO_IFACE;
	}

	/* Register a receive interface */
	cap = ASI_CAP_RX_SYNC | ASI_CAP_RX_MAKE188 |
		ASI_CAP_RX_BYTECOUNTER |
		ASI_CAP_RX_CD |
		ASI_CAP_RX_DSYNC | ASI_CAP_RX_DATA |
		ASI_CAP_RX_PIDCOUNTER;
	switch (version >> 4) {
	case 0:
		cap |= ASI_CAP_RX_PIDFILTER;
		if ((version & 0x0f) >= 2) {
			cap |= ASI_CAP_RX_27COUNTER;
		}
		if ((version & 0x0f) >= 5) {
			cap |= ASI_CAP_RX_4PIDCOUNTER |
				ASI_CAP_RX_TIMESTAMPS;
		}
		break;
	case 1:
		cap |= ASI_CAP_RX_4PIDCOUNTER;
		break;
	default:
		break;
	}
	if ((err = asi_register_iface (card,
		&plx_dma_ops,
		DVBM_FD_FIFO,
		MASTER_DIRECTION_RX,
		&dvbm_fd_rxfops,
		&dvbm_fd_rxops,
		cap,
		4,
		ASI_CTL_TRANSPORT_DVB_ASI)) < 0) {
		goto NO_IFACE;
	}

	return 0;

NO_IFACE:
NO_DEV:
NO_MEM:
	dvbm_fd_pci_remove (pdev);
NO_PCI:
	return err;
}

/**
 * dvbm_fd_pci_remove - PCI removal handler for a DVB Master FD
 * @pdev: PCI device
 *
 * Handle the removal of a DVB Master FD.
 * This function may be called during PCI probe error handling,
 * so don't mark it as __devexit.
 **/
void
dvbm_fd_pci_remove (struct pci_dev *pdev)
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
 * dvbm_fd_irq_handler - DVB Master FD interrupt service routine
 * @irq: interrupt number
 * @dev_id: pointer to the device data structure
 * @regs: processor context
 **/
static irqreturn_t
IRQ_HANDLER(dvbm_fd_irq_handler,irq,dev_id,regs)
{
	struct master_dev *card = dev_id;
	unsigned int intcsr = readl (card->bridge_addr + PLX_INTCSR);
	unsigned int status, interrupting_iface = 0;
	struct master_iface *txiface = list_entry (card->iface_list.next,
		struct master_iface, list);
	struct master_iface *rxiface = list_entry (card->iface_list.prev,
		struct master_iface, list);

	if (intcsr & PLX_INTCSR_DMA0INT_ACTIVE) {
		/* Read the interrupt type and clear it */
		status = readb (card->bridge_addr + PLX_DMACSR0);
		writeb (PLX_DMACSR_ENABLE | PLX_DMACSR_CLINT,
			card->bridge_addr + PLX_DMACSR0);

		/* Increment the buffer pointer */
		mdma_advance (txiface->dma);

		/* Flag end-of-chain */
		if (status & PLX_DMACSR_DONE) {
			set_bit (ASI_EVENT_TX_BUFFER_ORDER, &txiface->events);
			set_bit (0, &txiface->dma_done);
		}

		interrupting_iface |= 0x1;
	}
	if (intcsr & PLX_INTCSR_DMA1INT_ACTIVE) {
		struct master_dma *dma = rxiface->dma;

		/* Read the interrupt type and clear it */
		status = readb (card->bridge_addr + PLX_DMACSR1);
		writeb (PLX_DMACSR_ENABLE | PLX_DMACSR_CLINT,
			card->bridge_addr + PLX_DMACSR1);

		/* Increment the buffer pointer */
		mdma_advance (dma);

		if (mdma_rx_isempty (dma)) {
			set_bit (ASI_EVENT_RX_BUFFER_ORDER, &rxiface->events);
		}

		/* Flag end-of-chain */
		if (status & PLX_DMACSR_DONE) {
			set_bit (0, &rxiface->dma_done);
		}

		interrupting_iface |= 0x2;
	}
	if (intcsr & PLX_INTCSR_PCILOCINT_ACTIVE) {
		/* Clear the source of the interrupt */
		spin_lock (&card->irq_lock);
		status = master_inl (card, DVBM_FD_ICSR);
		master_outl (card, DVBM_FD_ICSR, status);
		spin_unlock (&card->irq_lock);

		if (status & DVBM_FD_ICSR_TXUIS) {
			set_bit (ASI_EVENT_TX_FIFO_ORDER,
				&txiface->events);
			interrupting_iface |= 0x1;
		}
		if (status & DVBM_FD_ICSR_TXDIS) {
			set_bit (ASI_EVENT_TX_DATA_ORDER,
				&txiface->events);
			interrupting_iface |= 0x1;
		}
		if (status & DVBM_FD_ICSR_RXCDIS) {
			set_bit (ASI_EVENT_RX_CARRIER_ORDER,
				&rxiface->events);
			interrupting_iface |= 0x2;
		}
		if (status & DVBM_FD_ICSR_RXAOSIS) {
			set_bit (ASI_EVENT_RX_AOS_ORDER,
				&rxiface->events);
			interrupting_iface |= 0x2;
		}
		if (status & DVBM_FD_ICSR_RXLOSIS) {
			set_bit (ASI_EVENT_RX_LOS_ORDER,
				&rxiface->events);
			interrupting_iface |= 0x2;
		}
		if (status & DVBM_FD_ICSR_RXOIS) {
			set_bit (ASI_EVENT_RX_FIFO_ORDER,
				&rxiface->events);
			interrupting_iface |= 0x2;
		}
		if (status & DVBM_FD_ICSR_RXDIS) {
			set_bit (ASI_EVENT_RX_DATA_ORDER,
				&rxiface->events);
			interrupting_iface |= 0x2;
		}
	}

	if (interrupting_iface) {
		/* Dummy read to flush PCI posted writes */
		readb (card->bridge_addr + PLX_DMACSR1);

		if (interrupting_iface & 0x1) {
			wake_up (&txiface->queue);
		}
		if (interrupting_iface & 0x2) {
			wake_up (&rxiface->queue);
		}
		return IRQ_HANDLED;
	}
	return IRQ_NONE;
}

/**
 * dvbm_fd_txinit - Initialize the DVB Master FD transmitter
 * @iface: interface
 **/
static void
dvbm_fd_txinit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;

	spin_lock (&card->reg_lock);
	reg = master_inl (card, DVBM_FD_CSR) & DVBM_FD_CSR_RXMASK;
	switch (iface->mode) {
	default:
	case ASI_CTL_TX_MODE_188:
		reg |= 0;
		break;
	case ASI_CTL_TX_MODE_204:
		reg |= DVBM_FD_CSR_TX204;
		break;
	case ASI_CTL_TX_MODE_MAKE204:
		reg |= DVBM_FD_CSR_TXMAKE204;
		break;
	}
	switch (iface->clksrc) {
	default:
	case ASI_CTL_TX_CLKSRC_ONBOARD:
		reg |= 0;
		break;
	case ASI_CTL_TX_CLKSRC_EXT:
		reg |= DVBM_FD_CSR_TXEXTCLK;
		break;
	case ASI_CTL_TX_CLKSRC_RX:
		reg |= DVBM_FD_CSR_TXRXCLK;
		break;
	}
	master_outl (card, DVBM_FD_CSR, reg | DVBM_FD_CSR_TXRST);
	wmb ();
	master_outl (card, DVBM_FD_CSR, reg);
	wmb ();
	reg = master_inl (card, DVBM_FD_DMATLR) &
		~DVBM_FD_DMATLR_TXMASK;
	master_outl (card, DVBM_FD_DMATLR,
		reg | DVBM_FD_TDMATL |
		(iface->timestamps ? DVBM_FD_DMATLR_TXTSS : 0));
	spin_unlock (&card->reg_lock);
	/* There will be no races on STR and FTR until
	 * this code returns, so we don't need to lock them */
	master_outl (card, DVBM_FD_STR, 0);
	master_outl (card, DVBM_FD_FTR, 0);

	/* Reset byte counter */
	master_inl (card, DVBM_FD_TXBCOUNTR);

	return;
}

/**
 * dvbm_fd_txstart - Activate the DVB Master FD transmitter
 * @iface: interface
 **/
static void
dvbm_fd_txstart (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;

	/* Enable DMA */
	writeb (PLX_DMACSR_ENABLE, card->bridge_addr + PLX_DMACSR0);

	/* Enable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = master_inl (card, DVBM_FD_ICSR) &
		DVBM_FD_ICSR_RXCTRLMASK;
	reg |= DVBM_FD_ICSR_TXUIE | DVBM_FD_ICSR_TXDIE;
	master_outl (card, DVBM_FD_ICSR, reg);
	spin_unlock_irq (&card->irq_lock);

	/* Enable the transmitter */
	spin_lock (&card->reg_lock);
	reg = master_inl (card, DVBM_FD_CSR);
	master_outl (card, DVBM_FD_CSR, reg | DVBM_FD_CSR_TXE);
	spin_unlock (&card->reg_lock);

	return;
}

/**
 * dvbm_fd_txstop - Deactivate the DVB Master FD transmitter
 * @iface: interface
 **/
static void
dvbm_fd_txstop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	struct master_dma *dma = iface->dma;
	unsigned int reg;

	plx_tx_link_all (dma);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	plx_reset (dma);

	/* Wait for the onboard FIFOs to empty */
	/* Atomic read of ICSR, so we don't need to lock */
	wait_event (iface->queue,
		!(master_inl (card, DVBM_FD_ICSR) & DVBM_FD_ICSR_TXD));

	/* Disable the transmitter */
	spin_lock (&card->reg_lock);
	reg = master_inl (card, DVBM_FD_CSR);
	master_outl (card, DVBM_FD_CSR, reg & ~DVBM_FD_CSR_TXE);
	spin_unlock (&card->reg_lock);

	/* Disable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = master_inl (card, DVBM_FD_ICSR) &
		DVBM_FD_ICSR_RXCTRLMASK;
	reg |= DVBM_FD_ICSR_TXUIS | DVBM_FD_ICSR_TXDIS;
	master_outl (card, DVBM_FD_ICSR, reg);
	spin_unlock_irq (&card->irq_lock);

	/* Disable DMA */
	writeb (0, card->bridge_addr + PLX_DMACSR0);

	return;
}

/**
 * dvbm_fd_start_tx_dma - start transmit DMA
 * @iface: interface
 **/
static void
dvbm_fd_start_tx_dma (struct master_iface *iface)
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
 * dvbm_fd_txunlocked_ioctl - DVB Master FD transmitter unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
dvbm_fd_txunlocked_ioctl (struct file *filp,
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
		if ((stuffing.ib >
			((iface->capabilities & ASI_CAP_TX_LARGEIB) ?
			0xffff : 0x00ff)) ||
			(stuffing.ip > 0xffffff) ||
			(stuffing.normal_ip > 0xff) ||
			(stuffing.big_ip > 0xff) ||
			((stuffing.il_normal + stuffing.il_big) > 0xf) ||
			(stuffing.il_normal > stuffing.normal_ip) ||
			(stuffing.il_big > stuffing.big_ip)) {
			return -EINVAL;
		}
		spin_lock (&card->reg_lock);
		if (iface->capabilities & ASI_CAP_TX_LARGEIB) {
			master_outl (card, DVBM_FD_CSR,
				(master_inl (card, DVBM_FD_CSR) &
				~DVBM_FD_CSR_TXLARGEIBMASK) |
				((stuffing.ib << 8) &
				DVBM_FD_CSR_TXLARGEIBMASK));
		}
		master_outl (card, DVBM_FD_STR,
			(stuffing.ip << 8) | (stuffing.ib & 0x00ff));
		master_outl (card, DVBM_FD_FTR,
			(stuffing.il_big << 24) |
			(stuffing.big_ip << 16) |
			(stuffing.il_normal << 8) |
			stuffing.normal_ip);
		spin_unlock (&card->reg_lock);
		break;
	case ASI_IOC_TXGETBYTECOUNT:
		if (put_user (master_inl (card, DVBM_FD_TXBCOUNTR),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_TXGETTXD:
		/* Atomic read of ICSR, so we don't need to lock */
		if (put_user ((master_inl (card, DVBM_FD_ICSR) &
			DVBM_FD_ICSR_TXD) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_TXGET27COUNT:
		if (!(iface->capabilities & ASI_CAP_TX_27COUNTER)) {
			return -ENOTTY;
		}
		if (put_user (master_inl (card, DVBM_FD_27COUNTR),
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
 * dvbm_fd_txfsync - DVB Master FD transmitter fsync() method
 * @filp: file to flush
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
FSYNC_HANDLER(dvbm_fd_txfsync,filp,datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	struct master_dma *dma = iface->dma;

	mutex_lock (&iface->buf_mutex);
	plx_tx_link_all (dma);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	plx_reset (dma);

	/* Wait for the onboard FIFOs to empty */
	/* Atomic read of ICSR, so we don't need to lock */
	wait_event (iface->queue,
		!(master_inl (card, DVBM_FD_ICSR) & DVBM_FD_ICSR_TXD));

	mutex_unlock (&iface->buf_mutex);
	return 0;
}

/**
 * dvbm_fd_rxinit - Initialize the DVB Master FD receiver
 * @iface: interface
 **/
static void
dvbm_fd_rxinit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;

	spin_lock (&card->reg_lock);
	reg = master_inl (card, DVBM_FD_CSR) & DVBM_FD_CSR_TXMASK;
	switch (iface->mode) {
	default:
	case ASI_CTL_RX_MODE_RAW:
		reg |= 0;
		break;
	case ASI_CTL_RX_MODE_188:
		reg |= DVBM_FD_CSR_RX188 | DVBM_FD_CSR_RXPFE;
		break;
	case ASI_CTL_RX_MODE_204:
		reg |= DVBM_FD_CSR_RX204 | DVBM_FD_CSR_RXPFE;
		break;
	case ASI_CTL_RX_MODE_AUTO:
		reg |= DVBM_FD_CSR_RXAUTO | DVBM_FD_CSR_RXPFE;
		break;
	case ASI_CTL_RX_MODE_AUTOMAKE188:
		reg |= DVBM_FD_CSR_RXAUTOMAKE188 | DVBM_FD_CSR_RXPFE;
		break;
	case ASI_CTL_RX_MODE_204MAKE188:
		reg |= DVBM_FD_CSR_RX204MAKE188 | DVBM_FD_CSR_RXPFE;
		break;
	}
	reg |= DVBM_FD_CSR_RXDSYNC | DVBM_FD_CSR_RXRF;
	master_outl (card, DVBM_FD_CSR, reg | DVBM_FD_CSR_RXRST);
	wmb ();
	master_outl (card, DVBM_FD_CSR, reg);
	wmb ();
	reg = master_inl (card, DVBM_FD_DMATLR) &
		~DVBM_FD_DMATLR_RXMASK;
	master_outl (card, DVBM_FD_DMATLR,
		reg | (DVBM_FD_RDMATL << 16) |
		(iface->timestamps ? DVBM_FD_DMATLR_RXTSE : 0));
	spin_unlock (&card->reg_lock);

	/* Reset byte counter */
	master_inl (card, DVBM_FD_RXBCOUNTR);

	/* Reset PID filter */
	if (iface->capabilities & ASI_CAP_RX_PIDFILTER) {
		unsigned int i;

		/* There will be no races on PFLUT until
		 * this code returns, so we don't need to lock it */
		for (i = 0; i < 256; i++) {
			master_outl (card, DVBM_FD_PFLUTAR, i);
			wmb ();
			master_outl (card, DVBM_FD_PFLUTR,
				0xffffffff);
			wmb ();
		}
	}

	/* Clear PID register 0 */
	master_outl (card, DVBM_FD_PIDR0, 0);

	/* Reset PID counter 0 */
	master_inl (card, DVBM_FD_PIDCOUNTR0);

	if (iface->capabilities & ASI_CAP_RX_4PIDCOUNTER) {
		/* Clear PID registers 1-3 */
		master_outl (card, DVBM_FD_PIDR1, 0);
		master_outl (card, DVBM_FD_PIDR2, 0);
		master_outl (card, DVBM_FD_PIDR3, 0);

		/* Reset PID counters 1-3 */
		master_inl (card, DVBM_FD_PIDCOUNTR1);
		master_inl (card, DVBM_FD_PIDCOUNTR2);
		master_inl (card, DVBM_FD_PIDCOUNTR3);
	}
	return;
}

/**
 * dvbm_fd_rxstart - Activate the DVB Master FD receiver
 * @iface: interface
 **/
static void
dvbm_fd_rxstart (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;

	/* Enable and start DMA.
	 * According to PLX Design Note 7 for the PCI 9080,
	 * there is a race between the enable and start bits
	 * so enable must be set first */
	writel (plx_head_desc_bus_addr (iface->dma) |
		PLX_DMADPR_DLOC_PCI | PLX_DMADPR_LB2PCI,
		card->bridge_addr + PLX_DMADPR1);
	writeb (PLX_DMACSR_ENABLE, card->bridge_addr + PLX_DMACSR1);
	clear_bit (0, &iface->dma_done);
	wmb ();
	writeb (PLX_DMACSR_ENABLE | PLX_DMACSR_START,
		card->bridge_addr + PLX_DMACSR1);
	/* Dummy read to flush PCI posted writes */
	readb (card->bridge_addr + PLX_DMACSR1);

	/* Enable receiver interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = master_inl (card, DVBM_FD_ICSR) &
		DVBM_FD_ICSR_TXCTRLMASK;
	reg |= DVBM_FD_ICSR_RXCDIE | DVBM_FD_ICSR_RXAOSIE |
		DVBM_FD_ICSR_RXLOSIE | DVBM_FD_ICSR_RXOIE |
		DVBM_FD_ICSR_RXDIE;
	master_outl (card, DVBM_FD_ICSR, reg);
	spin_unlock_irq (&card->irq_lock);

	/* Enable the receiver */
	spin_lock (&card->reg_lock);
	reg = master_inl (card, DVBM_FD_CSR);
	master_outl (card, DVBM_FD_CSR, reg | DVBM_FD_CSR_RXE);
	spin_unlock (&card->reg_lock);

	return;
}

/**
 * dvbm_fd_rxstop - Deactivate the DVB Master FD receiver
 * @iface: interface
 **/
static void
dvbm_fd_rxstop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;

	/* Disable the receiver */
	spin_lock (&card->reg_lock);
	reg = master_inl (card, DVBM_FD_CSR);
	master_outl (card, DVBM_FD_CSR, reg & ~DVBM_FD_CSR_RXE);
	spin_unlock (&card->reg_lock);

	/* Disable receiver interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = master_inl (card, DVBM_FD_ICSR) &
		DVBM_FD_ICSR_TXCTRLMASK;
	reg |= DVBM_FD_ICSR_RXCDIS | DVBM_FD_ICSR_RXAOSIS |
		DVBM_FD_ICSR_RXLOSIS | DVBM_FD_ICSR_RXOIS |
		DVBM_FD_ICSR_RXDIS;
	master_outl (card, DVBM_FD_ICSR, reg);
	spin_unlock_irq (&card->irq_lock);

	/* Abort and disable DMA.
	 * Despite PLX's documentation,
	 * do not disable DMA before aborting;
	 * it resets STR (bug in 9080?) */
	writeb (PLX_DMACSR_ENABLE | PLX_DMACSR_ABORT,
		card->bridge_addr + PLX_DMACSR1);
	/* Dummy read to flush PCI posted writes */
	readb (card->bridge_addr + PLX_DMACSR1);
	wait_event_timeout (iface->queue, test_bit (0, &iface->dma_done), HZ);
	writeb (0, card->bridge_addr + PLX_DMACSR1);
	return;
}

/**
 * dvbm_fd_rxunlocked_ioctl - DVB Master FD receiver unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
dvbm_fd_rxunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	int val;
	unsigned int reg = 0, *pflut, i;

	switch (cmd) {
	case ASI_IOC_RXGETSTATUS:
		/* Atomic reads of CSR and ICSR, so we don't need to lock */
		reg = master_inl (card, DVBM_FD_ICSR);
		switch (master_inl (card, DVBM_FD_CSR) &
			DVBM_FD_CSR_RXSYNCMASK) {
		case 0:
			val = 1;
			break;
		case DVBM_FD_CSR_RX188:
			val = (reg & DVBM_FD_ICSR_RXPASSING) ? 188 : 0;
			break;
		case DVBM_FD_CSR_RX204:
		case DVBM_FD_CSR_RX204MAKE188:
			val = (reg & DVBM_FD_ICSR_RXPASSING) ? 204 : 0;
			break;
		case DVBM_FD_CSR_RXAUTO:
		case DVBM_FD_CSR_RXAUTOMAKE188:
			if (reg & DVBM_FD_ICSR_RXPASSING) {
				val = (reg & DVBM_FD_ICSR_RX204) ? 204 : 188;
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
		if (put_user (master_inl (card, DVBM_FD_RXBCOUNTR),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETINVSYNC:
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
	case ASI_IOC_RXGETCARRIER:
		/* Atomic read of ICSR, so we don't need to lock */
		if (put_user ((master_inl (card, DVBM_FD_ICSR) &
			DVBM_FD_ICSR_RXCD) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETDSYNC:
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		switch (val) {
		case 0:
			reg |= 0;
			break;
		case 1:
			reg |= DVBM_FD_CSR_RXDSYNC;
			break;
		default:
			return -EINVAL;
		}
		spin_lock (&card->reg_lock);
		master_outl (card, DVBM_FD_CSR,
			(master_inl (card, DVBM_FD_CSR) &
			~DVBM_FD_CSR_RXDSYNC) | reg);
		spin_unlock (&card->reg_lock);
		break;
	case ASI_IOC_RXGETRXD:
		/* Atomic read of ICSR, so we don't need to lock */
		if (put_user ((master_inl (card, DVBM_FD_ICSR) &
			DVBM_FD_ICSR_RXD) ? 1 : 0, (int __user *)arg)) {
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
			master_outl (card, DVBM_FD_PFLUTAR, i);
			wmb ();
			master_outl (card, DVBM_FD_PFLUTR, pflut[i]);
			wmb ();
		}
		spin_unlock (&card->reg_lock);
		kfree (pflut);
		break;
	case ASI_IOC_RXSETPID0:
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		master_outl (card, DVBM_FD_PIDR0, val);
		/* Reset PID count */
		master_inl (card, DVBM_FD_PIDCOUNTR0);
		break;
	case ASI_IOC_RXGETPID0COUNT:
		if (put_user (master_inl (card, DVBM_FD_PIDCOUNTR0),
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
		master_outl (card, DVBM_FD_PIDR1, val);
		/* Reset PID count */
		master_inl (card, DVBM_FD_PIDCOUNTR1);
		break;
	case ASI_IOC_RXGETPID1COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (master_inl (card, DVBM_FD_PIDCOUNTR1),
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
		master_outl (card, DVBM_FD_PIDR2, val);
		/* Reset PID count */
		master_inl (card, DVBM_FD_PIDCOUNTR2);
		break;
	case ASI_IOC_RXGETPID2COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (master_inl (card, DVBM_FD_PIDCOUNTR2),
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
		master_outl (card, DVBM_FD_PIDR3, val);
		/* Reset PID count */
		master_inl (card, DVBM_FD_PIDCOUNTR3);
		break;
	case ASI_IOC_RXGETPID3COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (master_inl (card, DVBM_FD_PIDCOUNTR3),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXGET27COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_27COUNTER)) {
			return -ENOTTY;
		}
		if (put_user (master_inl (card, DVBM_FD_27COUNTR),
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
 * dvbm_fd_rxfsync - DVB Master FD receiver fsync() method
 * @filp: file to flush
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
FSYNC_HANDLER(dvbm_fd_rxfsync,filp,datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	unsigned int reg;

	mutex_lock (&iface->buf_mutex);

	/* Stop the receiver */
	dvbm_fd_rxstop (iface);

	/* Reset the onboard FIFO and driver buffers */
	spin_lock (&card->reg_lock);
	reg = master_inl (card, DVBM_FD_CSR);
	master_outl (card, DVBM_FD_CSR, reg | DVBM_FD_CSR_RXRST);
	wmb ();
	master_outl (card, DVBM_FD_CSR, reg);
	spin_unlock (&card->reg_lock);
	iface->events = 0;
	plx_reset (iface->dma);

	/* Start the receiver */
	dvbm_fd_rxstart (iface);

	mutex_unlock (&iface->buf_mutex);
	return 0;
}

