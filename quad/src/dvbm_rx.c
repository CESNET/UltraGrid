/* dvbm_rx.c
 *
 * Linux driver functions for Linear Systems Ltd. DVB Master Receive.
 *
 * Copyright (C) 1999 Tony Bolger <d7v@indigo.ie>
 * Copyright (C) 2000-2006 Linear Systems Ltd.
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
#include <linux/pci.h> /* pci_dev */
#include <linux/slab.h> /* kmalloc () */
#include <linux/list.h> /* INIT_LIST_HEAD () */
#include <linux/spinlock.h> /* spin_lock_init () */
#include <linux/init.h> /* __devinit */
#include <linux/errno.h> /* error codes */
#include <linux/interrupt.h> /* irqreturn_t */

#include <asm/semaphore.h> /* sema_init () */
#include <asm/uaccess.h> /* put_user () */
#include <asm/bitops.h> /* set_bit () */

#include "asicore.h"
#include "../include/master.h"
#include "miface.h"
#include "mdev.h"
#include "dvbm.h"
#include "plx9080.h"
#include "masterplx.h"
#include "dvbm_rx.h"

static const char dvbm_rx_name[] = DVBM_NAME_RX;

/* Static function prototypes */
static irqreturn_t IRQ_HANDLER(dvbm_rx_irq_handler,irq,dev_id,regs);
static void dvbm_rx_init (struct master_iface *iface);
static void dvbm_rx_start (struct master_iface *iface);
static void dvbm_rx_stop (struct master_iface *iface);
static int dvbm_rx_open (struct inode *inode, struct file *filp);
static long dvbm_rx_unlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int dvbm_rx_ioctl (struct inode *inode,
	struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int dvbm_rx_fsync (struct file *filp,
	struct dentry *dentry,
	int datasync);
static int dvbm_rx_release (struct inode *inode, struct file *filp);

static struct file_operations dvbm_rx_fops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.read = masterplx_read,
	.poll = masterplx_rxpoll,
	.ioctl = dvbm_rx_ioctl,
#ifdef HAVE_UNLOCKED_IOCTL
	.unlocked_ioctl = dvbm_rx_unlocked_ioctl,
#endif
#ifdef HAVE_COMPAT_IOCTL
	.compat_ioctl = asi_compat_ioctl,
#endif
	.open = dvbm_rx_open,
	.release = dvbm_rx_release,
	.fsync = dvbm_rx_fsync,
	.fasync = NULL
};

/**
 * dvbm_rx_pci_probe - PCI insertion handler for a DVB Master Receive
 * @dev: PCI device
 *
 * Handle the insertion of a DVB Master Receive.
 * Returns a negative error code on failure and 0 on success.
 **/
int __devinit
dvbm_rx_pci_probe (struct pci_dev *dev)
{
	int err;
	unsigned int version, cap;
	struct master_dev *card;

	/* Print the firmware version.
	 * Firmware versions 3 and earlier do not have
	 * a CFG2/STATUS2 register, and read STATUS instead.
	 * Clearing the register before reading it should
	 * produce a version number of 3 or less on these boards.
	 * This version number must be stored, since it cannot
	 * be read again without first clearing the register. */
	outl (0, pci_resource_start (dev, 2) + DVBM_RX_CFG2);
	version = inl (pci_resource_start (dev, 2) + DVBM_RX_STATUS2) >> 24;
	if (version < 4) {
		version = 3;
		printk (KERN_INFO "%s: %s detected, "
			"unknown firmware version "
			"(probably 0x03)\n",
			dvbm_driver_name, dvbm_rx_name);
	} else {
		printk (KERN_INFO "%s: %s detected, "
			"firmware version %u (0x%02X)\n",
			dvbm_driver_name, dvbm_rx_name, version, version);
	}

	/* Allocate a board info structure */
	if ((card = (struct master_dev *)
		kmalloc (sizeof (*card), GFP_KERNEL)) == NULL) {
		err = -ENOMEM;
		goto NO_MEM;
	}

	/* Initialize the board info structure */
	memset (card, 0, sizeof (*card));
	card->bridge_addr = ioremap_nocache (pci_resource_start (dev, 0),
		pci_resource_len (dev, 0));
	card->core.port = pci_resource_start (dev, 2);
	card->version = version << 8;
	card->name = dvbm_rx_name;
	card->irq_handler = dvbm_rx_irq_handler;
	INIT_LIST_HEAD(&card->iface_list);
	/* Lock for CSR, CSR2 */
	spin_lock_init (&card->irq_lock);
	spin_lock_init (&card->reg_lock); /* Unused */
	sema_init (&card->users_sem, 1);
	card->pdev = dev;

	/* Store the pointer to the board info structure
	 * in the PCI info structure */
	pci_set_drvdata (dev, card);

	/* Register a Master device */
	if ((err = mdev_register (card,
		&dvbm_card_list,
		dvbm_driver_name,
		&dvbm_class)) < 0) {
		goto NO_DEV;
	}

	/* Register a receive interface */
	cap = 0;
	if (version >= 4) {
		cap |= ASI_CAP_RX_SYNC |
			ASI_CAP_RX_BYTECOUNTER;
	}
	if (version >= 5) {
		cap |= ASI_CAP_RX_INVSYNC | ASI_CAP_RX_CD;
	}
	if (version >= 6) {
		cap |= ASI_CAP_RX_DSYNC;
	}
	if (version >= 7) {
		cap |= ASI_CAP_RX_FORCEDMA |
			ASI_CAP_RX_27COUNTER | ASI_CAP_RX_BYTESOR27;
	}
	if ((err = asi_register_iface (card,
		MASTER_DIRECTION_RX,
		&dvbm_rx_fops,
		cap,
		4,
		ASI_CTL_TRANSPORT_DVB_ASI)) < 0) {
		goto NO_IFACE;
	}

	return 0;

NO_IFACE:
	dvbm_pci_remove (dev);
NO_DEV:
NO_MEM:
	return err;
}

/**
 * dvbm_rx_irq_handler - DVB Master Receive interrupt service routine
 * @irq: interrupt number
 * @dev_id: pointer to the device data structure
 * @regs: processor context
 **/
static irqreturn_t
IRQ_HANDLER(dvbm_rx_irq_handler,irq,dev_id,regs)
{
	struct master_dev *card = dev_id;
	unsigned int intcsr = readl (card->bridge_addr + PLX_INTCSR);
	unsigned int status, interrupting_iface = 0;
	struct master_iface *iface = list_entry (card->iface_list.next,
		struct master_iface, list);

	if (intcsr & PLX_INTCSR_DMA1INT_ACTIVE) {
		struct plx_dma *dma = iface->dma;

		/* Read the interrupt type and clear it */
		status = readb (card->bridge_addr + PLX_DMACSR1);
		writeb (PLX_DMACSR_ENABLE | PLX_DMACSR_CLINT,
			card->bridge_addr + PLX_DMACSR1);

		/* Increment the buffer pointer */
		plx_advance (dma);

		if (plx_rx_isempty (dma)) {
			set_bit (ASI_EVENT_RX_BUFFER_ORDER, &iface->events);
		}

		/* Flag end-of-chain */
		if (status & PLX_DMACSR_DONE) {
			set_bit (0, &iface->dma_done);
		}

		interrupting_iface |= 0x2;
	}
	if (intcsr & PLX_INTCSR_PCILOCINT_ACTIVE) {
		spin_lock (&card->irq_lock);
		status = master_inl (card, DVBM_RX_STATUS) &
			(DVBM_RX_STATUS_REFRAME | DVBM_RX_STATUS_ENABLE |
			DVBM_RX_STATUS_OVERRUN);
		if (status & DVBM_RX_STATUS_OVERRUN) {
			master_outl (card, DVBM_RX_CFG,
				status | DVBM_RX_CFG_DEFAULT |
				DVBM_RX_CFG_OVERRUNINT);
			set_bit (ASI_EVENT_RX_FIFO_ORDER,
				&iface->events);
			interrupting_iface |= 0x2;
		}
		if (iface->capabilities & ASI_CAP_RX_SYNC) {
			unsigned int clear, set;

			status = master_inl (card, DVBM_RX_STATUS2) &
				~DVBM_RX_CFG2_PSTARTEDRST;
			set = status |
				DVBM_RX_CFG2_AOSINT | DVBM_RX_CFG2_LOSINT;
			clear = set;
			if (iface->capabilities & ASI_CAP_RX_CD) {
				set |= DVBM_RX_CFG2_CDIE;
				clear = set;
				if (status & DVBM_RX_STATUS2_CDIS) {
					clear &= ~DVBM_RX_CFG2_CDIE;
					set_bit (ASI_EVENT_RX_CARRIER_ORDER,
						&iface->events);
					interrupting_iface |= 0x2;
				}
			}
			if (status & DVBM_RX_STATUS2_AOSINT) {
				clear &= ~DVBM_RX_CFG2_AOSINT;
				set_bit (ASI_EVENT_RX_AOS_ORDER,
					&iface->events);
				interrupting_iface |= 0x2;
			}
			if (status & DVBM_RX_STATUS2_LOSINT) {
				clear &= ~DVBM_RX_CFG2_LOSINT;
				set_bit (ASI_EVENT_RX_LOS_ORDER,
					&iface->events);
				interrupting_iface |= 0x2;
			}
			master_outl (card, DVBM_RX_CFG2, clear);
			wmb ();
			master_outl (card, DVBM_RX_CFG2, set);
		}
		spin_unlock (&card->irq_lock);
	}

	if (interrupting_iface) {
		/* Dummy read to flush PCI posted writes */
		readb (card->bridge_addr + PLX_DMACSR1);

		wake_up (&iface->queue);
		return IRQ_HANDLED;
	}
	return IRQ_NONE;
}

/**
 * dvbm_rx_init - Initialize the DVB Master Receive
 * @iface: interface
 **/
static void
dvbm_rx_init (struct master_iface *iface)
{
	struct master_dev *card = iface->card;

	master_outl (card, DVBM_RX_CFG, DVBM_RX_CFG_NORMAL);

	/* Reset the PCI 9080 */
	masterplx_reset_bridge (card);

	/* Setup the PCI 9080 */
	writel (PLX_INTCSR_PCIINT_ENABLE |
		PLX_INTCSR_PCILOCINT_ENABLE |
		PLX_INTCSR_DMA1INT_ENABLE,
		card->bridge_addr + PLX_INTCSR);
	writel (PLX_DMAMODE_32BIT | PLX_DMAMODE_READY |
		PLX_DMAMODE_LOCALBURST | PLX_DMAMODE_CHAINED |
		PLX_DMAMODE_INT | PLX_DMAMODE_CLOC |
		PLX_DMAMODE_DEMAND | PLX_DMAMODE_INTPCI,
		card->bridge_addr + PLX_DMAMODE1);
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + PLX_INTCSR);

	/* Program onboard FIFO almost-full and almost-empty levels */
	master_outl (card, DVBM_RX_CFG, (DVBM_RX_PAF << 11) | DVBM_RX_PAE);
	master_outl (card, DVBM_RX_CFG, DVBM_RX_CFG_PGMFIFO |
		(DVBM_RX_PAF << 11) | DVBM_RX_PAE);
	while (master_inl (card, DVBM_RX_STATUS) &
		DVBM_RX_STATUS_PGMFIFO) {
	}

	master_outl (card, DVBM_RX_CFG, DVBM_RX_CFG_DEFAULT |
		DVBM_RX_CFG_CLOVERRUNINT | DVBM_RX_CFG_REFRAME);
	if (iface->capabilities & ASI_CAP_RX_SYNC) {
		unsigned int reg;

		switch (iface->mode) {
		case ASI_CTL_RX_MODE_RAW:
			reg = 0;
			break;
		case ASI_CTL_RX_MODE_188:
			reg = DVBM_RX_CFG2_SYNC;
			break;
		case ASI_CTL_RX_MODE_204:
			reg = DVBM_RX_CFG2_SYNC | DVBM_RX_CFG2_204;
			break;
		case ASI_CTL_RX_MODE_AUTO:
			reg = DVBM_RX_CFG2_SYNC | DVBM_RX_CFG2_APS;
			break;
		default:
			reg = 0;
			break;
		}
		if (iface->capabilities & ASI_CAP_RX_DSYNC) {
			reg |= DVBM_RX_CFG2_DSYNC;
		}
		if (iface->count27) {
			reg |= DVBM_RX_CFG2_27MHZ;
		}
		master_outl (card, DVBM_RX_CFG2, reg);

		/* Reset byte counter */
		master_inl (card, DVBM_RX_COUNTR);
	}
	return;
}

/**
 * dvbm_rx_start - Activate the DVB Master Receive
 * @iface: interface
 **/
static void
dvbm_rx_start (struct master_iface *iface)
{
	struct master_dev *card = iface->card;

	/* Enable and start DMA */
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

	/* Enable interrupts and the receiver */
	spin_lock_irq (&card->irq_lock);
	if (iface->capabilities & ASI_CAP_RX_SYNC) {
		unsigned int reg;

		reg = master_inl (card, DVBM_RX_CFG2) |
			DVBM_RX_CFG2_AOSINT | DVBM_RX_CFG2_LOSINT;
		if (iface->capabilities & ASI_CAP_RX_CD) {
			reg |= DVBM_RX_CFG2_CDIE;
		}
		master_outl (card, DVBM_RX_CFG2, reg);
	}
	master_outl (card, DVBM_RX_CFG, DVBM_RX_CFG_DEFAULT |
		DVBM_RX_CFG_REFRAME | DVBM_RX_CFG_ENABLE |
		DVBM_RX_CFG_OVERRUNINT);
	spin_unlock_irq (&card->irq_lock);

	return;
}

/**
 * dvbm_rx_stop - Deactivate the DVB Master Receive
 * @iface: interface
 **/
static void
dvbm_rx_stop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;

	/* Disable the receiver and interrupts, and clear status */
	spin_lock_irq (&card->irq_lock);
	master_outl (card, DVBM_RX_CFG,
		DVBM_RX_CFG_DEFAULT | DVBM_RX_CFG_CLOVERRUNINT |
		DVBM_RX_CFG_REFRAME);
	if (iface->capabilities & ASI_CAP_RX_SYNC) {
		master_outl (card, DVBM_RX_CFG2, 0);
	}
	spin_unlock_irq (&card->irq_lock);

	/* Abort and disable DMA */
	writeb (PLX_DMACSR_ENABLE | PLX_DMACSR_ABORT,
		card->bridge_addr + PLX_DMACSR1);
	/* Dummy read to flush PCI posted writes */
	readb (card->bridge_addr + PLX_DMACSR1);
	wait_event_timeout (iface->queue, test_bit (0, &iface->dma_done), HZ);
	writeb (0, card->bridge_addr + PLX_DMACSR1);
	return;
}

/**
 * dvbm_rx_open - DVB Master Receive open() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
dvbm_rx_open (struct inode *inode, struct file *filp)
{
	return masterplx_open (inode,
		filp,
		dvbm_rx_init,
		dvbm_rx_start,
		DVBM_RX_FIFO,
		0);
}

/**
 * dvbm_rx_unlocked_ioctl - DVB Master Receive unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
dvbm_rx_unlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	int val;
	unsigned int reg = 0;

	switch (cmd) {
	case ASI_IOC_RXGETBUFLEVEL:
		if (put_user (plx_rx_buflevel (iface->dma),
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXGETSTATUS:
		if (!(iface->capabilities & ASI_CAP_RX_SYNC)) {
			val = 1;
		} else {
			/* Atomic read of STATUS2, so we don't need to lock */
			reg = master_inl (card, DVBM_RX_STATUS2);
			if (reg & DVBM_RX_CFG2_SYNC) {
				if (reg & DVBM_RX_STATUS2_PASSING) {
					if (reg & DVBM_RX_CFG2_APS) {
						val = (reg &
							DVBM_RX_STATUS2_204) ?
							204 : 188;
					} else if (reg & DVBM_RX_CFG2_204) {
						val = 204;
					} else {
						val = 188;
					}
				} else {
					val = 0;
				}
			} else {
				val = 1;
			}
		}
		if (put_user (val, (int *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXGETBYTECOUNT:
		if (!(iface->capabilities & ASI_CAP_RX_BYTECOUNTER) ||
			iface->count27) {
			return -ENOTTY;
		}
		if (put_user (master_inl (card, DVBM_RX_COUNTR),
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETINVSYNC:
		if (!(iface->capabilities & ASI_CAP_RX_SYNC)) {
			return -ENOTTY;
		}
		if (get_user (val, (int *)arg)) {
			return -EFAULT;
		}
		if (!(iface->capabilities & ASI_CAP_RX_INVSYNC)) {
			if (val) {
				return -EINVAL;
			}
		} else {
			reg = DVBM_RX_CFG2_AOSINT | DVBM_RX_CFG2_LOSINT |
				DVBM_RX_CFG2_CDIE;
			switch (val) {
			case 0:
				reg |= 0;
				break;
			case 1:
				reg |= DVBM_RX_CFG2_INVSYNC;
				break;
			default:
				return -EINVAL;
			}
			spin_lock_irq (&card->irq_lock);
			master_outl (card, DVBM_RX_CFG2,
				(master_inl (card, DVBM_RX_STATUS2) &
				~DVBM_RX_CFG2_INVSYNC &
				~DVBM_RX_CFG2_PSTARTEDRST) | reg);
			spin_unlock_irq (&card->irq_lock);
		}
		break;
	case ASI_IOC_RXGETCARRIER:
		if (!(iface->capabilities & ASI_CAP_RX_CD)) {
			return -ENOTTY;
		}
		/* Atomic read of STATUS2, so we don't need to lock */
		if (put_user ((master_inl (card, DVBM_RX_STATUS2) &
			DVBM_RX_STATUS2_CD) ? 1 : 0, (int *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETDSYNC:
		if (!(iface->capabilities & ASI_CAP_RX_SYNC)) {
			return -ENOTTY;
		}
		if (get_user (val, (int *)arg)) {
			return -EFAULT;
		}
		if (!(iface->capabilities & ASI_CAP_RX_DSYNC)) {
			if (val) {
				return -EINVAL;
			}
		} else {
			reg = DVBM_RX_CFG2_AOSINT | DVBM_RX_CFG2_LOSINT |
				DVBM_RX_CFG2_CDIE;
			switch (val) {
			case 0:
				reg |= 0;
				break;
			case 1:
				reg |= DVBM_RX_CFG2_DSYNC;
				break;
			default:
				return -EINVAL;
			}
			spin_lock_irq (&card->irq_lock);
			master_outl (card, DVBM_RX_CFG2,
				(master_inl (card, DVBM_RX_STATUS2) &
				~DVBM_RX_CFG2_DSYNC &
				~DVBM_RX_CFG2_PSTARTEDRST) | reg);
			spin_unlock_irq (&card->irq_lock);
		}
		break;
	case ASI_IOC_RXGET27COUNT:
		if (!iface->count27) {
			return -ENOTTY;
		}
		if (put_user (master_inl (card, DVBM_RX_COUNTR),
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETINPUT:
		if (get_user (val, (int *)arg)) {
			return -EFAULT;
		}
		if (val) {
			return -EINVAL;
		}
		break;
	default:
		return asi_rxioctl (iface, cmd, arg);
	}
	return 0;
}

/**
 * dvbm_rx_ioctl - DVB Master Receive ioctl() method
 * @inode: inode
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
dvbm_rx_ioctl (struct inode *inode,
	struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	return dvbm_rx_unlocked_ioctl (filp, cmd, arg);
}

/**
 * dvbm_rx_fsync - DVB Master Receive fsync() method
 * @filp: file to flush
 * @dentry: directory entry associated with the file
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
dvbm_rx_fsync (struct file *filp,
	struct dentry *dentry,
	int datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;

	if (down_interruptible (&iface->buf_sem)) {
		return -ERESTARTSYS;
	}

	/* Stop the receiver */
	dvbm_rx_stop (iface);

	/* Reset the onboard FIFO and driver buffers */
	while (!(master_inl (card, DVBM_RX_STATUS) & DVBM_RX_STATUS_EMPTY)) {
		master_inl (card, DVBM_RX_FIFO);
		rmb ();
	}
	iface->events = 0;
	plx_reset (iface->dma);

	/* Start the receiver */
	dvbm_rx_start (iface);

	up (&iface->buf_sem);
	return 0;
}

/**
 * dvbm_rx_release - DVB Master Receive release() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
dvbm_rx_release (struct inode *inode, struct file *filp)
{
	struct master_iface *iface = filp->private_data;

	return masterplx_release (iface, dvbm_rx_stop, NULL);
}

