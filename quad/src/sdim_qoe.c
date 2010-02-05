/* sdim_qoe.c
 *
 * Linux driver for Linear Systems Ltd. SDI Master Q/o PCIe.
 *
 * Copyright (C) 2007-2008 Linear Systems Ltd.
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
#include <linux/module.h> /* THIS_MODULE */
#include <linux/fs.h> /* inode, file, file_operations */
#include <linux/sched.h> /* pt_regs */
#include <linux/pci.h> /* pci_dev */
#include <linux/slab.h> /* kmalloc () */
#include <linux/list.h> /* INIT_LIST_HEAD () */
#include <linux/poll.h> /* poll_table */
#include <linux/spinlock.h> /* spin_lock_init () */
#include <linux/init.h> /* __devinit */
#include <linux/errno.h> /* error codes */
#include <linux/delay.h> /* udelay () */
#include <linux/interrupt.h> /* irqreturn_t */
#include <linux/device.h> /* class_device_create file */
#include <asm/semaphore.h> /* sema_init () */
#include <asm/uaccess.h> /* put_user () */
#include <asm/bitops.h> /* set_bit () */

#include "sdicore.h"
#include "../include/master.h"
//Temporary fix for kernel 2.6.21
#include "mdev.c"
#include "sdim.h"
#include "miface.h"
#include "sdim_qoe.h"
#include "lsdma.h"
//Temporary fix for kernel 2.6.21
#include "masterlsdma.c"

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,8))
static inline int
nonseekable_open (struct inode *inode, struct file *filp)
{
	return 0;
}
#endif

#ifndef list_for_each_safe
#define list_for_each_safe(pos, n, head) \
	for (pos = (head)->next, n = pos->next; pos != (head); \
		pos = n, n = pos->next)
#endif

static const char sdim_qoe_name[] = SDIM_NAME_QOE;
static char sdimqoe_driver_name[] = "sdiqoe";

/* Static function prototypes */
static ssize_t sdim_qoe_show_uid (struct class_device *cd, char *buf);
static int sdim_qoe_probe (struct pci_dev *dev,
	const struct pci_device_id *id) __devinit;
void sdim_qoe_remove (struct pci_dev *dev);
static irqreturn_t IRQ_HANDLER (sdim_qoe_irq_handler, irq, dev_id, regs);
static void sdim_qoe_init (struct master_iface *iface);
static void sdim_qoe_start (struct master_iface *iface);
static void sdim_qoe_stop (struct master_iface *iface);
static void sdim_qoe_exit (struct master_iface *iface);
static int sdim_qoe_open (struct inode *inode, struct file *filp);
static long sdim_qoe_unlocked_ioctl (struct file *filp,
		unsigned int cmd,
		unsigned long arg);
static int sdim_qoe_ioctl (struct inode *inode,
		struct file *filp, unsigned int cmd, unsigned long arg);
static int sdim_qoe_fsync (struct file *filp,
		struct dentry *dentry, int datasync);
static int sdim_qoe_release (struct inode *inode,
		struct file *filp);
static int sdim_qoe_init_module (void) __init;
static void sdim_qoe_cleanup_module (void) __exit;

MODULE_AUTHOR("Linear Systems Ltd.");
MODULE_DESCRIPTION("SDI Master Q/o");
MODULE_LICENSE("GPL");

#ifdef MODULE_VERSION
MODULE_VERSION(MASTER_DRIVER_VERSION);
#endif

static struct pci_device_id sdim_pci_id_table[] = {
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			SDIM_PCI_DEVICE_ID_LINSYS_SDIQOE)
	},
	{0, }
};

static struct pci_driver sdim_qoe_pci_driver = {
	.name = sdimqoe_driver_name,
	.id_table = sdim_pci_id_table,
	.probe = sdim_qoe_probe,
	.remove = sdim_qoe_remove
};

MODULE_DEVICE_TABLE(pci, sdim_pci_id_table);

struct file_operations sdim_qoe_fops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.write = masterlsdma_write,
	.poll = masterlsdma_txpoll,
	.ioctl = sdim_qoe_ioctl,
#ifdef HAVE_UNLOCKED_IOCTL
	.unlocked_ioctl=sdim_qoe_unlocked_ioctl,
#endif
#ifdef HAVE_COMPAT_IOCTL
	.compat_ioctl = sdi_compat_ioctl,
#endif
	.mmap = masterlsdma_mmap,
	.open = sdim_qoe_open,
	.release = sdim_qoe_release,
	.fsync = sdim_qoe_fsync,
	.fasync = NULL
};

static LIST_HEAD(sdim_card_list);

static struct class sdimqoe_class = {
	.name = sdimqoe_driver_name,
	.release = mdev_class_device_release,
	.class_release = NULL
};

/**
 * sdim_qoe_show_uid - interface attribute read handler
 * @cd: class_device being read
 * @buf: output buffer
 **/
static ssize_t
sdim_qoe_show_uid (struct class_device *cd, char *buf)
{
	struct master_dev *card = to_master_dev(cd);
	return snprintf (buf, PAGE_SIZE, "0x%08X%08X\n",
		readl (card->core.addr + SDIM_QOE_SSN_HI),
		readl (card->core.addr + SDIM_QOE_SSN_LO));
}

static CLASS_DEVICE_ATTR(uid,S_IRUGO, sdim_qoe_show_uid, NULL);

/**
 * sdim_qoe_pci_probe - PCI insertion handler for a SDI Master Q/o
 * @dev: PCI device
 * @id: PCI ID
 *
 * Handle the insertion of a SDI Master Q/o.
 * Returns a negative error code on failure and 0 on success.
 **/
static int __devinit
sdim_qoe_probe (struct pci_dev *dev,
	const struct pci_device_id *id)
{
	int err, i;
	struct master_dev *card;
	const char *name;

	switch (dev->device) {
	case SDIM_PCI_DEVICE_ID_LINSYS_SDIQOE:
		name = sdim_qoe_name;
		break;
	default:
		name = "";
		break;
	}

	/* Initialize the driver_data pointer so that sdim_pci_remove()
	 * doesn't try to free it if an error occurs */
	pci_set_drvdata (dev, NULL);

	/* Wake a sleeping device */
	if ((err = pci_enable_device (dev)) < 0) {
		printk (KERN_WARNING "%s: unable to enable device\n",
			sdimqoe_driver_name);
		goto NO_PCI;
	}

	/* Set PCI DMA addressing limitations */
	if ((err = pci_set_dma_mask (dev, DMA_32BIT_MASK)) < 0) {
		printk (KERN_WARNING "%s: unable to set PCI DMA mask\n",
			sdimqoe_driver_name);
		goto NO_PCI;
	}

	/* Enable Bus Mastering */
	pci_set_master (dev);

	/* Request the PCI I/O Resources */
	if ((err = pci_request_regions (dev, sdimqoe_driver_name)) < 0) {
		goto NO_PCI;
	}

	/* Allocate a board info structure */
	if ((card = (struct master_dev *)
		kmalloc (sizeof (*card), GFP_KERNEL)) == NULL) {
		err = -ENOMEM;
		goto NO_MEM;
	}

	/* Initialize the board info structure */
	memset (card, 0, sizeof (*card));

	/* Remap bridge address to the DMA controller */
	card->bridge_addr = ioremap_nocache (pci_resource_start (dev, 2),
		pci_resource_len (dev, 2));
	card->core.addr = ioremap_nocache (pci_resource_start (dev, 0),
		pci_resource_len (dev, 0));
	card->version = readl(card->core.addr + SDIM_QOE_FPGAID) & 0xffff;
	card->name = name;
	card->irq_handler = sdim_qoe_irq_handler;
	card->capabilities = MASTER_CAP_UID;
	INIT_LIST_HEAD(&card->iface_list);
	/* Lock for ICSR[] */
	spin_lock_init (&card->irq_lock);
	/* Lock for TCSR */
	spin_lock_init (&card->reg_lock);
	sema_init (&card->users_sem, 1);
	card->pdev = dev;

	/* Get the firmware version and flush PCI posted writes */
	printk (KERN_INFO "%s: %s detected, firmware version %u.%u (0x%04X)\n",
			sdimqoe_driver_name, name, card->version >> 8,
			card->version & 0x00ff, card->version);

	/* Store the pointer to the board info structure
	 * in the PCI info structure */
	pci_set_drvdata (dev, card);

	/* Reset the FPGA */
	for (i = 0; i < 4; i++) {
		writel (SDIM_QOE_TCSR_TXRST,
			card->core.addr + SDIM_QOE_TCSR(i));
	}

	/* Setup the LS DMA controller */
	writel (LSDMA_INTMSK_CH(0) | LSDMA_INTMSK_CH(1) | LSDMA_INTMSK_CH(2)
		| LSDMA_INTMSK_CH(3),
		card->bridge_addr + LSDMA_INTMSK);

	for (i = 0; i < 4; i++) {
		writel (LSDMA_CH_CSR_INTDONEENABLE |
			LSDMA_CH_CSR_INTSTOPENABLE,
			card->bridge_addr + LSDMA_CSR(i));
	}

	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);

	/* Register a Master device */
	if ((err = mdev_register (card,
		&sdim_card_list,
		sdimqoe_driver_name,
		&sdimqoe_class)) < 0) {
		goto NO_DEV;
	}

	/* Add class_device attributes */
	if (card->capabilities & MASTER_CAP_UID) {
		if ((err = class_device_create_file (&card->class_dev,
			&class_device_attr_uid)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'uid'\n",
				sdimqoe_driver_name);
		}
	}

	/* Register the transmit interfaces */
	for (i = 0; i < 4; i++) {
		if ((err = sdi_register_iface (card,
			MASTER_DIRECTION_TX,
			&sdim_qoe_fops,
			SDI_CAP_TX_RXCLKSRC,
			4)) < 0) {
			goto NO_IFACE;
		}
 	}
	return 0;

NO_IFACE:
	sdim_qoe_remove (dev);
NO_DEV:
NO_MEM:
NO_PCI:
	return err;
}

/**
 * sdim_qoe_pci_remove - PCI removal handler for a SDI Master Q/o
 * @card: Master device
 *
 * Handle the removal of a SDI Master Q/o.
 **/
void sdim_qoe_remove (struct pci_dev *dev)
{
	struct master_dev *card = pci_get_drvdata (dev);

	if (card) {
		struct list_head *p, *n;
		struct master_iface *iface;

		list_for_each_safe (p, n, &card->iface_list) {
			iface = list_entry (p, struct master_iface, list);
			sdi_unregister_iface (iface);
		}
		iounmap (card->core.addr);
		iounmap (card->bridge_addr);
		list_for_each (p, &sdim_card_list) {
			if (p == &card->list) {
				mdev_unregister (card);
				break;
			}
		}
		pci_set_drvdata (dev, NULL);
	}
	pci_release_regions (dev);
	pci_disable_device (dev);

	return;
}

/**
 * sdim_qoe_irq_handler - SDI Master Q/o interrupt service routine
 * @irq: interrupt number
 * @dev_id: pointer to the device data structure
 * @regs: processor context
 **/
static irqreturn_t
IRQ_HANDLER (sdim_qoe_irq_handler, irq, dev_id, regs)
{
	struct master_dev *card = dev_id;
	struct list_head *p = &card->iface_list;
	struct master_iface *iface;
	unsigned int status, interrupting_iface = 0;
	unsigned int dmaintsrc = readl (card->bridge_addr + LSDMA_INTSRC);
	int i;

	for (i = 0; i < 4; i++) {
		p = p->next;
		iface = list_entry(p, struct master_iface, list);
		if (dmaintsrc & LSDMA_INTSRC_CH(i)) {
			/* Read the interrupt type and clear it */
			spin_lock(&card->irq_lock);
			status = readl(card->bridge_addr + LSDMA_CSR(i));
			writel(status,card->bridge_addr + LSDMA_CSR(i));
			spin_unlock(&card->irq_lock);
			/* Increment the buffer pointer */
			if (status & LSDMA_CH_CSR_INTSRCBUFFER) {
				lsdma_advance(iface->dma);
			}
			/* Flag end of chain */
			if (status & LSDMA_CH_CSR_INTSRCDONE) {
				set_bit(SDI_EVENT_TX_BUFFER_ORDER,
					&iface->events);
				set_bit(0, &iface->dma_done);
			}

			/* Flag DMA abort */
			if (status &LSDMA_CH_CSR_INTSRCSTOP) {
				set_bit(0, &iface->dma_done);
			}

			interrupting_iface |= (0x1 << i );
		}

		/* Check and clear the source of the interrupts */
		spin_lock (&card->irq_lock);
		status = readl (card->core.addr + SDIM_QOE_ICSR(i));
		writel (status, card->core.addr + SDIM_QOE_ICSR(i));
		spin_unlock (&card->irq_lock);

		if (status & SDIM_QOE_ICSR_TUIS) {
			set_bit (SDI_EVENT_TX_FIFO_ORDER,
				&iface->events);
			interrupting_iface |= 0x1 << i;
		}
		if (status & SDIM_QOE_ICSR_TXDIS) {
			set_bit (SDI_EVENT_TX_DATA_ORDER,
				&iface->events);
			interrupting_iface |= 0x1 << i;
		}

		if (interrupting_iface & (0x1 << i)) {
			wake_up (&iface->queue);
		}
	}
	if (interrupting_iface) {
		readl (card->bridge_addr + LSDMA_INTMSK);
		return IRQ_HANDLED;
	}

	return IRQ_NONE;
}

/**
 * sdim_qoe_init - Initialize the SDI Master Q/o Transmitter
 * @iface: interface
 **/
static void
sdim_qoe_init (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg = 0;

	switch (iface->mode) {
	default:
	case SDI_CTL_MODE_8BIT:
		reg |= 0;
		break;
	case SDI_CTL_MODE_10BIT:
		reg |= SDIM_TCSR_10BIT;
		break;
	}

	switch (iface->clksrc) {
	default:
	case SDI_CTL_TX_CLKSRC_ONBOARD:
		reg |= 0;
		break;
	}

	/* There will be no races on CSR
	 * until this code returns, so we don't need to lock it
	*/
	writel (reg | SDIM_QOE_TCSR_TXRST,
		card->core.addr + SDIM_QOE_TCSR(channel));
	wmb();
	writel (reg, card->core.addr + SDIM_QOE_TCSR(channel));
	wmb();
	/* Dummy read to flush PCI posted writes */
	readl (card->core.addr + SDIM_QOE_ICSR(channel));
	writel (SDIM_QOE_TFSL << 16,
		card->core.addr + SDIM_QOE_TFCR(channel));
	/* Disable RP178 pattern generation.
	 * There will be no races on TCSR
	 * until this code returns, so we don't need to lock it */
	reg = readl (card->core.addr + SDIM_QOE_TCSR(channel));
	writel (reg | SDIM_QOE_TCSR_RP178,
		card->core.addr + SDIM_QOE_TCSR(channel));
	return;
}

/**
 * sdim_qoe_start - Activate the SDI Master Q/o Transmitter
 * @iface: interface
 **/
static void
sdim_qoe_start (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg = 0;

	/* Enabling Channel DMA Explicitly */
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE,
		card->bridge_addr + LSDMA_CSR(channel));

	/* Enable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	reg |= SDIM_QOE_ICSR_TUIE | SDIM_QOE_ICSR_TXDIE;
	writel(reg, card->core.addr + SDIM_QOE_ICSR(channel));
	readl(card->core.addr + SDIM_QOE_FPGAID);
	spin_unlock_irq(&card->irq_lock);

	/* Enable the transmitter
	 * There will be no races on TCSR
	 * until this code returns, so we don't need to lock it */
	reg = readl(card->core.addr + SDIM_QOE_TCSR(channel));
	writel(reg | SDIM_QOE_TCSR_TXE,
		card->core.addr + SDIM_QOE_TCSR(channel));

	return;
}

/**
 * sdim_qoe_stop - Deactivate the SDI Master Q/o Transmitter
 * @iface: interface
 **/
static void
sdim_qoe_stop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	struct lsdma_dma *dma = iface->dma;
	unsigned int reg;

	lsdma_tx_link_all (dma);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	lsdma_reset (dma);

	/* Wait for the onboard FIFOs to empty */
	/* Atomic read of ICSR, so we don't need to lock */
	wait_event (iface->queue,
		!(readl (card->core.addr + SDIM_QOE_ICSR(channel)) &
 		SDIM_QOE_ICSR_TXD));

	/* Disable the Transmitter */
	/* Races will be taken care of here */
	reg = readl(card->core.addr + SDIM_QOE_TCSR(channel));
	writel(reg & ~SDIM_QOE_TCSR_TXE,
		card->core.addr + SDIM_QOE_TCSR(channel));

	/* Disable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = SDIM_QOE_ICSR_TUIS | SDIM_QOE_ICSR_TXDIS;
	writel (reg, card->core.addr + SDIM_QOE_ICSR(channel));
	writel ((LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_STOP) & ~LSDMA_CH_CSR_ENABLE,
		card->bridge_addr + LSDMA_CSR(channel));
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);
	spin_unlock_irq (&card->irq_lock);

	udelay (10L);
	return;
}

/**
 * sdim_qoe_exit - Clean up the SDI Master Q/o transmitter
 * @iface: interface
 **/
static void
sdim_qoe_exit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);

	/* Reset the transmitter.
	 * There will be no races on CSR here,
	 * so we don't need to lock it */
	writel (SDIM_QOE_TCSR_TXRST,
		card->core.addr + SDIM_QOE_TCSR(channel));

	return;
}

/**
 * sdim_qoe_open - SDI Master Q/o open() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
sdim_qoe_open (struct inode *inode, struct file *filp)
{
	return masterlsdma_open (inode,
		filp,
		sdim_qoe_init,
		sdim_qoe_start,
		0,
		LSDMA_MMAP);
}

/**
 * sdim_qoe_unlocked_ioctl - SDI Master Q/o unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
sdim_qoe_unlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);

	switch (cmd) {
	case SDI_IOC_TXGETBUFLEVEL:
		if (put_user (lsdma_tx_buflevel (iface->dma),
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_TXGETTXD:
	/* Atomic read of ICSR, so we don't need to lock */
		if (put_user ((readl (card->core.addr +
				SDIM_QOE_ICSR(channel)) &
				SDIM_QOE_ICSR_TXD) ? 1 : 0, (int *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_QBUF:
		return masterlsdma_txqbuf (filp, arg);
	case SDI_IOC_DQBUF:
		return masterlsdma_txdqbuf (filp, arg);
	default:
		return sdi_txioctl (iface, cmd, arg);
		}

	return 0;
}

/**
 * sdim_qoe_ioctl - SDI Master Q/o ioctl() method.
 * @inode: inode
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
sdim_qoe_ioctl (struct inode *inode,
	struct file *filp,
	unsigned int cmd,
	unsigned long arg)

{
	return sdim_qoe_unlocked_ioctl (filp, cmd, arg);
}


/**
 * sdim_qoe_fsync - SDI Master Q/o fsync() method
 * @filp: file to flush
 * @dentry: directory entry associated with the file
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
sdim_qoe_fsync (struct file *filp,
	struct dentry *dentry,
	int datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	struct lsdma_dma *dma = iface->dma;
	const unsigned int channel = mdev_index (card, &iface->list);

	down (&iface->buf_sem);
	lsdma_tx_link_all(dma);
	wait_event(iface->queue, test_bit(0, &iface->dma_done));
	lsdma_reset(dma);

	/* Wait for the onboard FIFOs to empty */
	/* Atomic read of ICSR, so we don't need to lock */
	wait_event(iface->queue,
		!(readl(card->core.addr +
			SDIM_QOE_ICSR(channel)) & SDIM_QOE_ICSR_TXD));

	up(&iface->buf_sem);

	return 0;
}

/**
 * sdim_qoe_release - SDI Master Q/o release() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
sdim_qoe_release (struct inode *inode, struct file *filp)
{
	struct master_iface *iface = filp->private_data;
	return masterlsdma_release(iface, sdim_qoe_stop, sdim_qoe_exit);
}

/**
 * sdim_init_module - register the module as a PCI driver
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int __init
sdim_qoe_init_module (void)
{
	printk (KERN_INFO "%s: Linear Systems Ltd. "
		"SDI Master driver from master-%s (%s)\n",
		sdimqoe_driver_name, MASTER_DRIVER_VERSION, MASTER_DRIVER_DATE);

	return mdev_init_module (&sdim_qoe_pci_driver,
		&sdimqoe_class,
		sdimqoe_driver_name);
}

/**
 * sdim_cleanup_module - unregister the module as a PCI driver
 **/
static void __exit
sdim_qoe_cleanup_module (void)
{
	mdev_cleanup_module (&sdim_qoe_pci_driver, &sdimqoe_class);
	return;
}

module_init (sdim_qoe_init_module);
module_exit (sdim_qoe_cleanup_module);

