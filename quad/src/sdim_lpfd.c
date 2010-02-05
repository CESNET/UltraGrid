/* sdim_lpfd.c
 *
 * Linux driver functions for Linear Systems Ltd. SDI Master LP FD.
 *
 * Copyright (C) 2003-2007 Linear Systems Ltd.
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

#include <linux/fs.h> /* inode, file, file_operations */
#include <linux/sched.h> /* pt_regs */
#include <linux/pci.h> /* pci_dev */
#include <linux/dma-mapping.h> /* DMA_32BIT_MASK */
#include <linux/slab.h> /* kmalloc () */
#include <linux/list.h> /* INIT_LIST_HEAD () */
#include <linux/spinlock.h> /* spin_lock_init () */
#include <linux/init.h> /* module_init () */
#include <linux/errno.h> /* error codes */
#include <linux/interrupt.h> /* irqreturn_t */

#include <asm/semaphore.h> /* sema_init () */
#include <asm/uaccess.h> /* put_user () */
#include <asm/bitops.h> /* set_bit () */

#include "sdicore.h"
#include "../include/master.h"
// Temporary fix for Linux kernel 2.6.21
#include "mdev.c"
#include "miface.h"
#include "lsdma.h"
// Temporary fix for Linux kernel 2.6.21
#include "masterlsdma.c"
#include "sdim.h"

#ifndef list_for_each_safe
#define list_for_each_safe(pos, n, head) \
	for (pos = (head)->next, n = pos->next; pos != (head); \
		pos = n, n = pos->next)
#endif

static const char sdim_lpfd_name[] = SDIM_NAME_LPFD;
static const char sdim_lpfde_name[] = SDIM_NAME_LPFDE;

/* Static function prototypes */
static ssize_t sdim_show_blackburst_type (struct class_device *cd,
	char *buf);
static ssize_t sdim_store_blackburst_type (struct class_device *cd,
	const char *buf,
	size_t count);
static ssize_t sdim_show_uid (struct class_device *cd,
	char *buf);
static int sdim_pci_probe (struct pci_dev *dev,
	const struct pci_device_id *id) __devinit;
static void sdim_pci_remove (struct pci_dev *dev);
static irqreturn_t IRQ_HANDLER(sdim_irq_handler,irq,dev_id,regs);
static void sdim_txinit (struct master_iface *iface);
static void sdim_txstart (struct master_iface *iface);
static void sdim_txstop (struct master_iface *iface);
static void sdim_txexit (struct master_iface *iface);
static int sdim_txopen (struct inode *inode, struct file *filp);
static long sdim_txunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int sdim_txioctl (struct inode *inode,
	struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int sdim_txfsync (struct file *filp,
	struct dentry *dentry,
	int datasync);
static int sdim_txrelease (struct inode *inode, struct file *filp);
static void sdim_rxinit (struct master_iface *iface);
static void sdim_rxstart (struct master_iface *iface);
static void sdim_rxstop (struct master_iface *iface);
static void sdim_rxexit (struct master_iface *iface);
static int sdim_rxopen (struct inode *inode, struct file *filp);
static long sdim_rxunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int sdim_rxioctl (struct inode *inode,
	struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int sdim_rxfsync (struct file *filp,
	struct dentry *dentry,
	int datasync);
static int sdim_rxrelease (struct inode *inode, struct file *filp);
static int sdim_init_module (void) __init;
static void sdim_cleanup_module (void) __exit;

MODULE_AUTHOR("Linear Systems Ltd.");
MODULE_DESCRIPTION("SDI Master driver");
MODULE_LICENSE("GPL");

#ifdef MODULE_VERSION
MODULE_VERSION(MASTER_DRIVER_VERSION);
#endif

static char sdim_driver_name[] = "sdilpm";

static struct pci_device_id sdim_pci_id_table[] = {
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			SDIM_PCI_DEVICE_ID_LINSYS_SDILPFD)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			SDIM_PCI_DEVICE_ID_LINSYS_SDILPFDE)
	},
	{0, }
};

static struct pci_driver sdim_pci_driver = {
	.name = sdim_driver_name,
	.id_table = sdim_pci_id_table,
	.probe = sdim_pci_probe,
	.remove = sdim_pci_remove
};

MODULE_DEVICE_TABLE(pci,sdim_pci_id_table);

static struct file_operations sdim_txfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.write = masterlsdma_write,
	.poll = masterlsdma_txpoll,
	.ioctl = sdim_txioctl,
#ifdef HAVE_UNLOCKED_IOCTL
	.unlocked_ioctl = sdim_txunlocked_ioctl,
#endif
#ifdef HAVE_COMPAT_IOCTL
	.compat_ioctl = sdi_compat_ioctl,
#endif
	.mmap = masterlsdma_mmap,
	.open = sdim_txopen,
	.release = sdim_txrelease,
	.fsync = sdim_txfsync,
	.fasync = NULL
};

static struct file_operations sdim_rxfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.read = masterlsdma_read,
	.poll = masterlsdma_rxpoll,
	.ioctl = sdim_rxioctl,
#ifdef HAVE_UNLOCKED_IOCTL
	.unlocked_ioctl = sdim_rxunlocked_ioctl,
#endif
#ifdef HAVE_COMPAT_IOCTL
	.compat_ioctl = sdi_compat_ioctl,
#endif
	.mmap = masterlsdma_mmap,
	.open = sdim_rxopen,
	.release = sdim_rxrelease,
	.fsync = sdim_rxfsync,
	.fasync = NULL
};

static LIST_HEAD(sdim_card_list);

static struct class sdim_class = {
	.name = sdim_driver_name,
	.release = mdev_class_device_release,
	.class_release = NULL
};

/**
 * sdim_show_blackburst_type - interface attribute read handler
 * @cd: class_device being read
 * @buf: output buffer
 **/
static ssize_t
sdim_show_blackburst_type (struct class_device *cd,
	char *buf)
{
	struct master_dev *card = to_master_dev(cd);

	return snprintf (buf, PAGE_SIZE, "%u\n",
		(master_readl (card, SDIM_TCSR) & SDIM_TCSR_PAL) >> 9);
}

/**
 * sdim_store_blackburst_type - interface attribute write handler
 * @cd: class_device being written
 * @buf: input buffer
 * @count:
 **/
static ssize_t
sdim_store_blackburst_type (struct class_device *cd,
	const char *buf,
	size_t count)
{
	struct master_dev *card = to_master_dev(cd);
	char *endp;
	unsigned long val = simple_strtoul (buf, &endp, 0);
	unsigned int reg;
	const unsigned long max = MASTER_CTL_BLACKBURST_PAL;
	int retcode = count;

	if ((endp == buf) || (val > max)) {
		return -EINVAL;
	}
	spin_lock (&card->reg_lock);
	reg = master_readl (card, SDIM_TCSR) & ~SDIM_TCSR_PAL;
	master_writel (card, SDIM_TCSR, reg | (val << 9));
	spin_unlock (&card->reg_lock);
	return retcode;
}

/**
 * sdim_show_uid - interface attribute read handler
 * @cd: class_device being read
 * @buf: output buffer
 **/
static ssize_t
sdim_show_uid (struct class_device *cd,
	char *buf)
{
	struct master_dev *card = to_master_dev(cd);

	return snprintf (buf, PAGE_SIZE, "0x%08X%08X\n",
		readl (card->core.addr + SDIM_UIDR_HI),
		readl (card->core.addr + SDIM_UIDR_LO));
}

static CLASS_DEVICE_ATTR(blackburst_type,S_IRUGO|S_IWUSR,
	sdim_show_blackburst_type,sdim_store_blackburst_type);
static CLASS_DEVICE_ATTR(uid,S_IRUGO,
	sdim_show_uid,NULL);

/**
 * sdim_pci_probe - PCI insertion handler for a SDI Master
 * @dev: PCI device
 * @id: PCI ID
 *
 * Handle the insertion of a SDI Master.
 * Returns a negative error code on failure and 0 on success.
 **/
static int __devinit
sdim_pci_probe (struct pci_dev *dev,
	const struct pci_device_id *id)
{
	int err;
	unsigned int version;
	const char *name;
	struct master_dev *card;


	switch (dev->device) {
	case SDIM_PCI_DEVICE_ID_LINSYS_SDILPFD:
		name = sdim_lpfd_name;
		break;
	case SDIM_PCI_DEVICE_ID_LINSYS_SDILPFDE:
		name = sdim_lpfde_name;
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
			sdim_driver_name);
		goto NO_PCI;
	}

	/* Set PCI DMA addressing limitations */
	if ((err = pci_set_dma_mask (dev, DMA_32BIT_MASK)) < 0) {
		printk (KERN_WARNING "%s: unable to set PCI DMA mask\n",
			sdim_driver_name);
		goto NO_PCI;
	}

	/* Enable bus mastering */
	pci_set_master (dev);

	/* Request the PCI I/O resources */
	if ((err = pci_request_regions (dev, sdim_driver_name)) < 0) {
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
	/* LS DMA Controller */
	switch (dev->device) {
	case SDIM_PCI_DEVICE_ID_LINSYS_SDILPFD:
		card->bridge_addr = ioremap_nocache (pci_resource_start (dev, 1),
			pci_resource_len (dev, 1));
		break;
	case SDIM_PCI_DEVICE_ID_LINSYS_SDILPFDE:
		card->bridge_addr = ioremap_nocache (pci_resource_start (dev, 2),
			pci_resource_len (dev, 2));
		break;
	}
	card->name = name;
	card->irq_handler = sdim_irq_handler;
	card->capabilities = MASTER_CAP_BLACKBURST | MASTER_CAP_UID;
	INIT_LIST_HEAD(&card->iface_list);
	/* Lock for ICSR, DMACSR1 */
	spin_lock_init (&card->irq_lock);
	/* Lock for TCSR, RCSR */
	spin_lock_init (&card->reg_lock);
	sema_init (&card->users_sem, 1);
	card->pdev = dev;

	/* SDI Core */
	card->core.addr = ioremap_nocache (pci_resource_start (dev, 0),
		pci_resource_len (dev, 0));

	/* Print the firmware version */
	version = readl (card->core.addr + SDIM_CSR) >> 16;
	printk (KERN_INFO "%s: %s detected, firmware version %u.%u (0x%04X)\n",
		sdim_driver_name, name,
		version >> 8, version & 0x00ff, version);
	card->version = version;

	/* Store the pointer to the board info structure
	 * in the PCI info structure */
	pci_set_drvdata (dev, card);

	/* Reset the FPGA */
	master_writel (card, SDIM_TCSR, SDIM_TCSR_RST);
	master_writel (card, SDIM_RCSR, SDIM_RCSR_RST);

	/* Setup the LS DMA */
	writel (LSDMA_INTMSK_CH(0) | LSDMA_INTMSK_CH(1),
		card->bridge_addr + LSDMA_INTMSK);
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE,
		card->bridge_addr + LSDMA_CSR(0));
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION,
		card->bridge_addr + LSDMA_CSR(1));
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);

	/* Register a Master device */
	if ((err = mdev_register (card,
		&sdim_card_list,
		sdim_driver_name,
		&sdim_class)) < 0) {
		goto NO_DEV;
	}

	if (card->capabilities & MASTER_CAP_BLACKBURST) {
		if ((err = class_device_create_file (&card->class_dev,
			&class_device_attr_blackburst_type)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'blackburst_type'\n",
				sdim_driver_name);
		}
	}
	if (card->capabilities & MASTER_CAP_UID) {
		if ((err = class_device_create_file (&card->class_dev,
			&class_device_attr_uid)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'uid'\n",
				sdim_driver_name);
		}
	}

	/* Register a transmit interface */
	if ((err = sdi_register_iface (card,
		MASTER_DIRECTION_TX,
		&sdim_txfops,
		SDI_CAP_TX_RXCLKSRC,
		4)) < 0) {
		goto NO_IFACE;
	}

	/* Register a receive interface */
	if ((err = sdi_register_iface (card,
		MASTER_DIRECTION_RX,
		&sdim_rxfops,
		0,
		4)) < 0) {
		goto NO_IFACE;
	}

	return 0;

NO_IFACE:
	sdim_pci_remove (dev);
NO_DEV:
NO_MEM:
NO_PCI:
	return err;
}

/**
 * sdim_pci_remove - PCI removal handler for a SDI Master
 * @dev: PCI device
 *
 * Handle the removal of a SDI Master.
 **/
static void
sdim_pci_remove (struct pci_dev *dev)
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
 * sdim_irq_handler - SDI Master interrupt service routine
 * @irq: interrupt number
 * @dev_id: pointer to the device data structure
 * @regs: processor context
 **/
static irqreturn_t
IRQ_HANDLER(sdim_irq_handler,irq,dev_id,regs)
{
	struct master_dev *card = dev_id;
	unsigned int dmaintsrc = readl (card->bridge_addr + LSDMA_INTSRC);
	unsigned int status, interrupting_iface = 0;
	struct master_iface *txiface = list_entry (card->iface_list.next,
		struct master_iface, list);
	struct master_iface *rxiface = list_entry (card->iface_list.prev,
		struct master_iface, list);

	if (dmaintsrc & LSDMA_INTSRC_CH(0)) {
		/* Read the interrupt type and clear it */
		spin_lock (&card->irq_lock);
		status = readl (card->bridge_addr + LSDMA_CSR(0));
		writel (status, card->bridge_addr + LSDMA_CSR(0));
		spin_unlock (&card->irq_lock);

		/* Increment the buffer pointer */
		if (status & LSDMA_CH_CSR_INTSRCBUFFER) {
			lsdma_advance (txiface->dma);
		}

		/* Flag end-of-chain */
		if (status & LSDMA_CH_CSR_INTSRCDONE) {
			set_bit (SDI_EVENT_TX_BUFFER_ORDER, &txiface->events);
			set_bit (0, &txiface->dma_done);
		}

		/* Flag DMA abort */
		if (status & LSDMA_CH_CSR_INTSRCSTOP) {
			set_bit (0, &txiface->dma_done);
		}

		interrupting_iface |= 0x1;
	}

	if (dmaintsrc & LSDMA_INTSRC_CH(1)) {
		struct lsdma_dma *dma = rxiface->dma;

		/* Read the interrupt type and clear it */
		spin_lock (&card->irq_lock);
		status = readl (card->bridge_addr + LSDMA_CSR(1));
		writel (status, card->bridge_addr + LSDMA_CSR(1));
		spin_unlock (&card->irq_lock);

		/* Increment the buffer pointer */
		if (status & LSDMA_CH_CSR_INTSRCBUFFER) {
			lsdma_advance (dma);
			if (lsdma_rx_isempty (dma)) {
				set_bit (SDI_EVENT_RX_BUFFER_ORDER, &rxiface->events);
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

		/* Clear the source of the interrupt */
		spin_lock (&card->irq_lock);
		status = master_readl (card, SDIM_ICSR);
		master_writel (card, SDIM_ICSR, status);
		spin_unlock (&card->irq_lock);

		if (status & SDIM_ICSR_TXUIS) {
			set_bit (SDI_EVENT_TX_FIFO_ORDER,
				&txiface->events);
			interrupting_iface |= 0x1;
		}
		if (status & SDIM_ICSR_TXDIS) {
			set_bit (SDI_EVENT_TX_DATA_ORDER,
				&txiface->events);
			interrupting_iface |= 0x1;
		}
		if (status & SDIM_ICSR_RXCDIS) {
			set_bit (SDI_EVENT_RX_CARRIER_ORDER,
				&rxiface->events);
			interrupting_iface |= 0x2;
		}
		if (status & SDIM_ICSR_RXOIS) {
			set_bit (SDI_EVENT_RX_FIFO_ORDER,
				&rxiface->events);
			interrupting_iface |= 0x2;
		}

	if (interrupting_iface) {
		/* Dummy read to flush PCI posted writes */
		readl (card->bridge_addr + LSDMA_INTMSK);

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
 * sdim_txinit - Initialize the SDI Master transmitter
 * @iface: interface
 **/
static void
sdim_txinit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
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
	case SDI_CTL_TX_CLKSRC_EXT:
		reg |= SDIM_TCSR_EXTCLK;
		break;
	case SDI_CTL_TX_CLKSRC_RX:
		reg |= SDIM_TCSR_RXCLK;
		break;
	}
	/* There will be no races on TCSR
	 * until this code returns, so we don't need to lock it */
	master_writel (card, SDIM_TCSR, reg | SDIM_TCSR_RST);
	wmb ();
	master_writel (card, SDIM_TCSR, reg);
	wmb ();
	master_writel (card, SDIM_TFCR,
		(SDIM_TFSL << 16) | SDIM_TDMATL);

	return;
}

/**
 * sdim_txstart - Activate the SDI Master transmitter
 * @iface: interface
 **/
static void
sdim_txstart (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;

	/* Enable DMA */
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE,
		card->bridge_addr + LSDMA_CSR(0));

	/* Enable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = master_readl (card, SDIM_ICSR) &
		SDIM_ICSR_RXCTRLMASK;
	reg |= SDIM_ICSR_TXUIE | SDIM_ICSR_TXDIE;
	master_writel (card, SDIM_ICSR, reg);
	spin_unlock_irq (&card->irq_lock);

	/* Enable the transmitter.
	 * There will be no races on TCSR
	 * until this code returns, so we don't need to lock it */
	reg = master_readl (card, SDIM_TCSR);
	master_writel (card, SDIM_TCSR, reg | SDIM_TCSR_EN);

	/* Disable RP178 pattern generation.
	 * There will be no races on TCSR
	 * until this code returns, so we don't need to lock it */
	reg = master_readl (card, SDIM_TCSR);
	master_writel (card, SDIM_TCSR, reg | SDIM_TCSR_RP178);

	return;
}

/**
 * sdim_txstop - Deactivate the SDI Master transmitter
 * @iface: interface
 **/
static void
sdim_txstop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	struct lsdma_dma *dma = iface->dma;
	unsigned int reg;

	lsdma_tx_link_all (dma);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	lsdma_reset (dma);

	/* Wait for the onboard FIFOs to empty */
	/* We don't lock since this should be an atomic read */
	wait_event (iface->queue,
		!(master_readl (card, SDIM_ICSR) & SDIM_ICSR_TXD));

	/* Disable the transmitter.
	 * There will be no races on TCSR here,
	 * so we don't need to lock it */
	reg = master_readl (card, SDIM_TCSR);
	master_writel (card, SDIM_TCSR, reg & ~SDIM_TCSR_EN);

	/* Re-Enable RP178 pattern generation.
	 * There will be no races on TCSR here,
	 * so we don't need to lock it */
	reg = master_readl (card, SDIM_TCSR);
	master_writel (card, SDIM_TCSR, reg & ~SDIM_TCSR_RP178);

	/* Disable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = master_readl (card, SDIM_ICSR) &
		SDIM_ICSR_RXCTRLMASK;
	reg |= SDIM_ICSR_TXUIS | SDIM_ICSR_TXDIS;
	master_writel (card, SDIM_ICSR, reg);
	spin_unlock_irq (&card->irq_lock);

	/* Disable DMA */
	writel ((LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE) &
		~LSDMA_CH_CSR_ENABLE,
		card->bridge_addr + LSDMA_CSR(0));

	return;
}

/**
 * sdim_txexit - Clean up the SDI Master transmitter
 * @iface: interface
 **/
static void
sdim_txexit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;

	/* Reset the transmitter */
	master_writel (card, SDIM_TCSR, SDIM_TCSR_RST);

	return;
}

/**
 * sdim_txopen - SDI Master transmitter open() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
sdim_txopen (struct inode *inode, struct file *filp)
{
	return masterlsdma_open (inode,
		filp,
		sdim_txinit,
		sdim_txstart,
		SDIM_FIFO,
		LSDMA_MMAP);
}

/**
 * sdim_txunlocked_ioctl - SDI Master transmitter unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
sdim_txunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;

	switch (cmd) {
	case SDI_IOC_TXGETBUFLEVEL:
		if (put_user (lsdma_tx_buflevel (iface->dma),
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_TXGETTXD:
		/* We don't lock since this should be an atomic read */
		if (put_user ((master_readl (card, SDIM_ICSR) &
			SDIM_ICSR_TXD) ? 1 : 0, (int *)arg)) {
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
 * sdim_txioctl - SDI Master transmitter ioctl() method
 * @inode: inode
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
sdim_txioctl (struct inode *inode,
	struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	return sdim_txunlocked_ioctl (filp, cmd, arg);
}

/**
 * sdim_txfsync - SDI Master transmitter fsync() method
 * @filp: file to flush
 * @dentry: directory entry associated with the file
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
sdim_txfsync (struct file *filp,
	struct dentry *dentry,
	int datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	struct lsdma_dma *dma = iface->dma;

	if (down_interruptible (&iface->buf_sem)) {
		return -ERESTARTSYS;
	}
	lsdma_tx_link_all (dma);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	lsdma_reset (dma);


	/* Wait for the onboard FIFOs to empty */
	/* We don't lock since this should be an atomic read */
	wait_event (iface->queue,
		!(master_readl (card, SDIM_ICSR) &
		SDIM_ICSR_TXD));

	up (&iface->buf_sem);
	return 0;
}

/**
 * sdim_txrelease - SDI Master transmitter release() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
sdim_txrelease (struct inode *inode, struct file *filp)
{
	struct master_iface *iface = filp->private_data;

	return masterlsdma_release (iface, sdim_txstop, sdim_txexit);
}

/**
 * sdim_rxinit - Initialize the SDI Master receiver
 * @iface: interface
 **/
static void
sdim_rxinit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg = 0;

	switch (iface->mode) {
	default:
	case SDI_CTL_MODE_8BIT:
		reg |= 0;
		break;
	case SDI_CTL_MODE_10BIT:
		reg |= SDIM_RCSR_10BIT;
		break;
	}
	/* There will be no races on RCSR
	 * until this code returns, so we don't need to lock it */
	master_writel (card, SDIM_RCSR, reg | SDIM_RCSR_RST);
	wmb ();
	master_writel (card, SDIM_RCSR, reg);
	wmb ();
	master_writel (card, SDIM_RFCR, SDIM_RDMATL);

	return;
}

/**
 * sdim_rxstart - Activate the SDI Master receiver
 * @iface: interface
 **/
static void
sdim_rxstart (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;

	/* Enable and start DMA */
	writel (lsdma_dma_to_desc_low (lsdma_head_desc_bus_addr (iface->dma)),
		card->bridge_addr + LSDMA_DESC(1));
	clear_bit (0, &iface->dma_done);
	wmb ();
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION | LSDMA_CH_CSR_ENABLE,
		card->bridge_addr + LSDMA_CSR(1));

	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);

	/* Enable receiver interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = master_readl (card, SDIM_ICSR) &
		SDIM_ICSR_TXCTRLMASK;
	reg |= SDIM_ICSR_RXCDIE | SDIM_ICSR_RXOIE;
	master_writel (card, SDIM_ICSR, reg);
	spin_unlock_irq (&card->irq_lock);

	/* Enable the receiver */
	spin_lock (&card->reg_lock);
	reg = master_readl (card, SDIM_RCSR);
	master_writel (card, SDIM_RCSR, reg | SDIM_RCSR_EN);
	spin_unlock (&card->reg_lock);

	return;
}

/**
 * sdim_rxstop - Deactivate the SDI Master receiver
 * @iface: interface
 **/
static void
sdim_rxstop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;

	/* Disable the receiver */
	spin_lock (&card->reg_lock);
	reg = master_readl (card, SDIM_RCSR);
	master_writel (card, SDIM_RCSR, reg & ~SDIM_RCSR_EN);
	spin_unlock (&card->reg_lock);

	/* Disable receiver interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = master_readl (card, SDIM_ICSR) &
		SDIM_ICSR_TXCTRLMASK;
	reg |= SDIM_ICSR_RXCDIS | SDIM_ICSR_RXOIS |
		SDIM_ICSR_RXDIS;
	master_writel (card, SDIM_ICSR, reg);
	spin_unlock_irq (&card->irq_lock);

	/* Disable and abort DMA */
	writel ((LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION) & ~LSDMA_CH_CSR_ENABLE,
		card->bridge_addr + LSDMA_CSR(1));
	wmb ();
	writel ((LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION | LSDMA_CH_CSR_STOP) &
		~LSDMA_CH_CSR_ENABLE,
		card->bridge_addr + LSDMA_CSR(1));

	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	writel ((LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION) & ~LSDMA_CH_CSR_ENABLE,
		card->bridge_addr + LSDMA_CSR(1));

	return;
}

/**
 * sdim_rxexit - Clean up the SDI Master receiver
 * @iface: interface
 **/
static void
sdim_rxexit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;

	/* Reset the receiver.
	 * There will be no races on RCSR here,
	 * so we don't need to lock it */
	master_writel (card, SDIM_RCSR, SDIM_RCSR_RST);

	return;
}

/**
 * sdim_rxopen - SDI Master receiver open() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
sdim_rxopen (struct inode *inode, struct file *filp)
{
	return masterlsdma_open (inode,
		filp,
		sdim_rxinit,
		sdim_rxstart,
		SDIM_FIFO,
		LSDMA_MMAP);
}

/**
 * sdim_rxunlocked_ioctl - SDI Master receiver unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
sdim_rxunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;

	switch (cmd) {
	case SDI_IOC_RXGETBUFLEVEL:
		if (put_user (lsdma_rx_buflevel (iface->dma),
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_RXGETCARRIER:
		/* We don't lock since this should be an atomic read */
		if (put_user ((master_readl (card, SDIM_ICSR) &
			SDIM_ICSR_RXCD) ? 1 : 0, (int *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_RXGETSTATUS:
		/* We don't lock since this should be an atomic read */
		if (put_user ((master_readl (card, SDIM_ICSR) &
			SDIM_ICSR_RXPASSING) ? 1 : 0, (int *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_QBUF:
		return masterlsdma_rxqbuf (filp, arg);
	case SDI_IOC_DQBUF:
		return masterlsdma_rxdqbuf (filp, arg);
	default:
		return sdi_rxioctl (iface, cmd, arg);
	}
	return 0;
}

/**
 * sdim_rxioctl - SDI Master receiver ioctl() method
 * @inode: inode
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
sdim_rxioctl (struct inode *inode,
	struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	return sdim_rxunlocked_ioctl (filp, cmd, arg);
}

/**
 * sdim_rxfsync - SDI Master receiver fsync() method
 * @filp: file to flush
 * @dentry: directory entry associated with the file
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
sdim_rxfsync (struct file *filp,
	struct dentry *dentry,
	int datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	unsigned int reg;

	if (down_interruptible (&iface->buf_sem)) {
		return -ERESTARTSYS;
	}

	/* Stop the receiver */
	sdim_rxstop (iface);

	/* Reset the onboard FIFO and driver buffers */
	spin_lock (&card->reg_lock);
	reg = master_readl (card, SDIM_RCSR);
	master_writel (card, SDIM_RCSR, reg | SDIM_RCSR_RST);
	wmb ();
	master_writel (card, SDIM_RCSR, reg);
	spin_unlock (&card->reg_lock);
	iface->events = 0;
	lsdma_reset (iface->dma);

	/* Start the receiver */
	sdim_rxstart (iface);

	up (&iface->buf_sem);
	return 0;
}

/**
 * sdim_rxrelease - SDI Master receiver release() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
sdim_rxrelease (struct inode *inode, struct file *filp)
{
	struct master_iface *iface = filp->private_data;

	return masterlsdma_release (iface, sdim_rxstop, sdim_rxexit);
}

/**
 * sdim_init_module - register the module as a PCI driver
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int __init
sdim_init_module (void)
{
	printk (KERN_INFO "%s: Linear Systems Ltd. "
		"SDI Master driver from master-%s (%s)\n",
		sdim_driver_name, MASTER_DRIVER_VERSION, MASTER_DRIVER_DATE);

	return mdev_init_module (&sdim_pci_driver,
		&sdim_class,
		sdim_driver_name);
}

/**
 * sdim_cleanup_module - unregister the module as a PCI driver
 **/
static void __exit
sdim_cleanup_module (void)
{
	mdev_cleanup_module (&sdim_pci_driver, &sdim_class);
	return;
}

module_init (sdim_init_module);
module_exit (sdim_cleanup_module);

