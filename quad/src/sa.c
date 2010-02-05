/* sa.c
 *
 * Linux driver for Linear Systems Ltd. MultiMaster SDI-R.
 *
 * Copyright (C) 2004-2007 Linear Systems Ltd.
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

#include <linux/types.h> /* size_t, loff_t, u32 */
#include <linux/init.h> /* __init */
#include <linux/pci.h> /* pci_dev */
#include <linux/dma-mapping.h> /* DMA_32BIT_MASK */
#include <linux/fs.h> /* inode, file, file_operations */
#include <linux/sched.h> /* pt_regs */
#include <linux/slab.h> /* kmalloc () */
#include <linux/spinlock.h> /* spin_lock_init () */
#include <linux/list.h> /* list_head */
#include <linux/errno.h> /* error codes */
#include <linux/interrupt.h> /* irqreturn_t */

#include <asm/semaphore.h> /* sema_init () */
#include <asm/uaccess.h> /* put_user () */
#include <asm/bitops.h> /* set_bit () */

#include "asicore.h"
#include "sdicore.h"
#include "../include/master.h"
// Temporary fix for Linux kernel 2.6.21
#include "mdev.c"
#include "miface.h"
#include "plx9080.h"
// Temporary fix for Linux kernel 2.6.21
#include "masterplx.c"
#include "mmsa.h"

static const char mmsa_name[] = MMSA_NAME;
static const char mmsae_name[] = MMSAE_NAME;

/* Static function prototypes */
static ssize_t mmsa_show_blackburst_type (struct class_device *cd,
	char *buf);
static ssize_t mmsa_store_blackburst_type (struct class_device *cd,
	const char *buf,
	size_t count);
static ssize_t mmsa_show_uid (struct class_device *cd,
	char *buf);
static int mmsa_pci_probe (struct pci_dev *dev,
	const struct pci_device_id *id) __devinit;
static void mmsa_pci_remove (struct pci_dev *dev);
static irqreturn_t IRQ_HANDLER(mmsa_irq_handler,irq,dev_id,regs);
static void mmsa_txinit (struct master_iface *iface);
static void mmsa_txstart (struct master_iface *iface);
static void mmsa_txstop (struct master_iface *iface);
static void mmsa_txexit (struct master_iface *iface);
static int mmsa_txopen (struct inode *inode, struct file *filp);
static long mmsa_txunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int mmsa_txioctl (struct inode *inode,
	struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int mmsa_txfsync (struct file *filp,
	struct dentry *dentry,
	int datasync);
static int mmsa_txrelease (struct inode *inode, struct file *filp);
static void mmsa_rxinit (struct master_iface *iface);
static void mmsa_rxstart (struct master_iface *iface);
static void mmsa_rxstop (struct master_iface *iface);
static void mmsa_rxexit (struct master_iface *iface);
static int mmsa_rxopen (struct inode *inode, struct file *filp);
static long mmsa_rxunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int mmsa_rxioctl (struct inode *inode,
	struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int mmsa_rxfsync (struct file *filp,
	struct dentry *dentry,
	int datasync);
static int mmsa_rxrelease (struct inode *inode, struct file *filp);
static int mmsa_init_module (void) __init;
static void mmsa_cleanup_module (void) __exit;

MODULE_AUTHOR("Linear Systems Ltd.");
MODULE_DESCRIPTION("MultiMaster SDI-R driver");
MODULE_LICENSE("GPL");

#ifdef MODULE_VERSION
MODULE_VERSION(MASTER_DRIVER_VERSION);
#endif

static char mmsa_driver_name[] = "mmsa";

static struct pci_device_id mmsa_pci_id_table[] = {
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			MMSA_PCI_DEVICE_ID_LINSYS)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			MMSAE_PCI_DEVICE_ID_LINSYS)
	},
	{0, }
};

static struct pci_driver mmsa_pci_driver = {
	.name = mmsa_driver_name,
	.id_table = mmsa_pci_id_table,
	.probe = mmsa_pci_probe,
	.remove = mmsa_pci_remove
};

MODULE_DEVICE_TABLE(pci,mmsa_pci_id_table);

static struct file_operations mmsa_txfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.write = masterplx_write,
	.poll = masterplx_txpoll,
	.ioctl = mmsa_txioctl,
#ifdef HAVE_UNLOCKED_IOCTL
	.unlocked_ioctl = mmsa_txunlocked_ioctl,
#endif
#ifdef HAVE_COMPAT_IOCTL
	.compat_ioctl = asi_compat_ioctl,
#endif
	.mmap = NULL,
	.open = mmsa_txopen,
	.release = mmsa_txrelease,
	.fsync = mmsa_txfsync,
	.fasync = NULL
};

static struct file_operations mmsa_rxfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.read = masterplx_read,
	.poll = masterplx_rxpoll,
	.ioctl = mmsa_rxioctl,
#ifdef HAVE_UNLOCKED_IOCTL
	.unlocked_ioctl = mmsa_rxunlocked_ioctl,
#endif
#ifdef HAVE_COMPAT_IOCTL
	.compat_ioctl = sdi_compat_ioctl,
#endif
	.mmap = masterplx_mmap,
	.open = mmsa_rxopen,
	.release = mmsa_rxrelease,
	.fsync = mmsa_rxfsync,
	.fasync = NULL
};

static LIST_HEAD(mmsa_card_list);

static struct class mmsa_class = {
	.name = mmsa_driver_name,
	.release = mdev_class_device_release,
	.class_release = NULL
};

/**
 * mmsa_show_blackburst_type - interface attribute read handler
 * @cd: class_device being read
 * @buf: output buffer
 **/
static ssize_t
mmsa_show_blackburst_type (struct class_device *cd,
	char *buf)
{
	struct master_dev *card = to_master_dev(cd);

	return snprintf (buf, PAGE_SIZE, "%u\n",
		(master_inl (card, MMSA_TCSR) & MMSA_TCSR_PAL) >> 13);
}

/**
 * mmsa_store_blackburst_type - interface attribute write handler
 * @cd: class_device being written
 * @buf: input buffer
 * @count:
 **/
static ssize_t
mmsa_store_blackburst_type (struct class_device *cd,
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
	reg = master_inl (card, MMSA_TCSR) & ~MMSA_TCSR_PAL;
	master_outl (card, MMSA_TCSR, reg | (val << 13));
	spin_unlock (&card->reg_lock);
	return retcode;
}

/**
 * mmsa_show_uid - interface attribute read handler
 * @cd: class_device being read
 * @buf: output buffer
 **/
static ssize_t
mmsa_show_uid (struct class_device *cd,
	char *buf)
{
	struct master_dev *card = to_master_dev(cd);

	return snprintf (buf, PAGE_SIZE, "0x%08X%08X\n",
		readl (card->core.addr + MMSA_UIDR_HI),
		readl (card->core.addr + MMSA_UIDR_LO));
}

static CLASS_DEVICE_ATTR(blackburst_type,S_IRUGO|S_IWUSR,
	mmsa_show_blackburst_type,mmsa_store_blackburst_type);
static CLASS_DEVICE_ATTR(uid,S_IRUGO,
	mmsa_show_uid,NULL);

/**
 * mmsa_pci_probe - PCI insertion handler
 * @dev: PCI device
 * @id: PCI ID
 *
 * Handle the insertion of a MultiMaster SDI-R.
 * Returns a negative error code on failure and 0 on success.
 **/
static int __devinit
mmsa_pci_probe (struct pci_dev *dev,
	const struct pci_device_id *id)
{
	int err;
	unsigned int version, cap;
	struct master_dev *card;
	const char *name;

	/* Initialize the driver_data pointer so that mmsa_pci_remove()
	 * doesn't try to free it if an error occurs */
	pci_set_drvdata (dev, NULL);

	/* Wake a sleeping device */
	if ((err = pci_enable_device (dev)) < 0) {
		printk (KERN_WARNING "%s: unable to enable device\n",
			mmsa_driver_name);
		return err;
	}

	/* Set PCI DMA addressing limitations */
	if ((err = pci_set_dma_mask (dev, DMA_32BIT_MASK)) < 0) {
		printk (KERN_WARNING "%s: unable to set PCI DMA mask\n",
			mmsa_driver_name);
		return err;
	}

	/* Enable bus mastering */
	pci_set_master (dev);

	/* Request the PCI I/O resources */
	if ((err = pci_request_regions (dev, mmsa_driver_name)) < 0) {
		goto NO_PCI;
	}

	/* Print the firmware version */
	switch (dev->device) {
	case MMSA_PCI_DEVICE_ID_LINSYS:
		name = mmsa_name;
		break;
	case MMSAE_PCI_DEVICE_ID_LINSYS:
		name = mmsae_name;
		break;
	default:
		name = "";
		break;
	}
	version = inl (pci_resource_start (dev, 2) + MMSA_CSR) >> 16;
	printk (KERN_INFO "%s: %s detected, firmware version %u.%u (0x%04X)\n",
		mmsa_driver_name, mmsa_name,
		version >> 8, version & 0x00ff, version);

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
	card->version = version;
	card->name = name;
	card->irq_handler = mmsa_irq_handler;

	switch (dev->device) {
	case MMSA_PCI_DEVICE_ID_LINSYS:
		card->capabilities = MASTER_CAP_BLACKBURST;
		break;
	case MMSAE_PCI_DEVICE_ID_LINSYS:
		card->capabilities = MASTER_CAP_BLACKBURST | MASTER_CAP_UID;
		break;
	}

	INIT_LIST_HEAD(&card->iface_list);
	/* Lock for ICSR, DMACSR1 */
	spin_lock_init (&card->irq_lock);
	/* Lock for TCSR, RCSR */
	spin_lock_init (&card->reg_lock);
	sema_init (&card->users_sem, 1);
	card->pdev = dev;

	/* Store the pointer to the board info structure
	 * in the PCI info structure */
	pci_set_drvdata (dev, card);

	/* Reset the PCI 9056 */
	masterplx_reset_bridge (card);

	/* Setup the PCI 9056 */
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

	/* Reset the FPGA */
	master_outl (card, MMSA_TCSR, MMSA_TCSR_RST);
	master_outl (card, MMSA_RCSR, MMSA_RCSR_RST);

	/* Register a Master device */
	if ((err = mdev_register (card,
		&mmsa_card_list,
		mmsa_driver_name,
		&mmsa_class)) < 0) {
		goto NO_DEV;
	}

	if (card->capabilities & MASTER_CAP_BLACKBURST) {
		if ((err = class_device_create_file (&card->class_dev,
			&class_device_attr_blackburst_type)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'blackburst_type'\n",
				mmsa_driver_name);
		}
	}
	if (card->capabilities & MASTER_CAP_UID) {
		if ((err = class_device_create_file (&card->class_dev,
			&class_device_attr_uid)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'uid'\n",
				mmsa_driver_name);
		}
	}

	/* Register a transmit interface */
	cap = ASI_CAP_TX_MAKE204 | ASI_CAP_TX_FINETUNING |
		ASI_CAP_TX_BYTECOUNTER | ASI_CAP_TX_SETCLKSRC |
		ASI_CAP_TX_FIFOUNDERRUN | ASI_CAP_TX_LARGEIB |
		ASI_CAP_TX_INTERLEAVING | ASI_CAP_TX_DATA |
		ASI_CAP_TX_27COUNTER |
		ASI_CAP_TX_TIMESTAMPS | ASI_CAP_TX_PTIMESTAMPS |
		ASI_CAP_TX_NULLPACKETS;
	cap |= (version >= 0x0302) ? ASI_CAP_TX_RXCLKSRC : 0;
	if ((err = asi_register_iface (card,
		MASTER_DIRECTION_TX,
		&mmsa_txfops,
		cap,
		4,
		ASI_CTL_TRANSPORT_DVB_ASI)) < 0) {
		goto NO_IFACE;
	}

	/* Register a receive interface */
	if ((err = sdi_register_iface (card,
		MASTER_DIRECTION_RX,
		&mmsa_rxfops,
		0,
		4)) < 0) {
		goto NO_IFACE;
	}

	return 0;

NO_IFACE:
	mmsa_pci_remove (dev);
NO_DEV:
NO_MEM:
NO_PCI:
	return err;
}

/**
 * mmsa_pci_remove - PCI removal handler
 * @dev: PCI device
 **/
static void
mmsa_pci_remove (struct pci_dev *dev)
{
	struct master_dev *card = pci_get_drvdata (dev);

	if (card) {
		struct list_head *p;
		struct master_iface *iface;

		if (!list_empty (&card->iface_list)) {
			iface = list_entry (card->iface_list.next,
					struct master_iface, list);
			asi_unregister_iface (iface);
		}
		if (!list_empty (&card->iface_list)) {
			iface = list_entry (card->iface_list.next,
					struct master_iface, list);
			sdi_unregister_iface (iface);
		}
		iounmap (card->bridge_addr);
		list_for_each (p, &mmsa_card_list) {
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
 * mmsa_irq_handler - MultiMaster SDI-R interrupt service routine
 * @irq: interrupt number
 * @dev_id: pointer to the device data structure
 * @regs: processor context
 **/
static irqreturn_t
IRQ_HANDLER(mmsa_irq_handler,irq,dev_id,regs)
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
		plx_advance (txiface->dma);

		/* Flag end-of-chain */
		if (status & PLX_DMACSR_DONE) {
			set_bit (ASI_EVENT_TX_BUFFER_ORDER, &txiface->events);
			set_bit (0, &txiface->dma_done);
		}

		interrupting_iface |= 0x1;
	}
	if (intcsr & PLX_INTCSR_DMA1INT_ACTIVE) {
		struct plx_dma *dma = rxiface->dma;

		/* Read the interrupt type and clear it */
		spin_lock (&card->irq_lock);
		status = readb (card->bridge_addr + PLX_DMACSR1);
		writeb (status | PLX_DMACSR_CLINT,
			card->bridge_addr + PLX_DMACSR1);
		spin_unlock (&card->irq_lock);

		/* Increment the buffer pointer */
		plx_advance (dma);

		if (plx_rx_isempty (dma)) {
			set_bit (SDI_EVENT_RX_BUFFER_ORDER, &rxiface->events);
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
		status = master_inl (card, MMSA_ICSR);
		master_outl (card, MMSA_ICSR, status);
		spin_unlock (&card->irq_lock);

		if (status & MMSA_ICSR_TXUIS) {
			set_bit (ASI_EVENT_TX_FIFO_ORDER,
				&txiface->events);
			interrupting_iface |= 0x1;
		}
		if (status & MMSA_ICSR_TXDIS) {
			set_bit (ASI_EVENT_TX_DATA_ORDER,
				&txiface->events);
			interrupting_iface |= 0x1;
		}
		if (status & MMSA_ICSR_RXCDIS) {
			set_bit (SDI_EVENT_RX_CARRIER_ORDER,
				&rxiface->events);
			interrupting_iface |= 0x2;
		}
		if (status & MMSA_ICSR_RXOIS) {
			set_bit (SDI_EVENT_RX_FIFO_ORDER,
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
 * mmsa_txinit - Initialize the MultiMaster SDI-R transmitter
 * @iface: interface
 **/
static void
mmsa_txinit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg = iface->null_packets ? MMSA_TCSR_NP : 0;

	switch (iface->timestamps) {
	default:
	case ASI_CTL_TSTAMP_NONE:
		reg |= 0;
		break;
	case ASI_CTL_TSTAMP_APPEND:
		reg |= MMSA_TCSR_TSS;
		break;
	case ASI_CTL_TSTAMP_PREPEND:
		reg |= MMSA_TCSR_PRC;
		break;
	}
	switch (iface->mode) {
	default:
	case ASI_CTL_TX_MODE_188:
		reg |= 0;
		break;
	case ASI_CTL_TX_MODE_204:
		reg |= MMSA_TCSR_204;
		break;
	case ASI_CTL_TX_MODE_MAKE204:
		reg |= MMSA_TCSR_MAKE204;
		break;
	}
	switch (iface->clksrc) {
	default:
	case ASI_CTL_TX_CLKSRC_ONBOARD:
		reg |= 0;
		break;
	case ASI_CTL_TX_CLKSRC_EXT:
		reg |= MMSA_TCSR_EXTCLK;
		break;
	case ASI_CTL_TX_CLKSRC_RX:
		reg |= MMSA_TCSR_RXCLK;
		break;
	}
	/* There will be no races on IBSTR, IPSTR, FTR, and TCSR
	 * until this code returns, so we don't need to lock them */
	master_outl (card, MMSA_TCSR, reg | MMSA_TCSR_RST);
	wmb ();
	master_outl (card, MMSA_TCSR, reg);
	wmb ();
	master_outl (card, MMSA_TFCR,
		(MMSA_TFSL << 16) | MMSA_TDMATL);
	master_outl (card, MMSA_IBSTR, 0);
	master_outl (card, MMSA_IBSTR, 0);
	master_outl (card, MMSA_FTR, 0);

	/* Reset byte counter */
	master_inl (card, MMSA_TXBCOUNTR);

	return;
}

/**
 * mmsa_txstart - Activate the MultiMaster SDI-R transmitter
 * @iface: interface
 **/
static void
mmsa_txstart (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;

	/* Enable DMA */
	writeb (PLX_DMACSR_ENABLE, card->bridge_addr + PLX_DMACSR0);

	/* Enable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = master_inl (card, MMSA_ICSR) &
		MMSA_ICSR_RXCTRLMASK;
	reg |= MMSA_ICSR_TXUIE | MMSA_ICSR_TXDIE;
	master_outl (card, MMSA_ICSR, reg);
	spin_unlock_irq (&card->irq_lock);

	/* Enable the transmitter.
	 * There will be no races on TCSR
	 * until this code returns, so we don't need to lock it */
	reg = master_inl (card, MMSA_TCSR);
	master_outl (card, MMSA_TCSR, reg | MMSA_TCSR_EN);

	return;
}

/**
 * mmsa_txstop - Deactivate the MultiMaster SDI-R transmitter
 * @iface: interface
 **/
static void
mmsa_txstop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	struct plx_dma *dma = iface->dma;
	unsigned int reg;

	plx_tx_link_all (dma);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	plx_reset (dma);

	if (!iface->null_packets) {
		/* Wait for the onboard FIFOs to empty */
		/* We don't lock since this should be an atomic read */
		wait_event (iface->queue,
			!(master_inl (card, MMSA_ICSR) & MMSA_ICSR_TXD));
	}

	/* Disable the transmitter.
	 * There will be no races on TCSR here,
	 * so we don't need to lock it */
	reg = master_inl (card, MMSA_TCSR);
	master_outl (card, MMSA_TCSR, reg & ~MMSA_TCSR_EN);

	/* Disable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = master_inl (card, MMSA_ICSR) &
		MMSA_ICSR_RXCTRLMASK;
	reg |= MMSA_ICSR_TXUIS | MMSA_ICSR_TXDIS;
	master_outl (card, MMSA_ICSR, reg);
	spin_unlock_irq (&card->irq_lock);

	/* Disable DMA */
	writeb (0, card->bridge_addr + PLX_DMACSR0);

	return;
}

/**
 * mmsa_txexit - Clean up the MultiMaster SDI-R transmitter
 * @iface: interface
 **/
static void
mmsa_txexit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;

	/* Reset the transmitter */
	master_outl (card, MMSA_TCSR, MMSA_TCSR_RST);

	return;
}

/**
 * mmsa_txopen - MultiMaster SDI-R transmitter open() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
mmsa_txopen (struct inode *inode, struct file *filp)
{
	return masterplx_open (inode,
		filp,
		mmsa_txinit,
		mmsa_txstart,
		MMSA_FIFO,
		0);
}

/**
 * mmsa_txunlocked_ioctl - MultiMaster SDI-R transmitter unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
mmsa_txunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	struct asi_txstuffing stuffing;

	switch (cmd) {
	case ASI_IOC_TXGETBUFLEVEL:
		if (put_user (plx_tx_buflevel (iface->dma),
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_TXSETSTUFFING:
		if (copy_from_user (&stuffing, (struct asi_txstuffing *)arg,
			sizeof (stuffing))) {
			return -EFAULT;
		}
		if ((stuffing.ib > 0xffff) ||
			(stuffing.ip > 0xffffff) ||
			(stuffing.normal_ip > 0xff) ||
			(stuffing.big_ip > 0xff) ||
			((stuffing.il_normal + stuffing.il_big) > 0xf) ||
			(stuffing.il_normal > stuffing.normal_ip) ||
			(stuffing.il_big > stuffing.big_ip)) {
			return -EINVAL;
		}
		spin_lock (&card->reg_lock);
		master_outl (card, MMSA_IBSTR, stuffing.ib);
		master_outl (card, MMSA_IPSTR, stuffing.ip);
		master_outl (card, MMSA_FTR,
			(stuffing.il_big << 24) |
			(stuffing.big_ip << 16) |
			(stuffing.il_normal << 8) |
			stuffing.normal_ip);
		spin_unlock (&card->reg_lock);
		break;
	case ASI_IOC_TXGETBYTECOUNT:
		if (put_user (master_inl (card, MMSA_TXBCOUNTR),
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_TXGETTXD:
		/* We don't lock since this should be an atomic read */
		if (put_user ((master_inl (card, MMSA_ICSR) &
			MMSA_ICSR_TXD) ? 1 : 0, (int *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_TXGET27COUNT:
		if (put_user (master_inl (card, MMSA_27COUNTR),
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return asi_txioctl (iface, cmd, arg);
	}
	return 0;
}

/**
 * mmsa_txioctl - MultiMaster SDI-R transmitter ioctl() method
 * @inode: inode
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
mmsa_txioctl (struct inode *inode,
	struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	return mmsa_txunlocked_ioctl (filp, cmd, arg);
}

/**
 * mmsa_txfsync - MultiMaster SDI-R transmitter fsync() method
 * @filp: file to flush
 * @dentry: directory entry associated with the file
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
mmsa_txfsync (struct file *filp,
	struct dentry *dentry,
	int datasync)
{
	struct master_iface *iface = filp->private_data;
	struct plx_dma *dma = iface->dma;

	if (down_interruptible (&iface->buf_sem)) {
		return -ERESTARTSYS;
	}
	plx_tx_link_all (dma);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	plx_reset (dma);

	if (!iface->null_packets) {
		struct master_dev *card = iface->card;

		/* Wait for the onboard FIFOs to empty */
		/* We don't lock since this should be an atomic read */
		wait_event (iface->queue,
			!(master_inl (card, MMSA_ICSR) &
			MMSA_ICSR_TXD));
	}

	up (&iface->buf_sem);
	return 0;
}

/**
 * mmsa_txrelease - MultiMaster SDI-R transmitter release() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
mmsa_txrelease (struct inode *inode, struct file *filp)
{
	struct master_iface *iface = filp->private_data;

	return masterplx_release (iface, mmsa_txstop, mmsa_txexit);
}

/**
 * mmsa_rxinit - Initialize the MultiMaster SDI-R receiver
 * @iface: interface
 **/
static void
mmsa_rxinit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg = 0;

	switch (iface->mode) {
	default:
	case SDI_CTL_MODE_8BIT:
		reg |= 0;
		break;
	case SDI_CTL_MODE_10BIT:
		reg |= MMSA_RCSR_10BIT;
		break;
	}
	/* There will be no races on RCSR
	 * until this code returns, so we don't need to lock it */
	master_outl (card, MMSA_RCSR, reg | MMSA_RCSR_RST);
	wmb ();
	master_outl (card, MMSA_RCSR, reg);
	wmb ();
	master_outl (card, MMSA_RFCR, MMSA_RDMATL);

	return;
}

/**
 * mmsa_rxstart - Activate the MultiMaster SDI-R receiver
 * @iface: interface
 **/
static void
mmsa_rxstart (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;

	/* Enable and start DMA */
	writel (plx_head_desc_bus_addr (iface->dma) |
		PLX_DMADPR_DLOC_PCI | PLX_DMADPR_LB2PCI,
		card->bridge_addr + PLX_DMADPR1);
	writeb (PLX_DMACSR_ENABLE,
		card->bridge_addr + PLX_DMACSR1);
	clear_bit (0, &iface->dma_done);
	wmb ();
	writeb (PLX_DMACSR_ENABLE | PLX_DMACSR_START,
		card->bridge_addr + PLX_DMACSR1);
	/* Dummy read to flush PCI posted writes */
	readb (card->bridge_addr + PLX_DMACSR1);

	/* Enable receiver interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = master_inl (card, MMSA_ICSR) &
		MMSA_ICSR_TXCTRLMASK;
	reg |= MMSA_ICSR_RXCDIE | MMSA_ICSR_RXOIE;
	master_outl (card, MMSA_ICSR, reg);
	spin_unlock_irq (&card->irq_lock);

	/* Enable the receiver */
	spin_lock (&card->reg_lock);
	reg = master_inl (card, MMSA_RCSR);
	master_outl (card, MMSA_RCSR, reg | MMSA_RCSR_EN);
	spin_unlock (&card->reg_lock);

	return;
}

/**
 * mmsa_rxstop - Deactivate the MultiMaster SDI-R receiver
 * @iface: interface
 **/
static void
mmsa_rxstop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;

	/* Disable the receiver */
	spin_lock (&card->reg_lock);
	reg = master_inl (card, MMSA_RCSR);
	master_outl (card, MMSA_RCSR, reg & ~MMSA_RCSR_EN);
	spin_unlock (&card->reg_lock);

	/* Disable receiver interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = master_inl (card, MMSA_ICSR) &
		MMSA_ICSR_TXCTRLMASK;
	reg |= MMSA_ICSR_RXCDIS | MMSA_ICSR_RXOIS |
		MMSA_ICSR_RXDIS;
	master_outl (card, MMSA_ICSR, reg);

	/* Disable and abort DMA */
	writeb (0, card->bridge_addr + PLX_DMACSR1);
	wmb ();
	writeb (PLX_DMACSR_START | PLX_DMACSR_ABORT,
		card->bridge_addr + PLX_DMACSR1);
	/* Dummy read to flush PCI posted writes */
	readb (card->bridge_addr + PLX_DMACSR1);
	spin_unlock_irq (&card->irq_lock);
	wait_event_timeout (iface->queue, test_bit (0, &iface->dma_done), HZ);

	return;
}

/**
 * mmsa_rxexit - Clean up the MultiMaster SDI-R receiver
 * @iface: interface
 **/
static void
mmsa_rxexit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;

	/* Reset the receiver.
	 * There will be no races on RCSR here,
	 * so we don't need to lock it */
	master_outl (card, MMSA_RCSR, MMSA_RCSR_RST);

	return;
}

/**
 * mmsa_rxopen - MultiMaster SDI-R receiver open() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
mmsa_rxopen (struct inode *inode, struct file *filp)
{
	return masterplx_open (inode,
		filp,
		mmsa_rxinit,
		mmsa_rxstart,
		MMSA_FIFO,
		PLX_MMAP);
}

/**
 * mmsa_rxunlocked_ioctl - MultiMaster SDI-R receiver unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
mmsa_rxunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;

	switch (cmd) {
	case SDI_IOC_RXGETBUFLEVEL:
		if (put_user (plx_rx_buflevel (iface->dma),
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_RXGETCARRIER:
		/* We don't lock since this should be an atomic read */
		if (put_user ((master_inl (card, MMSA_ICSR) &
			MMSA_ICSR_RXCD) ? 1 : 0, (int *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_RXGETSTATUS:
		/* We don't lock since this should be an atomic read */
		if (put_user ((master_inl (card, MMSA_ICSR) &
			MMSA_ICSR_RXPASSING) ? 1 : 0, (int *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_QBUF:
		return masterplx_rxqbuf (filp, arg);
	case SDI_IOC_DQBUF:
		return masterplx_rxdqbuf (filp, arg);
	default:
		return sdi_rxioctl (iface, cmd, arg);
	}
	return 0;
}

/**
 * mmsa_rxioctl - MultiMaster SDI-R receiver ioctl() method
 * @inode: inode
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
mmsa_rxioctl (struct inode *inode,
	struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	return mmsa_rxunlocked_ioctl (filp, cmd, arg);
}

/**
 * mmsa_rxfsync - MultiMaster SDI-R receiver fsync() method
 * @filp: file to flush
 * @dentry: directory entry associated with the file
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
mmsa_rxfsync (struct file *filp,
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
	mmsa_rxstop (iface);

	/* Reset the onboard FIFO and driver buffers */
	spin_lock (&card->reg_lock);
	reg = master_inl (card, MMSA_RCSR);
	master_outl (card, MMSA_RCSR, reg | MMSA_RCSR_RST);
	wmb ();
	master_outl (card, MMSA_RCSR, reg);
	spin_unlock (&card->reg_lock);
	iface->events = 0;
	plx_reset (iface->dma);

	/* Start the receiver */
	mmsa_rxstart (iface);

	up (&iface->buf_sem);
	return 0;
}

/**
 * mmsa_rxrelease - MultiMaster SDI-R receiver release() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
mmsa_rxrelease (struct inode *inode, struct file *filp)
{
	struct master_iface *iface = filp->private_data;

	return masterplx_release (iface, mmsa_rxstop, mmsa_rxexit);
}

/**
 * mmsa_init_module - register the module as a PCI driver
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int __init
mmsa_init_module (void)
{
	printk (KERN_INFO "%s: Linear Systems Ltd. "
		"MultiMaster SDI-R driver from master-%s (%s)\n",
		mmsa_driver_name, MASTER_DRIVER_VERSION, MASTER_DRIVER_DATE);

	return mdev_init_module (&mmsa_pci_driver,
		&mmsa_class,
		mmsa_driver_name);
}

/**
 * mmsa_cleanup_module - unregister the module as a PCI driver
 **/
static void __exit
mmsa_cleanup_module (void)
{
	mdev_cleanup_module (&mmsa_pci_driver, &mmsa_class);
	return;
}

module_init (mmsa_init_module);
module_exit (mmsa_cleanup_module);

