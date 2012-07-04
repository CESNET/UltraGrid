/* as.c
 *
 * Linux driver for Linear Systems Ltd. MultiMaster SDI-T.
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

#include <linux/kernel.h> /* KERN_INFO */
#include <linux/module.h> /* MODULE_LICENSE */

#include <linux/types.h> /* size_t, loff_t, u32 */
#include <linux/init.h> /* module_init () */
#include <linux/pci.h> /* pci_resource_start () */
#include <linux/dma-mapping.h> /* DMA_BIT_MASK */
#include <linux/fs.h> /* inode, file, file_operations */
#include <linux/sched.h> /* pt_regs */
#include <linux/slab.h> /* kzalloc () */
#include <linux/spinlock.h> /* spin_lock_init () */
#include <linux/list.h> /* list_head */
#include <linux/errno.h> /* error codes */
#include <linux/interrupt.h> /* irqreturn_t */
#include <linux/device.h> /* device_create_file () */
#include <linux/mutex.h> /* mutex_init () */

#include <asm/uaccess.h> /* put_user () */
#include <asm/bitops.h> /* set_bit () */

#include "asicore.h"
#include "sdicore.h"
#include "../include/master.h"
#include "mdev.h"
#include "mdma.h"
#include "plx9080.h"
#include "mmas.h"

#ifndef DEFINE_PCI_DEVICE_TABLE
#define DEFINE_PCI_DEVICE_TABLE(_table) \
	const struct pci_device_id _table[]
#endif

#ifndef DMA_BIT_MASK
#define DMA_BIT_MASK(n)	(((n) == 64) ? ~0ULL : ((1ULL<<(n))-1))
#endif

static const char mmas_name[] = MMAS_NAME;
static const char mmase_name[] = MMASE_NAME;

/* Static function prototypes */
static ssize_t mmas_show_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static ssize_t mmas_store_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	const char *buf,
	size_t count);
static ssize_t mmas_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static int mmas_pci_probe (struct pci_dev *pdev,
	const struct pci_device_id *id) __devinit;
static void mmas_pci_remove (struct pci_dev *pdev);
static irqreturn_t IRQ_HANDLER(mmas_irq_handler,irq,dev_id,regs);
static void mmas_txinit (struct master_iface *iface);
static void mmas_txstart (struct master_iface *iface);
static void mmas_txstop (struct master_iface *iface);
static void mmas_txexit (struct master_iface *iface);
static void mmas_start_tx_dma (struct master_iface *iface);
static long mmas_txunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int FSYNC_HANDLER(mmas_txfsync,filp,datasync);
static void mmas_rxinit (struct master_iface *iface);
static void mmas_rxstart (struct master_iface *iface);
static void mmas_rxstop (struct master_iface *iface);
static void mmas_rxexit (struct master_iface *iface);
static long mmas_rxunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int FSYNC_HANDLER(mmas_rxfsync,filp,datasync);
static int mmas_init_module (void) __init;
static void mmas_cleanup_module (void) __exit;

MODULE_AUTHOR("Linear Systems Ltd.");
MODULE_DESCRIPTION("MultiMaster SDI-T driver");
MODULE_LICENSE("GPL");
MODULE_VERSION(MASTER_DRIVER_VERSION);

static char mmas_driver_name[] = "mmas";

static DEFINE_PCI_DEVICE_TABLE(mmas_pci_id_table) = {
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			MMAS_PCI_DEVICE_ID_LINSYS)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			MMASE_PCI_DEVICE_ID_LINSYS)
	},
	{0, }
};

static struct pci_driver mmas_pci_driver = {
	.name = mmas_driver_name,
	.id_table = mmas_pci_id_table,
	.probe = mmas_pci_probe,
	.remove = mmas_pci_remove
};

MODULE_DEVICE_TABLE(pci,mmas_pci_id_table);

static struct file_operations mmas_txfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.write = sdi_write,
	.poll = sdi_txpoll,
	.unlocked_ioctl = mmas_txunlocked_ioctl,
	.compat_ioctl = mmas_txunlocked_ioctl,
	.mmap = sdi_mmap,
	.open = sdi_open,
	.release = sdi_release,
	.fsync = mmas_txfsync,
	.fasync = NULL
};

static struct file_operations mmas_rxfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.read = asi_read,
	.poll = asi_rxpoll,
	.unlocked_ioctl = mmas_rxunlocked_ioctl,
	.compat_ioctl = mmas_rxunlocked_ioctl,
	.mmap = NULL,
	.open = asi_open,
	.release = asi_release,
	.fsync = mmas_rxfsync,
	.fasync = NULL
};

static struct master_iface_operations mmas_txops = {
	.init = mmas_txinit,
	.start = mmas_txstart,
	.stop = mmas_txstop,
	.exit = mmas_txexit,
	.start_tx_dma = mmas_start_tx_dma
};

static struct master_iface_operations mmas_rxops = {
	.init = mmas_rxinit,
	.start = mmas_rxstart,
	.stop = mmas_rxstop,
	.exit = mmas_rxexit
};

static LIST_HEAD(mmas_card_list);

static struct class *mmas_class;

/**
 * mmas_show_blackburst_type - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
static ssize_t
mmas_show_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	return snprintf (buf, PAGE_SIZE, "%u\n",
		(master_inl (card, MMAS_TCSR) & MMAS_TCSR_PAL) >> 9);
}

/**
 * mmas_store_blackburst_type - interface attribute write handler
 * @dev: device being written
 * @attr: device attribute
 * @buf: input buffer
 * @count:
 **/
static ssize_t
mmas_store_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	const char *buf,
	size_t count)
{
	struct master_dev *card = dev_get_drvdata(dev);
	char *endp;
	unsigned long val = simple_strtoul (buf, &endp, 0);
	unsigned int reg;
	const unsigned long max = MASTER_CTL_BLACKBURST_PAL;
	int retcode = count;
	struct master_iface *txiface = list_entry (card->iface_list.next,
		struct master_iface, list);

	if ((endp == buf) || (val > max)) {
		return -EINVAL;
	}
	mutex_lock (&card->users_mutex);
	if (txiface->users) {
		retcode = -EBUSY;
		goto OUT;
	}
	spin_lock (&card->reg_lock);
	reg = master_inl (card, MMAS_TCSR) & ~MMAS_TCSR_PAL;
	master_outl (card, MMAS_TCSR, reg | (val << 9));
	spin_unlock (&card->reg_lock);
OUT:
	mutex_unlock (&card->users_mutex);
	return retcode;
}

/**
 * mmas_show_uid - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
static ssize_t
mmas_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	return snprintf (buf, PAGE_SIZE, "0x%08X%08X\n",
		readl (card->core.addr + MMAS_UIDR_HI),
		readl (card->core.addr + MMAS_UIDR_LO));
}

static DEVICE_ATTR(blackburst_type,S_IRUGO|S_IWUSR,
	mmas_show_blackburst_type,mmas_store_blackburst_type);
static DEVICE_ATTR(uid,S_IRUGO,
	mmas_show_uid,NULL);

/**
 * mmas_pci_probe - PCI insertion handler
 * @pdev: PCI device
 * @id: PCI ID
 *
 * Handle the insertion of a MultiMaster SDI-T.
 * Returns a negative error code on failure and 0 on success.
 **/
static int __devinit
mmas_pci_probe (struct pci_dev *pdev,
	const struct pci_device_id *id)
{
	int err;
	unsigned int cap;
	struct master_dev *card;

	/* Wake a sleeping device */
	if ((err = pci_enable_device (pdev)) < 0) {
		printk (KERN_WARNING "%s: unable to enable device\n",
			mmas_driver_name);
		return err;
	}

	/* Enable bus mastering */
	pci_set_master (pdev);

	/* Request the PCI I/O resources */
	if ((err = pci_request_regions (pdev, mmas_driver_name)) < 0) {
		printk (KERN_WARNING "%s: unable to get I/O resources\n",
			mmas_driver_name);
		pci_disable_device (pdev);
		return err;
	}

	/* Set PCI DMA addressing limitations */
	if ((err = pci_set_dma_mask (pdev, DMA_BIT_MASK(32))) < 0) {
		printk (KERN_WARNING "%s: unable to set PCI DMA mask\n",
			mmas_driver_name);
		pci_disable_device (pdev);
		pci_release_regions (pdev);
		return err;
	}

	/* Initialize the driver_data pointer so that mmas_pci_remove()
	 * doesn't try to free it if an error occurs */
	pci_set_drvdata (pdev, NULL);

	/* Allocate a board info structure */
	if ((card = (struct master_dev *)
		kzalloc (sizeof (*card), GFP_KERNEL)) == NULL) {
		err = -ENOMEM;
		goto NO_MEM;
	}

	/* Initialize the board info structure */
	card->bridge_addr = ioremap_nocache (pci_resource_start (pdev, 0),
		pci_resource_len (pdev, 0));
	card->core.port = pci_resource_start (pdev, 2);
	card->version = master_inl (card, MMAS_CSR) >> 16;
	switch (pdev->device) {
	default:
	case MMAS_PCI_DEVICE_ID_LINSYS:
		card->name = mmas_name;
		card->capabilities = MASTER_CAP_BLACKBURST;
		break;
	case MMASE_PCI_DEVICE_ID_LINSYS:
		card->name = mmase_name;
		card->capabilities = MASTER_CAP_BLACKBURST | MASTER_CAP_UID;
		break;
	}
	card->id = pdev->device;
	card->irq = pdev->irq;
	card->irq_handler = mmas_irq_handler;
	INIT_LIST_HEAD(&card->iface_list);
	/* Lock for ICSR, DMACSR1 */
	spin_lock_init (&card->irq_lock);
	/* Lock for TCSR, RCSR */
	spin_lock_init (&card->reg_lock);
	mutex_init (&card->users_mutex);
	card->parent = &pdev->dev;

	/* Print the firmware version */
	printk (KERN_INFO "%s: %s detected, firmware version %u.%u (0x%04X)\n",
		mmas_driver_name, card->name,
		card->version >> 8, card->version & 0x00ff, card->version);

	/* Store the pointer to the board info structure
	 * in the PCI info structure */
	pci_set_drvdata (pdev, card);

	/* Reset the PCI 9056 */
	plx_reset_bridge (card->bridge_addr);

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
	master_outl (card, MMAS_TCSR, MMAS_TCSR_RST);
	master_outl (card, MMAS_RCSR, MMAS_RCSR_RST);

	/* Register a Master device */
	if ((err = mdev_register (card,
		&mmas_card_list,
		mmas_driver_name,
		mmas_class)) < 0) {
		goto NO_DEV;
	}

	if (card->capabilities & MASTER_CAP_BLACKBURST) {
		if ((err = device_create_file (card->dev,
			&dev_attr_blackburst_type)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'blackburst_type'\n",
				mmas_driver_name);
		}
	}
	if (card->capabilities & MASTER_CAP_UID) {
		if ((err = device_create_file (card->dev,
			&dev_attr_uid)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'uid'\n",
				mmas_driver_name);
		}
	}

	/* Register a transmit interface */
	switch (pdev->device) {
	default:
	case MMAS_PCI_DEVICE_ID_LINSYS:
		if (card->version >= 0x0102) {
			cap = SDI_CAP_TX_RXCLKSRC;
		}
		break;
	case MMASE_PCI_DEVICE_ID_LINSYS:
		cap = SDI_CAP_TX_RXCLKSRC;
		break;
	}
	if ((err = sdi_register_iface (card,
		&plx_dma_ops,
		MMAS_FIFO,
		MASTER_DIRECTION_TX,
		&mmas_txfops,
		&mmas_txops,
		cap,
		4)) < 0) {
		goto NO_IFACE;
	}

	/* Register a receive interface */
	cap = ASI_CAP_RX_SYNC | ASI_CAP_RX_MAKE188 |
		ASI_CAP_RX_BYTECOUNTER |
		ASI_CAP_RX_INVSYNC | ASI_CAP_RX_CD |
		ASI_CAP_RX_DATA |
		ASI_CAP_RX_PIDFILTER | ASI_CAP_RX_PIDCOUNTER |
		ASI_CAP_RX_4PIDCOUNTER | ASI_CAP_RX_27COUNTER |
		ASI_CAP_RX_TIMESTAMPS | ASI_CAP_RX_PTIMESTAMPS |
		ASI_CAP_RX_NULLPACKETS;
	if ((err = asi_register_iface (card,
		&plx_dma_ops,
		MMAS_FIFO,
		MASTER_DIRECTION_RX,
		&mmas_rxfops,
		&mmas_rxops,
		cap,
		4,
		ASI_CTL_TRANSPORT_DVB_ASI)) < 0) {
		goto NO_IFACE;
	}

	return 0;

NO_IFACE:
NO_DEV:
NO_MEM:
	mmas_pci_remove (pdev);
	return err;
}

/**
 * mmas_pci_remove - PCI removal handler
 * @pdev: PCI device
 *
 * This function may be called during PCI probe error handling,
 * so don't mark it as __devexit.
 **/
static void
mmas_pci_remove (struct pci_dev *pdev)
{
	struct master_dev *card = pci_get_drvdata (pdev);

	if (card) {
		struct list_head *p;
		struct master_iface *iface;

		if (!list_empty (&card->iface_list)) {
			iface = list_entry (card->iface_list.next,
				struct master_iface, list);
			sdi_unregister_iface (iface);
		}
		if (!list_empty (&card->iface_list)) {
			iface = list_entry (card->iface_list.next,
				struct master_iface, list);
			asi_unregister_iface (iface);
		}

		/* Unregister the device if it was registered */
		list_for_each (p, &mmas_card_list) {
			if (p == &card->list) {
				mdev_unregister (card, mmas_class);
				break;
			}
		}

		iounmap (card->bridge_addr);
		kfree (card);
	}
	pci_disable_device (pdev);
	pci_release_regions (pdev);
	return;
}

/**
 * mmas_irq_handler - MultiMaster SDI-T interrupt service routine
 * @irq: interrupt number
 * @dev_id: pointer to the device data structure
 * @regs: processor context
 **/
static irqreturn_t
IRQ_HANDLER(mmas_irq_handler,irq,dev_id,regs)
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
			set_bit (SDI_EVENT_TX_BUFFER_ORDER, &txiface->events);
			set_bit (0, &txiface->dma_done);
		}

		interrupting_iface |= 0x1;
	}
	if (intcsr & PLX_INTCSR_DMA1INT_ACTIVE) {
		struct master_dma *dma = rxiface->dma;

		/* Read the interrupt type and clear it */
		spin_lock (&card->irq_lock);
		status = readb (card->bridge_addr + PLX_DMACSR1);
		writeb (status | PLX_DMACSR_CLINT,
			card->bridge_addr + PLX_DMACSR1);
		spin_unlock (&card->irq_lock);

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
		status = master_inl (card, MMAS_ICSR);
		master_outl (card, MMAS_ICSR, status);
		spin_unlock (&card->irq_lock);

		if (status & MMAS_ICSR_TXUIS) {
			set_bit (SDI_EVENT_TX_FIFO_ORDER,
				&txiface->events);
			interrupting_iface |= 0x1;
		}
		if (status & MMAS_ICSR_TXDIS) {
			set_bit (SDI_EVENT_TX_DATA_ORDER,
				&txiface->events);
			interrupting_iface |= 0x1;
		}
		if (status & MMAS_ICSR_RXCDIS) {
			set_bit (ASI_EVENT_RX_CARRIER_ORDER,
				&rxiface->events);
			interrupting_iface |= 0x2;
		}
		if (status & MMAS_ICSR_RXAOSIS) {
			set_bit (ASI_EVENT_RX_AOS_ORDER,
				&rxiface->events);
			interrupting_iface |= 0x2;
		}
		if (status & MMAS_ICSR_RXLOSIS) {
			set_bit (ASI_EVENT_RX_LOS_ORDER,
				&rxiface->events);
			interrupting_iface |= 0x2;
		}
		if (status & MMAS_ICSR_RXOIS) {
			set_bit (ASI_EVENT_RX_FIFO_ORDER,
				&rxiface->events);
			interrupting_iface |= 0x2;
		}
		if (status & MMAS_ICSR_RXDIS) {
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
 * mmas_txinit - Initialize the MultiMaster SDI-T transmitter
 * @iface: interface
 **/
static void
mmas_txinit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg = 0;

	switch (iface->mode) {
	default:
	case SDI_CTL_MODE_8BIT:
		reg |= 0;
		break;
	case SDI_CTL_MODE_10BIT:
		reg |= MMAS_TCSR_10BIT;
		break;
	}
	switch (iface->clksrc) {
	default:
	case SDI_CTL_TX_CLKSRC_ONBOARD:
		reg |= 0;
		break;
	case SDI_CTL_TX_CLKSRC_EXT:
		reg |= MMAS_TCSR_EXTCLK;
		break;
	case SDI_CTL_TX_CLKSRC_RX:
		reg |= MMAS_TCSR_RXCLK;
		break;
	}
	spin_lock (&card->reg_lock);
	reg |= master_inl (card, MMAS_TCSR) & MMAS_TCSR_PAL;
	master_outl (card, MMAS_TCSR, reg | MMAS_TCSR_RST);
	master_outl (card, MMAS_TCSR, reg);
	spin_unlock (&card->reg_lock);
	master_outl (card, MMAS_TFCR,
		(MMAS_TFSL << 16) | MMAS_TDMATL);

	return;
}

/**
 * mmas_txstart - Activate the MultiMaster SDI-T transmitter
 * @iface: interface
 **/
static void
mmas_txstart (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;

	/* Enable DMA */
	writeb (PLX_DMACSR_ENABLE, card->bridge_addr + PLX_DMACSR0);

	/* Enable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = master_inl (card, MMAS_ICSR) &
		MMAS_ICSR_RXCTRLMASK;
	reg |= MMAS_ICSR_TXUIE | MMAS_ICSR_TXDIE;
	master_outl (card, MMAS_ICSR, reg);
	spin_unlock_irq (&card->irq_lock);

	/* Enable the transmitter */
	spin_lock (&card->reg_lock);
	reg = master_inl (card, MMAS_TCSR);
	master_outl (card, MMAS_TCSR, reg | MMAS_TCSR_EN);
	spin_unlock (&card->reg_lock);

	return;
}

/**
 * mmas_txstop - Deactivate the MultiMaster SDI-T transmitter
 * @iface: interface
 **/
static void
mmas_txstop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	struct master_dma *dma = iface->dma;
	unsigned int reg;

	plx_tx_link_all (dma);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	plx_reset (dma);

	/* Wait for the onboard FIFOs to empty */
	/* We don't lock since this should be an atomic read */
	wait_event (iface->queue,
		!(master_inl (card, MMAS_ICSR) & MMAS_ICSR_TXD));

	/* Disable the transmitter */
	spin_lock (&card->reg_lock);
	reg = master_inl (card, MMAS_TCSR);
	master_outl (card, MMAS_TCSR, reg & ~MMAS_TCSR_EN);
	spin_unlock (&card->reg_lock);

	/* Disable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = master_inl (card, MMAS_ICSR) &
		MMAS_ICSR_RXCTRLMASK;
	reg |= MMAS_ICSR_TXUIS | MMAS_ICSR_TXDIS;
	master_outl (card, MMAS_ICSR, reg);
	spin_unlock_irq (&card->irq_lock);

	/* Disable DMA */
	writeb (0, card->bridge_addr + PLX_DMACSR0);

	return;
}

/**
 * mmas_txexit - Clean up the MultiMaster SDI-T transmitter
 * @iface: interface
 **/
static void
mmas_txexit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;

	/* Reset the transmitter */
	spin_lock (&card->reg_lock);
	reg = master_inl (card, MMAS_TCSR);
	master_outl (card, MMAS_TCSR, reg | MMAS_TCSR_RST);
	spin_unlock (&card->reg_lock);

	return;
}

/**
 * mmas_start_tx_dma - start transmit DMA
 * @iface: interface
 **/
static void
mmas_start_tx_dma (struct master_iface *iface)
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
 * mmas_txunlocked_ioctl - MultiMaster SDI-T transmitter unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
mmas_txunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;

	switch (cmd) {
	case SDI_IOC_TXGETTXD:
		/* We don't lock since this should be an atomic read */
		if (put_user ((master_inl (card, MMAS_ICSR) &
			MMAS_ICSR_TXD) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return sdi_txioctl (filp, cmd, arg);
	}
	return 0;
}

/**
 * mmas_txfsync - MultiMaster SDI-T transmitter fsync() method
 * @filp: file to flush
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
FSYNC_HANDLER(mmas_txfsync,filp,datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	struct master_dma *dma = iface->dma;

	mutex_lock (&iface->buf_mutex);
	plx_tx_link_all (dma);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	plx_reset (dma);

	/* Wait for the onboard FIFOs to empty */
	/* We don't lock since this should be an atomic read */
	wait_event (iface->queue,
		!(master_inl (card, MMAS_ICSR) &
		MMAS_ICSR_TXD));

	mutex_unlock (&iface->buf_mutex);
	return 0;
}

/**
 * mmas_rxinit - Initialize the MultiMaster SDI-T receiver
 * @iface: interface
 **/
static void
mmas_rxinit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int i, reg = MMAS_RCSR_RF |
		(iface->null_packets ? MMAS_RCSR_NP : 0);

	switch (iface->timestamps) {
	default:
	case ASI_CTL_TSTAMP_NONE:
		reg |= 0;
		break;
	case ASI_CTL_TSTAMP_APPEND:
		reg |= MMAS_RCSR_TSE;
		break;
	case ASI_CTL_TSTAMP_PREPEND:
		reg |= MMAS_RCSR_PTSE;
		break;
	}
	switch (iface->mode) {
	default:
	case ASI_CTL_RX_MODE_RAW:
		reg |= 0;
		break;
	case ASI_CTL_RX_MODE_188:
		reg |= MMAS_RCSR_188 | MMAS_RCSR_PFE;
		break;
	case ASI_CTL_RX_MODE_204:
		reg |= MMAS_RCSR_204 | MMAS_RCSR_PFE;
		break;
	case ASI_CTL_RX_MODE_AUTO:
		reg |= MMAS_RCSR_AUTO | MMAS_RCSR_PFE;
		break;
	case ASI_CTL_RX_MODE_AUTOMAKE188:
		reg |= MMAS_RCSR_AUTO | MMAS_RCSR_RSS |
			MMAS_RCSR_PFE;
		break;
	case ASI_CTL_RX_MODE_204MAKE188:
		reg |= MMAS_RCSR_204 | MMAS_RCSR_RSS |
			MMAS_RCSR_PFE;
		break;
	}
	/* There will be no races on RCSR
	 * until this code returns, so we don't need to lock it */
	master_outl (card, MMAS_RCSR, reg | MMAS_RCSR_RST);
	wmb ();
	master_outl (card, MMAS_RCSR, reg);
	wmb ();
	master_outl (card, MMAS_RFCR, MMAS_RDMATL);

	/* Reset byte counter */
	master_inl (card, MMAS_RXBCOUNTR);

	/* Reset PID filter.
	 * There will be no races on PFLUT
	 * until this code returns, so we don't need to lock it */
	for (i = 0; i < 256; i++) {
		master_outl (card, MMAS_PFLUTAR, i);
		wmb ();
		master_outl (card, MMAS_PFLUTR, 0xffffffff);
		wmb ();
	}

	/* Clear PID registers */
	master_outl (card, MMAS_PIDR0, 0);
	master_outl (card, MMAS_PIDR1, 0);
	master_outl (card, MMAS_PIDR2, 0);
	master_outl (card, MMAS_PIDR3, 0);

	/* Reset PID counters */
	master_inl (card, MMAS_PIDCOUNTR0);
	master_inl (card, MMAS_PIDCOUNTR1);
	master_inl (card, MMAS_PIDCOUNTR2);
	master_inl (card, MMAS_PIDCOUNTR3);

	return;
}

/**
 * mmas_rxstart - Activate the MultiMaster SDI-T receiver
 * @iface: interface
 **/
static void
mmas_rxstart (struct master_iface *iface)
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
	reg = master_inl (card, MMAS_ICSR) &
		MMAS_ICSR_TXCTRLMASK;
	reg |= MMAS_ICSR_RXCDIE | MMAS_ICSR_RXAOSIE |
		MMAS_ICSR_RXLOSIE | MMAS_ICSR_RXOIE |
		MMAS_ICSR_RXDIE;
	master_outl (card, MMAS_ICSR, reg);
	spin_unlock_irq (&card->irq_lock);

	/* Enable the receiver */
	spin_lock (&card->reg_lock);
	reg = master_inl (card, MMAS_RCSR);
	master_outl (card, MMAS_RCSR, reg | MMAS_RCSR_EN);
	spin_unlock (&card->reg_lock);

	return;
}

/**
 * mmas_rxstop - Deactivate the MultiMaster SDI-T receiver
 * @iface: interface
 **/
static void
mmas_rxstop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;

	/* Disable the receiver */
	spin_lock (&card->reg_lock);
	reg = master_inl (card, MMAS_RCSR);
	master_outl (card, MMAS_RCSR, reg & ~MMAS_RCSR_EN);
	spin_unlock (&card->reg_lock);

	/* Disable receiver interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = master_inl (card, MMAS_ICSR) &
		MMAS_ICSR_TXCTRLMASK;
	reg |= MMAS_ICSR_RXCDIS | MMAS_ICSR_RXAOSIS |
		MMAS_ICSR_RXLOSIS | MMAS_ICSR_RXOIS |
		MMAS_ICSR_RXDIS;
	master_outl (card, MMAS_ICSR, reg);

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
 * mmas_rxexit - Clean up the MultiMaster SDI-T receiver
 * @iface: interface
 **/
static void
mmas_rxexit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;

	/* Reset the receiver.
	 * There will be no races on RCSR here,
	 * so we don't need to lock it */
	master_outl (card, MMAS_RCSR, MMAS_RCSR_RST);

	return;
}

/**
 * mmas_rxunlocked_ioctl - MultiMaster SDI-T receiver unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
mmas_rxunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	int val;
	unsigned int reg = 0, *pflut, i;

	switch (cmd) {
	case ASI_IOC_RXGETSTATUS:
		/* We don't lock since these should be atomic reads,
		 * and the sync bits shouldn't change while the
		 * interface is open. */
		reg = master_inl (card, MMAS_ICSR);
		switch (master_inl (card, MMAS_RCSR) &
			MMAS_RCSR_SYNCMASK) {
		case 0:
			val = 1;
			break;
		case MMAS_RCSR_188:
			val = (reg & MMAS_ICSR_RXPASSING) ? 188 : 0;
			break;
		case MMAS_RCSR_204:
			val = (reg & MMAS_ICSR_RXPASSING) ? 204 : 0;
			break;
		case MMAS_RCSR_AUTO:
			if (reg & MMAS_ICSR_RXPASSING) {
				val = (reg & MMAS_ICSR_RX204) ? 204 : 188;
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
		if (put_user (master_inl (card, MMAS_RXBCOUNTR),
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
			reg |= MMAS_RCSR_INVSYNC;
			break;
		default:
			return -EINVAL;
		}
		spin_lock (&card->reg_lock);
		master_outl (card, MMAS_RCSR,
			(master_inl (card, MMAS_RCSR) &
			~MMAS_RCSR_INVSYNC) | reg);
		spin_unlock (&card->reg_lock);
		break;
	case ASI_IOC_RXGETCARRIER:
		/* We don't lock since this should be an atomic read */
		if (put_user ((master_inl (card, MMAS_ICSR) &
			MMAS_ICSR_RXCD) ? 1 : 0, (int __user *)arg)) {
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
		/* We don't lock since this should be an atomic read */
		if (put_user ((master_inl (card, MMAS_ICSR) &
			MMAS_ICSR_RXD) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPF:
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
			master_outl (card, MMAS_PFLUTAR, i);
			wmb ();
			master_outl (card, MMAS_PFLUTR, pflut[i]);
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
		master_outl (card, MMAS_PIDR0, val);
		/* Reset PID count */
		master_inl (card, MMAS_PIDCOUNTR0);
		break;
	case ASI_IOC_RXGETPID0COUNT:
		if (put_user (master_inl (card, MMAS_PIDCOUNTR0),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPID1:
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		master_outl (card, MMAS_PIDR1, val);
		/* Reset PID count */
		master_inl (card, MMAS_PIDCOUNTR1);
		break;
	case ASI_IOC_RXGETPID1COUNT:
		if (put_user (master_inl (card, MMAS_PIDCOUNTR1),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPID2:
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		master_outl (card, MMAS_PIDR2, val);
		/* Reset PID count */
		master_inl (card, MMAS_PIDCOUNTR2);
		break;
	case ASI_IOC_RXGETPID2COUNT:
		if (put_user (master_inl (card, MMAS_PIDCOUNTR2),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPID3:
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		master_outl (card, MMAS_PIDR3, val);
		/* Reset PID count */
		master_inl (card, MMAS_PIDCOUNTR3);
		break;
	case ASI_IOC_RXGETPID3COUNT:
		if (put_user (master_inl (card, MMAS_PIDCOUNTR3),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXGET27COUNT:
		if (put_user (master_inl (card, MMAS_27COUNTR),
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
 * mmas_rxfsync - MultiMaster SDI-T receiver fsync() method
 * @filp: file to flush
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
FSYNC_HANDLER(mmas_rxfsync,filp,datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	unsigned int reg;

	mutex_lock (&iface->buf_mutex);

	/* Stop the receiver */
	mmas_rxstop (iface);

	/* Reset the onboard FIFO and driver buffers */
	spin_lock (&card->reg_lock);
	reg = master_inl (card, MMAS_RCSR);
	master_outl (card, MMAS_RCSR, reg | MMAS_RCSR_RST);
	wmb ();
	master_outl (card, MMAS_RCSR, reg);
	spin_unlock (&card->reg_lock);
	iface->events = 0;
	plx_reset (iface->dma);

	/* Start the receiver */
	mmas_rxstart (iface);

	mutex_unlock (&iface->buf_mutex);
	return 0;
}

/**
 * mmas_init_module - register the module as a Master and PCI driver
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int __init
mmas_init_module (void)
{
	int err;

	printk (KERN_INFO "%s: Linear Systems Ltd. "
		"MultiMaster SDI-T driver from master-%s (%s)\n",
		mmas_driver_name, MASTER_DRIVER_VERSION, MASTER_DRIVER_DATE);

	/* Create the device class */
	mmas_class = mdev_init (mmas_driver_name);
	if (IS_ERR(mmas_class)) {
		err = PTR_ERR(mmas_class);
		goto NO_CLASS;
	}

	/* Register with the PCI subsystem */
	if ((err = pci_register_driver (&mmas_pci_driver)) < 0) {
		printk (KERN_WARNING
			"%s: unable to register with PCI subsystem\n",
			mmas_driver_name);
		goto NO_PCI;
	}

	return 0;

NO_PCI:
	mdev_cleanup (mmas_class);
NO_CLASS:
	return err;
}

/**
 * mmas_cleanup_module - unregister the module as a Master and PCI driver
 **/
static void __exit
mmas_cleanup_module (void)
{
	pci_unregister_driver (&mmas_pci_driver);
	mdev_cleanup (mmas_class);
	return;
}

module_init (mmas_init_module);
module_exit (mmas_cleanup_module);

