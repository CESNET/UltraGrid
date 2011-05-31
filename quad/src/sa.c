/* sa.c
 *
 * Linux driver for Linear Systems Ltd. MultiMaster SDI-R.
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
#include <linux/init.h> /* __init */
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
#include "mmsa.h"

#ifndef DEFINE_PCI_DEVICE_TABLE
#define DEFINE_PCI_DEVICE_TABLE(_table) \
	const struct pci_device_id _table[]
#endif

#ifndef DMA_BIT_MASK
#define DMA_BIT_MASK(n)	(((n) == 64) ? ~0ULL : ((1ULL<<(n))-1))
#endif

static const char mmsa_name[] = MMSA_NAME;
static const char mmsae_name[] = MMSAE_NAME;

/* Static function prototypes */
static ssize_t mmsa_show_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static ssize_t mmsa_store_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	const char *buf,
	size_t count);
static ssize_t mmsa_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static int mmsa_pci_probe (struct pci_dev *pdev,
	const struct pci_device_id *id) __devinit;
static void mmsa_pci_remove (struct pci_dev *pdev);
static irqreturn_t IRQ_HANDLER(mmsa_irq_handler,irq,dev_id,regs);
static void mmsa_txinit (struct master_iface *iface);
static void mmsa_txstart (struct master_iface *iface);
static void mmsa_txstop (struct master_iface *iface);
static void mmsa_txexit (struct master_iface *iface);
static void mmsa_start_tx_dma (struct master_iface *iface);
static long mmsa_txunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int FSYNC_HANDLER(mmsa_txfsync,filp,datasync);
static void mmsa_rxinit (struct master_iface *iface);
static void mmsa_rxstart (struct master_iface *iface);
static void mmsa_rxstop (struct master_iface *iface);
static void mmsa_rxexit (struct master_iface *iface);
static long mmsa_rxunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int FSYNC_HANDLER(mmsa_rxfsync,filp,datasync);
static int mmsa_init_module (void) __init;
static void mmsa_cleanup_module (void) __exit;

MODULE_AUTHOR("Linear Systems Ltd.");
MODULE_DESCRIPTION("MultiMaster SDI-R driver");
MODULE_LICENSE("GPL");
MODULE_VERSION(MASTER_DRIVER_VERSION);

static char mmsa_driver_name[] = "mmsa";

static DEFINE_PCI_DEVICE_TABLE(mmsa_pci_id_table) = {
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
	.write = asi_write,
	.poll = asi_txpoll,
	.unlocked_ioctl = mmsa_txunlocked_ioctl,
	.compat_ioctl = mmsa_txunlocked_ioctl,
	.mmap = NULL,
	.open = asi_open,
	.release = asi_release,
	.fsync = mmsa_txfsync,
	.fasync = NULL
};

static struct file_operations mmsa_rxfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.read = sdi_read,
	.poll = sdi_rxpoll,
	.unlocked_ioctl = mmsa_rxunlocked_ioctl,
	.compat_ioctl = mmsa_rxunlocked_ioctl,
	.mmap = sdi_mmap,
	.open = sdi_open,
	.release = sdi_release,
	.fsync = mmsa_rxfsync,
	.fasync = NULL
};

static struct master_iface_operations mmsa_txops = {
	.init = mmsa_txinit,
	.start = mmsa_txstart,
	.stop = mmsa_txstop,
	.exit = mmsa_txexit,
	.start_tx_dma = mmsa_start_tx_dma
};

static struct master_iface_operations mmsa_rxops = {
	.init = mmsa_rxinit,
	.start = mmsa_rxstart,
	.stop = mmsa_rxstop,
	.exit = mmsa_rxexit
};

static LIST_HEAD(mmsa_card_list);

static struct class *mmsa_class;

/**
 * mmsa_show_blackburst_type - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
static ssize_t
mmsa_show_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	return snprintf (buf, PAGE_SIZE, "%u\n",
		(master_inl (card, MMSA_TCSR) & MMSA_TCSR_PAL) >> 13);
}

/**
 * mmsa_store_blackburst_type - interface attribute write handler
 * @dev: device being written
 * @attr: device attribute
 * @buf: input buffer
 * @count:
 **/
static ssize_t
mmsa_store_blackburst_type (struct device *dev,
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
	reg = master_inl (card, MMSA_TCSR) & ~MMSA_TCSR_PAL;
	master_outl (card, MMSA_TCSR, reg | (val << 13));
	spin_unlock (&card->reg_lock);
OUT:
	mutex_unlock (&card->users_mutex);
	return retcode;
}

/**
 * mmsa_show_uid - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
static ssize_t
mmsa_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	return snprintf (buf, PAGE_SIZE, "0x%08X%08X\n",
		readl (card->core.addr + MMSA_UIDR_HI),
		readl (card->core.addr + MMSA_UIDR_LO));
}

static DEVICE_ATTR(blackburst_type,S_IRUGO|S_IWUSR,
	mmsa_show_blackburst_type,mmsa_store_blackburst_type);
static DEVICE_ATTR(uid,S_IRUGO,
	mmsa_show_uid,NULL);

/**
 * mmsa_pci_probe - PCI insertion handler
 * @pdev: PCI device
 * @id: PCI ID
 *
 * Handle the insertion of a MultiMaster SDI-R.
 * Returns a negative error code on failure and 0 on success.
 **/
static int __devinit
mmsa_pci_probe (struct pci_dev *pdev,
	const struct pci_device_id *id)
{
	int err;
	unsigned int cap;
	struct master_dev *card;

	/* Wake a sleeping device */
	if ((err = pci_enable_device (pdev)) < 0) {
		printk (KERN_WARNING "%s: unable to enable device\n",
			mmsa_driver_name);
		return err;
	}

	/* Enable bus mastering */
	pci_set_master (pdev);

	/* Request the PCI I/O resources */
	if ((err = pci_request_regions (pdev, mmsa_driver_name)) < 0) {
		printk (KERN_WARNING "%s: unable to get I/O resources\n",
			mmsa_driver_name);
		pci_disable_device (pdev);
		return err;
	}

	/* Set PCI DMA addressing limitations */
	if ((err = pci_set_dma_mask (pdev, DMA_BIT_MASK(32))) < 0) {
		printk (KERN_WARNING "%s: unable to set PCI DMA mask\n",
			mmsa_driver_name);
		pci_disable_device (pdev);
		pci_release_regions (pdev);
		return err;
	}

	/* Initialize the driver_data pointer so that mmsa_pci_remove()
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
	card->version = master_inl (card, MMSA_CSR) >> 16;
	switch (pdev->device) {
	default:
	case MMSA_PCI_DEVICE_ID_LINSYS:
		card->name = mmsa_name;
		card->capabilities = MASTER_CAP_BLACKBURST;
		break;
	case MMSAE_PCI_DEVICE_ID_LINSYS:
		card->name = mmsae_name;
		card->capabilities = MASTER_CAP_BLACKBURST | MASTER_CAP_UID;
		break;
	}
	card->id = pdev->device;
	card->irq = pdev->irq;
	card->irq_handler = mmsa_irq_handler;
	INIT_LIST_HEAD(&card->iface_list);
	/* Lock for ICSR, DMACSR1 */
	spin_lock_init (&card->irq_lock);
	/* Lock for TCSR, RCSR */
	spin_lock_init (&card->reg_lock);
	mutex_init (&card->users_mutex);
	card->parent = &pdev->dev;

	/* Print the firmware version */
	printk (KERN_INFO "%s: %s detected, firmware version %u.%u (0x%04X)\n",
		mmsa_driver_name, card->name,
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
	master_outl (card, MMSA_TCSR, MMSA_TCSR_RST);
	master_outl (card, MMSA_RCSR, MMSA_RCSR_RST);

	/* Register a Master device */
	if ((err = mdev_register (card,
		&mmsa_card_list,
		mmsa_driver_name,
		mmsa_class)) < 0) {
		goto NO_DEV;
	}

	if (card->capabilities & MASTER_CAP_BLACKBURST) {
		if ((err = device_create_file (card->dev,
			&dev_attr_blackburst_type)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'blackburst_type'\n",
				mmsa_driver_name);
		}
	}
	if (card->capabilities & MASTER_CAP_UID) {
		if ((err = device_create_file (card->dev,
			&dev_attr_uid)) < 0) {
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
	switch (pdev->device) {
	default:
	case MMSA_PCI_DEVICE_ID_LINSYS:
		if (card->version >= 0x0302) {
			cap |= ASI_CAP_TX_RXCLKSRC;
		}
		break;
	case MMSAE_PCI_DEVICE_ID_LINSYS:
		cap |= ASI_CAP_TX_RXCLKSRC;
		break;
	}
	if ((err = asi_register_iface (card,
		&plx_dma_ops,
		MMSA_FIFO,
		MASTER_DIRECTION_TX,
		&mmsa_txfops,
		&mmsa_txops,
		cap,
		4,
		ASI_CTL_TRANSPORT_DVB_ASI)) < 0) {
		goto NO_IFACE;
	}

	/* Register a receive interface */
	if ((err = sdi_register_iface (card,
		&plx_dma_ops,
		MMSA_FIFO,
		MASTER_DIRECTION_RX,
		&mmsa_rxfops,
		&mmsa_rxops,
		0,
		4)) < 0) {
		goto NO_IFACE;
	}

	return 0;

NO_IFACE:
NO_DEV:
NO_MEM:
	mmsa_pci_remove (pdev);
	return err;
}

/**
 * mmsa_pci_remove - PCI removal handler
 * @pdev: PCI device
 **/
static void
mmsa_pci_remove (struct pci_dev *pdev)
{
	struct master_dev *card = pci_get_drvdata (pdev);

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
		/* Unregister the device if it was registered */
		list_for_each (p, &mmsa_card_list) {
			if (p == &card->list) {
				mdev_unregister (card, mmsa_class);
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
		spin_lock (&card->irq_lock);
		status = readb (card->bridge_addr + PLX_DMACSR1);
		writeb (status | PLX_DMACSR_CLINT,
			card->bridge_addr + PLX_DMACSR1);
		spin_unlock (&card->irq_lock);

		/* Increment the buffer pointer */
		mdma_advance (dma);

		if (mdma_rx_isempty (dma)) {
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
	spin_lock (&card->reg_lock);
	reg |= master_inl (card, MMSA_TCSR) & MMSA_TCSR_PAL;
	master_outl (card, MMSA_TCSR, reg | MMSA_TCSR_RST);
	master_outl (card, MMSA_TCSR, reg);
	spin_unlock (&card->reg_lock);
	master_outl (card, MMSA_TFCR,
		(MMSA_TFSL << 16) | MMSA_TDMATL);
	/* There will be no races on IBSTR, IPSTR, and FTR
	 * until this code returns, so we don't need to lock them */
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

	/* Enable the transmitter */
	spin_lock (&card->reg_lock);
	reg = master_inl (card, MMSA_TCSR);
	master_outl (card, MMSA_TCSR, reg | MMSA_TCSR_EN);
	spin_unlock (&card->reg_lock);

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
	struct master_dma *dma = iface->dma;
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

	/* Disable the transmitter */
	spin_lock (&card->reg_lock);
	reg = master_inl (card, MMSA_TCSR);
	master_outl (card, MMSA_TCSR, reg & ~MMSA_TCSR_EN);
	spin_unlock (&card->reg_lock);

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
	unsigned int reg;

	/* Reset the transmitter */
	spin_lock (&card->reg_lock);
	reg = master_inl (card, MMSA_TCSR);
	master_outl (card, MMSA_TCSR, reg | MMSA_TCSR_RST);
	spin_unlock (&card->reg_lock);

	return;
}

/**
 * mmsa_start_tx_dma - start transmit DMA
 * @iface: interface
 **/
static void
mmsa_start_tx_dma (struct master_iface *iface)
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
	case ASI_IOC_TXSETSTUFFING:
		if (copy_from_user (&stuffing,
			(struct asi_txstuffing __user *)arg,
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
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_TXGETTXD:
		/* We don't lock since this should be an atomic read */
		if (put_user ((master_inl (card, MMSA_ICSR) &
			MMSA_ICSR_TXD) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_TXGET27COUNT:
		if (put_user (master_inl (card, MMSA_27COUNTR),
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
 * mmsa_txfsync - MultiMaster SDI-R transmitter fsync() method
 * @filp: file to flush
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
FSYNC_HANDLER(mmsa_txfsync,filp,datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dma *dma = iface->dma;

	mutex_lock (&iface->buf_mutex);
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

	mutex_unlock (&iface->buf_mutex);
	return 0;
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
	case SDI_IOC_RXGETCARRIER:
		/* We don't lock since this should be an atomic read */
		if (put_user ((master_inl (card, MMSA_ICSR) &
			MMSA_ICSR_RXCD) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_RXGETSTATUS:
		/* We don't lock since this should be an atomic read */
		if (put_user ((master_inl (card, MMSA_ICSR) &
			MMSA_ICSR_RXPASSING) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return sdi_rxioctl (filp, cmd, arg);
	}
	return 0;
}

/**
 * mmsa_rxfsync - MultiMaster SDI-R receiver fsync() method
 * @filp: file to flush
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
FSYNC_HANDLER(mmsa_rxfsync,filp,datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	unsigned int reg;

	mutex_lock (&iface->buf_mutex);

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

	mutex_unlock (&iface->buf_mutex);
	return 0;
}

/**
 * mmsa_init_module - register the module as a Master and PCI driver
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int __init
mmsa_init_module (void)
{
	int err;

	printk (KERN_INFO "%s: Linear Systems Ltd. "
		"MultiMaster SDI-R driver from master-%s (%s)\n",
		mmsa_driver_name, MASTER_DRIVER_VERSION, MASTER_DRIVER_DATE);

	/* Create the device class */
	mmsa_class = mdev_init (mmsa_driver_name);
	if (IS_ERR(mmsa_class)) {
		err = PTR_ERR(mmsa_class);
		goto NO_CLASS;
	}

	/* Register with the PCI subsystem */
	if ((err = pci_register_driver (&mmsa_pci_driver)) < 0) {
		printk (KERN_WARNING
			"%s: unable to register with PCI subsystem\n",
			mmsa_driver_name);
		goto NO_PCI;
	}

	return 0;

NO_PCI:
	mdev_cleanup (mmsa_class);
NO_CLASS:
	return err;
}

/**
 * mmsa_cleanup_module - unregister the module as a Master and PCI driver
 **/
static void __exit
mmsa_cleanup_module (void)
{
	pci_unregister_driver (&mmsa_pci_driver);
	mdev_cleanup (mmsa_class);
	return;
}

module_init (mmsa_init_module);
module_exit (mmsa_cleanup_module);

