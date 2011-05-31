/* sdim_lpfd.c
 *
 * Linux driver functions for Linear Systems Ltd. SDI Master LP FD.
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
#include <linux/module.h> /* MODULE_LICENSE */

#include <linux/fs.h> /* inode, file, file_operations */
#include <linux/sched.h> /* pt_regs */
#include <linux/pci.h> /* pci_resource_start () */
#include <linux/dma-mapping.h> /* DMA_BIT_MASK */
#include <linux/slab.h> /* kzalloc () */
#include <linux/list.h> /* INIT_LIST_HEAD () */
#include <linux/spinlock.h> /* spin_lock_init () */
#include <linux/init.h> /* module_init () */
#include <linux/errno.h> /* error codes */
#include <linux/interrupt.h> /* irqreturn_t */
#include <linux/device.h> /* device_create_file () */
#include <linux/mutex.h> /* mutex_init () */

#include <asm/uaccess.h> /* put_user () */
#include <asm/bitops.h> /* set_bit () */

#include "sdicore.h"
#include "../include/master.h"
#include "mdev.h"
#include "mdma.h"
#include "lsdma.h"
#include "sdim.h"

#ifndef DEFINE_PCI_DEVICE_TABLE
#define DEFINE_PCI_DEVICE_TABLE(_table) \
	const struct pci_device_id _table[]
#endif

#ifndef DMA_BIT_MASK
#define DMA_BIT_MASK(n)	(((n) == 64) ? ~0ULL : ((1ULL<<(n))-1))
#endif

static const char sdim_lpfd_name[] = SDIM_NAME_LPFD;
static const char sdim_lpfde_name[] = SDIM_NAME_LPFDE;

/* Static function prototypes */
static ssize_t sdim_show_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static ssize_t sdim_store_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	const char *buf,
	size_t count);
static ssize_t sdim_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static int sdim_pci_probe (struct pci_dev *pdev,
	const struct pci_device_id *id) __devinit;
static void sdim_pci_remove (struct pci_dev *pdev);
static irqreturn_t IRQ_HANDLER(sdim_irq_handler,irq,dev_id,regs);
static void sdim_txinit (struct master_iface *iface);
static void sdim_txstart (struct master_iface *iface);
static void sdim_txstop (struct master_iface *iface);
static void sdim_txexit (struct master_iface *iface);
static void sdim_start_tx_dma (struct master_iface *iface);
static long sdim_txunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int FSYNC_HANDLER(sdim_txfsync,filp,datasync);
static void sdim_rxinit (struct master_iface *iface);
static void sdim_rxstart (struct master_iface *iface);
static void sdim_rxstop (struct master_iface *iface);
static void sdim_rxexit (struct master_iface *iface);
static long sdim_rxunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int FSYNC_HANDLER(sdim_rxfsync,filp,datasync);
static int sdim_init_module (void) __init;
static void sdim_cleanup_module (void) __exit;

MODULE_AUTHOR("Linear Systems Ltd.");
MODULE_DESCRIPTION("SDI Master driver");
MODULE_LICENSE("GPL");
MODULE_VERSION(MASTER_DRIVER_VERSION);

static char sdim_driver_name[] = "sdilpm";

static DEFINE_PCI_DEVICE_TABLE(sdim_pci_id_table) = {
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
	.write = sdi_write,
	.poll = sdi_txpoll,
	.unlocked_ioctl = sdim_txunlocked_ioctl,
	.compat_ioctl = sdim_txunlocked_ioctl,
	.mmap = sdi_mmap,
	.open = sdi_open,
	.release = sdi_release,
	.fsync = sdim_txfsync,
	.fasync = NULL
};

static struct file_operations sdim_rxfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.read = sdi_read,
	.poll = sdi_rxpoll,
	.unlocked_ioctl = sdim_rxunlocked_ioctl,
	.compat_ioctl = sdim_rxunlocked_ioctl,
	.mmap = sdi_mmap,
	.open = sdi_open,
	.release = sdi_release,
	.fsync = sdim_rxfsync,
	.fasync = NULL
};

static struct master_iface_operations sdim_txops = {
	.init = sdim_txinit,
	.start = sdim_txstart,
	.stop = sdim_txstop,
	.exit = sdim_txexit,
	.start_tx_dma = sdim_start_tx_dma
};

static struct master_iface_operations sdim_rxops = {
	.init = sdim_rxinit,
	.start = sdim_rxstart,
	.stop = sdim_rxstop,
	.exit = sdim_rxexit
};

static LIST_HEAD(sdim_card_list);

static struct class *sdim_class;

/**
 * sdim_show_blackburst_type - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
static ssize_t
sdim_show_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	return snprintf (buf, PAGE_SIZE, "%u\n",
		(readl (card->core.addr + SDIM_TCSR) & SDIM_TCSR_PAL) >> 9);
}

/**
 * sdim_store_blackburst_type - interface attribute write handler
 * @dev: device being written
 * @attr: device attribute
 * @buf: input buffer
 * @count:
 **/
static ssize_t
sdim_store_blackburst_type (struct device *dev,
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
	reg = readl (card->core.addr + SDIM_TCSR) & ~SDIM_TCSR_PAL;
	writel (reg | (val << 9), card->core.addr + SDIM_TCSR);
OUT:
	mutex_unlock (&card->users_mutex);
	return retcode;
}

/**
 * sdim_show_uid - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
static ssize_t
sdim_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	return snprintf (buf, PAGE_SIZE, "0x%08X%08X\n",
		readl (card->core.addr + SDIM_UIDR_HI),
		readl (card->core.addr + SDIM_UIDR_LO));
}

static DEVICE_ATTR(blackburst_type,S_IRUGO|S_IWUSR,
	sdim_show_blackburst_type,sdim_store_blackburst_type);
static DEVICE_ATTR(uid,S_IRUGO,
	sdim_show_uid,NULL);

/**
 * sdim_pci_probe - PCI insertion handler for a SDI Master
 * @pdev: PCI device
 * @id: PCI ID
 *
 * Handle the insertion of a SDI Master.
 * Returns a negative error code on failure and 0 on success.
 **/
static int __devinit
sdim_pci_probe (struct pci_dev *pdev,
	const struct pci_device_id *id)
{
	int err;
	struct master_dev *card;

	/* Wake a sleeping device */
	if ((err = pci_enable_device (pdev)) < 0) {
		printk (KERN_WARNING "%s: unable to enable device\n",
			sdim_driver_name);
		return err;
	}

	/* Enable bus mastering */
	pci_set_master (pdev);

	/* Request the PCI I/O resources */
	if ((err = pci_request_regions (pdev, sdim_driver_name)) < 0) {
		printk (KERN_WARNING "%s: unable to get I/O resources\n",
			sdim_driver_name);
		pci_disable_device (pdev);
		return err;
	}

	/* Set PCI DMA addressing limitations */
	if ((err = pci_set_dma_mask (pdev, DMA_BIT_MASK(32))) < 0) {
		printk (KERN_WARNING "%s: unable to set PCI DMA mask\n",
			sdim_driver_name);
		pci_disable_device (pdev);
		pci_release_regions (pdev);
		return err;
	}

	/* Initialize the driver_data pointer so that sdim_pci_remove()
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
	switch (pdev->device) {
	default:
	case SDIM_PCI_DEVICE_ID_LINSYS_SDILPFD:
		card->bridge_addr = ioremap_nocache (pci_resource_start (pdev, 1),
			pci_resource_len (pdev, 1));
		card->name = sdim_lpfd_name;
		break;
	case SDIM_PCI_DEVICE_ID_LINSYS_SDILPFDE:
		card->bridge_addr = ioremap_nocache (pci_resource_start (pdev, 2),
			pci_resource_len (pdev, 2));
		card->name = sdim_lpfde_name;
		break;
	}
	/* SDI Core */
	card->core.addr = ioremap_nocache (pci_resource_start (pdev, 0),
		pci_resource_len (pdev, 0));
	card->version = readl (card->core.addr + SDIM_CSR) >> 16;
	card->id = pdev->device;
	card->irq = pdev->irq;
	card->irq_handler = sdim_irq_handler;
	card->capabilities = MASTER_CAP_BLACKBURST | MASTER_CAP_UID;
	INIT_LIST_HEAD(&card->iface_list);
	/* Lock for ICSR, DMACSR1 */
	spin_lock_init (&card->irq_lock);
	/* Lock for TCSR, RCSR */
	spin_lock_init (&card->reg_lock);
	mutex_init (&card->users_mutex);
	card->parent = &pdev->dev;

	/* Print the firmware version */
	printk (KERN_INFO "%s: %s detected, firmware version %u.%u (0x%04X)\n",
		sdim_driver_name, card->name,
		card->version >> 8, card->version & 0x00ff, card->version);

	/* Store the pointer to the board info structure
	 * in the PCI info structure */
	pci_set_drvdata (pdev, card);

	/* Reset the FPGA */
	writel (SDIM_TCSR_RST, card->core.addr + SDIM_TCSR);
	writel (SDIM_RCSR_RST, card->core.addr + SDIM_RCSR);

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
		sdim_class)) < 0) {
		goto NO_DEV;
	}

	if (card->capabilities & MASTER_CAP_BLACKBURST) {
		if ((err = device_create_file (card->dev,
			&dev_attr_blackburst_type)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'blackburst_type'\n",
				sdim_driver_name);
		}
	}
	if (card->capabilities & MASTER_CAP_UID) {
		if ((err = device_create_file (card->dev,
			&dev_attr_uid)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'uid'\n",
				sdim_driver_name);
		}
	}

	/* Register a transmit interface */
	if ((err = sdi_register_iface (card,
		&lsdma_dma_ops,
		0,
		MASTER_DIRECTION_TX,
		&sdim_txfops,
		&sdim_txops,
		SDI_CAP_TX_RXCLKSRC,
		4)) < 0) {
		goto NO_IFACE;
	}

	/* Register a receive interface */
	if ((err = sdi_register_iface (card,
		&lsdma_dma_ops,
		0,
		MASTER_DIRECTION_RX,
		&sdim_rxfops,
		&sdim_rxops,
		0,
		4)) < 0) {
		goto NO_IFACE;
	}

	return 0;

NO_IFACE:
NO_DEV:
NO_MEM:
	sdim_pci_remove (pdev);
	return err;
}

/**
 * sdim_pci_remove - PCI removal handler for a SDI Master
 * @pdev: PCI device
 *
 * Handle the removal of a SDI Master.
 **/
static void
sdim_pci_remove (struct pci_dev *pdev)
{
	struct master_dev *card = pci_get_drvdata (pdev);

	if (card) {
		struct list_head *p, *n;
		struct master_iface *iface;

		list_for_each_safe (p, n, &card->iface_list) {
			iface = list_entry (p, struct master_iface, list);
			sdi_unregister_iface (iface);
		}

		/* Unregister the device if it was registered */
		list_for_each (p, &sdim_card_list) {
			if (p == &card->list) {
				mdev_unregister (card, sdim_class);
				break;
			}
		}

		iounmap (card->core.addr);
		iounmap (card->bridge_addr);
		kfree (card);
	}
	pci_disable_device (pdev);
	pci_release_regions (pdev);
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
			mdma_advance (txiface->dma);
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
	status = readl (card->core.addr + SDIM_ICSR);
	writel (status, card->core.addr + SDIM_ICSR);
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
	reg |= readl (card->core.addr + SDIM_TCSR) & SDIM_TCSR_PAL;
	writel (reg | SDIM_TCSR_RST, card->core.addr + SDIM_TCSR);
	writel (reg, card->core.addr + SDIM_TCSR);
	writel ((SDIM_TFSL << 16) | SDIM_TDMATL, card->core.addr + SDIM_TFCR);

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

	/* Enable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = readl (card->core.addr + SDIM_ICSR) &
		SDIM_ICSR_RXCTRLMASK;
	reg |= SDIM_ICSR_TXUIE | SDIM_ICSR_TXDIE;
	writel (reg, card->core.addr + SDIM_ICSR);
	spin_unlock_irq (&card->irq_lock);

	/* Disable RP178 pattern generation and enable the transmitter */
	reg = readl (card->core.addr + SDIM_TCSR);
	writel (reg | SDIM_TCSR_EN | SDIM_TCSR_RP178,
		card->core.addr + SDIM_TCSR);

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
	struct master_dma *dma = iface->dma;
	unsigned int reg;

	lsdma_tx_link_all (dma);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	lsdma_reset (dma);

	/* Wait for the onboard FIFOs to empty */
	/* We don't lock since this should be an atomic read */
	wait_event (iface->queue,
		!(readl (card->core.addr + SDIM_ICSR) & SDIM_ICSR_TXD));

	/* Disable the transmitter and enable RP178 pattern generation */
	reg = readl (card->core.addr + SDIM_TCSR);
	writel (reg & ~SDIM_TCSR_EN & ~SDIM_TCSR_RP178,
		card->core.addr + SDIM_TCSR);

	/* Disable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = readl (card->core.addr + SDIM_ICSR) &
		SDIM_ICSR_RXCTRLMASK;
	reg |= SDIM_ICSR_TXUIS | SDIM_ICSR_TXDIS;
	writel (reg, card->core.addr + SDIM_ICSR);
	spin_unlock_irq (&card->irq_lock);

	/* Disable DMA */
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE,
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
	unsigned int reg;

	/* Reset the transmitter */
	reg = readl (card->core.addr + SDIM_TCSR);
	writel (reg | SDIM_TCSR_RST, card->core.addr + SDIM_TCSR);

	return;
}

/**
 * sdim_start_tx_dma - start transmit DMA
 * @iface: interface
 **/
static void
sdim_start_tx_dma (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	struct master_dma *dma = iface->dma;

	writel (LSDMA_CH_CSR_INTDONEENABLE |
		LSDMA_CH_CSR_INTSTOPENABLE,
		card->bridge_addr + LSDMA_CSR(0));
	writel (mdma_dma_to_desc_low (lsdma_head_desc_bus_addr (dma)),
		card->bridge_addr + LSDMA_DESC(0));
	writel (mdma_dma_to_desc_high (lsdma_head_desc_bus_addr (dma)),
		card->bridge_addr + LSDMA_DESC_H(0));
	clear_bit (0, &iface->dma_done);
	writel (LSDMA_CH_CSR_INTDONEENABLE |
		LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_ENABLE,
		card->bridge_addr + LSDMA_CSR(0));
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);
	return;
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
	case SDI_IOC_TXGETTXD:
		/* We don't lock since this should be an atomic read */
		if (put_user ((readl (card->core.addr + SDIM_ICSR) &
			SDIM_ICSR_TXD) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return sdi_txioctl (filp, cmd, arg);
	}
	return 0;
}

/**
 * sdim_txfsync - SDI Master transmitter fsync() method
 * @filp: file to flush
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
FSYNC_HANDLER(sdim_txfsync,filp,datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	struct master_dma *dma = iface->dma;

	mutex_lock (&iface->buf_mutex);
	lsdma_tx_link_all (dma);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	lsdma_reset (dma);


	/* Wait for the onboard FIFOs to empty */
	/* We don't lock since this should be an atomic read */
	wait_event (iface->queue,
		!(readl (card->core.addr + SDIM_ICSR) &
		SDIM_ICSR_TXD));

	mutex_unlock (&iface->buf_mutex);
	return 0;
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
	writel (reg | SDIM_RCSR_RST, card->core.addr + SDIM_RCSR);
	writel (reg, card->core.addr + SDIM_RCSR);
	writel (SDIM_RDMATL, card->core.addr + SDIM_RFCR);

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
	struct master_dma *dma = iface->dma;
	unsigned int reg;

	/* Enable and start DMA */
	writel (mdma_dma_to_desc_low (lsdma_head_desc_bus_addr (dma)),
		card->bridge_addr + LSDMA_DESC(1));
	writel (mdma_dma_to_desc_high (lsdma_head_desc_bus_addr (dma)),
		card->bridge_addr + LSDMA_DESC_H(1));
	clear_bit (0, &iface->dma_done);
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION | LSDMA_CH_CSR_ENABLE,
		card->bridge_addr + LSDMA_CSR(1));

	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);

	/* Enable receiver interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = readl (card->core.addr + SDIM_ICSR) &
		SDIM_ICSR_TXCTRLMASK;
	reg |= SDIM_ICSR_RXCDIE | SDIM_ICSR_RXOIE;
	writel (reg, card->core.addr + SDIM_ICSR);
	spin_unlock_irq (&card->irq_lock);

	/* Enable the receiver */
	reg = readl (card->core.addr + SDIM_RCSR);
	writel (reg | SDIM_RCSR_EN, card->core.addr + SDIM_RCSR);

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
	reg = readl (card->core.addr + SDIM_RCSR);
	writel (reg & ~SDIM_RCSR_EN, card->core.addr + SDIM_RCSR);

	/* Disable receiver interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = readl (card->core.addr + SDIM_ICSR) &
		SDIM_ICSR_TXCTRLMASK;
	reg |= SDIM_ICSR_RXCDIS | SDIM_ICSR_RXOIS |
		SDIM_ICSR_RXDIS;
	writel (reg, card->core.addr + SDIM_ICSR);
	spin_unlock_irq (&card->irq_lock);

	/* Disable and abort DMA */
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION,
		card->bridge_addr + LSDMA_CSR(1));
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION | LSDMA_CH_CSR_STOP,
		card->bridge_addr + LSDMA_CSR(1));
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);

	wait_event (iface->queue, test_bit (0, &iface->dma_done));

	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION,
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
	writel (SDIM_RCSR_RST, card->core.addr + SDIM_RCSR);

	return;
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
	case SDI_IOC_RXGETCARRIER:
		/* We don't lock since this should be an atomic read */
		if (put_user ((readl (card->core.addr + SDIM_ICSR) &
			SDIM_ICSR_RXCD) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_RXGETSTATUS:
		/* We don't lock since this should be an atomic read */
		if (put_user ((readl (card->core.addr + SDIM_ICSR) &
			SDIM_ICSR_RXPASSING) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return sdi_rxioctl (filp, cmd, arg);
	}
	return 0;
}

/**
 * sdim_rxfsync - SDI Master receiver fsync() method
 * @filp: file to flush
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
FSYNC_HANDLER(sdim_rxfsync,filp,datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	unsigned int reg;

	mutex_lock (&iface->buf_mutex);

	/* Stop the receiver */
	sdim_rxstop (iface);

	/* Reset the onboard FIFO and driver buffers */
	spin_lock (&card->reg_lock);
	reg = readl (card->core.addr + SDIM_RCSR);
	writel (reg | SDIM_RCSR_RST, card->core.addr + SDIM_RCSR);
	writel (reg, card->core.addr + SDIM_RCSR);
	spin_unlock (&card->reg_lock);
	iface->events = 0;
	lsdma_reset (iface->dma);

	/* Start the receiver */
	sdim_rxstart (iface);

	mutex_unlock (&iface->buf_mutex);
	return 0;
}

/**
 * sdim_init_module - register the module as a Master and PCI driver
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int __init
sdim_init_module (void)
{
	int err;

	printk (KERN_INFO "%s: Linear Systems Ltd. "
		"SDI Master driver from master-%s (%s)\n",
		sdim_driver_name, MASTER_DRIVER_VERSION, MASTER_DRIVER_DATE);

	/* Create the device class */
	sdim_class = mdev_init (sdim_driver_name);
	if (IS_ERR(sdim_class)) {
		err = PTR_ERR(sdim_class);
		goto NO_CLASS;
	}

	/* Register with the PCI subsystem */
	if ((err = pci_register_driver (&sdim_pci_driver)) < 0) {
		printk (KERN_WARNING
			"%s: unable to register with PCI subsystem\n",
			sdim_driver_name);
		goto NO_PCI;
	}

	return 0;

NO_PCI:
	mdev_cleanup (sdim_class);
NO_CLASS:
	return err;
}

/**
 * sdim_cleanup_module - unregister the module as a Master and PCI driver
 **/
static void __exit
sdim_cleanup_module (void)
{
	pci_unregister_driver (&sdim_pci_driver);
	mdev_cleanup (sdim_class);
	return;
}

module_init (sdim_init_module);
module_exit (sdim_cleanup_module);

