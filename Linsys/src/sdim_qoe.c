/* sdim_qoe.c
 *
 * Linux driver for Linear Systems Ltd. SDI Master Q/o PCIe.
 *
 * Copyright (C) 2007-2010 Linear Systems Ltd.
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
#include <linux/dma-mapping.h> /* DMA_BIT_MASK */
#include <linux/slab.h> /* kzalloc () */
#include <linux/list.h> /* INIT_LIST_HEAD () */
#include <linux/spinlock.h> /* spin_lock_init () */
#include <linux/init.h> /* __devinit */
#include <linux/errno.h> /* error codes */
#include <linux/interrupt.h> /* irqreturn_t */
#include <linux/device.h> /* device_create_file */
#include <linux/mutex.h> /* mutex_init () */

#include <asm/uaccess.h> /* put_user () */
#include <asm/bitops.h> /* set_bit () */

#include "sdicore.h"
#include "../include/master.h"
#include "mdev.h"
#include "sdim.h"
#include "mdma.h"
#include "sdim_qoe.h"
#include "lsdma.h"

#ifndef DEFINE_PCI_DEVICE_TABLE
#define DEFINE_PCI_DEVICE_TABLE(_table) \
	const struct pci_device_id _table[]
#endif

#ifndef DMA_BIT_MASK
#define DMA_BIT_MASK(n)	(((n) == 64) ? ~0ULL : ((1ULL<<(n))-1))
#endif

static const char sdim_qoe_name[] = SDIM_NAME_QOE;

/* Static function prototypes */
static ssize_t sdim_qoe_show_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static ssize_t sdim_qoe_store_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	const char *buf,
	size_t count);
static ssize_t sdim_qoe_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static int sdim_qoe_probe (struct pci_dev *pdev,
	const struct pci_device_id *id) __devinit;
void sdim_qoe_remove (struct pci_dev *pdev);
static irqreturn_t IRQ_HANDLER (sdim_qoe_irq_handler, irq, dev_id, regs);
static void sdim_qoe_init (struct master_iface *iface);
static void sdim_qoe_start (struct master_iface *iface);
static void sdim_qoe_stop (struct master_iface *iface);
static void sdim_qoe_exit (struct master_iface *iface);
static void sdim_qoe_start_tx_dma (struct master_iface *iface);
static long sdim_qoe_unlocked_ioctl (struct file *filp,
		unsigned int cmd,
		unsigned long arg);
static int FSYNC_HANDLER(sdim_qoe_fsync,filp,datasync);
static int sdim_qoe_init_module (void) __init;
static void sdim_qoe_cleanup_module (void) __exit;

MODULE_AUTHOR("Linear Systems Ltd.");
MODULE_DESCRIPTION("SDI Master Q/o driver");
MODULE_LICENSE("GPL");
MODULE_VERSION(MASTER_DRIVER_VERSION);

static char sdim_qoe_driver_name[] = "sdiqoe";

static DEFINE_PCI_DEVICE_TABLE(sdim_qoe_pci_id_table) = {
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			SDIM_PCI_DEVICE_ID_LINSYS_SDIQOE)
	},
	{0, }
};

static struct pci_driver sdim_qoe_pci_driver = {
	.name = sdim_qoe_driver_name,
	.id_table = sdim_qoe_pci_id_table,
	.probe = sdim_qoe_probe,
	.remove = sdim_qoe_remove
};

MODULE_DEVICE_TABLE(pci, sdim_qoe_pci_id_table);

static struct file_operations sdim_qoe_fops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.write = sdi_write,
	.poll = sdi_txpoll,
	.unlocked_ioctl = sdim_qoe_unlocked_ioctl,
	.compat_ioctl = sdim_qoe_unlocked_ioctl,
	.mmap = sdi_mmap,
	.open = sdi_open,
	.release = sdi_release,
	.fsync = sdim_qoe_fsync,
	.fasync = NULL
};

static struct master_iface_operations sdim_qoe_ops = {
	.init = sdim_qoe_init,
	.start = sdim_qoe_start,
	.stop = sdim_qoe_stop,
	.exit = sdim_qoe_exit,
	.start_tx_dma = sdim_qoe_start_tx_dma
};

static LIST_HEAD(sdim_qoe_card_list);

static struct class *sdim_qoe_class;

/**
 * sdim_qoe_show_blackburst_type - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
static ssize_t
sdim_qoe_show_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	return snprintf (buf, PAGE_SIZE, "%u\n",
		(readl (card->core.addr + SDIM_QOE_CSR) & SDIM_QOE_CSR_PAL) >> 2);
}

/**
 * sdim_qoe_store_blackburst_type - interface attribute write handler
 * @dev: device being written
 * @attr: device attribute
 * @buf: input buffer
 * @count:
 **/
static ssize_t
sdim_qoe_store_blackburst_type (struct device *dev,
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
	struct list_head *p;
	struct master_iface *iface;
	unsigned int total_users = 0;

	if ((endp == buf) || (val > max)) {
		return -EINVAL;
	}
	mutex_lock (&card->users_mutex);
	list_for_each (p, &card->iface_list) {
		iface = list_entry (p, struct master_iface, list);
		total_users += iface->users;
	}
	if (total_users) {
		retcode = -EBUSY;
		goto OUT;
	}
	reg = readl (card->core.addr + SDIM_QOE_CSR) & ~SDIM_QOE_CSR_PAL;
	writel (reg | (val << 2), card->core.addr + SDIM_QOE_CSR);
OUT:
	mutex_unlock (&card->users_mutex);
	return retcode;
}

/**
 * sdim_qoe_show_uid - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
static ssize_t
sdim_qoe_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	return snprintf (buf, PAGE_SIZE, "0x%08X%08X\n",
		readl (card->core.addr + SDIM_QOE_SSN_HI),
		readl (card->core.addr + SDIM_QOE_SSN_LO));
}

static DEVICE_ATTR(blackburst_type,S_IRUGO|S_IWUSR,
	sdim_qoe_show_blackburst_type,sdim_qoe_store_blackburst_type);
static DEVICE_ATTR(uid,S_IRUGO, sdim_qoe_show_uid, NULL);

/**
 * sdim_qoe_pci_probe - PCI insertion handler for a SDI Master Q/o
 * @pdev: PCI device
 * @id: PCI ID
 *
 * Handle the insertion of a SDI Master Q/o.
 * Returns a negative error code on failure and 0 on success.
 **/
static int __devinit
sdim_qoe_probe (struct pci_dev *pdev,
	const struct pci_device_id *id)
{
	int err, i;
	struct master_dev *card;

	/* Wake a sleeping device */
	if ((err = pci_enable_device (pdev)) < 0) {
		printk (KERN_WARNING "%s: unable to enable device\n",
			sdim_qoe_driver_name);
		return err;
	}

	/* Enable bus mastering */
	pci_set_master (pdev);

	/* Request the PCI I/O Resources */
	if ((err = pci_request_regions (pdev, sdim_qoe_driver_name)) < 0) {
		printk (KERN_WARNING "%s: unable to get I/O resources\n",
			sdim_qoe_driver_name);
		pci_disable_device (pdev);
		return err;
	}

	/* Set PCI DMA addressing limitations */
	if ((err = pci_set_dma_mask (pdev, DMA_BIT_MASK(32))) < 0) {
		printk (KERN_WARNING "%s: unable to set PCI DMA mask\n",
			sdim_qoe_driver_name);
		pci_disable_device (pdev);
		pci_release_regions (pdev);
		return err;
	}

	/* Initialize the driver_data pointer so that sdim_qoe_pci_remove()
	 * doesn't try to free it if an error occurs */
	pci_set_drvdata (pdev, NULL);

	/* Allocate a board info structure */
	if ((card = (struct master_dev *)
		kzalloc (sizeof (*card), GFP_KERNEL)) == NULL) {
		err = -ENOMEM;
		goto NO_MEM;
	}

	/* Initialize the board info structure */
	/* Remap bridge address to the DMA controller */
	card->bridge_addr = ioremap_nocache (pci_resource_start (pdev, 2),
		pci_resource_len (pdev, 2));
	card->core.addr = ioremap_nocache (pci_resource_start (pdev, 0),
		pci_resource_len (pdev, 0));
	card->version = readl(card->core.addr + SDIM_QOE_FPGAID) & 0xffff;
	card->name = sdim_qoe_name;
	card->id = pdev->device;
	card->irq = pdev->irq;
	card->irq_handler = sdim_qoe_irq_handler;
	card->capabilities = MASTER_CAP_BLACKBURST | MASTER_CAP_UID;
	INIT_LIST_HEAD(&card->iface_list);
	/* Lock for ICSR[] */
	spin_lock_init (&card->irq_lock);
	/* Lock for TCSR */
	spin_lock_init (&card->reg_lock);
	mutex_init (&card->users_mutex);
	card->parent = &pdev->dev;

	/* Get the firmware version and flush PCI posted writes */
	printk (KERN_INFO "%s: %s detected, firmware version %u.%u (0x%04X)\n",
		sdim_qoe_driver_name, card->name, card->version >> 8,
		card->version & 0x00ff, card->version);

	/* Store the pointer to the board info structure
	 * in the PCI info structure */
	pci_set_drvdata (pdev, card);

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
		&sdim_qoe_card_list,
		sdim_qoe_driver_name,
		sdim_qoe_class)) < 0) {
		goto NO_DEV;
	}

	/* Add device attributes */
	if (card->capabilities & MASTER_CAP_BLACKBURST) {
		if ((err = device_create_file (card->dev,
			&dev_attr_blackburst_type)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'blackburst_type'\n",
				sdim_qoe_driver_name);
		}
	}
	if (card->capabilities & MASTER_CAP_UID) {
		if ((err = device_create_file (card->dev,
			&dev_attr_uid)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'uid'\n",
				sdim_qoe_driver_name);
		}
	}

	/* Register the transmit interfaces */
	for (i = 0; i < 4; i++) {
		if ((err = sdi_register_iface (card,
			&lsdma_dma_ops,
			0,
			MASTER_DIRECTION_TX,
			&sdim_qoe_fops,
			&sdim_qoe_ops,
			SDI_CAP_TX_RXCLKSRC,
			4)) < 0) {
			goto NO_IFACE;
		}
	}
	return 0;

NO_IFACE:
NO_DEV:
NO_MEM:
	sdim_qoe_remove (pdev);
	return err;
}

/**
 * sdim_qoe_pci_remove - PCI removal handler for a SDI Master Q/o
 * @pdev: PCI device
 *
 * Handle the removal of a SDI Master Q/o.
 **/
void sdim_qoe_remove (struct pci_dev *pdev)
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
		list_for_each (p, &sdim_qoe_card_list) {
			if (p == &card->list) {
				mdev_unregister (card, sdim_qoe_class);
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
				mdma_advance(iface->dma);
			}
			/* Flag end of chain */
			if (status & LSDMA_CH_CSR_INTSRCDONE) {
				set_bit(SDI_EVENT_TX_BUFFER_ORDER,
					&iface->events);
				set_bit(0, &iface->dma_done);
			}

			/* Flag DMA abort */
			if (status & LSDMA_CH_CSR_INTSRCSTOP) {
				set_bit(0, &iface->dma_done);
			}

			interrupting_iface |= 0x1 << i;
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
	unsigned int clkreg =
		readl (card->core.addr + SDIM_QOE_CSR) & ~SDIM_QOE_CSR_EXTCLK;

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
		clkreg &= ~SDIM_QOE_CSR_EXTCLK;
		break;
	case SDI_CTL_TX_CLKSRC_EXT:
		clkreg |= SDIM_QOE_CSR_EXTCLK;
		break;
	}

	/* There will be no races on CSR
	 * until this code returns, so we don't need to lock it
	 */
	writel (reg | SDIM_QOE_TCSR_TXRST,
		card->core.addr + SDIM_QOE_TCSR(channel));
	/* XXX Set the transmit clock source (shared by all 4 channels)
	 * This is broken because each channel can set a different clock source
	 * for all channels */
	writel (clkreg, card->core.addr + SDIM_QOE_CSR);
	/* Dummy read to flush PCI posted writes */
	readl (card->core.addr + SDIM_QOE_FPGAID);
	writel (reg, card->core.addr + SDIM_QOE_TCSR(channel));

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

	/* Enable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	writel(SDIM_QOE_ICSR_TUIE | SDIM_QOE_ICSR_TXDIE,
		card->core.addr + SDIM_QOE_ICSR(channel));
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
	struct master_dma *dma = iface->dma;
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
	reg = readl(card->core.addr + SDIM_QOE_TCSR(channel));
	writel(reg & ~SDIM_QOE_TCSR_TXE,
		card->core.addr + SDIM_QOE_TCSR(channel));

	/* Disable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = SDIM_QOE_ICSR_TUIS | SDIM_QOE_ICSR_TXDIS;
	writel (reg, card->core.addr + SDIM_QOE_ICSR(channel));
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);
	spin_unlock_irq (&card->irq_lock);

	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE,
		card->bridge_addr + LSDMA_CSR(channel));
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
 * sdim_qoe_start_tx_dma - start transmit DMA
 * @iface: interface
 **/
static void
sdim_qoe_start_tx_dma (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	struct master_dma *dma = iface->dma;
	const unsigned int dma_channel = mdev_index (card, &iface->list);

	writel (LSDMA_CH_CSR_INTDONEENABLE |
		LSDMA_CH_CSR_INTSTOPENABLE,
		card->bridge_addr + LSDMA_CSR(dma_channel));
	writel (mdma_dma_to_desc_low (lsdma_head_desc_bus_addr (dma)),
		card->bridge_addr + LSDMA_DESC(dma_channel));
	writel (mdma_dma_to_desc_high (lsdma_head_desc_bus_addr (dma)),
		card->bridge_addr + LSDMA_DESC_H(dma_channel));
	clear_bit (0, &iface->dma_done);
	writel (LSDMA_CH_CSR_INTDONEENABLE |
		LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_ENABLE,
		card->bridge_addr + LSDMA_CSR(dma_channel));
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);
	return;
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
	case SDI_IOC_TXGETTXD:
	/* Atomic read of ICSR, so we don't need to lock */
		if (put_user ((readl (card->core.addr +
				SDIM_QOE_ICSR(channel)) &
				SDIM_QOE_ICSR_TXD) ? 1 : 0,
			(int __user *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return sdi_txioctl (filp, cmd, arg);
		}

	return 0;
}

/**
 * sdim_qoe_fsync - SDI Master Q/o fsync() method
 * @filp: file to flush
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
FSYNC_HANDLER(sdim_qoe_fsync,filp,datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	struct master_dma *dma = iface->dma;
	const unsigned int channel = mdev_index (card, &iface->list);

	mutex_lock (&iface->buf_mutex);
	lsdma_tx_link_all(dma);
	wait_event(iface->queue, test_bit(0, &iface->dma_done));
	lsdma_reset(dma);

	/* Wait for the onboard FIFOs to empty */
	/* Atomic read of ICSR, so we don't need to lock */
	wait_event(iface->queue,
		!(readl(card->core.addr +
			SDIM_QOE_ICSR(channel)) & SDIM_QOE_ICSR_TXD));

	mutex_unlock (&iface->buf_mutex);

	return 0;
}

/**
 * sdim_qoe_init_module - register the module as a Master and PCI driver
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int __init
sdim_qoe_init_module (void)
{
	int err;

	printk (KERN_INFO "%s: Linear Systems Ltd. "
		"SDI Master driver from master-%s (%s)\n",
		sdim_qoe_driver_name,
		MASTER_DRIVER_VERSION, MASTER_DRIVER_DATE);

	/* Create the device class */
	sdim_qoe_class = mdev_init (sdim_qoe_driver_name);
	if (IS_ERR(sdim_qoe_class)) {
		err = PTR_ERR(sdim_qoe_class);
		goto NO_CLASS;
	}

	/* Register with the PCI subsystem */
	if ((err = pci_register_driver (&sdim_qoe_pci_driver)) < 0) {
		printk (KERN_WARNING
			"%s: unable to register with PCI subsystem\n",
			sdim_qoe_driver_name);
		goto NO_PCI;
	}

	return 0;

NO_PCI:
	mdev_cleanup (sdim_qoe_class);
NO_CLASS:
	return err;
}

/**
 * sdim_qoe_cleanup_module - unregister the module as a Master and PCI driver
 **/
static void __exit
sdim_qoe_cleanup_module (void)
{
	pci_unregister_driver (&sdim_qoe_pci_driver);
	mdev_cleanup (sdim_qoe_class);
	return;
}

module_init (sdim_qoe_init_module);
module_exit (sdim_qoe_cleanup_module);

