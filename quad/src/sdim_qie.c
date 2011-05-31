/* sdim_qie.c
 *
 * Linux driver functions for Linear Systems Ltd. SDI Master Q/i.
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
#include <linux/device.h> /* device_create_file () */
#include <linux/mutex.h> /* mutex_init () */

#include <asm/uaccess.h> /* put_user () */
#include <asm/bitops.h> /* set_bit () */

#include "sdicore.h"
#include "../include/master.h"
#include "mdev.h"
#include "mdma.h"
#include "plx9080.h"
#include "lsdma.h"
#include "sdim_qie.h"
#include "sdim.h"

#ifndef DEFINE_PCI_DEVICE_TABLE
#define DEFINE_PCI_DEVICE_TABLE(_table) \
	const struct pci_device_id _table[]
#endif

#ifndef DMA_BIT_MASK
#define DMA_BIT_MASK(n)	(((n) == 64) ? ~0ULL : ((1ULL<<(n))-1))
#endif

static const char sdim_qie_name[] = SDIM_NAME_QIE;
static const char sdim_qi_name[] = SDIM_NAME_QI;

/* Static function prototypes */
static ssize_t sdim_qie_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static int sdim_qie_pci_probe (struct pci_dev *pdev,
	const struct pci_device_id *id) __devinit;
static void sdim_qie_pci_remove (struct pci_dev *pdev);
static irqreturn_t IRQ_HANDLER(sdim_qie_irq_handler,irq,dev_id,regs);
static void sdim_qie_init (struct master_iface *iface);
static void sdim_qie_start (struct master_iface *iface);
static void sdim_qie_stop (struct master_iface *iface);
static void sdim_qie_exit (struct master_iface *iface);
static long sdim_qie_unlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int FSYNC_HANDLER(sdim_qie_fsync,filp,datasync);
static int sdim_qie_init_module (void) __init;
static void sdim_qie_cleanup_module (void) __exit;

MODULE_AUTHOR("Linear Systems Ltd.");
MODULE_DESCRIPTION("SDI Master Quad i driver");
MODULE_LICENSE("GPL");
MODULE_VERSION(MASTER_DRIVER_VERSION);

static char sdim_qie_driver_name[] = "sdiqie";

static DEFINE_PCI_DEVICE_TABLE(sdim_qie_pci_id_table) = {
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			SDIM_PCI_DEVICE_ID_LINSYS_SDIQIE)
	},
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			SDIM_PCI_DEVICE_ID_LINSYS_SDIQI)
	},
	{0, }
};

static struct pci_driver sdim_qie_pci_driver = {
	.name = sdim_qie_driver_name,
	.id_table = sdim_qie_pci_id_table,
	.probe = sdim_qie_pci_probe,
	.remove = sdim_qie_pci_remove
};

MODULE_DEVICE_TABLE(pci, sdim_qie_pci_id_table);

static struct file_operations sdim_qie_fops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.read = sdi_read,
	.poll = sdi_rxpoll,
	.unlocked_ioctl = sdim_qie_unlocked_ioctl,
	.compat_ioctl = sdim_qie_unlocked_ioctl,
	.mmap = sdi_mmap,
	.open = sdi_open,
	.release = sdi_release,
	.fsync = sdim_qie_fsync,
	.fasync = NULL
};

static struct master_iface_operations sdim_qie_ops = {
	.init = sdim_qie_init,
	.start = sdim_qie_start,
	.stop = sdim_qie_stop,
	.exit = sdim_qie_exit
};

static LIST_HEAD(sdim_qie_card_list);

static struct class *sdim_qie_class;

/**
 * sdim_qie_show_uid - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
static ssize_t
sdim_qie_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	return snprintf (buf, PAGE_SIZE, "0x%08X%08X\n",
		readl (card->core.addr + SDIM_QIE_UIDR_HI),
		readl (card->core.addr + SDIM_QIE_UIDR_LO));
}

static DEVICE_ATTR(uid,S_IRUGO,
	sdim_qie_show_uid,NULL);

/**
 * sdim_qie_pci_probe - PCI insertion handler for a SDI Master Q/i
 * @pdev: PCI device
 *
 * Handle the insertion of a SDI Master Q/i.
 * Returns a negative error code on failure and 0 on success.
 **/
static int __devinit
sdim_qie_pci_probe (struct pci_dev *pdev,
	const struct pci_device_id *id)
{
	int err;
	unsigned int i;
	struct master_dev *card;
	void __iomem *p;

	/* Wake a sleeping device */
	if ((err = pci_enable_device (pdev)) < 0) {
		printk (KERN_WARNING "%s: unable to enable device\n",
			sdim_qie_driver_name);
		return err;
	}

	/* Enable bus mastering */
	pci_set_master (pdev);

	/* Request the PCI I/O resources */
	if ((err = pci_request_regions (pdev, sdim_qie_driver_name)) < 0) {
		printk (KERN_WARNING "%s: unable to get I/O resources\n",
			sdim_qie_driver_name);
		pci_disable_device (pdev);
		return err;
	}

	/* Set PCI DMA addressing limitations */
	if ((err = pci_set_dma_mask (pdev, DMA_BIT_MASK(32))) < 0) {
		printk (KERN_WARNING "%s: unable to set PCI DMA mask\n",
			sdim_qie_driver_name);
		pci_disable_device (pdev);
		pci_release_regions (pdev);
		return err;
	}

	/* Initialize the driver_data pointer so that sdim_qie_pci_remove()
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
	card->bridge_addr = ioremap_nocache (pci_resource_start (pdev, 3),
		pci_resource_len (pdev, 3));
	/* SDI Core */
	card->core.addr = ioremap_nocache (pci_resource_start (pdev, 2),
		pci_resource_len (pdev, 2));
	card->version = readl (card->core.addr + SDIM_QIE_FPGAID) & 0xffff;
	switch (pdev->device) {
	default:
	case SDIM_PCI_DEVICE_ID_LINSYS_SDIQIE:
		card->name = sdim_qie_name;
		break;
	case SDIM_PCI_DEVICE_ID_LINSYS_SDIQI:
		card->name = sdim_qi_name;
		break;
	}
	card->id = pdev->device;
	card->irq = pdev->irq;
	card->irq_handler = sdim_qie_irq_handler;
	INIT_LIST_HEAD(&card->iface_list);
	card->capabilities = MASTER_CAP_UID;
	/* Lock for LSDMA_CSR, ICSR */
	spin_lock_init (&card->irq_lock);
	/* Lock for PFLUT, RCR */
	spin_lock_init (&card->reg_lock);
	mutex_init (&card->users_mutex);
	card->parent = &pdev->dev;

	/* Print the firmware version */
	printk (KERN_INFO "%s: %s detected, firmware version %u.%u (0x%04X)\n",
		sdim_qie_driver_name, card->name,
		card->version >> 8, card->version & 0x00ff, card->version);

	/* Store the pointer to the board info structure
	 * in the PCI info structure */
	pci_set_drvdata (pdev, card);

	/* PLX */
	p = ioremap_nocache (pci_resource_start (pdev, 0),
		pci_resource_len (pdev, 0));

	/* Reset the PCI 9056 */
	plx_reset_bridge (p);

	/* Setup the PCI 9056 */
	writel (PLX_INTCSR_PCIINT_ENABLE |
		PLX_INTCSR_PCILOCINT_ENABLE,
		p + PLX_INTCSR);
	/* Dummy read to flush PCI posted writes */
	readl (p + PLX_INTCSR);

	/* Unmap PLX */
	iounmap (p);

	/* Reset the FPGA */
	for (i = 0; i < 4; i++) {
		writel (SDIM_QIE_RCSR_RST, card->core.addr + SDIM_QIE_RCR(i));
	}

	/* Setup the LS DMA controller */
	writel (LSDMA_INTMSK_CH(0) | LSDMA_INTMSK_CH(1) |
		LSDMA_INTMSK_CH(2) | LSDMA_INTMSK_CH(3),
		card->bridge_addr + LSDMA_INTMSK);
	for (i = 0; i < 4; i++) {
		writel (LSDMA_CH_CSR_INTDONEENABLE |
			LSDMA_CH_CSR_INTSTOPENABLE |
			LSDMA_CH_CSR_DIRECTION,
			card->bridge_addr + LSDMA_CSR(i));
	}

	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);

	/* Register a Master device */
	if ((err = mdev_register (card,
		&sdim_qie_card_list,
		sdim_qie_driver_name,
		sdim_qie_class)) < 0) {
		goto NO_DEV;
	}

	/* Add device attributes */
	if (card->capabilities & MASTER_CAP_UID) {
		if ((err = device_create_file (card->dev,
			&dev_attr_uid)) < 0) {
			printk (KERN_WARNING
				"%s: Unable to create file 'uid'\n",
				sdim_qie_driver_name);
		}
	}

	/* Register receiver interfaces */
	for (i = 0; i < 4; i++) {
		if ((err = sdi_register_iface (card,
			&lsdma_dma_ops,
			0,
			MASTER_DIRECTION_RX,
			&sdim_qie_fops,
			&sdim_qie_ops,
			0,
			4)) < 0) {
			goto NO_IFACE;
		}
	}

	return 0;

NO_IFACE:
NO_DEV:
NO_MEM:
	sdim_qie_pci_remove (pdev);
	return err;
}

/**
 * sdim_qie_pci_remove - PCI removal handler for a SDI Master Q/i.
 * @pdev: PCI device
 *
 * Handle the removal of a SDI Master Q/i.
 **/
static void
sdim_qie_pci_remove (struct pci_dev *pdev)
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
		list_for_each (p, &sdim_qie_card_list) {
			if (p == &card->list) {
				mdev_unregister (card, sdim_qie_class);
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
 * sdim_qie_irq_handler - SDI Master Q/i interrupt service routine
 * @irq: interrupt number
 * @dev_id: pointer to the device data structure
 * @regs: processor context
 **/
static irqreturn_t
IRQ_HANDLER(sdim_qie_irq_handler,irq,dev_id,regs)
{
	struct master_dev *card = dev_id;
	struct list_head *p = &card->iface_list;
	struct master_iface *iface;
	unsigned int dmaintsrc = readl (card->bridge_addr + LSDMA_INTSRC);
	unsigned int status, interrupting_iface = 0, i;

	for (i = 0; i < 4; i++) {
		p = p->next;
		iface = list_entry (p, struct master_iface, list);

		/* Clear DMA interrupts */
		if (dmaintsrc & LSDMA_INTSRC_CH(i)) {
			/* Read the interrupt type and clear it */
			spin_lock (&card->irq_lock);
			status = readl (card->bridge_addr + LSDMA_CSR(i));
			writel (status, card->bridge_addr + LSDMA_CSR(i));
			spin_unlock (&card->irq_lock);
			/* Increment the buffer pointer */
			if (status & LSDMA_CH_CSR_INTSRCBUFFER) {
				mdma_advance (iface->dma);
				if (mdma_rx_isempty (iface->dma)) {
					set_bit (SDI_EVENT_RX_BUFFER_ORDER,
						&iface->events);
				}
			}

			/* Flag end-of-chain */
			if (status & LSDMA_CH_CSR_INTSRCDONE) {
				set_bit (0, &iface->dma_done);
			}

			/* Flag DMA abort */
			if (status & LSDMA_CH_CSR_INTSRCSTOP) {
				set_bit (0, &iface->dma_done);
			}

			interrupting_iface |= (0x1 << i);
		}

		/* Clear SDI interrupts */
		spin_lock (&card->irq_lock);
		status = readl (card->core.addr + SDIM_QIE_ICSR(i));
		writel (status, card->core.addr + SDIM_QIE_ICSR(i));
		spin_unlock (&card->irq_lock);
			if (status & SDIM_QIE_ICSR_RXCDIS) {
				set_bit (SDI_EVENT_RX_CARRIER_ORDER,
					&iface->events);
				interrupting_iface |= (0x1 << i);
			}
			if (status & SDIM_QIE_ICSR_RXOIS) {
				set_bit (SDI_EVENT_RX_FIFO_ORDER,
					&iface->events);
				interrupting_iface |= (0x1 << i);
			}
		if (interrupting_iface & (0x1 << i)) {
			wake_up (&iface->queue);
		}
	}

	if (interrupting_iface) {
		/* Dummy read to flush PCI posted writes */
		readl (card->bridge_addr + LSDMA_INTMSK);
		return IRQ_HANDLED;
	}
	return IRQ_NONE;
}

/**
 * sdim_qie_init - Initialize a SDI Master Q/i receiver
 * @iface: interface
 **/
static void
sdim_qie_init (struct master_iface *iface)
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
		reg |= SDIM_QIE_RCSR_10BIT;
		break;
	}

	/* There will be no races on RCR
	 * until this code returns, so we don't need to lock it */
	writel (reg | SDIM_QIE_RCSR_RST, card->core.addr + SDIM_QIE_RCR(channel));
	writel (reg, card->core.addr + SDIM_QIE_RCR(channel));
	writel(SDIM_QIE_RDMATL, card->core.addr + SDIM_QIE_RDMATLR(channel));

	return;
}

/**
 * sdim_qie_start - Activate the SDI Master Q/i receiver
 * @iface: interface
 **/
static void
sdim_qie_start (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	struct master_dma *dma = iface->dma;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	/* Enable and start DMA */
	writel (mdma_dma_to_desc_low (lsdma_head_desc_bus_addr (dma)),
		card->bridge_addr + LSDMA_DESC(channel));
	writel (mdma_dma_to_desc_high (lsdma_head_desc_bus_addr (dma)),
		card->bridge_addr + LSDMA_DESC_H(channel));
	clear_bit (0, &iface->dma_done);
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION | LSDMA_CH_CSR_ENABLE,
		card->bridge_addr + LSDMA_CSR(channel));
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);

	/* Enable receiver interrupts */
	spin_lock_irq (&card->irq_lock);
	writel (SDIM_QIE_ICSR_RXCDIE | SDIM_QIE_ICSR_RXOIE |
		SDIM_QIE_ICSR_RXDIE, card->core.addr + SDIM_QIE_ICSR(channel));
	spin_unlock_irq (&card->irq_lock);

	/* Enable the receiver */
	reg = readl (card->core.addr + SDIM_QIE_RCR(channel));
	writel (reg | SDIM_QIE_RCSR_EN,
		card->core.addr + SDIM_QIE_RCR(channel));

	return;
}

/**
 * sdim_qie_stop - Deactivate the SDI Master Q/i receiver
 * @iface: interface
 **/
static void
sdim_qie_stop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	/* Disable the receiver */
	reg = readl (card->core.addr + SDIM_QIE_RCR(channel));
	writel (reg & ~SDIM_QIE_RCSR_EN,
		card->core.addr + SDIM_QIE_RCR(channel));

	/* Disable receiver interrupts */
	spin_lock_irq (&card->irq_lock);
	writel (SDIM_QIE_ICSR_RXCDIS | SDIM_QIE_ICSR_RXOIS |
		SDIM_QIE_ICSR_RXDIS, card->core.addr + SDIM_QIE_ICSR(channel));
	spin_unlock_irq (&card->irq_lock);

	/* Disable and abort DMA */
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION | LSDMA_CH_CSR_STOP,
		card->bridge_addr + LSDMA_CSR(channel));
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION | LSDMA_CH_CSR_STOP,
		card->bridge_addr + LSDMA_CSR(channel));
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);

	wait_event (iface->queue, test_bit (0, &iface->dma_done));

	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION | LSDMA_CH_CSR_STOP,
		card->bridge_addr + LSDMA_CSR(channel));

	return;
}

/**
 * sdim_qie_exit - Clean up the SDI Master Q/i receiver
 * @iface: interface
 **/
static void
sdim_qie_exit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);

	/* Reset the receiver.
	 * There will be no races on RCR here,
	 * so we don't need to lock it */
	writel (SDIM_QIE_RCSR_RST, card->core.addr + SDIM_QIE_RCR(channel));

	return;
}

/**
 * sdim_qie_unlocked_ioctl - SDI Master Q/i receiver unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
sdim_qie_unlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);

	switch (cmd) {
	case SDI_IOC_RXGETSTATUS:
		/* We don't lock since this should be an atomic read */
		if (put_user ((readl (card->core.addr + SDIM_QIE_ICSR(channel)) &
			SDIM_QIE_ICSR_RXPASSING) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_RXGETCARRIER:
		/* We don't lock since this should be an atomic read */
		if (put_user ((readl (card->core.addr + SDIM_QIE_ICSR(channel)) &
			SDIM_QIE_ICSR_RXCD) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return sdi_rxioctl (filp, cmd, arg);
	}

	return 0;
}

/**
 * sdim_qie_fsync - SDI Master Q/i receiver fsync() method
 * @filp: file to flush
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
FSYNC_HANDLER(sdim_qie_fsync,filp,datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	mutex_lock (&iface->buf_mutex);

	/* Stop the receiver */
	sdim_qie_stop (iface);

	/* Reset the onboard FIFO and driver buffers */
	reg = readl (card->core.addr + SDIM_QIE_RCR(channel));
	writel (reg | SDIM_QIE_RCSR_RST, card->core.addr + SDIM_QIE_RCR(channel));
	writel (reg, card->core.addr + SDIM_QIE_RCR(channel));
	iface->events = 0;
	lsdma_reset (iface->dma);

	/* Start the receiver */
	sdim_qie_start (iface);

	mutex_unlock (&iface->buf_mutex);
	return 0;
}

/**
 * sdim_qie_init_module - register the module as a Master and PCI driver
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int __init
sdim_qie_init_module (void)
{
	int err;

	printk (KERN_INFO "%s: Linear Systems Ltd. "
		"SDI Master driver from master-%s (%s)\n",
		sdim_qie_driver_name,
		MASTER_DRIVER_VERSION, MASTER_DRIVER_DATE);

	/* Create the device class */
	sdim_qie_class = mdev_init (sdim_qie_driver_name);
	if (IS_ERR(sdim_qie_class)) {
		err = PTR_ERR(sdim_qie_class);
		goto NO_CLASS;
	}

	/* Register with the PCI subsystem */
	if ((err = pci_register_driver (&sdim_qie_pci_driver)) < 0) {
		printk (KERN_WARNING
			"%s: unable to register with PCI subsystem\n",
			sdim_qie_driver_name);
		goto NO_PCI;
	}

	return 0;

NO_PCI:
	mdev_cleanup (sdim_qie_class);
NO_CLASS:
	return err;
}

/**
 * sdim_qie_cleanup_module - unregister the module as a Master and PCI driver
 **/
static void __exit
sdim_qie_cleanup_module (void)
{
	pci_unregister_driver (&sdim_qie_pci_driver);
	mdev_cleanup (sdim_qie_class);

	return;
}

module_init (sdim_qie_init_module);
module_exit (sdim_qie_cleanup_module);

