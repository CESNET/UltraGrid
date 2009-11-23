/* sdim_qie.c
 *
 * Linux driver functions for Linear Systems Ltd. SDI Master Q/i.
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
#include <linux/device.h> /* class_device_create_file () */

#include <asm/semaphore.h> /* sema_init () */
#include <asm/uaccess.h> /* put_user () */
#include <asm/bitops.h> /* set_bit () */

#include "sdicore.h"
#include "../include/master.h"
#include "miface.h"
//Temporary fix for kernel 2.6.21
#include "mdev.c"
#include "plx9080.h"
//Temporary fix for kernel 2.6.21
#include "masterplx.c"
#include "lsdma.h"
//Temporary fix for kernel 2.6.21
#include "masterlsdma.c"
#include "sdim_qie.h"
#include "sdim.h"

#ifndef list_for_each_safe
#define list_for_each_safe(pos, n, head) \
	for (pos = (head)->next, n = pos->next; pos != (head); \
		pos = n, n = pos->next)
#endif

static const char sdim_qie_name[] = SDIM_NAME_QIE;
static const char sdim_qi_name[] = SDIM_NAME_QI;

/* Static function prototypes */
static ssize_t sdim_qie_show_uid (struct class_device *cd, char *buf);
static int sdi_qie_pci_probe (struct pci_dev *dev,
	const struct pci_device_id *id) __devinit;
static void sdi_qie_pci_remove (struct pci_dev *dev);
static irqreturn_t IRQ_HANDLER(sdim_qie_irq_handler,irq,dev_id,regs);
static void sdim_qie_init (struct master_iface *iface);
static void sdim_qie_start (struct master_iface *iface);
static void sdim_qie_stop (struct master_iface *iface);
static void sdim_qie_exit (struct master_iface *iface);
static int sdim_qie_open (struct inode *inode, struct file *filp);
static long sdim_qie_unlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int sdim_qie_ioctl (struct inode *inode,
	struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int sdim_qie_fsync (struct file *filp,
	struct dentry *dentry,
	int datasync);
static int sdim_qie_release (struct inode *inode, struct file *filp);
static int sdim_qie_init_module (void) __init;
static void sdim_qie_cleanup_module (void) __exit;

MODULE_AUTHOR("Linear Systems Ltd.");
MODULE_DESCRIPTION("SDI Master Quad i Driver");
MODULE_LICENSE("GPL");

#ifdef MODULE_VERSION
MODULE_VERSION(MASTER_DRIVER_VERSION);
#endif

static char sdiqie_driver_name[] = "sdim_qie";

static struct pci_device_id sdiqie_pci_id_table[] = {
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

static struct pci_driver sdiqie_pci_driver = {
	.name = sdiqie_driver_name,
	.id_table = sdiqie_pci_id_table,
	.probe = sdi_qie_pci_probe,
	.remove = sdi_qie_pci_remove
};

struct file_operations sdim_qie_rxfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.read = masterlsdma_read,
	.poll = masterlsdma_rxpoll,
	.ioctl = sdim_qie_ioctl,
#ifdef HAVE_UNLOCKED_IOCTL
	.unlocked_ioctl = sdim_qie_unlocked_ioctl,
#endif
#ifdef HAVE_COMPAT_IOCTL
	.compat_ioctl = sdi_compat_ioctl,
#endif
	.mmap = masterlsdma_mmap,
	.open = sdim_qie_open,
	.release = sdim_qie_release,
	.fsync = sdim_qie_fsync,
	.fasync = NULL
};

MODULE_DEVICE_TABLE(pci, sdiqie_pci_id_table);
static LIST_HEAD(sdim_card_list);

static struct class sdiqie_class = {
	.name = sdiqie_driver_name,
	.release = mdev_class_device_release,
	.class_release = NULL
};

/**
 * sdim_qie_show_uid - interface attribute read handler
 * @cd: class_device being read
 * @buf: output buffer
 **/
static ssize_t
sdim_qie_show_uid (struct class_device *cd,
	char *buf)
{
	struct master_dev *card = to_master_dev(cd);

	return snprintf (buf, PAGE_SIZE, "0x%08X%08X\n",
		master_readl (card, SDIM_QIE_UIDR_HI),
		master_readl (card, SDIM_QIE_UIDR_LO));
}

static CLASS_DEVICE_ATTR(uid,S_IRUGO,
	sdim_qie_show_uid,NULL);

/**
 * sdim_pci_probe - PCI insertion handler for a SDI Master Q/i
 * @dev: PCI device
 *
 * Handle the insertion of a SDI Master Q/i.
 * Returns a negative error code on failure and 0 on success.
 **/
int __devinit
sdi_qie_pci_probe (struct pci_dev *dev,
	const struct pci_device_id *id)
{

	int err;
	unsigned int i;
	const char *name;
	struct master_dev *card;

	switch (dev->device) {
	case SDIM_PCI_DEVICE_ID_LINSYS_SDIQIE:
		name = sdim_qie_name;
		break;
	case SDIM_PCI_DEVICE_ID_LINSYS_SDIQI:
		name = sdim_qi_name;
		break;
	default:
		name = "";
		break;
	}

	/* Store the pointer to the board info structure
	 * in the PCI info structure */
	pci_set_drvdata (dev, NULL);

	/* Wake a sleeping device */
	if ((err = pci_enable_device (dev)) < 0) {
		printk (KERN_WARNING "%s: unable to enable device\n",
			sdiqie_driver_name);
		goto NO_PCI;
	}

	/* Enable bus mastering */
	pci_set_master (dev);

	/* Request the PCI I/O resources */
	if ((err = pci_request_regions (dev, sdiqie_driver_name)) < 0) {
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
	/* PLX 9056 */
	card->bridge_addr = ioremap_nocache (pci_resource_start (dev, 0),
		pci_resource_len (dev, 0));
	/* SDI Core */
	card->core.addr = ioremap_nocache (pci_resource_start (dev, 2),
		pci_resource_len (dev, 2));
	card->version = master_readl (card, SDIM_QIE_FPGAID) & 0xffff;
	card->name = sdim_qie_name;
	card->irq_handler = sdim_qie_irq_handler;
	INIT_LIST_HEAD(&card->iface_list);
	card->capabilities = MASTER_CAP_UID;
	/* Lock for LSDMA_CSR, ICSR */
	spin_lock_init (&card->irq_lock);
	/* Lock for PFLUT, RCR */
	spin_lock_init (&card->reg_lock);
	sema_init (&card->users_sem, 1);
	card->pdev = dev;

	/* Print the firmware version */
	printk (KERN_INFO "%s: %s detected, firmware version %u.%u (0x%04X)\n",
		sdiqie_driver_name, name,
		card->version >> 8, card->version & 0x00ff, card->version);
	pci_set_drvdata (dev, card);
	/* Reset the PCI 9056 */
	masterplx_reset_bridge (card);

	/* Setup the PCI 9056 */
	writel (PLX_INTCSR_PCIINT_ENABLE |
		PLX_INTCSR_PCILOCINT_ENABLE,
		card->bridge_addr + PLX_INTCSR);
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + PLX_INTCSR);

	/* Remap bridge address to the DMA controller */
	iounmap (card->bridge_addr);
	/* LS DMA Controller */
	card->bridge_addr = ioremap_nocache (pci_resource_start (dev, 3),
		pci_resource_len (dev, 3));
	/* Reset the FPGA */
	for (i = 0; i < 4; i++) {
		master_writel (card, SDIM_QIE_RCR(i), SDIM_QIE_RCSR_RST);
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
		&sdim_card_list,
		sdiqie_driver_name,
		&sdiqie_class)) < 0) {
		goto NO_DEV;
	}

	/* Add class_device attributes */
	if (card->capabilities & MASTER_CAP_UID) {
		if ((err = class_device_create_file (&card->class_dev,
			&class_device_attr_uid)) < 0) {
			printk (KERN_WARNING
				"%s: Unable to create file 'uid'\n",
				sdiqie_driver_name);
		}
	}

	/* Register receiver interfaces */
	for (i = 0; i < 4; i++) {
		if ((err = sdi_register_iface (card,
			MASTER_DIRECTION_RX,
			&sdim_qie_rxfops,
			0,
			4)) < 0) {
			goto NO_IFACE;
		}
	}

	return 0;

NO_IFACE:
	sdi_qie_pci_remove (dev);
NO_DEV:
NO_MEM:
NO_PCI:
	return err;
}

/**
 * sdim_pci_remove - PCI removal handler for a SDI Master Q/i.
 * @card: Master device
 *
 * Handle the removal of a SDI Master Q/i.
 **/
void
sdi_qie_pci_remove (struct pci_dev *dev)
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
				lsdma_advance (iface->dma);
				if (lsdma_rx_isempty (iface->dma)) {
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
		status = master_readl (card, SDIM_QIE_ICSR(i));
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
	wmb ();
	writel (reg, card->core.addr + SDIM_QIE_RCR(channel));
	wmb ();
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
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	/* Enable and start DMA */
	writel (lsdma_dma_to_desc_low (lsdma_head_desc_bus_addr (iface->dma)),
		card->bridge_addr + LSDMA_DESC(channel));
	clear_bit (0, &iface->dma_done);
	wmb ();
	writel (0x00000001 | LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
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
	spin_lock (&card->reg_lock);
	reg = master_readl (card, SDIM_QIE_RCR(channel));
	writel (reg | SDIM_QIE_RCSR_EN, card->core.addr + SDIM_QIE_RCR(channel));
	spin_unlock (&card->reg_lock);

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
	spin_lock (&card->reg_lock);
	reg = master_readl (card, SDIM_QIE_RCR(channel));
	writel (reg & ~SDIM_QIE_RCSR_EN, card->core.addr + SDIM_QIE_RCR(channel));
	spin_unlock (&card->reg_lock);

	/* Disable receiver interrupts */
	spin_lock_irq (&card->irq_lock);
	writel (SDIM_QIE_ICSR_RXCDIS | SDIM_QIE_ICSR_RXOIS |
		SDIM_QIE_ICSR_RXDIS, card->core.addr + SDIM_QIE_ICSR(channel));

	/* Disable and abort DMA */
	writel ((LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION | LSDMA_CH_CSR_STOP) & ~LSDMA_CH_CSR_ENABLE,
		card->bridge_addr + LSDMA_CSR(channel));
	wmb ();
	writel ((LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION | LSDMA_CH_CSR_STOP) & ~LSDMA_CH_CSR_ENABLE,
		card->bridge_addr + LSDMA_CSR(channel));

	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);
	spin_unlock_irq (&card->irq_lock);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	writel ((LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION | LSDMA_CH_CSR_STOP) & ~LSDMA_CH_CSR_ENABLE,
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
 * sdim_qie_open - SDI Master Q/i receiver open() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
sdim_qie_open (struct inode *inode, struct file *filp)
{

	return masterlsdma_open (inode,
		filp,
		sdim_qie_init,
		sdim_qie_start,
		0,
		LSDMA_MMAP);
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
	case SDI_IOC_RXGETBUFLEVEL:
		if (put_user (lsdma_rx_buflevel (iface->dma),
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_RXGETSTATUS:
		/* We don't lock since this should be an atomic read */
		if (put_user ((master_readl (card, SDIM_QIE_ICSR(channel)) &
			SDIM_QIE_ICSR_RXPASSING) ? 1 : 0, (int *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_RXGETCARRIER:
		/* We don't lock since this should be an atomic read */
		if (put_user ((master_readl (card, SDIM_QIE_ICSR(channel)) &
			SDIM_QIE_ICSR_RXCD) ? 1 : 0, (int *)arg)) {
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
 * sdim_qie_ioctl - SDI Master Q/i ioctl() method
 * @inode: inode
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
sdim_qie_ioctl (struct inode *inode,
	struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	return sdim_qie_unlocked_ioctl (filp, cmd, arg);
}

/**
 * sdim_qie_fsync - SDI Master Q/i receiver fsync() method
 * @filp: file to flush
 * @dentry: directory entry associated with the file
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
sdim_qie_fsync (struct file *filp,
	struct dentry *dentry,
	int datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	if (down_interruptible (&iface->buf_sem)) {
		return -ERESTARTSYS;
	}

	/* Stop the receiver */
	sdim_qie_stop (iface);

	/* Reset the onboard FIFO and driver buffers */
	spin_lock (&card->reg_lock);
	reg = master_readl (card, SDIM_QIE_RCR(channel));
	writel (reg | SDIM_QIE_RCSR_RST, card->core.addr + SDIM_QIE_RCR(channel));
	wmb ();
	writel (reg, card->core.addr + SDIM_QIE_RCR(channel));
	spin_unlock (&card->reg_lock);
	iface->events = 0;
	lsdma_reset (iface->dma);

	/* Start the receiver */
	sdim_qie_start (iface);

	up (&iface->buf_sem);
	return 0;
}

/**
 * sdim_qie_release - SDI Master Q/i receiver release() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
sdim_qie_release (struct inode *inode, struct file *filp)
{
	struct master_iface *iface = filp->private_data;

	return masterlsdma_release (iface, sdim_qie_stop, sdim_qie_exit);
}

/**
 * sdim_qie_init_module - register the module as a PCI driver
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int __init
sdim_qie_init_module (void)
{
	printk (KERN_INFO "%s: Linear Systems Ltd. "
		"SDI Master driver from master-%s (%s)\n",
		sdiqie_driver_name, MASTER_DRIVER_VERSION, MASTER_DRIVER_DATE);

	return mdev_init_module (&sdiqie_pci_driver,
		&sdiqie_class,
		sdiqie_driver_name);
}

/**
 * sdim_qie_cleanup_module - unregister the module as a PCI driver
 **/
static void __exit
sdim_qie_cleanup_module (void)
{
	mdev_cleanup_module (&sdiqie_pci_driver, &sdiqie_class);

	return;
}

module_init (sdim_qie_init_module);
module_exit (sdim_qie_cleanup_module);
