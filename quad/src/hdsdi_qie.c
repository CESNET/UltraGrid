/* hdsdi_qi.c
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
#include "sdim.h"
#include "../include/master.h"
#include "miface.h"
//Temporary fix for kernel 2.6.21
#include "mdev.c"
//Temporary fix for kernel 2.6.21
#include "lsdma.h"
//Temporary fix for kernel 2.6.21
#include "masterlsdma.c"
#include "hdsdi_qie.h"

#ifndef list_for_each_safe
#define list_for_each_safe(pos, n, head) \
	for (pos = (head)->next, n = pos->next; pos != (head); \
		pos = n, n = pos->next)
#endif

static const char hdsdi_qi_name[] = HDSDI_NAME;

/* Static function prototypes */
static ssize_t hdsdi_qi_show_uid (struct class_device *cd, char *buf);
static int hdsdi_qi_pci_probe (struct pci_dev *dev,
	const struct pci_device_id *id) __devinit;
static void hdsdi_qi_pci_remove (struct pci_dev *dev);
static irqreturn_t IRQ_HANDLER(hdsdi_qi_irq_handler,irq,dev_id,regs);
static void hdsdi_qi_init (struct master_iface *iface);
static void hdsdi_qi_start (struct master_iface *iface);
static void hdsdi_qi_stop (struct master_iface *iface);
static void hdsdi_qi_exit (struct master_iface *iface);
static int hdsdi_qi_open (struct inode *inode, struct file *filp);
static long hdsdi_qi_unlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int hdsdi_qi_ioctl (struct inode *inode,
	struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int hdsdi_qi_fsync (struct file *filp,
	struct dentry *dentry,
	int datasync);
static int hdsdi_qi_release (struct inode *inode, struct file *filp);
static int hdsdi_qi_init_module (void) __init;
static void hdsdi_qi_cleanup_module (void) __exit;

MODULE_AUTHOR("Linear Systems Ltd.");
MODULE_DESCRIPTION("HD-SDI Q/i Driver");
MODULE_LICENSE("GPL");

#ifdef MODULE_VERSION
MODULE_VERSION(MASTER_DRIVER_VERSION);
#endif

static char hdsdi_driver_name[] = "hdsdiqi";

static struct pci_device_id hdsdi_pci_id_table[] = {
	{
		PCI_DEVICE(MASTER_PCI_VENDOR_ID_LINSYS,
			HDSDI_PCI_DEVICE_ID_LINSYS)
	},
	{0, }
};

static struct pci_driver hdsdi_pci_driver = {
	.name = hdsdi_driver_name,
	.id_table = hdsdi_pci_id_table,
	.probe = hdsdi_qi_pci_probe,
	.remove = hdsdi_qi_pci_remove
};

struct file_operations hdsdi_qi_rxfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.read = masterlsdma_read,
	.poll = masterlsdma_rxpoll,
	.ioctl = hdsdi_qi_ioctl,
#ifdef HAVE_UNLOCKED_IOCTL
	.unlocked_ioctl = hdsdi_qi_unlocked_ioctl,
#endif
#ifdef HAVE_COMPAT_IOCTL
	.compat_ioctl = sdi_compat_ioctl,
#endif
	.mmap = masterlsdma_mmap,
	.open = hdsdi_qi_open,
	.release = hdsdi_qi_release,
	.fsync = hdsdi_qi_fsync,
	.fasync = NULL
};

MODULE_DEVICE_TABLE(pci, hdsdi_pci_id_table);
static LIST_HEAD(hdsdi_card_list);

static struct class hdsdi_class = {
	.name = hdsdi_driver_name,
	.release = mdev_class_device_release,
	.class_release = NULL
};

/**
 * hdsdi_qi_show_uid - interface attribute read handler
 * @cd: class_device being read
 * @buf: output buffer
 **/
static ssize_t
hdsdi_qi_show_uid (struct class_device *cd,
	char *buf)
{
	struct master_dev *card = to_master_dev(cd);

	return snprintf (buf, PAGE_SIZE, "0x%08X%08X\n",
		master_readl (card, HDSDI_QI_UIDR_HI),
		master_readl (card, HDSDI_QI_UIDR_LO));
}

static CLASS_DEVICE_ATTR(uid,S_IRUGO,
	hdsdi_qi_show_uid,NULL);

/**
 * hdsdi_pci_probe - PCI insertion handler for a SDI Master Q/i
 * @dev: PCI device
 *
 * Handle the insertion of a SDI Master Q/i.
 * Returns a negative error code on failure and 0 on success.
 **/
int __devinit
hdsdi_qi_pci_probe (struct pci_dev *dev,
	const struct pci_device_id *id)
{

	int err;
	unsigned int i;
	//unsigned int cap;
	const char *name;
	struct master_dev *card;

	switch (dev->device) {
	case HDSDI_PCI_DEVICE_ID_LINSYS:
		name = hdsdi_qi_name;
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
			hdsdi_driver_name);
		goto NO_PCI;
	}

	/* Enable bus mastering */
	pci_set_master (dev);

	/* Request the PCI I/O resources */
	if ((err = pci_request_regions (dev, hdsdi_driver_name)) < 0) {
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
	card->core.addr = ioremap_nocache (pci_resource_start (dev, 0),
		pci_resource_len (dev, 0));
	card->version = master_readl (card, HDSDI_QI_FPGAID) & 0xffff;
	card->name = hdsdi_qi_name;
	card->irq_handler = hdsdi_qi_irq_handler;
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
		hdsdi_driver_name, name,
		card->version >> 8, card->version & 0x00ff, card->version);
	pci_set_drvdata (dev, card);
	
	/* Remap bridge address to the DMA controller */
	iounmap (card->bridge_addr);
	/* LS DMA Controller */
	card->bridge_addr = ioremap_nocache (pci_resource_start (dev, 2),
		pci_resource_len (dev, 2));
	/* Reset the FPGA */
	for (i = 0; i < 4; i++) {
		master_writel (card, HDSDI_QI_RCSR(i), HDSDI_QI_RCSR_RST);
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
		&hdsdi_card_list,
		hdsdi_driver_name,
		&hdsdi_class)) < 0) {
		goto NO_DEV;
	}

	/* Add class_device attributes */
	if (card->capabilities & MASTER_CAP_UID) {
		if ((err = class_device_create_file (&card->class_dev,
			&class_device_attr_uid)) < 0) {
			printk (KERN_WARNING
				"%s: Unable to create file 'uid'\n",
				hdsdi_driver_name);
		}
	}

	/* Register receiver interfaces */
	/*
	if (card->version >= 0x0042) { //debug: double check this 
		cap = SDI_CAP_RX_27COUNTER | SDI_CAP_RX_TIMESTAMP ; //debug
	}
	*/
	
	for (i = 0; i < 4; i++) {
		if ((err = sdi_register_iface (card,
			MASTER_DIRECTION_RX,
			&hdsdi_qi_rxfops,
			0,
			4)) < 0) {
			goto NO_IFACE;
		}
	}

	return 0;

NO_IFACE:
	hdsdi_qi_pci_remove (dev);
NO_DEV:
NO_MEM:
NO_PCI:
	return err;
}

/**
 * hdsdi_pci_remove - PCI removal handler for a SDI Master Q/i.
 * @card: Master device
 *
 * Handle the removal of a SDI Master Q/i.
 **/
void
hdsdi_qi_pci_remove (struct pci_dev *dev)
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

		list_for_each (p, &hdsdi_card_list) {
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
 * hdsdi_qi_irq_handler - SDI Master Q/i interrupt service routine
 * @irq: interrupt number
 * @dev_id: pointer to the device data structure
 * @regs: processor context
 **/
static irqreturn_t
IRQ_HANDLER(hdsdi_qi_irq_handler,irq,dev_id,regs)
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
		status = master_readl (card, HDSDI_QI_ICSR(i));
		writel (status, card->core.addr + HDSDI_QI_ICSR(i));
		spin_unlock (&card->irq_lock);
			if (status & HDSDI_QI_ICSR_CDIS) {
				set_bit (SDI_EVENT_RX_CARRIER_ORDER,
					&iface->events);
				interrupting_iface |= (0x1 << i);
			}
			if (status & HDSDI_QI_ICSR_ROIS) {
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
 * hdsdi_qi_init - Initialize a SDI Master Q/i receiver
 * @iface: interface
 **/
static void
hdsdi_qi_init (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg = 0;

	switch (iface->mode) {
	default:
	case HDSDI_QI_RCSR_MODE_RAW:
		reg |= 0;
		break;
	case HDSDI_QI_RCSR_MODE_SYNC:
		reg |= HDSDI_QI_RCSR_MODE_SYNC;
		break;
	case HDSDI_QI_RCSR_MODE_DEINTERLACE:
		reg |= HDSDI_QI_RCSR_MODE_DEINTERLACE;
		break;
	}
/*
	switch (iface->standard) {
	default:
	case HDSDI_QI_STD_260M_1035i:
			reg |= 0;
			printk ("Detected HDSDI_QI_STD_260M_1035i\n");
			break;
	case HDSDI_QI_STD_295M_1080i:
			reg |= HDSDI_QI_STD_295M_1080i;
			printk ("Detected HDSDI_QI_STD_295M_1080i\n");
			break;
	case HDSDI_QI_STD_274M_1080i:
			reg |= HDSDI_QI_STD_274M_1080i;
			printk ("Detected HDSDI_QI_STD_274M_1080i\n");
			break;
	case HDSDI_QI_STD_274M_1080i_25HZ:
			reg |= HDSDI_QI_STD_274M_1080i_25HZ;
			printk ("Detected HDSDI_QI_STD_274M_1080i_25HZ\n");
			break;
	case HDSDI_QI_STD_274M_1080p:
			reg |= HDSDI_QI_STD_274M_1080p;
			printk ("Detected HDSDI_QI_STD_274M_1080p\n");
			break;
	case HDSDI_QI_STD_274M_1080p_25HZ:
			reg |= HDSDI_QI_STD_274M_1080p_25HZ;
			printk ("Detected HDSDI_QI_STD_274M_1080p_25HZ\n");
			break;
	case HDSDI_QI_STD_274M_1080p_24HZ:
			reg |= HDSDI_QI_STD_274M_1080p_24HZ;
			printk ("Detected HDSDI_QI_STD_274M_1080p_24HZ\n");
			break;
	case HDSDI_QI_STD_296M_720p:
			reg |= HDSDI_QI_STD_296M_720p;
			printk ("Detected HDSDI_QI_STD_296M_720p\n");
			break;
	case HDSDI_QI_STD_274M_1080sf:
			reg |= HDSDI_QI_STD_274M_1080sf;
			printk ("Detected HDSDI_QI_STD_274M_1080sf\n");
			break;
	case HDSDI_QI_STD_296M_720p_50HZ:
			reg |= HDSDI_QI_STD_296M_720p_50HZ;
			printk ("Detected HDSDI_QI_STD_296M_720p_50HZ\n");
			break;
	}
*/

	/* There will be no races on RCR
	 * until this code returns, so we don't need to lock it */
	writel (reg | HDSDI_QI_RCSR_RST, card->core.addr + HDSDI_QI_RCSR(channel));
	wmb ();
	writel (reg, card->core.addr + HDSDI_QI_RCSR(channel));
	wmb ();
	writel(HDSDI_QI_RDMATL, card->core.addr + HDSDI_QI_RDMATLR(channel));

	return;
}

/**
 * hdsdi_qi_start - Activate the SDI Master Q/i receiver
 * @iface: interface
 **/
static void
hdsdi_qi_start (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	/* Enable and start DMA */
	writel (lsdma_dma_to_desc_low (lsdma_head_desc_bus_addr (iface->dma)),
		card->bridge_addr + LSDMA_DESC(channel));
	clear_bit (0, &iface->dma_done);
	wmb ();
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION | LSDMA_CH_CSR_ENABLE,
		card->bridge_addr + LSDMA_CSR(channel));
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);

	/* Enable receiver interrupts */
	spin_lock_irq (&card->irq_lock);
	writel (HDSDI_QI_ICSR_CDIE | HDSDI_QI_ICSR_ROIE |
		HDSDI_QI_ICSR_RXDIE, card->core.addr + HDSDI_QI_ICSR(channel));
	spin_unlock_irq (&card->irq_lock);

	/* Enable the receiver */
	spin_lock (&card->reg_lock);
	reg = master_readl (card, HDSDI_QI_RCSR(channel));
	writel (reg | HDSDI_QI_RCSR_RXE | 0x10000000, card->core.addr + HDSDI_QI_RCSR(channel));
	spin_unlock (&card->reg_lock);

	return;
}

/**
 * hdsdi_qi_stop - Deactivate the SDI Master Q/i receiver
 * @iface: interface
 **/
static void
hdsdi_qi_stop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	/* Disable the receiver */
	spin_lock (&card->reg_lock);
	reg = master_readl (card, HDSDI_QI_RCSR(channel));
	writel (reg & ~HDSDI_QI_RCSR_RXE, card->core.addr + HDSDI_QI_RCSR(channel));
	spin_unlock (&card->reg_lock);

	/* Disable receiver interrupts */
	spin_lock_irq (&card->irq_lock);
	writel (HDSDI_QI_ICSR_CDIS | HDSDI_QI_ICSR_ROIS |
		HDSDI_QI_ICSR_RXDIS, card->core.addr + HDSDI_QI_ICSR(channel));

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
 * hdsdi_qi_exit - Clean up the SDI Master Q/i receiver
 * @iface: interface
 **/
static void
hdsdi_qi_exit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);

	/* Reset the receiver.
	 * There will be no races on RCR here,
	 * so we don't need to lock it */
	writel (HDSDI_QI_RCSR_RST, card->core.addr + HDSDI_QI_RCSR(channel));

	return;
}

/**
 * hdsdi_qi_open - SDI Master Q/i receiver open() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
hdsdi_qi_open (struct inode *inode, struct file *filp)
{

	return masterlsdma_open (inode,
		filp,
		hdsdi_qi_init,
		hdsdi_qi_start,
		0,
		LSDMA_MMAP);
}

/**
 * hdsdi_qi_unlocked_ioctl - SDI Master Q/i receiver unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
hdsdi_qi_unlocked_ioctl (struct file *filp,
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
		if (put_user ((master_readl (card, HDSDI_QI_ICSR(channel)) &
			HDSDI_QI_ICSR_RXPASSING) ? 1 : 0, (int *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_RXGETCARRIER:
		/* We don't lock since this should be an atomic read */
		if (put_user ((master_readl (card, HDSDI_QI_ICSR(channel)) &
			HDSDI_QI_ICSR_CD) ? 1 : 0, (int *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_QBUF:
		return masterlsdma_rxqbuf (filp, arg);
	case SDI_IOC_DQBUF:
		return masterlsdma_rxdqbuf (filp, arg);
	case SDI_IOC_RXGET27COUNT:
		if (put_user (master_readl (card, HDSDI_QI_CNT27),
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case SDI_IOC_RXGETTIMESTAMP:
		if (put_user ((master_readl (card, HDSDI_QI_CFAT(channel))), 
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return sdi_rxioctl (iface, cmd, arg);
	}

	return 0;
}

/**
 * hdsdi_qi_ioctl - HD-SDI Master Q/i ioctl() method
 * @inode: inode
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
hdsdi_qi_ioctl (struct inode *inode,
	struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	return hdsdi_qi_unlocked_ioctl (filp, cmd, arg);
}

/**
 * hdsdi_qi_fsync - SDI Master Q/i receiver fsync() method
 * @filp: file to flush
 * @dentry: directory entry associated with the file
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
hdsdi_qi_fsync (struct file *filp,
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
	hdsdi_qi_stop (iface);

	/* Reset the onboard FIFO and driver buffers */
	spin_lock (&card->reg_lock);
	reg = master_readl (card, HDSDI_QI_RCSR(channel));
	writel (reg | HDSDI_QI_RCSR_RST, card->core.addr + HDSDI_QI_RCSR(channel));
	wmb ();
	writel (reg, card->core.addr + HDSDI_QI_RCSR(channel));
	spin_unlock (&card->reg_lock);
	iface->events = 0;
	lsdma_reset (iface->dma);

	/* Start the receiver */
	hdsdi_qi_start (iface);

	up (&iface->buf_sem);
	return 0;
}

/**
 * hdsdi_qi_release - SDI Master Q/i receiver release() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
hdsdi_qi_release (struct inode *inode, struct file *filp)
{
	struct master_iface *iface = filp->private_data;

	return masterlsdma_release (iface, hdsdi_qi_stop, hdsdi_qi_exit);
}

/**
 * hdsdi_qi_init_module - register the module as a PCI driver
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int __init
hdsdi_qi_init_module (void)
{
	printk (KERN_INFO "%s: Linear Systems Ltd. "
		"HD-SDI Master driver from master-%s (%s)\n",
		hdsdi_driver_name, MASTER_DRIVER_VERSION, MASTER_DRIVER_DATE);

	return mdev_init_module (&hdsdi_pci_driver,
		&hdsdi_class,
		hdsdi_driver_name);
}

/**
 * hdsdi_qi_cleanup_module - unregister the module as a PCI driver
 **/
static void __exit
hdsdi_qi_cleanup_module (void)
{
	mdev_cleanup_module (&hdsdi_pci_driver, &hdsdi_class);

	return;
}

module_init (hdsdi_qi_init_module);
module_exit (hdsdi_qi_cleanup_module);
