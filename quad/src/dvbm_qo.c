/* dvbm_qo.c
 *
 * Linux driver for Linear Systems Ltd. DVB Master Q/o.
 *
 * Copyright (C) 2006-2008 Linear Systems Ltd.
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
#include <linux/delay.h>

#include "asicore.h"
#include "../include/master.h"
#include "mdev.h"
#include "dvbm.h"
#include "miface.h"
#include "dvbm_qo.h"
#include "plx9080.h"
#include "masterplx.h"
#include "lsdma.h"
#include "masterlsdma.h"

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,8))
static inline int
nonseekable_open (struct inode *inode, struct file *filp)
{
	return 0;
}
#endif

static const char dvbm_qo_name[] = DVBM_NAME_QO;
static const char dvbm_qoe_name[] = DVBM_NAME_QOE;
static const char dvbm_lpqoe_name[] = DVBM_NAME_LPQOE;

/* Static function prototypes */
static ssize_t dvbm_qo_show_uid (struct class_device *cd, char *buf);
static irqreturn_t IRQ_HANDLER (dvbm_qo_irq_handler, irq, dev_id, regs);
static void dvbm_qo_init (struct master_iface *iface);
static void dvbm_qo_start (struct master_iface *iface);
static void dvbm_qo_stop (struct master_iface *iface);
static void dvbm_qo_exit (struct master_iface *iface);
static int dvbm_qo_open (struct inode *inode, struct file *filp);
static long dvbm_qo_unlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int dvbm_qo_ioctl (struct inode *inode,
	struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int dvbm_qo_fsync (struct file *filp,
	struct dentry *dentry,
	int datasync);
static int dvbm_qo_release (struct inode *inode, struct file *filp);

struct file_operations dvbm_qo_fops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.write = masterlsdma_write,
	.poll = masterlsdma_txpoll,
	.ioctl = dvbm_qo_ioctl,
#ifdef HAVE_UNLOCKED_IOCTL
	.unlocked_ioctl = dvbm_qo_unlocked_ioctl,
#endif
#ifdef HAVE_COMPAT_IOCTL
	.compat_ioctl = asi_compat_ioctl,
#endif
	.open = dvbm_qo_open,
	.release = dvbm_qo_release,
	.fsync = dvbm_qo_fsync,
	.fasync = NULL
};

/**
 * dvbm_qlf_show_uid - interface attribute read handler
 * @cd: class_device being read
 * @buf: output buffer
 **/
static ssize_t
dvbm_qo_show_uid (struct class_device *cd, char *buf)
{
	struct master_dev *card = to_master_dev(cd);

	return snprintf (buf, PAGE_SIZE, "0x%08X%08X\n",
		readl (card->core.addr + DVBM_QO_SSN_HI),
		readl (card->core.addr + DVBM_QO_SSN_LO));
}

static CLASS_DEVICE_ATTR(uid,S_IRUGO, dvbm_qo_show_uid,NULL);

/**
 * dvbm_qo_pci_probe - PCI insertion handler for a DVB Master Q/O
 * @dev: PCI device
 * @id: PCI ID
 *
 * Handle the insertion of a DVB Master Q/O.
 * Returns a negative error code on failure and 0 on success.
 **/
int __devinit
dvbm_qo_pci_probe (struct pci_dev *dev)
{
	int err;
	unsigned int i, cap;
	unsigned int transport;
	struct master_dev *card;
	const char *name;

	switch (dev->device) {
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQO:
		name = dvbm_qo_name;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQOE:
		name = dvbm_qoe_name;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQOE:
		name = dvbm_lpqoe_name;
		break;
	default:
		name = "";
		break;
	}

	/* Allocate a board info structure */
	if ((card = (struct master_dev *)
		kmalloc (sizeof (*card), GFP_KERNEL)) == NULL) {
		err = -ENOMEM;
		goto NO_MEM;
	}

	/* Initialize the board info structure */
	memset (card, 0, sizeof (*card));
	switch (dev->device) {
	default:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQO:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQOE:
		card->bridge_addr = ioremap_nocache (pci_resource_start (dev, 0),
			pci_resource_len (dev, 0));
		card->core.addr = ioremap_nocache (pci_resource_start (dev, 2),
			pci_resource_len (dev, 2));
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQOE:
		card->bridge_addr = ioremap_nocache (pci_resource_start (dev, 2),
			pci_resource_len (dev, 2));
		card->core.addr = ioremap_nocache (pci_resource_start (dev, 0),
			pci_resource_len (dev, 0));
		break;
	}
	card->version = readl(card->core.addr + DVBM_QO_FPGAID) & 0xffff;
	card->name = dvbm_qo_name;

	card->irq_handler = dvbm_qo_irq_handler;
	INIT_LIST_HEAD(&card->iface_list);

	switch (dev->device) {
	default:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQO:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQOE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQOE:
		card->capabilities= MASTER_CAP_UID;
		break;
	}

	/* Lock for ICSR[] */
	spin_lock_init (&card->irq_lock);

	spin_lock_init (&card->reg_lock);
	sema_init (&card->users_sem, 1);
	card->pdev = dev;

	/* Print the firmware version */
	printk (KERN_INFO "%s: %s detected, firmware version %u.%u (0x%04X)\n",
		dvbm_driver_name, card->name,
		card->version >> 8, card->version & 0x00ff, card->version);

	/* Store the pointer to the board info structure
	 * in the PCI info structure */
	pci_set_drvdata (dev, card);

	switch (dev->device) {
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQO:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQOE:
		/* Reset PCI 9056 */
		masterplx_reset_bridge(card);

		/* Setup the PCI 9056 */
		writel(PLX_INTCSR_PCIINT_ENABLE |
			PLX_INTCSR_PCILOCINT_ENABLE,
			card->bridge_addr + PLX_INTCSR);

		/* Dummy read to flush PCI posted wires */
		readl(card->bridge_addr + PLX_INTCSR);

		/* Remap bridge address to the DMA controller */
		iounmap (card->bridge_addr);
		/* LS DMA Controller */
		card->bridge_addr = ioremap_nocache (pci_resource_start (dev, 3),
			pci_resource_len (dev, 3));
		break;
	default:
		break;
	}

	/* Reset the FPGA */
	for (i = 0; i < 4; i++) {
		writel (DVBM_QO_TCSR_TXRST, card->core.addr + DVBM_QO_TCSR(i));
	}

	/* Setup the LS DMA controller */
	writel (LSDMA_INTMSK_CH(0) | LSDMA_INTMSK_CH(1) | LSDMA_INTMSK_CH(2)
		| LSDMA_INTMSK_CH(3),
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
		&dvbm_card_list,
		dvbm_driver_name,
		&dvbm_class)) < 0) {
		goto NO_DEV;
	}

	/* Add class_device attributes */
	if (card->capabilities & MASTER_CAP_UID) {
		if ((err = class_device_create_file (&card->class_dev,
			&class_device_attr_uid)) < 0) {
			printk (KERN_WARNING
				"%s: Unable to create file 'uid'\n",
				dvbm_driver_name);
		}
	}

	/* Setup device capabilities */
	cap = ASI_CAP_TX_SETCLKSRC | ASI_CAP_TX_FIFOUNDERRUN | ASI_CAP_TX_DATA;
	switch (dev->device) {
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQO:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQOE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQOE:
		cap |= ASI_CAP_TX_MAKE204 |
		ASI_CAP_TX_FINETUNING | ASI_CAP_TX_BYTECOUNTER |
		ASI_CAP_TX_LARGEIB | ASI_CAP_TX_INTERLEAVING |
		ASI_CAP_TX_TIMESTAMPS |
		ASI_CAP_TX_NULLPACKETS |
		ASI_CAP_TX_PTIMESTAMPS;
		transport = ASI_CTL_TRANSPORT_DVB_ASI;
		break;
	default:
		transport = 0xff;
		break;
	}

	/* Register the transmit interfaces */
	for (i = 0; i < 4; i++) {
		if ((err = asi_register_iface (card,
			MASTER_DIRECTION_TX,
			&dvbm_qo_fops,
			cap,
			4,
			transport)) < 0) {
			goto NO_IFACE;
		}
 	}
	return 0;

NO_IFACE:
	dvbm_pci_remove (dev);
NO_DEV:
NO_MEM:
	return err;
}

/**
 * dvbm_qo_pci_remove - PCI removal handler for DVB Master Q/o
 * @card: Master device
 *
 * Handle the removal of a DVB Master Q/o.
 **/
void dvbm_qo_pci_remove (struct master_dev *card)
{
	int i;

	for (i = 0; i < 4; i++)	{
		if (card->capabilities) {
		writel(0, card->core.addr + DVBM_QO_ICSR(i));
		}
	}

	iounmap (card->core.addr);
	return;
}

/**
 * dvbm_qo_irq_handler - DVB Master Q/o interrupt service routine
 * @irq: interrupt number
 * @dev_id: pointer to the device data structure
 * @regs: processor context
 **/
static irqreturn_t
IRQ_HANDLER (dvbm_qo_irq_handler, irq, dev_id, regs)
{
	struct master_dev *card = dev_id;
	struct list_head *p = &card->iface_list;
	struct master_iface *iface;
	unsigned int status, interrupting_iface = 0, i;
	unsigned int dmaintsrc = readl (card->bridge_addr + LSDMA_INTSRC);

	for (i = 0; i < 4; i++)	{
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
				set_bit(ASI_EVENT_TX_BUFFER_ORDER,
					&iface->events);
				set_bit(0, &iface->dma_done);
			}

			/* Flag DMA abort */
			if (status & LSDMA_CH_CSR_INTSRCSTOP) {
				set_bit(0, &iface->dma_done);
			}

			interrupting_iface |= (0x1 << i);
		}

		/* Check and clear the source of the interrupts */
		spin_lock (&card->irq_lock);
		status = readl (card->core.addr + DVBM_QO_ICSR(i));
		writel (status, card->core.addr + DVBM_QO_ICSR(i));

		if (status & DVBM_QO_ICSR_TUIS) {
			set_bit (ASI_EVENT_TX_FIFO_ORDER,
				&iface->events);
			interrupting_iface |= 0x1 << i;
		}
		if (status & DVBM_QO_ICSR_TXDIS) {
			set_bit (ASI_EVENT_TX_DATA_ORDER,
				&iface->events);
			interrupting_iface |= 0x1 << i;
		}
		spin_unlock (&card->irq_lock);

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
 * dvbm_qo_init - Initialize the DVB Master Q/o Transmitter
 * @iface: interface
 **/
static void
dvbm_qo_init (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg = iface->null_packets ? DVBM_QO_TCSR_TNP : 0;

	switch (iface->timestamps) {
	default:
	case ASI_CTL_TSTAMP_NONE:
		reg |= 0;
		break;
	case ASI_CTL_TSTAMP_APPEND:
		reg |= DVBM_QO_TCSR_TTSS;
		break;
	case ASI_CTL_TSTAMP_PREPEND:
		reg |= DVBM_QO_TCSR_TPRC;
		break;
	}

	switch (iface->mode) {
	default:
	case ASI_CTL_TX_MODE_188:
		reg |= DVBM_QO_TCSR_188;
		break;
	case ASI_CTL_TX_MODE_204:
		reg |= DVBM_QO_TCSR_204;
		break;
	case ASI_CTL_TX_MODE_MAKE204:
		reg |= DVBM_QO_TCSR_AUTO;
		break;
	}

	switch (iface->clksrc) {
	default:
	case ASI_CTL_TX_CLKSRC_ONBOARD:
		reg |= 0;
		break;
	case ASI_CTL_TX_CLKSRC_EXT:
		reg |= DVBM_QO_TCSR_EXTCLK;
		break;
	}

	/* There will be no races on CSR
	 * until this code returns, so we don't need to lock it */
	writel (reg | DVBM_QO_TCSR_TXRST,
		card->core.addr + DVBM_QO_TCSR(channel));
	wmb();
	writel (reg, card->core.addr + DVBM_QO_TCSR(channel));
	readl (card->core.addr + DVBM_QO_FPGAID);
	writel (DVBM_QO_TFSL << 16, card->core.addr + DVBM_QO_TFCR(channel));

	/* Reset byte counter */
	readl (card->core.addr + DVBM_QO_TXBCOUNT(channel));

	writel(0, card->core.addr + DVBM_QO_IBSTREG(channel));
	writel(0, card->core.addr + DVBM_QO_IPSTREG(channel));
	writel(0, card->core.addr + DVBM_QO_FTREG(channel));

	return;
}

/**
 * dvbm_qo_start - Activate the DVB Master Q/o Transmitter
 * @iface: interface
 **/
static void
dvbm_qo_start (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	/* Enable DMA */
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE,
		card->bridge_addr + LSDMA_CSR(channel));

	/* Enable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = readl (card->core.addr + DVBM_QO_ICSR(channel));
	reg |= DVBM_QO_ICSR_TUIE | DVBM_QO_ICSR_TXDIE;
	writel(reg, card->core.addr + DVBM_QO_ICSR(channel));
	spin_unlock_irq(&card->irq_lock);

	/* Enable the transmitter */
	spin_lock(&card->reg_lock);
	reg = readl(card->core.addr + DVBM_QO_TCSR(channel));
	writel(reg | DVBM_QO_TCSR_TXE,
		card->core.addr + DVBM_QO_TCSR(channel));
	spin_unlock(&card->reg_lock);
	return;
}

/**
 * dvbm_qo_stop - Deactivate the DVB Master Q/O transmitter
 * @iface: interface
 **/
static void
dvbm_qo_stop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	struct lsdma_dma *dma = iface->dma;
	unsigned int reg;

	lsdma_tx_link_all (dma);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	lsdma_reset (dma);

	if (!iface->null_packets) {
		/* Wait for the onboard FIFOs to empty */
		/* Atomic read of ICSR, so we don't need to lock */
		wait_event (iface->queue,
			!(readl (card->core.addr + DVBM_QO_ICSR(channel)) &
			DVBM_QO_ICSR_TXD));
	}

	/* Disable the Transmitter */
	spin_lock (&card->reg_lock);
	reg = readl(card->core.addr + DVBM_QO_TCSR(channel));
	writel (reg & ~DVBM_QO_TCSR_TXE,
		card->core.addr + DVBM_QO_TCSR(channel));
	spin_unlock (&card->reg_lock);

	/*Disable Transmitter Interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = readl (card->core.addr + DVBM_QO_ICSR(channel));
	reg |= DVBM_QO_ICSR_TUIS | DVBM_QO_ICSR_TXDIS;
	writel (reg, card->core.addr + DVBM_QO_ICSR(channel));

	writel ((LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION | LSDMA_CH_CSR_STOP),
		card->bridge_addr + LSDMA_CSR(channel));
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);
	spin_unlock_irq (&card->irq_lock);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));

	writel ((LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE) &
		~LSDMA_CH_CSR_ENABLE,
		card->bridge_addr + LSDMA_CSR(channel));
	udelay (10L);
	return;
}

/**
 * dvbm_qo_exit - Clean up the DVB Master Q/o transmitter
 * @iface: interface
 **/
static void
dvbm_qo_exit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);

	/* Reset the transmitter.
	 * There will be no races on CSR here,
	 * so we don't need to lock it */
	writel (DVBM_QO_TCSR_TXRST, card->core.addr + DVBM_QO_TCSR(channel));

	return;
}

/**
 * dvbm_qo_open - DVB Master Q/i open() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
dvbm_qo_open (struct inode *inode, struct file *filp)
{
	return masterlsdma_open (inode,
		filp,
		dvbm_qo_init,
		dvbm_qo_start,
		0,
		0);
}

/**
 * dvbm_qo_unlocked_ioctl - DVB Master Q/o unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
dvbm_qo_unlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	struct asi_txstuffing stuffing;
	const unsigned int channel = mdev_index (card, &iface->list);

	switch (cmd) {
	case ASI_IOC_TXGETBUFLEVEL:
		if (put_user (lsdma_tx_buflevel (iface->dma),
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_TXSETSTUFFING:
		if (iface->transport != ASI_CTL_TRANSPORT_DVB_ASI) {
				return -ENOTTY;
			}
			if (copy_from_user (&stuffing,
				(struct asi_txstuffing *)arg,
				sizeof (stuffing))) {
				return -EFAULT;
			}
			if ((stuffing.ib > 0x00ff) ||
				(stuffing.ip > 0xffffff) ||
				(stuffing.normal_ip > 0xff) ||
				(stuffing.big_ip > 0xff) ||
				((stuffing.il_normal + stuffing.il_big) > 0xf) ||
				(stuffing.il_normal > stuffing.normal_ip) ||
				(stuffing.il_big > stuffing.big_ip)) {
				return -EINVAL;
			}

			spin_lock (&card->reg_lock);
			writel (stuffing.ib,
				card->core.addr + DVBM_QO_IBSTREG(channel));
			writel (stuffing.ip,
				card->core.addr + DVBM_QO_IPSTREG(channel));
			writel ((stuffing.il_big << DVBM_QO_FTR_ILBIG_SHIFT) |
				(stuffing.big_ip << DVBM_QO_FTR_BIGIP_SHIFT) |
				(stuffing.il_normal << DVBM_QO_FTR_ILNORMAL_SHIFT) |
				stuffing.normal_ip,
				card->core.addr + DVBM_QO_FTREG(channel));
			spin_unlock (&card->reg_lock);
			break;
		case ASI_IOC_TXGETBYTECOUNT:
			if (!(iface->capabilities & ASI_CAP_TX_BYTECOUNTER)) {
				return -ENOTTY;
			}
			if (put_user (readl (card->core.addr + DVBM_QO_TXBCOUNT(channel)),
			(unsigned int *)arg)) {
			return -EFAULT;
			}
			break;
		case ASI_IOC_TXGETTXD:
			/* Atomic read of ICSR, so we don't need to lock */
			if (put_user ((readl (card->core.addr + DVBM_QO_ICSR(channel))
				& DVBM_QO_ICSR_TXD) ? 1 : 0, (int *)arg)) {
				return -EFAULT;
			}
			break;
		default:
			return asi_txioctl (iface, cmd, arg);
		}

	return 0;
}

/**
 * dvbm_qo_ioctl - DVB Master Q/o ioctl() method
 * @inode: inode
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
dvbm_qo_ioctl (struct inode *inode,
	struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	return dvbm_qo_unlocked_ioctl (filp, cmd, arg);
}

/**
 * dvbm_qo_fsync - DVB Master Q/o fsync() method
 * @filp: file to flush
 * @dentry: directory entry associated with the file
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
dvbm_qo_fsync (struct file *filp,
	struct dentry *dentry,
	int datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	struct lsdma_dma *dma = iface->dma;
	const unsigned int channel = mdev_index (card, &iface->list);

	if (down_interruptible (&iface->buf_sem)) {
		return -ERESTARTSYS;
	}

	lsdma_tx_link_all(dma);
	wait_event(iface->queue, test_bit(0, &iface->dma_done));
	lsdma_reset(dma);

	if (!iface->null_packets)
	{
		struct master_dev *card = iface->card;
		/* Wait for the onboard FIFOs to empty */
		/* Atomic read of ICSR, so we don't need to lock */
		wait_event(iface->queue,
			!(readl(card->core.addr + DVBM_QO_ICSR(channel))
			& DVBM_QO_ICSR_TXD));
	}

	up(&iface->buf_sem);

	return 0;
}

/**
 * dvbm_qo_release - DVB Master Q/o release() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
dvbm_qo_release (struct inode *inode, struct file *filp)
{
	struct master_iface *iface = filp->private_data;

	return masterlsdma_release(iface, dvbm_qo_stop, dvbm_qo_exit);
}

