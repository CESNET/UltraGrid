/* dvbm_lpqo.c
 *
 * Linux driver for Linear Systems Ltd. DVB Master Q/o LP PCIe.
 *
 * Copyright (C) 2006-2010 Linear Systems Ltd.
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
#include <linux/slab.h> /* kzalloc () */
#include <linux/list.h> /* INIT_LIST_HEAD () */
#include <linux/spinlock.h> /* spin_lock_init () */
#include <linux/init.h> /* __devinit */
#include <linux/errno.h> /* error codes */
#include <linux/interrupt.h> /* irqreturn_t */
#include <linux/device.h> /* device_create file */
#include <linux/mutex.h> /* mutex_init () */

#include <asm/uaccess.h> /* put_user () */
#include <asm/bitops.h> /* set_bit () */

#include "asicore.h"
#include "../include/master.h"
#include "mdev.h"
#include "dvbm.h"
#include "mdma.h"
#include "dvbm_lpqo.h"
#include "plx9080.h"
#include "lsdma.h"

static const char dvbm_lpqoe_name[] = DVBM_NAME_LPQOE;
static const char dvbm_lpqoe_minibnc_name[] = DVBM_NAME_LPQOE_MINIBNC;

/* Static function prototypes */
static ssize_t dvbm_lpqo_show_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static ssize_t dvbm_lpqo_store_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	const char *buf,
	size_t count);
static ssize_t dvbm_lpqo_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static irqreturn_t IRQ_HANDLER (dvbm_lpqo_irq_handler, irq, dev_id, regs);
static void dvbm_lpqo_init (struct master_iface *iface);
static void dvbm_lpqo_start (struct master_iface *iface);
static void dvbm_lpqo_stop (struct master_iface *iface);
static void dvbm_lpqo_exit (struct master_iface *iface);
static void dvbm_lpqo_start_tx_dma (struct master_iface *iface);
static long dvbm_lpqo_unlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int FSYNC_HANDLER(dvbm_lpqo_fsync,filp,datasync);

static struct file_operations dvbm_lpqo_fops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.write = asi_write,
	.poll = asi_txpoll,
	.unlocked_ioctl = dvbm_lpqo_unlocked_ioctl,
	.compat_ioctl = dvbm_lpqo_unlocked_ioctl,
	.open = asi_open,
	.release = asi_release,
	.fsync = dvbm_lpqo_fsync,
	.fasync = NULL
};

static struct master_iface_operations dvbm_lpqo_ops = {
	.init = dvbm_lpqo_init,
	.start = dvbm_lpqo_start,
	.stop = dvbm_lpqo_stop,
	.exit = dvbm_lpqo_exit,
	.start_tx_dma = dvbm_lpqo_start_tx_dma
};

/**
 * dvbm_lpqo_show_blackburst_type - interface attribute read handler
 * @dev: device being read
 * @attr: device_attribute
 * @buf: output buffer
 **/
static ssize_t
dvbm_lpqo_show_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	return snprintf (buf, PAGE_SIZE, "%u\n",
		(readl (card->core.addr + DVBM_LPQO_CSR) & DVBM_LPQO_CSR_PLLFS) >> 2);
}

/**
 * dvbm_lpqo_store_blackburst_type - interface attribute write handler
 * @dev: device being written
 * @attr: device attribute
 * @buf: input buffer
 * @count:
 **/
static ssize_t
dvbm_lpqo_store_blackburst_type (struct device *dev,
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
	reg = readl (card->core.addr + DVBM_LPQO_CSR) & ~DVBM_LPQO_CSR_PLLFS;
	writel (reg | (val << 2), card->core.addr + DVBM_LPQO_CSR);
OUT:
	mutex_unlock (&card->users_mutex);
	return retcode;
}

/**
 * dvbm_lpqo_show_uid - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
static ssize_t
dvbm_lpqo_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	return snprintf (buf, PAGE_SIZE, "0x%08X%08X\n",
		readl (card->core.addr + DVBM_LPQO_SSN_HI),
		readl (card->core.addr + DVBM_LPQO_SSN_LO));
}

static DEVICE_ATTR(blackburst_type,S_IRUGO|S_IWUSR,
	dvbm_lpqo_show_blackburst_type,dvbm_lpqo_store_blackburst_type);
static DEVICE_ATTR(uid,S_IRUGO, dvbm_lpqo_show_uid,NULL);

/**
 * dvbm_lpqo_pci_probe - PCI insertion handler for a DVB Master Q/o LP PCIe
 * @pdev: PCI device
 *
 * Handle the insertion of a DVB Master Q/o LP PCIe.
 * Returns a negative error code on failure and 0 on success.
 **/
int __devinit
dvbm_lpqo_pci_probe (struct pci_dev *pdev)
{
	int err;
	unsigned int i, cap;
	struct master_dev *card;

	err = dvbm_pci_probe_generic (pdev);
	if (err < 0) {
		goto NO_PCI;
	}

	/* Initialize the driver_data pointer so that dvbm_lpqo_pci_remove()
	 * doesn't try to free it if an error occurs */
	pci_set_drvdata (pdev, NULL);

	/* Allocate a board info structure */
	if ((card = (struct master_dev *)
		kzalloc (sizeof (*card), GFP_KERNEL)) == NULL) {
		err = -ENOMEM;
		goto NO_MEM;
	}

	/* Initialize the board info structure */
	card->bridge_addr = ioremap_nocache (pci_resource_start (pdev, 2),
		pci_resource_len (pdev, 2));
	card->core.addr = ioremap_nocache (pci_resource_start (pdev, 0),
		pci_resource_len (pdev, 0));
	switch (pdev->device) {
	default:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQOE:
		card->name = dvbm_lpqoe_name;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQOE_MINIBNC:
		card->name = dvbm_lpqoe_minibnc_name;
		break;
	}
	card->version = readl(card->core.addr + DVBM_LPQO_FPGAID) & 0xffff;
	card->id = pdev->device;
	card->irq = pdev->irq;
	card->irq_handler = dvbm_lpqo_irq_handler;
	INIT_LIST_HEAD(&card->iface_list);
	card->capabilities = MASTER_CAP_BLACKBURST | MASTER_CAP_UID;
	/* Lock for ICSR[] */
	spin_lock_init (&card->irq_lock);
	spin_lock_init (&card->reg_lock);
	mutex_init (&card->users_mutex);
	card->parent = &pdev->dev;

	/* Print the firmware version */
	printk (KERN_INFO "%s: %s detected, firmware version %u.%u (0x%04X)\n",
		dvbm_driver_name, card->name,
		card->version >> 8, card->version & 0x00ff, card->version);

	/* Store the pointer to the board info structure
	 * in the PCI info structure */
	pci_set_drvdata (pdev, card);

	/* Reset the FPGA */
	for (i = 0; i < 4; i++) {
		writel (DVBM_LPQO_TCSR_TXRST,
			card->core.addr + DVBM_LPQO_TCSR(i));
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

	/* Register a DVB Master device */
	if ((err = dvbm_register (card)) < 0) {
		goto NO_DEV;
	}

	/* Add device attributes */
	if (card->capabilities & MASTER_CAP_BLACKBURST) {
		if ((err = device_create_file (card->dev,
			&dev_attr_blackburst_type)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'blackburst_type'\n",
				dvbm_driver_name);
		}
	}
	if (card->capabilities & MASTER_CAP_UID) {
		if ((err = device_create_file (card->dev,
			&dev_attr_uid)) < 0) {
			printk (KERN_WARNING
				"%s: Unable to create file 'uid'\n",
				dvbm_driver_name);
		}
	}

	/* Register the transmit interfaces */
	cap = ASI_CAP_TX_SETCLKSRC | ASI_CAP_TX_FIFOUNDERRUN | ASI_CAP_TX_DATA |
		ASI_CAP_TX_MAKE204 |
		ASI_CAP_TX_FINETUNING | ASI_CAP_TX_BYTECOUNTER |
		ASI_CAP_TX_LARGEIB | ASI_CAP_TX_INTERLEAVING |
		ASI_CAP_TX_TIMESTAMPS |
		ASI_CAP_TX_NULLPACKETS |
		ASI_CAP_TX_PTIMESTAMPS;
	if (pdev->device == DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQOE_MINIBNC) {
		if (card->version >= 0x0101) {
			cap |= ASI_CAP_TX_27COUNTER;
		}
	}
	for (i = 0; i < 4; i++) {
		if ((err = asi_register_iface (card,
			&lsdma_dma_ops,
			0,
			MASTER_DIRECTION_TX,
			&dvbm_lpqo_fops,
			&dvbm_lpqo_ops,
			cap,
			4,
			ASI_CTL_TRANSPORT_DVB_ASI)) < 0) {
			goto NO_IFACE;
		}
	}
	return 0;

NO_IFACE:
NO_DEV:
NO_MEM:
	dvbm_lpqo_pci_remove (pdev);
NO_PCI:
	return err;
}

/**
 * dvbm_lpqo_pci_remove - PCI removal handler for DVB Master Q/o LP PCIe
 * @pdev: PCI device
 *
 * Handle the removal of a DVB Master Q/o LP PCIe.
 * This function may be called during PCI probe error handling,
 * so don't mark it as __devexit.
 **/
void dvbm_lpqo_pci_remove (struct pci_dev *pdev)
{
	struct master_dev *card = pci_get_drvdata (pdev);

	if (card) {
		unsigned int i;

		/* Unregister the device and all interfaces */
		dvbm_unregister_all (card);

		for (i = 0; i < 4; i++)	{
			writel(0, card->core.addr + DVBM_LPQO_ICSR(i));
		}
		iounmap (card->core.addr);
		iounmap (card->bridge_addr);
		kfree (card);
	}
	dvbm_pci_remove_generic (pdev);
	return;
}

/**
 * dvbm_lpqo_irq_handler - DVB Master Q/o LP PCIe interrupt service routine
 * @irq: interrupt number
 * @dev_id: pointer to the device data structure
 * @regs: processor context
 **/
static irqreturn_t
IRQ_HANDLER (dvbm_lpqo_irq_handler, irq, dev_id, regs)
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
				mdma_advance(iface->dma);
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
		status = readl (card->core.addr + DVBM_LPQO_ICSR(i));
		writel (status, card->core.addr + DVBM_LPQO_ICSR(i));
		spin_unlock (&card->irq_lock);
		if (status & DVBM_LPQO_ICSR_TUIS) {
			set_bit (ASI_EVENT_TX_FIFO_ORDER,
				&iface->events);
			interrupting_iface |= 0x1 << i;
		}
		if (status & DVBM_LPQO_ICSR_TXDIS) {
			set_bit (ASI_EVENT_TX_DATA_ORDER,
				&iface->events);
			interrupting_iface |= 0x1 << i;
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
 * dvbm_lpqo_init - Initialize the DVB Master Q/o LP PCIe transmitter
 * @iface: interface
 **/
static void
dvbm_lpqo_init (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg = iface->null_packets ? DVBM_LPQO_TCSR_TNP : 0;
	unsigned int clkreg =
		readl(card->core.addr + DVBM_LPQO_CSR) & ~DVBM_LPQO_CSR_EXTCLK;

	switch (iface->timestamps) {
	default:
	case ASI_CTL_TSTAMP_NONE:
		reg |= 0;
		break;
	case ASI_CTL_TSTAMP_APPEND:
		reg |= DVBM_LPQO_TCSR_TTSS;
		break;
	case ASI_CTL_TSTAMP_PREPEND:
		reg |= DVBM_LPQO_TCSR_TPRC;
		break;
	}
	switch (iface->mode) {
	default:
	case ASI_CTL_TX_MODE_188:
		reg |= DVBM_LPQO_TCSR_188;
		break;
	case ASI_CTL_TX_MODE_204:
		reg |= DVBM_LPQO_TCSR_204;
		break;
	case ASI_CTL_TX_MODE_MAKE204:
		reg |= DVBM_LPQO_TCSR_AUTO;
		break;
	}
	switch (iface->clksrc) {
	default:
	case ASI_CTL_TX_CLKSRC_ONBOARD:
		clkreg |= 0;
		break;
	case ASI_CTL_TX_CLKSRC_EXT:
		clkreg |= DVBM_LPQO_CSR_EXTCLK;
		break;
	}

	/* There will be no races on CSR
	 * until this code returns, so we don't need to lock it */
	writel (reg | DVBM_LPQO_TCSR_TXRST,
		card->core.addr + DVBM_LPQO_TCSR(channel));
	/* XXX Set the transmit clock source (shared by all 4 channels)
	 * This is broken because each channel can set a different clock source
	 * for all channels */
	writel (clkreg, card->core.addr + DVBM_LPQO_CSR);
	/* Dummy read to flush PCI posted writes */
	readl (card->core.addr + DVBM_LPQO_FPGAID);
	writel (reg, card->core.addr + DVBM_LPQO_TCSR(channel));
	writel (DVBM_LPQO_TFSL << 16,
		card->core.addr + DVBM_LPQO_TFCR(channel));

	/* Reset byte counter */
	readl (card->core.addr + DVBM_LPQO_TXBCOUNT(channel));

	writel(0, card->core.addr + DVBM_LPQO_IBSTREG(channel));
	writel(0, card->core.addr + DVBM_LPQO_IPSTREG(channel));
	writel(0, card->core.addr + DVBM_LPQO_FTREG(channel));

	return;
}

/**
 * dvbm_lpqo_start - Activate the DVB Master Q/o LP PCIe transmitter
 * @iface: interface
 **/
static void
dvbm_lpqo_start (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	/* Enable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	writel(DVBM_LPQO_ICSR_TUIE | DVBM_LPQO_ICSR_TXDIE,
		card->core.addr + DVBM_LPQO_ICSR(channel));
	spin_unlock_irq(&card->irq_lock);

	/* Enable the transmitter */
	reg = readl(card->core.addr + DVBM_LPQO_TCSR(channel));
	writel(reg | DVBM_LPQO_TCSR_TXE,
		card->core.addr + DVBM_LPQO_TCSR(channel));
	return;
}

/**
 * dvbm_lpqo_stop - Deactivate the DVB Master Q/o LP PCIe transmitter
 * @iface: interface
 **/
static void
dvbm_lpqo_stop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	struct master_dma *dma = iface->dma;
	unsigned int reg;

	lsdma_tx_link_all (dma);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	lsdma_reset (dma);

	if (!iface->null_packets) {
		/* Wait for the onboard FIFOs to empty */
		/* Atomic read of ICSR, so we don't need to lock */
		wait_event (iface->queue,
			!(readl (card->core.addr + DVBM_LPQO_ICSR(channel)) &
			DVBM_LPQO_ICSR_TXD));
	}

	/* Disable the Transmitter */
	reg = readl(card->core.addr + DVBM_LPQO_TCSR(channel));
	writel (reg & ~DVBM_LPQO_TCSR_TXE,
		card->core.addr + DVBM_LPQO_TCSR(channel));

	/* Disable Transmitter Interrupts */
	spin_lock_irq (&card->irq_lock);
	writel (DVBM_LPQO_ICSR_TUIS | DVBM_LPQO_ICSR_TXDIS,
		card->core.addr + DVBM_LPQO_ICSR(channel));
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);
	spin_unlock_irq (&card->irq_lock);

	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE,
		card->bridge_addr + LSDMA_CSR(channel));
	return;
}

/**
 * dvbm_lpqo_exit - Clean up the DVB Master Q/o LP PCIe transmitter
 * @iface: interface
 **/
static void
dvbm_lpqo_exit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);

	/* Reset the transmitter.
	 * There will be no races on CSR here,
	 * so we don't need to lock it */
	writel (DVBM_LPQO_TCSR_TXRST,
		card->core.addr + DVBM_LPQO_TCSR(channel));

	return;
}

/**
 * dvbm_lpqo_start_tx_dma - start transmit DMA
 * @iface: interface
 **/
static void
dvbm_lpqo_start_tx_dma (struct master_iface *iface)
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
 * dvbm_lpqo_unlocked_ioctl - DVB Master Q/o LP PCIe unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
dvbm_lpqo_unlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	struct asi_txstuffing stuffing;
	const unsigned int channel = mdev_index (card, &iface->list);

	switch (cmd) {
	case ASI_IOC_TXSETSTUFFING:
		if (iface->transport != ASI_CTL_TRANSPORT_DVB_ASI) {
				return -ENOTTY;
		}
		if (copy_from_user (&stuffing,
			(struct asi_txstuffing __user *)arg,
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
			card->core.addr + DVBM_LPQO_IBSTREG(channel));
		writel (stuffing.ip,
			card->core.addr + DVBM_LPQO_IPSTREG(channel));
		writel ((stuffing.il_big << DVBM_LPQO_FTR_ILBIG_SHIFT) |
			(stuffing.big_ip << DVBM_LPQO_FTR_BIGIP_SHIFT) |
			(stuffing.il_normal << DVBM_LPQO_FTR_ILNORMAL_SHIFT) |
			stuffing.normal_ip,
			card->core.addr + DVBM_LPQO_FTREG(channel));
		spin_unlock (&card->reg_lock);
		break;
	case ASI_IOC_TXGETBYTECOUNT:
		if (!(iface->capabilities & ASI_CAP_TX_BYTECOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr + DVBM_LPQO_TXBCOUNT(channel)),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_TXGETTXD:
		/* Atomic read of ICSR, so we don't need to lock */
		if (put_user ((readl (card->core.addr + DVBM_LPQO_ICSR(channel))
			& DVBM_LPQO_ICSR_TXD) ? 1 : 0,
			(int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_TXGET27COUNT:
		if (!(iface->capabilities & ASI_CAP_TX_27COUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr + DVBM_LPQO_27COUNTR),
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
 * dvbm_lpqo_fsync - DVB Master Q/o LP PCIe fsync() method
 * @filp: file to flush
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
FSYNC_HANDLER(dvbm_lpqo_fsync,filp,datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	struct master_dma *dma = iface->dma;
	const unsigned int channel = mdev_index (card, &iface->list);

	mutex_lock (&iface->buf_mutex);

	lsdma_tx_link_all(dma);
	wait_event(iface->queue, test_bit(0, &iface->dma_done));
	lsdma_reset(dma);

	if (!iface->null_packets) {
		/* Wait for the onboard FIFOs to empty */
		/* Atomic read of ICSR, so we don't need to lock */
		wait_event(iface->queue,
			!(readl(card->core.addr + DVBM_LPQO_ICSR(channel))
			& DVBM_LPQO_ICSR_TXD));
	}

	mutex_unlock (&iface->buf_mutex);

	return 0;
}

