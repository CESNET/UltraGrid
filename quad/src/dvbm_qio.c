/* dvbm_qio.c
 *
 * Generic Linux driver functions for Linear Systems Ltd. DVB Master Q/io.
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
#include <linux/slab.h> /* kzalloc () */
#include <linux/list.h> /* INIT_LIST_HEAD () */
#include <linux/spinlock.h> /* spin_lock_init () */
#include <linux/init.h> /* __devinit */
#include <linux/errno.h> /* error codes */
#include <linux/interrupt.h> /* irqreturn_t */
#include <linux/mutex.h> /* mutex_lock () */

#include <asm/uaccess.h> /* put_user () */

#include "asicore.h"
#include "../include/master.h"
#include "mdev.h"
#include "dvbm.h"
#include "mdma.h"
#include "dvbm_qio.h"
#include "plx9080.h"
#include "lsdma.h"

/* static function prototypes */
static void dvbm_qio_txinit (struct master_iface *iface);
static void dvbm_qio_txstart (struct master_iface *iface);
static void dvbm_qio_txstop (struct master_iface *iface);
static void dvbm_qio_txexit (struct master_iface *iface);
static void dvbm_qio_start_tx_dma (struct master_iface *iface);
static long dvbm_qio_txunlocked_ioctl (struct file *filp,
		unsigned int cmd,
		unsigned long arg);
static int FSYNC_HANDLER(dvbm_qio_txfsync,filp,datasync);
static void dvbm_qio_rxinit (struct master_iface *iface);
static void dvbm_qio_rxstart (struct master_iface *iface);
static void dvbm_qio_rxstop (struct master_iface *iface);
static void dvbm_qio_rxexit (struct master_iface *iface);
static long dvbm_qio_rxunlocked_ioctl (struct file *filp,
		unsigned int cmd,
		unsigned long arg);
static int FSYNC_HANDLER(dvbm_qio_rxfsync,filp,datasync);

struct file_operations dvbm_qio_txfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.write = asi_write,
	.poll = asi_txpoll,
	.unlocked_ioctl = dvbm_qio_txunlocked_ioctl,
	.compat_ioctl = dvbm_qio_txunlocked_ioctl,
	.open = asi_open,
	.release = asi_release,
	.fsync = dvbm_qio_txfsync,
	.fasync = NULL
};

struct file_operations dvbm_qio_rxfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.read = asi_read,
	.poll = asi_rxpoll,
	.unlocked_ioctl = dvbm_qio_rxunlocked_ioctl,
	.compat_ioctl = dvbm_qio_rxunlocked_ioctl,
	.open = asi_open,
	.release = asi_release,
	.fsync = dvbm_qio_rxfsync,
	.fasync = NULL
};

struct master_iface_operations dvbm_qio_txops = {
	.init = dvbm_qio_txinit,
	.start = dvbm_qio_txstart,
	.stop = dvbm_qio_txstop,
	.exit = dvbm_qio_txexit,
	.start_tx_dma = dvbm_qio_start_tx_dma
};

struct master_iface_operations dvbm_qio_rxops = {
	.init = dvbm_qio_rxinit,
	.start = dvbm_qio_rxstart,
	.stop = dvbm_qio_rxstop,
	.exit = dvbm_qio_rxexit
};

/**
 * dvbm_qio_show_blackburst_type - interface attribute read handler
 * @dev: device being read
 * @attr: device_attribute
 * @buf: output buffer
 **/
ssize_t
dvbm_qio_show_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	return snprintf (buf, PAGE_SIZE, "%u\n",
		(readl (card->core.addr + DVBM_QIO_CSR) & DVBM_QIO_CSR_PLLFS) >> 2);
}

/**
 * dvbm_qio_store_blackburst_type - interface attribute write handler
 * @dev: device being written
 * @attr: device attribute
 * @buf: input buffer
 * @count:
 **/
ssize_t
dvbm_qio_store_blackburst_type (struct device *dev,
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
	unsigned int tx_users = 0;

	if ((endp == buf) || (val > max)) {
		return -EINVAL;
	}
	mutex_lock (&card->users_mutex);
	list_for_each (p, &card->iface_list) {
		iface = list_entry (p, struct master_iface, list);
		if (iface->direction == MASTER_DIRECTION_RX) {
			break;
		}
		tx_users += iface->users;
	}
	if (tx_users) {
		retcode = -EBUSY;
		goto OUT;
	}
	reg = readl (card->core.addr + DVBM_QIO_CSR) & ~DVBM_QIO_CSR_PLLFS;
	writel (reg | (val << 2), card->core.addr + DVBM_QIO_CSR);
OUT:
	mutex_unlock (&card->users_mutex);
	return retcode;
}

/**
 * dvbm_qio_show_uid - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
ssize_t
dvbm_qio_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	return snprintf (buf, PAGE_SIZE, "0x%08X%08X\n",
		readl (card->core.addr + DVBM_QIO_SSN_HI),
		readl (card->core.addr + DVBM_QIO_SSN_LO));
}

/**
 * dvbm_qio_pci_remove - PCI removal handler for a DVB Master Q/io
 * @pdev: PCI device
 *
 * Handle the removal of a DVB Master Q/io.
 * This function may be called during PCI probe error handling,
 * so don't mark it as __devexit.
 **/
void
dvbm_qio_pci_remove (struct pci_dev *pdev)
{
	struct master_dev *card = pci_get_drvdata (pdev);

	if (card) {
		unsigned int i;

		/* Unregister the device and all interfaces */
		dvbm_unregister_all (card);

		for (i = 0; i < 4; i++) {
			writel (0, card->core.addr + DVBM_QIO_ICSR(i));
		}
		iounmap (card->core.addr);
		iounmap (card->bridge_addr);
		kfree (card);
	}
	dvbm_pci_remove_generic (pdev);
	return;
}

/**
 * dvbm_qio_txinit - Initialize the DVB Master Q/io transmitter
 * @iface: interface
 **/
static void
dvbm_qio_txinit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg = iface->null_packets ? DVBM_QIO_TCSR_TNP : 0;

	switch (iface->timestamps) {
	default:
	case ASI_CTL_TSTAMP_NONE:
		reg |= 0;
		break;
	case ASI_CTL_TSTAMP_APPEND:
		reg |= DVBM_QIO_TCSR_TTSS;
		break;
	case ASI_CTL_TSTAMP_PREPEND:
		reg |= DVBM_QIO_TCSR_TPRC;
		break;
	}
	switch (iface->mode) {
	default:
	case ASI_CTL_TX_MODE_188:
		reg |= DVBM_QIO_TCSR_188;
		break;
	case ASI_CTL_TX_MODE_204:
		reg |= DVBM_QIO_TCSR_204;
		break;
	case ASI_CTL_TX_MODE_MAKE204:
		reg |= DVBM_QIO_TCSR_MAKE204;
		break;
	}
	switch (iface->clksrc) {
	default:
	case ASI_CTL_TX_CLKSRC_ONBOARD:
		reg |= 0;
		break;
	case ASI_CTL_TX_CLKSRC_EXT:
		reg |= DVBM_QIO_TCSR_EXTCLK;
		break;
	}

	/* There will be no races on IBSTR, IPSTR, FTR, and TCSR
	 * until this code returns, so we don't need to lock them */
	writel (reg | DVBM_QIO_TCSR_TXRST,
		card->core.addr + DVBM_QIO_TCSR(channel));
	writel (reg, card->core.addr + DVBM_QIO_TCSR(channel));
	writel ((DVBM_QIO_TFSL << 16) | DVBM_QIO_TDMATL,
		card->core.addr + DVBM_QIO_TFCR(channel));
	writel (0, card->core.addr + DVBM_QIO_IBSTREG(channel));
	writel (0, card->core.addr + DVBM_QIO_IPSTREG(channel));
	writel (0, card->core.addr + DVBM_QIO_FTREG(channel));
	/* Reset byte counter */
	readl (card->core.addr + DVBM_QIO_TXBCOUNT (channel));
	return;
}

/**
 * dvbm_qio_txstart - Activate the DVB Master Q/io transmitter
 * @iface: interface
 **/
static void
dvbm_qio_txstart (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;
	const unsigned int channel = mdev_index (card, &iface->list);

	/* Enable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	writel (DVBM_QIO_ICSR_TXUIE | DVBM_QIO_ICSR_TXDIE,
		card->core.addr + DVBM_QIO_ICSR(channel));
	spin_unlock_irq(&card->irq_lock);

	/* Enable the transmitter */
	reg = readl(card->core.addr + DVBM_QIO_TCSR(channel));
	writel(reg | DVBM_QIO_TCSR_TXE,
		card->core.addr + DVBM_QIO_TCSR(channel));

	return;
}

/**
 * dvbm_qio_txstop - Deactivate the DVB Master Q/io transmitter
 * @iface: interface
 **/
static void
dvbm_qio_txstop (struct master_iface *iface)
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
			!(readl (card->core.addr + DVBM_QIO_ICSR(channel)) &
			DVBM_QIO_ICSR_TXD));
	}

	/* Disable the transmitter.
	 * There will be no races on TCSR here,
	 * so we don't need to lock it */
	reg = readl(card->core.addr + DVBM_QIO_TCSR(channel));
	writel(reg & ~DVBM_QIO_TCSR_TXE,
		card->core.addr + DVBM_QIO_TCSR(channel));

	/* Disable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	writel (DVBM_QIO_ICSR_TXUIS | DVBM_QIO_ICSR_TXDIS,
		card->core.addr + DVBM_QIO_ICSR(channel));
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);
	spin_unlock_irq (&card->irq_lock);

	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE,
		card->bridge_addr + LSDMA_CSR(channel));

	return;
}

/**
 * dvbm_qio_txexit - Clean up the DVB Master Q/io transmitter
 * @iface: interface
 **/
static void
dvbm_qio_txexit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);

	/* Reset the transmitter */
	writel (DVBM_QIO_TCSR_TXRST,
		card->core.addr + DVBM_QIO_TCSR(channel));

	return;
}

/**
 * dvbm_qio_start_tx_dma - start transmit DMA
 * @iface: interface
 **/
static void
dvbm_qio_start_tx_dma (struct master_iface *iface)
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
 * dvbm_qio_txunlocked_ioctl - DVB Master Q/io transmitter unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
dvbm_qio_txunlocked_ioctl (struct file *filp,
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
		writel (stuffing.ib, card->core.addr + DVBM_QIO_IBSTREG(channel));
		writel (stuffing.ip, card->core.addr + DVBM_QIO_IPSTREG(channel));
		writel ((stuffing.il_big << DVBM_QIO_FTR_ILBIG_SHIFT) |
			(stuffing.big_ip << DVBM_QIO_FTR_BIGIP_SHIFT) |
			(stuffing.il_normal << DVBM_QIO_FTR_ILNORMAL_SHIFT) |
			stuffing.normal_ip, card->core.addr + DVBM_QIO_FTREG(channel));
		spin_unlock (&card->reg_lock);
		break;
	case ASI_IOC_TXGETBYTECOUNT:
		if (!(iface->capabilities & ASI_CAP_TX_BYTECOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr + DVBM_QIO_TXBCOUNT(channel)),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_TXGETTXD:
		/* Atomic read of ICSR, so we don't need to lock */
		if (put_user ((readl (card->core.addr + DVBM_QIO_ICSR(channel)) &
			DVBM_QIO_ICSR_TXD) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return asi_txioctl (filp, cmd, arg);
	}
	return 0;
}

/**
 * dvbm_qio_txfsync - DVB Master Q/io transmitter fsync() method
 * @filp: file to flush
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
FSYNC_HANDLER(dvbm_qio_txfsync,filp,datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	struct master_dma *dma = iface->dma;
	struct master_iface *txiface = list_entry (card->iface_list.next,
		struct master_iface, list);
	const unsigned int channel = mdev_index (card, &iface->list);

	mutex_lock (&iface->buf_mutex);
	lsdma_tx_link_all (dma);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	lsdma_reset (dma);

	if (!txiface->null_packets) {
		/* Wait for the onboard FIFOs to empty */
		/* Atomic read of ICSR, so we don't need to lock */
		wait_event (iface->queue,
			!(readl (card->core.addr + DVBM_QIO_ICSR(channel)) &
			DVBM_QIO_ICSR_TXD));
	}

	mutex_unlock (&iface->buf_mutex);
	return 0;
}

/**
 * dvbm_qio_rxinit - Initialize the DVB Master Q/io receiver
 * @iface: interface
 **/
static void
dvbm_qio_rxinit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int i, reg = (iface->null_packets ? DVBM_QIO_RCSR_RNP : 0);

	switch (iface->timestamps) {
	default:
	case ASI_CTL_TSTAMP_NONE:
		reg |= 0;
		break;
	case ASI_CTL_TSTAMP_APPEND:
		reg |= DVBM_QIO_RCSR_APPEND;
		break;
	case ASI_CTL_TSTAMP_PREPEND:
		reg |= DVBM_QIO_RCSR_PREPEND;
		break;
	}
	switch (iface->mode) {
	default:
	case ASI_CTL_RX_MODE_RAW:
		reg |= 0;
		break;
	case ASI_CTL_RX_MODE_188:
		reg |= DVBM_QIO_RCSR_188 | DVBM_QIO_RCSR_PFE;
		break;
	case ASI_CTL_RX_MODE_204:
		reg |= DVBM_QIO_RCSR_204 | DVBM_QIO_RCSR_PFE;
		break;
	case ASI_CTL_RX_MODE_AUTO:
		reg |= DVBM_QIO_RCSR_AUTO | DVBM_QIO_RCSR_PFE;
		break;
	case ASI_CTL_RX_MODE_AUTOMAKE188:
		reg |= DVBM_QIO_RCSR_AUTO | DVBM_QIO_RCSR_RSS |
			DVBM_QIO_RCSR_PFE;
		break;
	case ASI_CTL_RX_MODE_204MAKE188:
		reg |= DVBM_QIO_RCSR_204 | DVBM_QIO_RCSR_RSS |
			DVBM_QIO_RCSR_PFE;
		break;
	}

	/* There will be no races on RCSR
	 * until this code returns, so we don't need to lock it */
	writel (reg | DVBM_QIO_RCSR_RXRST,
		card->core.addr + DVBM_QIO_RCSR(channel));
	writel (reg, card->core.addr + DVBM_QIO_RCSR(channel));
	writel (DVBM_QIO_RDMATL, card->core.addr + DVBM_QIO_RFCR(channel));

	/* Reset byte counter */
	readl (card->core.addr + DVBM_QIO_RXBCOUNT(channel));

	/* Reset PID Filter.
	 * There will be no races on PFLUT
	 * until this code returns, so we don't need to lock it */
	for (i = 0; i < 256; i++) {
		writel (i, card->core.addr + DVBM_QIO_PFLUTWA(channel));
		/* Dummy read to flush PCI posted writes */
		readl (card->core.addr + DVBM_QIO_FPGAID);
		writel (0xffffffff, card->core.addr + DVBM_QIO_PFLUT(channel));
	}

	/* Clear PID registers */
	writel (0, card->core.addr + DVBM_QIO_PID0(channel));
	writel (0, card->core.addr + DVBM_QIO_PID1(channel));
	writel (0, card->core.addr + DVBM_QIO_PID2(channel));
	writel (0, card->core.addr + DVBM_QIO_PID3(channel));

	/* Reset PID counters */
	readl (card->core.addr + DVBM_QIO_PIDCOUNT0(channel));
	readl (card->core.addr + DVBM_QIO_PIDCOUNT1(channel));
	readl (card->core.addr + DVBM_QIO_PIDCOUNT2(channel));
	readl (card->core.addr + DVBM_QIO_PIDCOUNT3(channel));

	return;
}

/**
 * dvbm_qio_rxstart - Activate the DVB Master Q/io receiver
 * @iface: interface
 **/
static void
dvbm_qio_rxstart (struct master_iface *iface)
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
	writel (DVBM_QIO_ICSR_RXCDIE | DVBM_QIO_ICSR_RXAOSIE |
		DVBM_QIO_ICSR_RXLOSIE | DVBM_QIO_ICSR_RXOIE |
		DVBM_QIO_ICSR_RXDIE,
		card->core.addr + DVBM_QIO_ICSR(channel));
	spin_unlock_irq (&card->irq_lock);

	/* Enable the receiver */
	reg = readl (card->core.addr + DVBM_QIO_RCSR(channel));
	writel (reg | DVBM_QIO_RCSR_RXE,
		card->core.addr + DVBM_QIO_RCSR(channel));

	return;
}

/**
 * dvbm_qio_rxstop - Deactivate the DVB Master Q/io receiver
 * @iface: interface
 **/
static void
dvbm_qio_rxstop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	/* Disable the receiver */
	reg = readl (card->core.addr + DVBM_QIO_RCSR(channel));
	writel (reg & ~DVBM_QIO_RCSR_RXE,
		card->core.addr + DVBM_QIO_RCSR(channel));

	/* Disable receiver interrupts */
	spin_lock_irq (&card->irq_lock);
	writel (DVBM_QIO_ICSR_RXCDIS | DVBM_QIO_ICSR_RXAOSIS |
		DVBM_QIO_ICSR_RXLOSIS | DVBM_QIO_ICSR_RXOIS |
		DVBM_QIO_ICSR_RXDIS,
		card->core.addr + DVBM_QIO_ICSR(channel));
	spin_unlock_irq (&card->irq_lock);

	/* Disable and abort DMA */
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION,
		card->bridge_addr + LSDMA_CSR(channel));
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION | LSDMA_CH_CSR_STOP,
		card->bridge_addr + LSDMA_CSR(channel));
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);

	wait_event (iface->queue, test_bit (0, &iface->dma_done));

	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION,
		card->bridge_addr + LSDMA_CSR(channel));

	return;
}

/**
 * dvbm_qio_rxexit - Clean up the DVB Master Q/io receiver
 * @iface: interface
 **/
static void
dvbm_qio_rxexit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);

	/* Reset the receiver.
	 * There will be no races on RCSR here,
	 * so we don't need to lock it */
	writel (DVBM_QIO_RCSR_RXRST,
		card->core.addr + DVBM_QIO_RCSR(channel));

	return;
}

/**
 * dvbm_qio_rxunlocked_ioctl - DVB Master Q/io receiver unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
dvbm_qio_rxunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	int val;
	unsigned int reg = 0, *pflut, i;

	switch (cmd) {
	case ASI_IOC_RXGETSTATUS:
		/* Atomic reads of ICSR and RCSR, so we don't need to lock */
		reg = readl (card->core.addr + DVBM_QIO_ICSR(channel));
		switch (readl (card->core.addr + DVBM_QIO_RCSR(channel)) & DVBM_QIO_RCSR_SYNC_MASK) {
		case 0:
			val = 1;
			break;
		case DVBM_QIO_RCSR_188:
			val = (reg & DVBM_QIO_ICSR_RXPASSING) ? 188 : 0;
			break;
		case DVBM_QIO_RCSR_204:
			val = (reg & DVBM_QIO_ICSR_RXPASSING) ? 204 : 0;
			break;
		case DVBM_QIO_RCSR_AUTO:
			if (reg & DVBM_QIO_ICSR_RXPASSING) {
				val = (reg & DVBM_QIO_ICSR_RX204) ? 204 : 188;
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
		if (!(iface->capabilities & ASI_CAP_RX_BYTECOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr + DVBM_QIO_RXBCOUNT(channel)),
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
			reg |= DVBM_QIO_RCSR_INVSYNC;
			break;
		default:
			return -EINVAL;
		}
		spin_lock (&card->reg_lock);
		writel ((readl (card->core.addr + DVBM_QIO_RCSR(channel)) &
			~DVBM_QIO_RCSR_INVSYNC) | reg,
			card->core.addr + DVBM_QIO_RCSR(channel));
		spin_unlock (&card->reg_lock);
		break;
	case ASI_IOC_RXGETCARRIER:
		/* Atomic read of ICSR, so we don't need to lock */
		if (put_user ((readl (card->core.addr + DVBM_QIO_ICSR(channel)) &
			DVBM_QIO_ICSR_RXCD) ? 1 : 0, (int __user *)arg)) {
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
		/* Atomic read of ICSR, so we don't need to lock */
		if (put_user ((readl (card->core.addr + DVBM_QIO_ICSR(channel)) &
			DVBM_QIO_ICSR_RXD) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPF:
		if (!(iface->capabilities & ASI_CAP_RX_PIDFILTER)) {
			return -ENOTTY;
		}
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
			writel (i, card->core.addr + DVBM_QIO_PFLUTWA(channel));
			/* Dummy read to flush PCI posted writes */
			readl (card->core.addr + DVBM_QIO_FPGAID);
			writel (pflut[i], card->core.addr + DVBM_QIO_PFLUT(channel));
		}
		spin_unlock (&card->reg_lock);
		kfree (pflut);
		break;
	case ASI_IOC_RXSETPID0:
		if (!(iface->capabilities & ASI_CAP_RX_PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		writel (val, card->core.addr + DVBM_QIO_PID0(channel));
		/* Reset PID count */
		readl (card->core.addr + DVBM_QIO_PIDCOUNT0(channel));
		break;
	case ASI_IOC_RXGETPID0COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr +
			DVBM_QIO_PIDCOUNT0(channel)),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPID1:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		writel (val, card->core.addr + DVBM_QIO_PID1(channel));
		/* Reset PID count */
		readl (card->core.addr + DVBM_QIO_PIDCOUNT1(channel));
		break;
	case ASI_IOC_RXGETPID1COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr +
			DVBM_QIO_PIDCOUNT1(channel)),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPID2:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		writel (val, card->core.addr + DVBM_QIO_PID2(channel));
		/* Reset PID count */
		readl (card->core.addr + DVBM_QIO_PIDCOUNT2(channel));
		break;
	case ASI_IOC_RXGETPID2COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr +
			DVBM_QIO_PIDCOUNT2(channel)),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPID3:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		writel (val, card->core.addr + DVBM_QIO_PID3(channel));
		/* Reset PID count */
		readl (card->core.addr + DVBM_QIO_PIDCOUNT3(channel));
		break;
	case ASI_IOC_RXGETPID3COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr +
			DVBM_QIO_PIDCOUNT3(channel)),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXGET27COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_27COUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr + DVBM_QIO_27COUNTR),
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
 * dvbm_qio_rxfsync - DVB Master Q/io receiver fsync() method
 * @filp: file to flush
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
FSYNC_HANDLER(dvbm_qio_rxfsync,filp,datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	mutex_lock (&iface->buf_mutex);

	/* Stop the receiver */
	dvbm_qio_rxstop (iface);

	/* Reset the onboard FIFO and driver buffers */
	spin_lock (&card->reg_lock);
	reg = readl (card->core.addr + DVBM_QIO_RCSR(channel));
	writel (reg | DVBM_QIO_RCSR_RXRST,
		card->core.addr + DVBM_QIO_RCSR(channel));
	writel (reg, card->core.addr + DVBM_QIO_RCSR(channel));
	spin_unlock (&card->reg_lock);
	iface->events = 0;
	lsdma_reset (iface->dma);

	/* Start the receiver */
	dvbm_qio_rxstart (iface);

	mutex_unlock (&iface->buf_mutex);
	return 0;
}

