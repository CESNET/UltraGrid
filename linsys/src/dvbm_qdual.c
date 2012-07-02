/* dvbm_qdual.c
 *
 * Linux driver functions for Linear Systems Ltd. DVB Master Quad-2in2out.
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
#include <linux/poll.h> /* poll_table */
#include <linux/spinlock.h> /* spin_lock_init () */
#include <linux/init.h> /* __devinit */
#include <linux/errno.h> /* error codes */
#include <linux/interrupt.h> /* irqreturn_t */
#include <linux/device.h> /* device_create_file () */
#include <linux/mutex.h> /* mutex_init () */

#include <asm/uaccess.h> /* put_user () */
#include <asm/bitops.h> /* set_bit () */

#include "asicore.h"
#include "../include/master.h"
#include "mdev.h"
#include "dvbm.h"
#include "mdma.h"
#include "dvbm_qdual.h"
#include "plx9080.h"
#include "lsdma.h"

static const char dvbm_qdual_name[] = DVBM_NAME_QDUAL;
static const char dvbm_qduale_name[] = DVBM_NAME_QDUALE;
static const char dvbm_lpqduale_name[] = DVBM_NAME_LPQDUALE;
static const char dvbm_lpqduale_minibnc_name[] = DVBM_NAME_LPQDUALE_MINIBNC;

/* static function prototypes */
static ssize_t dvbm_qdual_show_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static ssize_t dvbm_qdual_store_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	const char *buf,
	size_t count);
static ssize_t dvbm_qdual_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static irqreturn_t IRQ_HANDLER (dvbm_qdual_irq_handler, irq, dev_id, regs);
static void dvbm_qdual_txinit (struct master_iface *iface);
static void dvbm_qdual_txstart (struct master_iface *iface);
static void dvbm_qdual_txstop (struct master_iface *iface);
static void dvbm_qdual_txexit (struct master_iface *iface);
static void dvbm_qdual_start_tx_dma (struct master_iface *iface);
static long dvbm_qdual_txunlocked_ioctl (struct file *filp,
		unsigned int cmd,
		unsigned long arg);
static int FSYNC_HANDLER(dvbm_qdual_txfsync,filp,datasync);
static void dvbm_qdual_rxinit (struct master_iface *iface);
static void dvbm_qdual_rxstart (struct master_iface *iface);
static void dvbm_qdual_rxstop (struct master_iface *iface);
static void dvbm_qdual_rxexit (struct master_iface *iface);
static long dvbm_qdual_rxunlocked_ioctl (struct file *filp,
		unsigned int cmd,
		unsigned long arg);
static int FSYNC_HANDLER(dvbm_qdual_rxfsync,filp,datasync);

static struct file_operations dvbm_qdual_txfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.write = asi_write,
	.poll = asi_txpoll,
	.unlocked_ioctl = dvbm_qdual_txunlocked_ioctl,
	.compat_ioctl = dvbm_qdual_txunlocked_ioctl,
	.open = asi_open,
	.release = asi_release,
	.fsync = dvbm_qdual_txfsync,
	.fasync = NULL
};

static struct file_operations dvbm_qdual_rxfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.read = asi_read,
	.poll = asi_rxpoll,
	.unlocked_ioctl = dvbm_qdual_rxunlocked_ioctl,
	.compat_ioctl = dvbm_qdual_rxunlocked_ioctl,
	.open = asi_open,
	.release = asi_release,
	.fsync = dvbm_qdual_rxfsync,
	.fasync = NULL
};

static struct master_iface_operations dvbm_qdual_txops = {
	.init = dvbm_qdual_txinit,
	.start = dvbm_qdual_txstart,
	.stop = dvbm_qdual_txstop,
	.exit = dvbm_qdual_txexit,
	.start_tx_dma = dvbm_qdual_start_tx_dma
};

static struct master_iface_operations dvbm_qdual_rxops = {
	.init = dvbm_qdual_rxinit,
	.start = dvbm_qdual_rxstart,
	.stop = dvbm_qdual_rxstop,
	.exit = dvbm_qdual_rxexit
};

/**
 * dvbm_qdual_show_blackburst_type - interface attribute read handler
 * @dev: device being read
 * @attr: device_attribute
 * @buf: output buffer
 **/
static ssize_t
dvbm_qdual_show_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	return snprintf (buf, PAGE_SIZE, "%u\n",
		(readl (card->core.addr + DVBM_QDUAL_HL2CSR) & DVBM_QDUAL_HL2CSR_PLLFS) >> 2);
}

/**
 * dvbm_qdual_store_blackburst_type - interface attribute write handler
 * @dev: device being written
 * @attr: device attribute
 * @buf: input buffer
 * @count:
 **/
static ssize_t
dvbm_qdual_store_blackburst_type (struct device *dev,
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
	struct list_head *p = &card->iface_list;
	struct master_iface *iface;
	unsigned int i;
	unsigned int tx_users = 0;

	if ((endp == buf) || (val > max)) {
		return -EINVAL;
	}
	mutex_lock (&card->users_mutex);
	for (i = 0; i < 2; i++) {
		p = p->next;
		iface = list_entry (p, struct master_iface, list);
		tx_users += iface->users;
	}
	if (tx_users) {
		retcode = -EBUSY;
		goto OUT;
	}
	reg = readl (card->core.addr + DVBM_QDUAL_HL2CSR) & ~DVBM_QDUAL_HL2CSR_PLLFS;
	writel (reg | (val << 2), card->core.addr + DVBM_QDUAL_HL2CSR);
OUT:
	mutex_unlock (&card->users_mutex);
	return retcode;
}

/**
 * dvbm_qdual_show_uid - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
static ssize_t
dvbm_qdual_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	return snprintf (buf, PAGE_SIZE, "0x%08X%08X\n",
		readl (card->core.addr + DVBM_QDUAL_SSN_HI),
		readl (card->core.addr + DVBM_QDUAL_SSN_LO));
}

static DEVICE_ATTR(blackburst_type,S_IRUGO|S_IWUSR,
	dvbm_qdual_show_blackburst_type,dvbm_qdual_store_blackburst_type);
static DEVICE_ATTR(uid,S_IRUGO,
	dvbm_qdual_show_uid,NULL);

/**
 * dvbm_qdual_pci_probe - PCI insertion handler for a DVB Master Quad-2in2out
 * @pdev: PCI device
 *
 * Handle the insertion of a DVB Master Quad-2in2out.
 * Returns a negative error code on failure and 0 on success.
 **/
int __devinit
dvbm_qdual_pci_probe (struct pci_dev *pdev)
{
	int err, i;
	unsigned int cap;
	struct master_dev *card;
	void __iomem *p;

	err = dvbm_pci_probe_generic (pdev);
	if (err < 0) {
		goto NO_PCI;
	}

	/* Initialize the driver_data pointer so that dvbm_qdual_pci_remove()
	 * doesn't try to free it if an error occurs */
	pci_set_drvdata (pdev, NULL);

	/* Allocate a board info structure */
	if ((card = (struct master_dev *)
		kzalloc (sizeof (*card), GFP_KERNEL)) == NULL) {
		err = -ENOMEM;
		goto NO_MEM;
	}

	/* Initialize the board info structure */
	switch (pdev->device) {
	default:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQDUAL:
		/* LS DMA Controller */
		card->bridge_addr =
			ioremap_nocache (pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		card->core.addr = ioremap_nocache
			(pci_resource_start (pdev, 2),
			pci_resource_len (pdev, 2));
		card->name = dvbm_qdual_name;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQDUALE:
		/* LS DMA Controller */
		card->bridge_addr =
			ioremap_nocache (pci_resource_start (pdev, 3),
			pci_resource_len (pdev, 3));
		card->core.addr = ioremap_nocache
			(pci_resource_start (pdev, 2),
			pci_resource_len (pdev, 2));
		card->name = dvbm_qduale_name;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQDUALE:
		/* LS DMA Controller */
		card->bridge_addr = ioremap_nocache
			(pci_resource_start (pdev, 2),
			pci_resource_len (pdev, 2));
		card->core.addr = ioremap_nocache
			(pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		card->name = dvbm_lpqduale_name;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQDUALE_MINIBNC:
		/* LS DMA Controller */
		card->bridge_addr = ioremap_nocache
			(pci_resource_start (pdev, 2),
			pci_resource_len (pdev, 2));
		card->core.addr = ioremap_nocache
			(pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));
		card->name = dvbm_lpqduale_minibnc_name;
		break;
	}
	card->version = readl(card->core.addr +
		DVBM_QDUAL_FPGAID) & 0xffff;
	card->id = pdev->device;
	card->irq = pdev->irq;
	card->irq_handler = dvbm_qdual_irq_handler;
	INIT_LIST_HEAD(&card->iface_list);
	card->capabilities = MASTER_CAP_BLACKBURST | MASTER_CAP_UID;
	/* Lock for ICSR */
	spin_lock_init (&card->irq_lock);
	/* Lock for IBSTR, IPSTR, FTR, PFLUT, TCSR, RCSR */
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

	switch (pdev->device) {
	default:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQDUAL:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQDUALE:
		/* PLX */
		p = ioremap_nocache (pci_resource_start (pdev, 0),
			pci_resource_len (pdev, 0));

		/* Reset PCI 9056 */
		plx_reset_bridge(p);

		/* Setup the PCI 9056 */
		writel(PLX_INTCSR_PCIINT_ENABLE |
			PLX_INTCSR_PCILOCINT_ENABLE,
			p + PLX_INTCSR);

		/* Dummy read to flush PCI posted wires */
		readl(p + PLX_INTCSR);

		/* Unmap PLX */
		iounmap (p);
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQDUALE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQDUALE_MINIBNC:
		break;
	}

	/* Reset the FPGA */
	for (i = 0; i < 2; i++) {
		writel (DVBM_QDUAL_TCSR_TXRST,
			card->core.addr + DVBM_QDUAL_TCSR(i));
	}

	for (i = 2; i < 4; i++) {
		writel (DVBM_QDUAL_RCSR_RXRST,
			card->core.addr + DVBM_QDUAL_RCSR(i));
	}

	/* Setup the LS DMA Controller */
	writel (LSDMA_INTMSK_CH(0) | LSDMA_INTMSK_CH(1) | LSDMA_INTMSK_CH(2)
		| LSDMA_INTMSK_CH(3),
		card->bridge_addr + LSDMA_INTMSK);

	for (i = 0; i < 2; i++) {
		writel (LSDMA_CH_CSR_INTDONEENABLE |
			LSDMA_CH_CSR_INTSTOPENABLE,
			card->bridge_addr + LSDMA_CSR(i));
	}
	for (i = 2; i < 4; i++) {
		writel (LSDMA_CH_CSR_INTDONEENABLE |
			LSDMA_CH_CSR_INTSTOPENABLE | LSDMA_CH_CSR_DIRECTION,
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
			printk(KERN_WARNING
				"%s: Unable to create file 'uid'\n",
				dvbm_driver_name);
		}
	}

	/* Register a transmit interface */
	cap = ASI_CAP_TX_SETCLKSRC | ASI_CAP_TX_FIFOUNDERRUN |
		ASI_CAP_TX_DATA |
		ASI_CAP_TX_MAKE204 | ASI_CAP_TX_FINETUNING |
		ASI_CAP_TX_BYTECOUNTER |
		ASI_CAP_TX_LARGEIB |
		ASI_CAP_TX_INTERLEAVING |
		ASI_CAP_TX_TIMESTAMPS |
		ASI_CAP_TX_NULLPACKETS |
		ASI_CAP_TX_PTIMESTAMPS;
	switch (pdev->device) {
	default:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQDUAL:
		cap |= ASI_CAP_TX_RXCLKSRC | ASI_CAP_TX_27COUNTER;
		if (card->version >= 0x0301) {
			cap |= ASI_CAP_TX_EXTCLKSRC2;
		}
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQDUALE:
		cap |= ASI_CAP_TX_RXCLKSRC;
		if (card->version >= 0x0301) {
			cap |= ASI_CAP_TX_EXTCLKSRC2;
		}
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQDUALE:
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQDUALE_MINIBNC:
		if (card->version >= 0x0102) {
			cap |= ASI_CAP_TX_27COUNTER;
		}
		break;
	}
	for (i = 0; i < 2; i++) {
		if ((err = asi_register_iface (card,
			&lsdma_dma_ops,
			0,
			MASTER_DIRECTION_TX,
			&dvbm_qdual_txfops,
			&dvbm_qdual_txops,
			cap,
			4,
			ASI_CTL_TRANSPORT_DVB_ASI)) < 0) {
			goto NO_IFACE;
		}
	}

	/* Register a receive interface */
	cap = ASI_CAP_RX_SYNC | ASI_CAP_RX_INVSYNC | ASI_CAP_RX_CD |
		ASI_CAP_RX_MAKE188 |
		ASI_CAP_RX_BYTECOUNTER |
		ASI_CAP_RX_DATA |
		ASI_CAP_RX_PIDFILTER |
		ASI_CAP_RX_TIMESTAMPS |
		ASI_CAP_RX_PTIMESTAMPS |
		ASI_CAP_RX_NULLPACKETS;
	switch (pdev->device) {
	default:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQDUAL:
		cap |= ASI_CAP_RX_27COUNTER;
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQDUALE:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQDUALE:
		break;
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQDUALE_MINIBNC:
		if (card->version >= 0x0102) {
			cap |= ASI_CAP_RX_27COUNTER;
		}
		break;
	}
	for (i = 2; i < 4; i++) {
		if ((err = asi_register_iface (card,
			&lsdma_dma_ops,
			0,
			MASTER_DIRECTION_RX,
			&dvbm_qdual_rxfops,
			&dvbm_qdual_rxops,
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
	dvbm_qdual_pci_remove (pdev);
NO_PCI:
	return err;
}

/**
 * dvbm_qdual_pci_remove - PCI removal handler for a DVB Master Quad-2in2out
 * @pdev: PCI device
 *
 * Handle the removal of a DVB Master Quad-2in2out.
 * This function may be called during PCI probe error handling,
 * so don't mark it as __devexit.
 **/
void
dvbm_qdual_pci_remove (struct pci_dev *pdev)
{
	struct master_dev *card = pci_get_drvdata (pdev);

	if (card) {
		int i;

		/* Unregister the device and all interfaces */
		dvbm_unregister_all (card);

		for (i = 0; i < 4; i++) {
			writel (0, card->core.addr + DVBM_QDUAL_ICSR(i));
		}
		iounmap (card->core.addr);
		iounmap (card->bridge_addr);
		kfree (card);
	}
	dvbm_pci_remove_generic (pdev);
	return;
}

/**
 * dvbm_qdual_irq_handler - DVB Master Quad-2in2out interrupt service routine
 * @irq: interrupt number
 * @dev_id: pointer to the device data structure
 * @regs: processor context
 **/
static irqreturn_t
IRQ_HANDLER (dvbm_qdual_irq_handler, irq, dev_id, regs)
{
	struct master_dev *card = dev_id;
	struct list_head *p = &card->iface_list;
	struct master_iface *iface;
	unsigned int dmaintsrc = readl (card->bridge_addr + LSDMA_INTSRC);
	unsigned int status, interrupting_iface = 0;
	unsigned int i;

	for (i = 0 ; i < 2; i++) {
		p = p->next;
		iface = list_entry(p, struct master_iface, list);

		if (dmaintsrc & LSDMA_INTSRC_CH(i)) {
			/* Read the interrupt type and clear it */
			spin_lock (&card->irq_lock);
			status = readl (card->bridge_addr + LSDMA_CSR(i));
			writel (status, card->bridge_addr + LSDMA_CSR(i));
			spin_unlock (&card->irq_lock);
			/* Increment the buffer pointer */
			if (status & LSDMA_CH_CSR_INTSRCBUFFER) {
				mdma_advance (iface->dma);
			}

			/* Flag end-of-chain */
			if (status & LSDMA_CH_CSR_INTSRCDONE) {
				set_bit (ASI_EVENT_TX_BUFFER_ORDER,
					&iface->events);
				set_bit (0, &iface->dma_done);
			}

			/* Flag DMA abort */
			if (status & LSDMA_CH_CSR_INTSRCSTOP) {
				set_bit (0, &iface->dma_done);
			}

			interrupting_iface |= 0x1 << i;
		}

		spin_lock (&card->irq_lock);
		status = readl (card->core.addr + DVBM_QDUAL_ICSR(i));
		writel (status, card->core.addr + DVBM_QDUAL_ICSR(i));

		if (status & DVBM_QDUAL_ICSR_TXUIS) {
			set_bit (ASI_EVENT_TX_FIFO_ORDER,
				&iface->events);
			interrupting_iface |= 0x1 << i;
		}
		if (status & DVBM_QDUAL_ICSR_TXDIS) {
			set_bit (ASI_EVENT_TX_DATA_ORDER,
				&iface->events);
			interrupting_iface |= 0x1 << i;
		}
		spin_unlock (&card->irq_lock);

		if (interrupting_iface & 0x1 << i) {
			wake_up (&iface->queue);
		}
	}

	for (i = 2 ; i < 4; i++) {
		p = p->next;
		iface = list_entry(p, struct master_iface, list);

		if (dmaintsrc & LSDMA_INTSRC_CH(i)) {
			struct master_dma *dma = iface->dma;

			/* Read the interrupt type and clear it */
			spin_lock (&card->irq_lock);
			status = readl (card->bridge_addr + LSDMA_CSR(i));
			writel (status, card->bridge_addr + LSDMA_CSR(i));
			spin_unlock (&card->irq_lock);

			/* Increment the buffer pointer */
			if (status & LSDMA_CH_CSR_INTSRCBUFFER) {
				mdma_advance (dma);
				if (mdma_rx_isempty (dma)) {
					set_bit (ASI_EVENT_RX_BUFFER_ORDER,
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

			interrupting_iface |= 0x1 << i;
		}

		spin_lock (&card->irq_lock);
		status = readl (card->core.addr + DVBM_QDUAL_ICSR(i));
		writel (status, card->core.addr + DVBM_QDUAL_ICSR(i));

		if (status & DVBM_QDUAL_ICSR_RXCDIS) {
			set_bit (ASI_EVENT_RX_CARRIER_ORDER,
				&iface->events);
			interrupting_iface |= 0x1 << i;
		}
		if (status & DVBM_QDUAL_ICSR_RXAOSIS) {
			set_bit (ASI_EVENT_RX_AOS_ORDER,
				&iface->events);
			interrupting_iface |= 0x1 << i;
		}
		if (status & DVBM_QDUAL_ICSR_RXLOSIS) {
			set_bit (ASI_EVENT_RX_LOS_ORDER,
				&iface->events);
			interrupting_iface |= 0x1 << i;
		}
		if (status & DVBM_QDUAL_ICSR_RXOIS) {
			set_bit (ASI_EVENT_RX_FIFO_ORDER,
				&iface->events);
			interrupting_iface |= 0x1 << i;
		}
		if (status & DVBM_QDUAL_ICSR_RXDIS) {
			set_bit (ASI_EVENT_RX_DATA_ORDER,
				&iface->events);
			interrupting_iface |= 0x1 << i;
		}

		if (interrupting_iface & 0x1 << i) {
			wake_up (&iface->queue);
		}
		spin_unlock (&card->irq_lock);
	}

	/* Check and clear the source of the interrupt */
	if (interrupting_iface) {
		/* Dummy read to flush PCI posted writes */
		readl (card->bridge_addr + LSDMA_INTMSK);

		return IRQ_HANDLED;
	}
	return IRQ_NONE;
}

/**
 * dvbm_qdual_txinit - Initialize the DVB Master Quad-2in2out transmitter
 * @iface: interface
 **/
static void
dvbm_qdual_txinit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg = iface->null_packets ? DVBM_QDUAL_TCSR_TNP : 0;

	switch (iface->timestamps) {
	default:
	case ASI_CTL_TSTAMP_NONE:
		reg |= 0;
		break;
	case ASI_CTL_TSTAMP_APPEND:
		reg |= DVBM_QDUAL_TCSR_TTSS;
		break;
	case ASI_CTL_TSTAMP_PREPEND:
		reg |= DVBM_QDUAL_TCSR_TPRC;
		break;
	}
	switch (iface->mode) {
	default:
	case ASI_CTL_TX_MODE_188:
		reg |= DVBM_QDUAL_TCSR_188;
		break;
	case ASI_CTL_TX_MODE_204:
		reg |= DVBM_QDUAL_TCSR_204;
		break;
	case ASI_CTL_TX_MODE_MAKE204:
		reg |= DVBM_QDUAL_TCSR_MAKE204;
		break;
	}
	switch (iface->clksrc) {
	default:
	case ASI_CTL_TX_CLKSRC_ONBOARD:
		reg |= 0;
		break;
	case ASI_CTL_TX_CLKSRC_EXT:
		reg |= DVBM_QDUAL_TCSR_EXTCLK;
		break;
	case ASI_CTL_TX_CLKSRC_RX:
		reg |= DVBM_QDUAL_TCSR_RXCLK;
		break;
	case ASI_CTL_TX_CLKSRC_EXT2:
		reg |= DVBM_QDUAL_TCSR_EXTCLK2;
		break;
	}

	/* There will be no races on IBSTR, IPSTR, FTR, and TCSR
	 * until this code returns, so we don't need to lock them */
	writel (reg | DVBM_QDUAL_TCSR_TXRST,
		card->core.addr + DVBM_QDUAL_TCSR(channel));
	writel (reg, card->core.addr + DVBM_QDUAL_TCSR(channel));
	writel ((DVBM_QDUAL_TFL << 16) | DVBM_QDUAL_TDMATL,
		card->core.addr + DVBM_QDUAL_TFCR(channel));
	writel (0, card->core.addr + DVBM_QDUAL_IBSTREG(channel));
	writel (0, card->core.addr + DVBM_QDUAL_IPSTREG(channel));
	writel (0, card->core.addr + DVBM_QDUAL_FTREG(channel));
	/* Reset byte counter */
	readl (card->core.addr + DVBM_QDUAL_TXBCOUNT (channel));
	return;
}

/**
 * dvbm_qdual_txstart - Activate the DVB Master Quad-2in2out transmitter
 * @iface: interface
 **/
static void
dvbm_qdual_txstart (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;
	const unsigned int channel = mdev_index (card, &iface->list);

	/* Enable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	writel (DVBM_QDUAL_ICSR_TXUIE | DVBM_QDUAL_ICSR_TXDIE,
		card->core.addr + DVBM_QDUAL_ICSR(channel));
	spin_unlock_irq(&card->irq_lock);

	/* Enable the transmitter */
	reg = readl(card->core.addr + DVBM_QDUAL_TCSR(channel));
	writel(reg | DVBM_QDUAL_TCSR_TXE,
		card->core.addr + DVBM_QDUAL_TCSR(channel));

	return;
}

/**
 * dvbm_qdual_txstop - Deactivate the DVB Master Quad-2in2out transmitter
 * @iface: interface
 **/
static void
dvbm_qdual_txstop (struct master_iface *iface)
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
			!(readl (card->core.addr + DVBM_QDUAL_ICSR(channel)) &
			DVBM_QDUAL_ICSR_TXD));
	}

	/* Disable the transmitter.
	 * There will be no races on TCSR here,
	 * so we don't need to lock it */
	reg = readl(card->core.addr + DVBM_QDUAL_TCSR(channel));
	writel(reg & ~DVBM_QDUAL_TCSR_TXE,
		card->core.addr + DVBM_QDUAL_TCSR(channel));

	/* Disable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	writel (DVBM_QDUAL_ICSR_TXUIS | DVBM_QDUAL_ICSR_TXDIS,
		card->core.addr + DVBM_QDUAL_ICSR(channel));
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);
	spin_unlock_irq (&card->irq_lock);

	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE,
		card->bridge_addr + LSDMA_CSR(channel));

	return;
}

/**
 * dvbm_qdual_txexit - Clean up the DVB Master Quad-2in2out transmitter
 * @iface: interface
 **/
static void
dvbm_qdual_txexit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);

	/* Reset the transmitter */
	writel (DVBM_QDUAL_TCSR_TXRST,
		card->core.addr + DVBM_QDUAL_TCSR(channel));

	return;
}

/**
 * dvbm_qdual_start_tx_dma - start transmit DMA
 * @iface: interface
 **/
static void
dvbm_qdual_start_tx_dma (struct master_iface *iface)
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
 * dvbm_qdual_txunlocked_ioctl - DVB Master Quad-2in2out transmitter unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
dvbm_qdual_txunlocked_ioctl (struct file *filp,
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
		writel (stuffing.ib,
			card->core.addr + DVBM_QDUAL_IBSTREG(channel));
		writel (stuffing.ip,
			card->core.addr + DVBM_QDUAL_IPSTREG(channel));
		writel ((stuffing.il_big << DVBM_QDUAL_FTR_ILBIG_SHIFT) |
			(stuffing.big_ip << DVBM_QDUAL_FTR_BIGIP_SHIFT) |
			(stuffing.il_normal << DVBM_QDUAL_FTR_ILNORMAL_SHIFT) |
			stuffing.normal_ip,
			card->core.addr + DVBM_QDUAL_FTREG(channel));
		spin_unlock (&card->reg_lock);
		break;
	case ASI_IOC_TXGETBYTECOUNT:
		if (!(iface->capabilities & ASI_CAP_TX_BYTECOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr +
			DVBM_QDUAL_TXBCOUNT(channel)),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_TXGETTXD:
		/* Atomic read of ICSR, so we don't need to lock */
		if (put_user ((readl (card->core.addr +
			DVBM_QDUAL_ICSR(channel)) &
			DVBM_QDUAL_ICSR_TXD) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_TXGET27COUNT:
		if (!(iface->capabilities & ASI_CAP_TX_27COUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr + DVBM_QDUAL_27COUNTR),
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
 * dvbm_qdual_txfsync - DVB Master Quad-2in2out transmitter fsync() method
 * @filp: file to flush
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
FSYNC_HANDLER(dvbm_qdual_txfsync,filp,datasync)
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
			!(readl (card->core.addr + DVBM_QDUAL_ICSR(channel)) &
			DVBM_QDUAL_ICSR_TXD));
	}

	mutex_unlock (&iface->buf_mutex);
	return 0;
}

/**
 * dvbm_qdual_rxinit - Initialize the DVB Master Quad-2in2out receiver
 * @iface: interface
 **/
static void
dvbm_qdual_rxinit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int i, reg = (iface->null_packets ? DVBM_QDUAL_RCSR_RNP : 0);

	switch (iface->timestamps) {
	default:
	case ASI_CTL_TSTAMP_NONE:
		reg |= 0;
		break;
	case ASI_CTL_TSTAMP_APPEND:
		reg |= DVBM_QDUAL_RCSR_TSE;
		break;
	case ASI_CTL_TSTAMP_PREPEND:
		reg |= DVBM_QDUAL_RCSR_PTSE;
		break;
	}
	switch (iface->mode) {
	default:
	case ASI_CTL_RX_MODE_RAW:
		reg |= 0;
		break;
	case ASI_CTL_RX_MODE_188:
		reg |= DVBM_QDUAL_RCSR_188 | DVBM_QDUAL_RCSR_PFE;
		break;
	case ASI_CTL_RX_MODE_204:
		reg |= DVBM_QDUAL_RCSR_204 | DVBM_QDUAL_RCSR_PFE;
		break;
	case ASI_CTL_RX_MODE_AUTO:
		reg |= DVBM_QDUAL_RCSR_AUTO | DVBM_QDUAL_RCSR_PFE;
		break;
	case ASI_CTL_RX_MODE_AUTOMAKE188:
		reg |= DVBM_QDUAL_RCSR_AUTO | DVBM_QDUAL_RCSR_RSS |
			DVBM_QDUAL_RCSR_PFE;
		break;
	case ASI_CTL_RX_MODE_204MAKE188:
		reg |= DVBM_QDUAL_RCSR_204 | DVBM_QDUAL_RCSR_RSS |
			DVBM_QDUAL_RCSR_PFE;
		break;
	}

	/* There will be no races on RCSR
	 * until this code returns, so we don't need to lock it */
	writel (reg | DVBM_QDUAL_RCSR_RXRST,
		card->core.addr + DVBM_QDUAL_RCSR(channel));
	writel (reg, card->core.addr + DVBM_QDUAL_RCSR(channel));
	writel (DVBM_QDUAL_RDMATL,
		card->core.addr + DVBM_QDUAL_RFCR(channel));

	/* Reset byte counter */
	readl (card->core.addr + DVBM_QDUAL_RXBCOUNT(channel));

	/* Reset PID Filter.
	 * There will be no races on PFLUT
	 * until this code returns, so we don't need to lock it */

	for (i = 0; i < 256; i++) {
		writel (i, card->core.addr + DVBM_QDUAL_PFLUTWA(channel));
		/* Dummy read to flush PCI posted writes */
		readl (card->core.addr + DVBM_QDUAL_FPGAID);
		writel (0xffffffff,
			card->core.addr + DVBM_QDUAL_PFLUT(channel));
	}

	/* Clear PID registers */
	writel (0, card->core.addr + DVBM_QDUAL_PID0(channel));
	writel (0, card->core.addr + DVBM_QDUAL_PID1(channel));
	writel (0, card->core.addr + DVBM_QDUAL_PID2(channel));
	writel (0, card->core.addr + DVBM_QDUAL_PID3(channel));

	/* Reset PID counters */
	readl (card->core.addr + DVBM_QDUAL_PIDCOUNT0(channel));
	readl (card->core.addr + DVBM_QDUAL_PIDCOUNT1(channel));
	readl (card->core.addr + DVBM_QDUAL_PIDCOUNT2(channel));
	readl (card->core.addr + DVBM_QDUAL_PIDCOUNT3(channel));

	return;
}

/**
 * dvbm_qdual_rxstart - Activate the DVB Master Quad-2in2out receiver
 * @iface: interface
 **/
static void
dvbm_qdual_rxstart (struct master_iface *iface)
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
	writel (DVBM_QDUAL_ICSR_RXCDIE | DVBM_QDUAL_ICSR_RXAOSIE |
		DVBM_QDUAL_ICSR_RXLOSIE | DVBM_QDUAL_ICSR_RXOIE |
		DVBM_QDUAL_ICSR_RXDIE,
		card->core.addr + DVBM_QDUAL_ICSR(channel));
	spin_unlock_irq (&card->irq_lock);

	/* Enable the receiver */
	reg = readl (card->core.addr + DVBM_QDUAL_RCSR(channel));
	writel (reg | DVBM_QDUAL_RCSR_RXE,
		card->core.addr + DVBM_QDUAL_RCSR(channel));

	return;
}

/**
 * dvbm_qdual_rxstop - Deactivate the DVB Master Quad-2in2out receiver
 * @iface: interface
 **/
static void
dvbm_qdual_rxstop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	/* Disable the receiver */
	reg = readl (card->core.addr + DVBM_QDUAL_RCSR(channel));
	writel (reg & ~DVBM_QDUAL_RCSR_RXE,
		card->core.addr + DVBM_QDUAL_RCSR(channel));

	/* Disable receiver interrupts */
	spin_lock_irq (&card->irq_lock);
	writel (DVBM_QDUAL_ICSR_RXCDIS | DVBM_QDUAL_ICSR_RXAOSIS |
		DVBM_QDUAL_ICSR_RXLOSIS | DVBM_QDUAL_ICSR_RXOIS |
		DVBM_QDUAL_ICSR_RXDIS,
		card->core.addr + DVBM_QDUAL_ICSR(channel));
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
 * dvbm_qdual_rxexit - Clean up the DVB Master Quad-2in2out receiver
 * @iface: interface
 **/
static void
dvbm_qdual_rxexit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);

	/* Reset the receiver.
	 * There will be no races on RCSR here,
	 * so we don't need to lock it */
	writel (DVBM_QDUAL_RCSR_RXRST,
		card->core.addr + DVBM_QDUAL_RCSR(channel));

	return;
}

/**
 * dvbm_qdual_rxunlocked_ioctl - DVB Master Quad-2in2out receiver unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
dvbm_qdual_rxunlocked_ioctl (struct file *filp,
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
		reg = readl (card->core.addr + DVBM_QDUAL_ICSR(channel));
		switch (readl (card->core.addr + DVBM_QDUAL_RCSR(channel)) &
			DVBM_QDUAL_RCSR_SYNC_MASK) {
		case 0:
			val = 1;
			break;
		case DVBM_QDUAL_RCSR_188:
			val = (reg & DVBM_QDUAL_ICSR_RXPASSING) ? 188 : 0;
			break;
		case DVBM_QDUAL_RCSR_204:
			val = (reg & DVBM_QDUAL_ICSR_RXPASSING) ? 204 : 0;
			break;
		case DVBM_QDUAL_RCSR_AUTO:
			if (reg & DVBM_QDUAL_ICSR_RXPASSING) {
				val = (reg & DVBM_QDUAL_ICSR_RX204) ? 204 : 188;
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
		if (put_user (readl (card->core.addr +
			DVBM_QDUAL_RXBCOUNT(channel)),
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
			reg |= DVBM_QDUAL_RCSR_INVSYNC;
			break;
		default:
			return -EINVAL;
		}
		spin_lock (&card->reg_lock);
		writel ((readl (card->core.addr + DVBM_QDUAL_RCSR(channel)) &
			~DVBM_QDUAL_RCSR_INVSYNC) | reg,
			card->core.addr + DVBM_QDUAL_RCSR(channel));
		spin_unlock (&card->reg_lock);
		break;
	case ASI_IOC_RXGETCARRIER:
		/* Atomic read of ICSR, so we don't need to lock */
		if (put_user ((readl (card->core.addr +
			DVBM_QDUAL_ICSR(channel)) &
			DVBM_QDUAL_ICSR_RXCD) ? 1 : 0, (int __user *)arg)) {
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
		if (put_user ((readl (card->core.addr +
			DVBM_QDUAL_ICSR(channel)) &
			DVBM_QDUAL_ICSR_RXD) ? 1 : 0, (int __user *)arg)) {
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
			writel (i,
				card->core.addr + DVBM_QDUAL_PFLUTWA(channel));
			/* Dummy read to flush PCI posted writes */
			readl (card->core.addr + DVBM_QDUAL_FPGAID);
			writel (pflut[i],
				card->core.addr + DVBM_QDUAL_PFLUT(channel));
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
		writel (val, card->core.addr + DVBM_QDUAL_PID0(channel));
		/* Reset PID count */
		readl (card->core.addr + DVBM_QDUAL_PIDCOUNT0(channel));
		break;
	case ASI_IOC_RXGETPID0COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr +
			DVBM_QDUAL_PIDCOUNT0(channel)),
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
		writel (val, card->core.addr + DVBM_QDUAL_PID1(channel));
		/* Reset PID count */
		readl (card->core.addr + DVBM_QDUAL_PIDCOUNT1(channel));
		break;
	case ASI_IOC_RXGETPID1COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr +
			DVBM_QDUAL_PIDCOUNT1(channel)),
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
		writel (val, card->core.addr + DVBM_QDUAL_PID2(channel));
		/* Reset PID count */
		readl (card->core.addr + DVBM_QDUAL_PIDCOUNT2(channel));
		break;
	case ASI_IOC_RXGETPID2COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr +
			DVBM_QDUAL_PIDCOUNT2(channel)),
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
		writel (val, card->core.addr + DVBM_QDUAL_PID3(channel));
		/* Reset PID count */
		readl (card->core.addr + DVBM_QDUAL_PIDCOUNT3(channel));
		break;
	case ASI_IOC_RXGETPID3COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr +
			DVBM_QDUAL_PIDCOUNT3(channel)),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXGET27COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_27COUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr + DVBM_QDUAL_27COUNTR),
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
 * dvbm_qdual_rxfsync - DVB Master Quad-2in2out receiver fsync() method
 * @filp: file to flush
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
FSYNC_HANDLER(dvbm_qdual_rxfsync,filp,datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	mutex_lock (&iface->buf_mutex);

	/* Stop the receiver */
	dvbm_qdual_rxstop (iface);

	/* Reset the onboard FIFO and driver buffers */
	spin_lock (&card->reg_lock);
	reg = readl (card->core.addr + DVBM_QDUAL_RCSR(channel));
	writel (reg | DVBM_QDUAL_RCSR_RXRST,
		card->core.addr + DVBM_QDUAL_RCSR(channel));
	writel (reg, card->core.addr + DVBM_QDUAL_RCSR(channel));
	spin_unlock (&card->reg_lock);
	iface->events = 0;
	lsdma_reset (iface->dma);

	/* Start the receiver */
	dvbm_qdual_rxstart (iface);

	mutex_unlock (&iface->buf_mutex);
	return 0;
}

