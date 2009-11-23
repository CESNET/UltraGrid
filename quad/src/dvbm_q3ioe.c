/* dvbm_q3io.c
 *
 * Linux driver functions for Linear Systems Ltd. DVB Master Quad-1in3out.
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
#include <linux/interrupt.h> /* irqreturn_t */
#include <linux/device.h> /* class_device_create_file () */
#include <linux/delay.h>

#include <asm/semaphore.h> /* sema_init () */
#include <asm/uaccess.h> /* put_user () */
#include <asm/bitops.h> /* set_bit () */

#include "asicore.h"
#include "../include/master.h"
#include "mdev.h"
#include "dvbm.h"
#include "miface.h"
#include "dvbm_q3ioe.h"
#include "plx9080.h"
#include "masterplx.h"
#include "lsdma.h"
#include "masterlsdma.h"

static const char dvbm_q3ioe_name[] = DVBM_NAME_Q3IOE;

/* static function prototypes */
static ssize_t dvbm_q3io_show_uid (struct class_device *cd,
	char *buf);

static irqreturn_t IRQ_HANDLER (dvbm_q3io_irq_handler, irq, dev_id, regs);
static void dvbm_q3io_txinit (struct master_iface *iface);
static void dvbm_q3io_txstart (struct master_iface *iface);
static void dvbm_q3io_txstop (struct master_iface *iface);
static void dvbm_q3io_txexit (struct master_iface *iface);
static int dvbm_q3io_txopen (struct inode *inode, struct file *filp);
static long dvbm_q3io_txunlocked_ioctl (struct file *filp,
		unsigned int cmd,
		unsigned long arg);
static int dvbm_q3io_txioctl (struct inode *inode,
	struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int dvbm_q3io_txfsync (struct file *filp,
	struct dentry *dentry,
	int datasync);
static int dvbm_q3io_txrelease (struct inode *inode, struct file *filp);
static void dvbm_q3io_rxinit (struct master_iface *iface);
static void dvbm_q3io_rxstart (struct master_iface *iface);
static void dvbm_q3io_rxstop (struct master_iface *iface);
static void dvbm_q3io_rxexit (struct master_iface *iface);
static int dvbm_q3io_rxopen (struct inode *inode, struct file *filp);
static long dvbm_q3io_rxunlocked_ioctl (struct file *filp,
		unsigned int cmd,
		unsigned long arg);
static int dvbm_q3io_rxioctl (struct inode *inode,
	struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int dvbm_q3io_rxfsync (struct file *filp,
	struct dentry *dentry,
	int datasync);
static int dvbm_q3io_rxrelease (struct inode *inode, struct file *filp);

struct file_operations dvbm_q3io_txfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.write = masterlsdma_write,
	.poll = masterlsdma_txpoll,
	.ioctl = dvbm_q3io_txioctl,
#ifdef  HAVE_UNLOCKED_IOCTL
	.unlocked_ioctl = dvbm_q3io_txunlocked_ioctl,
#endif
#ifdef  HAVE_COMPAT_IOCTL
	.compat_ioctl = asi_compat_ioctl,
#endif
	.open = dvbm_q3io_txopen,
	.release = dvbm_q3io_txrelease,
	.fsync = dvbm_q3io_txfsync,
	.fasync = NULL
};

struct file_operations dvbm_q3io_rxfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.read = masterlsdma_read,
	.poll = masterlsdma_rxpoll,
	.ioctl = dvbm_q3io_rxioctl,
#ifdef  HAVE_UNLOCKED_IOCTL
	.unlocked_ioctl = dvbm_q3io_rxunlocked_ioctl,
#endif
#ifdef  HAVE_COMPAT_IOCTL
	.compat_ioctl = asi_compat_ioctl,
#endif
	.open = dvbm_q3io_rxopen,
	.release = dvbm_q3io_rxrelease,
	.fsync = dvbm_q3io_rxfsync,
	.fasync = NULL
};


/**
 * dvbm_q3io_show_uid - interface attribute read handler
 * @cd: class_device being read
 * @buf: output buffer
 **/
static ssize_t
dvbm_q3io_show_uid (struct class_device *cd,
	char *buf)
{
	struct master_dev *card = to_master_dev(cd);

	return snprintf (buf, PAGE_SIZE, "0x%08X%08X\n",
		readl (card->core.addr + DVBM_Q3IO_SSN_HI),
		readl (card->core.addr + DVBM_Q3IO_SSN_LO));
}

static CLASS_DEVICE_ATTR(uid,S_IRUGO,
	dvbm_q3io_show_uid,NULL);


/**
 * dvbm_q3io_pci_probe - PCI insertion handler for a DVB Master Quad-1in3out
 * @dev: PCI device
 *
 * Handle the insertion of a DVB Master Quad-1in3out.
 * Returns a negative error code on failure and 0 on success.
 **/

int __devinit
dvbm_q3io_pci_probe (struct pci_dev *dev)
{
	int err, i;
	unsigned int cap, transport;
	const char *name;
	struct master_dev *card;

	switch (dev->device) {
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQ3IOE:
		name = dvbm_q3ioe_name;
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
	/* LS DMA Controller */
	card->bridge_addr = ioremap_nocache (pci_resource_start (dev, 0),
		pci_resource_len (dev, 0));
	/* ASI Core */
	card->core.addr = ioremap_nocache (pci_resource_start (dev, 2),
		pci_resource_len (dev, 2));
	card->version = readl(card->core.addr + DVBM_Q3IO_FPGAID) & 0xffff;
	card->name = name;
	card->irq_handler = dvbm_q3io_irq_handler;
	INIT_LIST_HEAD(&card->iface_list);
	switch (dev->device) {
	default:
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQ3IOE:
		card->capabilities = MASTER_CAP_UID;
		break;
	}
	/* Lock for ICSR */
	spin_lock_init (&card->irq_lock);
	/* Lock for IBSTR, IPSTR, FTR, PFLUT, TCSR, RCSR */
	spin_lock_init (&card->reg_lock);
	sema_init (&card->users_sem, 1);
	card->pdev = dev;

	/* Print the firmware version */
	printk (KERN_INFO "%s: %s detected, firmware version %u.%u (0x%04X)\n",
		dvbm_driver_name, name,
		card->version >> 8, card->version & 0x00ff, card->version);

	/* Store the pointer to the board info structure
	 * in the PCI info structure */
	pci_set_drvdata (dev, card);

	/* Reset PCI 9056 */
	masterplx_reset_bridge(card);

	/* Setup the PCI 9056 */
	writel(PLX_INTCSR_PCIINT_ENABLE |
		   PLX_INTCSR_PCILOCINT_ENABLE, card->bridge_addr + PLX_INTCSR);

	/* Dummy read to flush PCI posted wires */
	readl(card->bridge_addr + PLX_INTCSR);

	/* Remap bridge address to the DMA controller */
	iounmap (card->bridge_addr);

	/* LS DMA Controller */
	card->bridge_addr = ioremap_nocache (pci_resource_start (dev, 3),pci_resource_len (dev, 3));

	/* Reset the FPGA */

	for (i = 0; i < 3; i++)
		writel (DVBM_Q3IO_TCSR_TXRST, card->core.addr + DVBM_Q3IO_TCSR(i));

	for (i = 3; i < 4; i++)
		writel (DVBM_Q3IO_RCSR_RXRST, card->core.addr + DVBM_Q3IO_RCSR(i));

	/* Setup the LS DMA Controller*/
	writel (LSDMA_INTMSK_CH(0) | LSDMA_INTMSK_CH(1) | LSDMA_INTMSK_CH(2)
		| LSDMA_INTMSK_CH(3),
		card->bridge_addr + LSDMA_INTMSK);

	for (i = 0; i< 4; i++) {
		writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE
			| LSDMA_CH_CSR_DIRECTION, card->bridge_addr + LSDMA_CSR(i));
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
			printk(KERN_WARNING
				"%s: Unable to create file 'uid'\n",
				card->name);
		}
	}

	/* Register a transmit interface */
	cap = ASI_CAP_TX_SETCLKSRC | ASI_CAP_TX_FIFOUNDERRUN |
		ASI_CAP_TX_DATA | ASI_CAP_TX_RXCLKSRC;
	switch (dev->device) {
		case DVBM_PCI_DEVICE_ID_LINSYS_DVBQ3IOE:
			cap |= ASI_CAP_TX_MAKE204 | ASI_CAP_TX_FINETUNING |
			ASI_CAP_TX_BYTECOUNTER |
			ASI_CAP_TX_LARGEIB |
			ASI_CAP_TX_INTERLEAVING |
			ASI_CAP_TX_TIMESTAMPS |
			ASI_CAP_TX_NULLPACKETS |
			ASI_CAP_TX_PTIMESTAMPS;
		transport = ASI_CTL_TRANSPORT_DVB_ASI;
		break;
	default:
		transport = 0xff;
		break;
	}

	for (i = 0; i < 3; i++) {
		if ((err = asi_register_iface (card,
			MASTER_DIRECTION_TX,
			&dvbm_q3io_txfops,
			cap,
			4,
			transport)) < 0) {
			goto NO_IFACE;
		}
	}

	/* Register a receive interface */
	cap = ASI_CAP_RX_SYNC | ASI_CAP_RX_INVSYNC | ASI_CAP_RX_CD;
	switch (dev->device) {
	case DVBM_PCI_DEVICE_ID_LINSYS_DVBQ3IOE:
		cap |= ASI_CAP_RX_MAKE188 |
			ASI_CAP_RX_BYTECOUNTER |
			ASI_CAP_RX_DATA |
			ASI_CAP_RX_PIDFILTER |
			ASI_CAP_RX_TIMESTAMPS |
			ASI_CAP_RX_PTIMESTAMPS |
			ASI_CAP_RX_NULLPACKETS;
		transport = ASI_CTL_TRANSPORT_DVB_ASI;
		break;
	default:
		transport = 0xff;
		break;
	}

	for (i = 3; i < 4; i++) {
		if ((err = asi_register_iface (card,
			MASTER_DIRECTION_RX,
			&dvbm_q3io_rxfops,
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
 * dvbm_q3io_pci_remove - PCI removal handler for a DVB Master Quad-1in3out
 * @card: Master device
 *
 * Handle the removal of a DVB Master Quad-1in3out.
 **/
void
dvbm_q3io_pci_remove (struct master_dev *card)
{
	int i;
	for (i = 0; i< 4; i++) {
		if (card->capabilities) {
			writel (0, card->core.addr + DVBM_Q3IO_ICSR(i));
		}
	}

	iounmap (card->core.addr);

	return;
}

/**
 * dvbm_q3io_irq_handler - DVB Master Quad-1in3out interrupt service routine
 * @irq: interrupt number
 * @dev_id: pointer to the device data structure
 * @regs: processor context
 **/
static irqreturn_t
IRQ_HANDLER (dvbm_q3io_irq_handler, irq, dev_id, regs)

{
	struct master_dev *card = dev_id;
	struct list_head *p = &card->iface_list;
	struct master_iface *iface;
	unsigned int dmaintsrc = readl (card->bridge_addr + LSDMA_INTSRC);
	unsigned int status, interrupting_iface = 0;
	int i;

	for (i = 0 ; i < 3; i++) {
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
				lsdma_advance (iface->dma);
		}

		/* Flag end-of-chain */
		if (status & LSDMA_CH_CSR_INTSRCDONE) {
			set_bit (ASI_EVENT_TX_BUFFER_ORDER, &iface->events);
			set_bit (0, &iface->dma_done);
		}
		/* Flag DMA abort */
		if (status & LSDMA_CH_CSR_INTSRCSTOP) {
			set_bit (0, &iface->dma_done);
		}

		interrupting_iface |= 0x1 << i;

	}

	spin_lock (&card->irq_lock);
	status = readl (card->core.addr + DVBM_Q3IO_ICSR(i));
	writel (status, card->core.addr + DVBM_Q3IO_ICSR(i));

	if (status & DVBM_Q3IO_ICSR_TXUIS) {
		set_bit (ASI_EVENT_TX_FIFO_ORDER,
			&iface->events);
		interrupting_iface |= 0x1 << i;
	}
	if (status & DVBM_Q3IO_ICSR_TXDIS) {
		set_bit (ASI_EVENT_TX_DATA_ORDER,
			&iface->events);
		interrupting_iface |= 0x1 << i;
	}
	spin_unlock (&card->irq_lock);

	if (interrupting_iface & 0x1 << i) {
			wake_up (&iface->queue);
		}

	}

	for (i = 3; i < 4; i++) {
		p = p->next;
		iface = list_entry(p, struct master_iface, list);

	if (dmaintsrc & LSDMA_INTSRC_CH(i)) {
		struct lsdma_dma *dma = iface->dma;

		/* Read the interrupt type and clear it */
		spin_lock (&card->irq_lock);
		status = readl (card->bridge_addr + LSDMA_CSR(i));
		writel (status, card->bridge_addr + LSDMA_CSR(i));
		spin_unlock (&card->irq_lock);

		/* Increment the buffer pointer */
		if (status & LSDMA_CH_CSR_INTSRCBUFFER) {
			lsdma_advance (dma);
			if (lsdma_rx_isempty (dma)) {
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

		interrupting_iface |= 0x2;
	}

	spin_lock (&card->irq_lock);
	status = readl (card->core.addr + DVBM_Q3IO_ICSR(i));
	writel (status, card->core.addr + DVBM_Q3IO_ICSR(i));

	if (status & DVBM_Q3IO_ICSR_RXCDIS) {
		set_bit (ASI_EVENT_RX_CARRIER_ORDER,
			&iface->events);
		interrupting_iface |= 0x2;
	}
	if (status & DVBM_Q3IO_ICSR_RXAOSIS) {
		set_bit (ASI_EVENT_RX_AOS_ORDER,
			&iface->events);
		interrupting_iface |= 0x2;
	}
	if (status & DVBM_Q3IO_ICSR_RXLOSIS) {
		set_bit (ASI_EVENT_RX_LOS_ORDER,
			&iface->events);
		interrupting_iface |= 0x2;
	}
	if (status & DVBM_Q3IO_ICSR_RXOIS) {
		set_bit (ASI_EVENT_RX_FIFO_ORDER,
			&iface->events);
		interrupting_iface |= 0x2;
	}
	if (status & DVBM_Q3IO_ICSR_RXDIS) {
		set_bit (ASI_EVENT_RX_DATA_ORDER,
			&iface->events);
		interrupting_iface |= 0x2;
	}

	if (interrupting_iface & 0x2) {
		wake_up (&iface->queue);
	}
	 spin_unlock (&card->irq_lock);
	}


	/* Check and clear the source of the interrupt*/
	if (interrupting_iface) {
		/* Dummy read to flush PCI posted writes */
		readl (card->bridge_addr + LSDMA_INTMSK);

		return IRQ_HANDLED;
	}
	return IRQ_NONE;
}

/**
 * dvbm_q3io_txinit - Initialize the DVB Master Quad-1in3out transmitter
 * @iface: interface
 **/
static void
dvbm_q3io_txinit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg = iface->null_packets ? DVBM_Q3IO_TCSR_TNP : 0;


	switch (iface->timestamps) {
	default:
	case ASI_CTL_TSTAMP_NONE:
		reg |= 0;
		break;
	case ASI_CTL_TSTAMP_APPEND:
		reg |= DVBM_Q3IO_TCSR_TTSS;
		break;
	case ASI_CTL_TSTAMP_PREPEND:
		reg |= DVBM_Q3IO_TCSR_TPRC;
		break;
	}

	switch (iface->mode) {
	default:
	case ASI_CTL_TX_MODE_188:
		reg |= DVBM_Q3IO_TCSR_188;
		break;
	case ASI_CTL_TX_MODE_204:
		reg |= DVBM_Q3IO_TCSR_204;
		break;
	case ASI_CTL_TX_MODE_MAKE204:
		reg |= DVBM_Q3IO_TCSR_MAKE204;
		break;
	}
	switch (iface->clksrc) {
	default:
	case ASI_CTL_TX_CLKSRC_ONBOARD:
		reg |= 0;
		break;
	case ASI_CTL_TX_CLKSRC_EXT:
		reg |= DVBM_Q3IO_TCSR_EXTCLK;
		break;
	}

	/* There will be no races on IBSTR, IPSTR, FTR, and TCSR
	 * until this code returns, so we don't need to lock them */
	writel (reg | DVBM_Q3IO_TCSR_TXRST, card->core.addr + DVBM_Q3IO_TCSR(channel));
	wmb ();
	writel (reg, card->core.addr + DVBM_Q3IO_TCSR(channel));
	wmb ();
	writel ((DVBM_Q3IO_TFL << 16) | DVBM_Q3IO_TDMATL, card->core.addr + DVBM_Q3IO_TFCR(channel));
	writel (0, card->core.addr + DVBM_Q3IO_IBSTREG(channel));
	writel (0, card->core.addr + DVBM_Q3IO_IPSTREG(channel));
	writel (0, card->core.addr + DVBM_Q3IO_FTREG(channel));
	/* Reset byte counter */
	readl (card->core.addr + DVBM_Q3IO_TXBCOUNT (channel));
	return;
}

/**
 * dvbm_q3io_txstart - Activate the DVB Master Quad-1in3out transmitter
 * @iface: interface
 **/
static void
dvbm_q3io_txstart (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;
	const unsigned int channel = mdev_index (card, &iface->list);

	/* Enable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = readl (card->core.addr + DVBM_Q3IO_ICSR(channel)) &
		DVBM_Q3IO_ICSR_RX_IE_MASK;
	reg |= DVBM_Q3IO_ICSR_TXUIE | DVBM_Q3IO_ICSR_TXDIE;
	writel (reg, card->core.addr + DVBM_Q3IO_ICSR(channel));
	spin_unlock_irq(&card->irq_lock);

	/* Enable the transmitter */
	spin_lock(&card->reg_lock);
	reg = readl(card->core.addr + DVBM_Q3IO_TCSR(channel));
	writel(reg | DVBM_Q3IO_TCSR_TXE, card->core.addr + DVBM_Q3IO_TCSR(channel));
	spin_unlock(&card->reg_lock);

	return;

}

/**
 * dvbm_q3io_txstop - Deactivate the DVB Master Quad-1in3out transmitter
 * @iface: interface
 **/
static void
dvbm_q3io_txstop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	/* Disable the transmitter.
	 * There will be no races on TCSR here,
	 * so we don't need to lock it */
	spin_lock(&card->reg_lock);
	reg = readl(card->core.addr + DVBM_Q3IO_TCSR(channel));
	writel(reg & ~DVBM_Q3IO_TCSR_TXE, card->core.addr + DVBM_Q3IO_TCSR(channel));
	spin_unlock(&card->reg_lock);

	/* Disable transmitter interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = readl (card->core.addr + DVBM_Q3IO_ICSR(channel)) &
		DVBM_Q3IO_ICSR_RX_IE_MASK;
	reg |= DVBM_Q3IO_ICSR_TXUIS | DVBM_Q3IO_ICSR_TXDIS;
	writel (reg, card->core.addr + DVBM_Q3IO_ICSR(channel));
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
 * dvbm_q3io_txexit - Clean up the DVB Master Quad-1in3out transmitter
 * @iface: interface
 **/
static void
dvbm_q3io_txexit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);

	/* Reset the transmitter */
	writel (DVBM_Q3IO_TCSR_TXRST, card->core.addr + DVBM_Q3IO_TCSR(channel));

	return;
}

/**
 * dvbm_q3io_txopen - DVB Master Quad-1in3out transmitter open() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
dvbm_q3io_txopen (struct inode *inode, struct file *filp)
{
	return masterlsdma_open (inode,
		filp,
		dvbm_q3io_txinit,
		dvbm_q3io_txstart,
		0,
		0);
}

/**
 * dvbm_q3io_txunlocked_ioctl - DVB Master Quad-1in3out transmitter unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
dvbm_q3io_txunlocked_ioctl (struct file *filp,
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
		if (copy_from_user (&stuffing, (struct asi_txstuffing *)arg,
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
		writel (stuffing.ib, card->core.addr + DVBM_Q3IO_IBSTREG(channel));
		writel (stuffing.ip, card->core.addr + DVBM_Q3IO_IPSTREG(channel));
		writel ((stuffing.il_big << DVBM_Q3IO_FTR_ILBIG_SHIFT) |
			(stuffing.big_ip << DVBM_Q3IO_FTR_BIGIP_SHIFT) |
			(stuffing.il_normal << DVBM_Q3IO_FTR_ILNORMAL_SHIFT) |
			stuffing.normal_ip, card->core.addr + DVBM_Q3IO_FTREG(channel));
		spin_unlock (&card->reg_lock);
		break;
	case ASI_IOC_TXGETBYTECOUNT:
		if (!(iface->capabilities & ASI_CAP_TX_BYTECOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr + DVBM_Q3IO_TXBCOUNT(channel)),
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_TXGETTXD:
		/* Atomic read of ICSR, so we don't need to lock */
		if (put_user ((readl (card->core.addr + DVBM_Q3IO_ICSR(channel)) &
			DVBM_Q3IO_ICSR_TXD) ? 1 : 0, (int *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return asi_txioctl (iface, cmd, arg);
	}
	return 0;
}

/**
 * dvbm_q3io_txioctl- DVB Master Quad-1in3out transmitter ioctl() method
 * @inode: inode
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/

static int
dvbm_q3io_txioctl (struct inode *inode,
	struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	return dvbm_q3io_txunlocked_ioctl (filp, cmd, arg);
}


/**
 * dvbm_q3io_txfsync - DVB Master Quad-1in3out transmitter fsync() method
 * @filp: file to flush
 * @dentry: directory entry associated with the file
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
dvbm_q3io_txfsync (struct file *filp,
	struct dentry *dentry,
	int datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	struct lsdma_dma *dma = iface->dma;
	struct master_iface *txiface = list_entry (card->iface_list.next,
		struct master_iface, list);
	const unsigned int channel = mdev_index (card, &iface->list);

	if (down_interruptible (&iface->buf_sem)) {
		return -ERESTARTSYS;
	}
	lsdma_tx_link_all (dma);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	lsdma_reset (dma);

	if (!txiface->null_packets) {
		struct master_dev *card = iface->card;

		/* Wait for the onboard FIFOs to empty */
		/* Atomic read of ICSR, so we don't need to lock */
		wait_event (iface->queue,
			!(readl (card->core.addr + DVBM_Q3IO_ICSR(channel)) &
			DVBM_Q3IO_ICSR_TXD));
	}

	up (&iface->buf_sem);
	return 0;
}

/**
 * dvbm_q3io_txrelease - DVB Master Quad-1in3out transmitter release() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
dvbm_q3io_txrelease (struct inode *inode, struct file *filp)
{
	struct master_iface *iface = filp->private_data;

	return masterlsdma_release (iface, dvbm_q3io_txstop, dvbm_q3io_txexit);
}

/**
 * dvbm_q3io_rxinit - Initialize the DVB Master Quad-1in3out receiver
 * @iface: interface
 **/
static void
dvbm_q3io_rxinit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int i, reg = (iface->null_packets ? DVBM_Q3IO_RCSR_RNP : 0);

	switch (iface->timestamps) {
	default:
	case ASI_CTL_TSTAMP_NONE:
		reg |= 0;
		break;
	case ASI_CTL_TSTAMP_APPEND:
		reg |= DVBM_Q3IO_RCSR_APPEND;
		break;
	case ASI_CTL_TSTAMP_PREPEND:
		reg |= DVBM_Q3IO_RCSR_PREPEND;
		break;
	}

	switch (iface->mode) {
	default:
	case ASI_CTL_RX_MODE_RAW:
		reg |= 0;
		break;
	case ASI_CTL_RX_MODE_188:
		reg |= DVBM_Q3IO_RCSR_188 | DVBM_Q3IO_RCSR_PFE;
		break;
	case ASI_CTL_RX_MODE_204:
		reg |= DVBM_Q3IO_RCSR_204 | DVBM_Q3IO_RCSR_PFE;
		break;
	case ASI_CTL_RX_MODE_AUTO:
		reg |= DVBM_Q3IO_RCSR_AUTO | DVBM_Q3IO_RCSR_PFE;
		break;
	case ASI_CTL_RX_MODE_AUTOMAKE188:
		reg |= DVBM_Q3IO_RCSR_AUTO | DVBM_Q3IO_RCSR_RSS |
			DVBM_Q3IO_RCSR_PFE;
		break;
	case ASI_CTL_RX_MODE_204MAKE188:
		reg |= DVBM_Q3IO_RCSR_204 | DVBM_Q3IO_RCSR_RSS |
			DVBM_Q3IO_RCSR_PFE;
		break;
	}


	/* There will be no races on RCSR
	 * until this code returns, so we don't need to lock it */
	writel (reg | DVBM_Q3IO_RCSR_RXRST, card->core.addr + DVBM_Q3IO_RCSR(channel));
	wmb ();
	writel (reg, card->core.addr + DVBM_Q3IO_RCSR(channel));
	wmb ();
	writel (DVBM_Q3IO_RDMATL, card->core.addr + DVBM_Q3IO_RDMATLR(channel));

	/* Reset byte counter */
	readl (card->core.addr + DVBM_Q3IO_RXBCOUNT(channel));

	 /* Reset PID Filter.
	  * There will be no races on PFLUT
	  * until this code returns, so we don't need to lock it */
	for (i = 0; i < 256; i++) {
		writel (i, card->core.addr + DVBM_Q3IO_PFLUTWA(channel));
		/* Dummy read to flush PCI posted writes */
		readl (card->core.addr + DVBM_Q3IO_FPGAID);
		writel (0xffffffff, card->core.addr + DVBM_Q3IO_PFLUT(channel));
		/* Dummy read to flush PCI posted writes */
		readl (card->core.addr + DVBM_Q3IO_FPGAID);
	}

	/* Clear PID registers */
	writel (0, card->core.addr + DVBM_Q3IO_PID0(channel));
	writel (0, card->core.addr + DVBM_Q3IO_PID1(channel));
	writel (0, card->core.addr + DVBM_Q3IO_PID2(channel));
	writel (0, card->core.addr + DVBM_Q3IO_PID3(channel));

	/* Reset PID counters */
	readl (card->core.addr + DVBM_Q3IO_PIDCOUNT0(channel));
	readl (card->core.addr + DVBM_Q3IO_PIDCOUNT1(channel));
	readl (card->core.addr + DVBM_Q3IO_PIDCOUNT2(channel));
	readl (card->core.addr + DVBM_Q3IO_PIDCOUNT3(channel));

	return;
}

/**
 * dvbm_q3io_rxstart - Activate the DVB Master Quad-1in3out receiver
 * @iface: interface
 **/
static void
dvbm_q3io_rxstart (struct master_iface *iface)
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
	reg = readl (card->core.addr + DVBM_Q3IO_ICSR(channel)) &
		DVBM_Q3IO_ICSR_TX_IEIS_MASK;
	reg |= DVBM_Q3IO_ICSR_RXCDIE | DVBM_Q3IO_ICSR_RXAOSIE |
		DVBM_Q3IO_ICSR_RXLOSIE | DVBM_Q3IO_ICSR_RXOIE |
		DVBM_Q3IO_ICSR_RXDIE;
	writel (reg, card->core.addr + DVBM_Q3IO_ICSR(channel));
	spin_unlock_irq (&card->irq_lock);

	/* Enable the receiver */
	spin_lock (&card->reg_lock);
	reg = readl (card->core.addr + DVBM_Q3IO_RCSR(channel));
	writel (reg | DVBM_Q3IO_RCSR_RXE, card->core.addr + DVBM_Q3IO_RCSR(channel));
	spin_unlock (&card->reg_lock);

	return;
}

/**
 * dvbm_q3io_rxstop - Deactivate the DVB Master Quad-1in3out receiver
 * @iface: interface
 **/
static void
dvbm_q3io_rxstop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	/* Disable the receiver */
	spin_lock (&card->reg_lock);
	reg = readl (card->core.addr + DVBM_Q3IO_RCSR(channel));
	writel (reg & ~DVBM_Q3IO_RCSR_RXE, card->core.addr + DVBM_Q3IO_RCSR(channel));
	spin_unlock (&card->reg_lock);

	/* Disable receiver interrupts */
	spin_lock_irq (&card->irq_lock);
	reg = readl (card->core.addr + DVBM_Q3IO_ICSR(channel)) &
		DVBM_Q3IO_ICSR_TX_IEIS_MASK;
	reg |= DVBM_Q3IO_ICSR_RXCDIS | DVBM_Q3IO_ICSR_RXAOSIS |
		DVBM_Q3IO_ICSR_RXLOSIS | DVBM_Q3IO_ICSR_RXOIS |
		DVBM_Q3IO_ICSR_RXDIS;
	writel (reg, card->core.addr + DVBM_Q3IO_ICSR(channel));
	spin_unlock_irq (&card->irq_lock);

	/* Disable and abort DMA */
	writel ((LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION) & ~LSDMA_CH_CSR_ENABLE,
		card->bridge_addr + LSDMA_CSR(channel));
	wmb ();
	writel ((LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION | LSDMA_CH_CSR_STOP) &
		~LSDMA_CH_CSR_ENABLE,
		card->bridge_addr + LSDMA_CSR(channel));

	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	writel ((LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_DIRECTION) & ~LSDMA_CH_CSR_ENABLE,
		card->bridge_addr + LSDMA_CSR(channel));

	return;
}

/**
 * dvbm_q3io_rxexit - Clean up the DVB Master Quad-1in3out receiver
 * @iface: interface
 **/
static void
dvbm_q3io_rxexit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);

	/* Reset the receiver.
	 * There will be no races on RCSR here,
	 * so we don't need to lock it */
	writel (DVBM_Q3IO_RCSR_RXRST, card->core.addr + DVBM_Q3IO_RCSR(channel));

	return;
}

/**
 * dvbm_q3io_rxopen - DVB Master Quad-1in3out receiver open() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
dvbm_q3io_rxopen (struct inode *inode, struct file *filp)
{
	return masterlsdma_open (inode,
		filp,
		dvbm_q3io_rxinit,
		dvbm_q3io_rxstart,
		0,
		0);
}

/**
 * dvbm_q3io_rxunlocked_ioctl - DVB Master Quad-1in3out receiver unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
dvbm_q3io_rxunlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	int val;
	unsigned int reg = 0, pflut[256], i;

	switch (cmd) {
	case ASI_IOC_RXGETBUFLEVEL:
		if (put_user (lsdma_rx_buflevel (iface->dma),
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXGETSTATUS:
		/* Atomic reads of ICSR and RCSR, so we don't need to lock */
		reg = readl (card->core.addr + DVBM_Q3IO_ICSR(channel));
		switch (readl (card->core.addr + DVBM_Q3IO_RCSR(channel))  & DVBM_Q3IO_RCSR_SYNC_MASK)
			{
		case 0:
			val = 1;
			break;
		case DVBM_Q3IO_RCSR_188:
			val = (reg & DVBM_Q3IO_ICSR_RXPASSING) ? 188 : 0;
			break;
		case DVBM_Q3IO_RCSR_204:
			val = (reg & DVBM_Q3IO_ICSR_RXPASSING) ? 204 : 0;
			break;
		case DVBM_Q3IO_RCSR_AUTO:
			if (reg & DVBM_Q3IO_ICSR_RXPASSING) {
				val = (reg & DVBM_Q3IO_ICSR_RX204) ? 204 : 188;
			} else {
				val = 0;
			}
			break;

		default:
			return -EIO;
		}
		if (put_user (val, (int *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXGETBYTECOUNT:
		if (!(iface->capabilities & ASI_CAP_RX_BYTECOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr + DVBM_Q3IO_RXBCOUNT(channel)),
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETINVSYNC:
		if (get_user (val, (int *)arg)) {
			return -EFAULT;
		}
		switch (val) {
		case 0:
			reg |= 0;
			break;
		case 1:
			reg |= DVBM_Q3IO_RCSR_INVSYNC;
			break;
		default:
			return -EINVAL;
		}
		spin_lock (&card->reg_lock);
		writel ((readl (card->core.addr + DVBM_Q3IO_RCSR(channel)) &
			~DVBM_Q3IO_RCSR_INVSYNC) | reg,
			card->core.addr + DVBM_Q3IO_RCSR(channel));
		spin_unlock (&card->reg_lock);
		break;
	case ASI_IOC_RXGETCARRIER:
		/* Atomic read of ICSR, so we don't need to lock */
		if (put_user ((readl (card->core.addr + DVBM_Q3IO_ICSR(channel)) &
			DVBM_Q3IO_ICSR_RXCD) ? 1 : 0, (int *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETDSYNC:
		if (get_user (val, (int *)arg)) {
			return -EFAULT;
		}
		if (val) {
			return -EINVAL;
		}
		break;
	case ASI_IOC_RXGETRXD:
		/* Atomic read of ICSR, so we don't need to lock */
		if (put_user ((readl (card->core.addr + DVBM_Q3IO_ICSR(channel)) &
			DVBM_Q3IO_ICSR_RXD) ? 1 : 0, (int *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPF:
		if (!(iface->capabilities & ASI_CAP_RX_PIDFILTER)) {
			return -ENOTTY;
		}
		if (copy_from_user (pflut, (unsigned int *)arg,
			sizeof (unsigned int [256]))) {
			return -EFAULT;
		}
		spin_lock (&card->reg_lock);
		for (i = 0; i < 256; i++) {
			writel (i, card->core.addr + DVBM_Q3IO_PFLUTWA(channel));
			wmb ();
			writel (pflut[i], card->core.addr + DVBM_Q3IO_PFLUT(channel));
			wmb ();
		}
		spin_unlock (&card->reg_lock);
		break;
	case ASI_IOC_RXSETPID0:
		if (!(iface->capabilities & ASI_CAP_RX_PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (get_user (val, (int *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		writel (val, card->core.addr + DVBM_Q3IO_PID0(channel));
		/* Reset PID count */
		readl (card->core.addr + DVBM_Q3IO_PIDCOUNT0(channel));
		break;
	case ASI_IOC_RXGETPID0COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr +
			DVBM_Q3IO_PIDCOUNT0(channel)),
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPID1:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (get_user (val, (int *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		writel (val, card->core.addr + DVBM_Q3IO_PID1(channel));
		/* Reset PID count */
		readl (card->core.addr + DVBM_Q3IO_PIDCOUNT1(channel));
		break;
	case ASI_IOC_RXGETPID1COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr +
			DVBM_Q3IO_PIDCOUNT1(channel)),
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPID2:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (get_user (val, (int *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		writel (val, card->core.addr + DVBM_Q3IO_PID2(channel));
		/* Reset PID count */
		readl (card->core.addr + DVBM_Q3IO_PIDCOUNT2(channel));
		break;
	case ASI_IOC_RXGETPID2COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr +
			DVBM_Q3IO_PIDCOUNT2(channel)),
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPID3:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (get_user (val, (int *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		writel (val, card->core.addr + DVBM_Q3IO_PID3(channel));
		/* Reset PID count */
		readl (card->core.addr + DVBM_Q3IO_PIDCOUNT3(channel));
		break;
	case ASI_IOC_RXGETPID3COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_4PIDCOUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr +
			DVBM_Q3IO_PIDCOUNT3(channel)),
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXGET27COUNT:
		if (!(iface->capabilities & ASI_CAP_RX_27COUNTER)) {
			return -ENOTTY;
		}
		if (put_user (readl (card->core.addr + DVBM_Q3IO_27COUNTR),
			(unsigned int *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return asi_rxioctl (iface, cmd, arg);
	}
	return 0;
}

/**
 * dvbm_q3io_rxioctl- DVB Master Quad-1in3out Receiver ioctl() method
 * @inode: inode
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/

static int
dvbm_q3io_rxioctl (struct inode *inode,
	struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	return dvbm_q3io_rxunlocked_ioctl (filp, cmd, arg);
}


/**
 * dvbm_q3io_rxfsync - DVB Master Quad-1in3out receiver fsync() method
 * @filp: file to flush
 * @dentry: directory entry associated with the file
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
dvbm_q3io_rxfsync (struct file *filp,
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
	dvbm_q3io_rxstop (iface);

	/* Reset the onboard FIFO and driver buffers */
	spin_lock (&card->reg_lock);
	reg = readl (card->core.addr + DVBM_Q3IO_RCSR(channel));
	writel (reg | DVBM_Q3IO_RCSR_RXRST, card->core.addr + DVBM_Q3IO_RCSR(channel));
	wmb ();
	writel (reg, card->core.addr + DVBM_Q3IO_RCSR(channel));
	spin_unlock (&card->reg_lock);
	iface->events = 0;
	lsdma_reset (iface->dma);

	/* Start the receiver */
	dvbm_q3io_rxstart (iface);

	up (&iface->buf_sem);
	return 0;
}

/**
 * dvbm_q3io_rxrelease - DVB Master Quad-1in3out receiver release() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
dvbm_q3io_rxrelease (struct inode *inode, struct file *filp)
{
	struct master_iface *iface = filp->private_data;

	return masterlsdma_release (iface, dvbm_q3io_rxstop, dvbm_q3io_rxexit);
}

