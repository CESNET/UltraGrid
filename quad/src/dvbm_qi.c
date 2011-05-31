/* dvbm_qi.c
 *
 * Linux driver for Linear Systems Ltd. DVB Master Q/i.
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
#include <linux/module.h> /* THIS_MODULE */

#include <linux/fs.h> /* inode, file, file_operations */
#include <linux/sched.h> /* pt_regs */
#include <linux/pci.h> /* pci_resource_start () */
#include <linux/slab.h> /* kzalloc () */
#include <linux/list.h> /* INIT_LIST_HEAD () */
#include <linux/spinlock.h> /* spin_lock_init () */
#include <linux/init.h> /* __devinit */
#include <linux/errno.h> /* error codes */
#include <linux/delay.h> /* udelay () */
#include <linux/interrupt.h> /* irqreturn_t */
#include <linux/mutex.h> /* mutex_init () */

#include <asm/uaccess.h> /* put_user () */
#include <asm/bitops.h> /* set_bit () */

#include "asicore.h"
#include "../include/master.h"
#include "mdev.h"
#include "dvbm.h"
#include "mdma.h"
#include "gt64131.h"
#include "dvbm_qi.h"

static const char dvbm_qi_name[] = DVBM_NAME_DVBQI;

/* Static function prototypes */
static irqreturn_t IRQ_HANDLER(dvbm_qi_irq_handler,irq,dev_id,regs);
static void dvbm_qi_init (struct master_iface *iface);
static void dvbm_qi_start (struct master_iface *iface);
static void dvbm_qi_stop (struct master_iface *iface);
static void dvbm_qi_exit (struct master_iface *iface);
static long dvbm_qi_unlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int FSYNC_HANDLER(dvbm_qi_fsync,filp,datasync);

static struct file_operations dvbm_qi_fops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.read = asi_read,
	.poll = asi_rxpoll,
	.unlocked_ioctl = dvbm_qi_unlocked_ioctl,
	.compat_ioctl = dvbm_qi_unlocked_ioctl,
	.open = asi_open,
	.release = asi_release,
	.fsync = dvbm_qi_fsync,
	.fasync = NULL
};

static struct master_iface_operations dvbm_qi_ops = {
	.init = dvbm_qi_init,
	.start = dvbm_qi_start,
	.stop = dvbm_qi_stop,
	.exit = dvbm_qi_exit
};

/**
 * dvbm_qi_pci_probe - PCI insertion handler for a DVB Master Q/i
 * @pdev: PCI device
 *
 * Handle the insertion of a DVB Master Q/i.
 * Returns a negative error code on failure and 0 on success.
 **/
int __devinit
dvbm_qi_pci_probe (struct pci_dev *pdev)
{
	int err;
	unsigned int i, cap;
	struct master_dev *card;
	void __iomem *p;

	err = dvbm_pci_probe_generic (pdev);
	if (err < 0) {
		goto NO_PCI;
	}

	/* Initialize the driver_data pointer so that dvbm_qi_pci_remove()
	 * doesn't try to free it if an error occurs */
	pci_set_drvdata (pdev, NULL);

	/* Allocate a board info structure */
	if ((card = (struct master_dev *)
		kzalloc (sizeof (*card), GFP_KERNEL)) == NULL) {
		err = -ENOMEM;
		goto NO_MEM;
	}

	/* Initialize the board info structure */
	card->bridge_addr = ioremap_nocache (pci_resource_start (pdev, 4),
		pci_resource_len (pdev, 4));
	card->core.addr = ioremap_nocache (pci_resource_start (pdev, 2),
		pci_resource_len (pdev, 2));
	card->name = dvbm_qi_name;
	card->id = pdev->device;
	card->irq = pdev->irq;
	card->irq_handler = dvbm_qi_irq_handler;
	INIT_LIST_HEAD(&card->iface_list);
	/* Lock for ICSR[] */
	spin_lock_init (&card->irq_lock);
	/* Lock for HL2CR, CSR[], PFLUT[] */
	spin_lock_init (&card->reg_lock);
	mutex_init (&card->users_mutex);
	card->parent = &pdev->dev;

	/* Setup the bridge */
	writel (DVBM_QI_TURNOFF |
		DVBM_QI_ACCTOFIRST << 3 |
		DVBM_QI_ACCTONEXT << 7 |
		DVBM_QI_ALETOWR << 11 |
		DVBM_QI_WRACTIVE << 14 |
		DVBM_QI_WRHIGH << 17 |
		DVBM_QI_DEVWIDTH << 20 |
		DVBM_QI_BOOTDMAFLYBY << 22 |
		DVBM_QI_DEVLOC << 23 |
		DVBM_QI_DMAFLYBY << 26,
		card->bridge_addr + GT64_DEVICE(0));
	writel (DVBM_QI_TIMEOUT0 |
		DVBM_QI_TIMEOUT1 << 8 |
		DVBM_QI_RETRYCTR << 16,
		card->bridge_addr + GT64_PCITOR);
	writel (0x000000ff, card->bridge_addr + GT64_SCS0LDA);
	writel (0x00000000, card->bridge_addr + GT64_SCS0HDA);
	writel (0x000000ff, card->bridge_addr + GT64_SCS1LDA);
	writel (0x00000000, card->bridge_addr + GT64_SCS1HDA);
	writel (0x000000ff, card->bridge_addr + GT64_SCS2LDA);
	writel (0x00000000, card->bridge_addr + GT64_SCS2HDA);
	writel (0x000000ff, card->bridge_addr + GT64_SCS3LDA);
	writel (0x00000000, card->bridge_addr + GT64_SCS3HDA);
	writel (DVBM_QI_LDA, card->bridge_addr + GT64_CS0LDA);
	writel (DVBM_QI_HDA, card->bridge_addr + GT64_CS0HDA);
	writel (0x000000ff, card->bridge_addr + GT64_CS1LDA);
	writel (0x00000000, card->bridge_addr + GT64_CS1HDA);
	writel (0x000000ff, card->bridge_addr + GT64_CS2LDA);
	writel (0x00000000, card->bridge_addr + GT64_CS2HDA);
	writel (0x000000ff, card->bridge_addr + GT64_CS3LDA);
	writel (0x00000000, card->bridge_addr + GT64_CS3HDA);
	writel (0x00000015, card->bridge_addr + GT64_SDRAM(1));

	/* Reconfigure the FPGA */
	writel (0x00000000, card->bridge_addr + GT64_BOOTCSLDA);
	writel (0x000000ff, card->bridge_addr + GT64_BOOTCSHDA);
	p = ioremap_nocache (pci_resource_start (pdev, 3),
		pci_resource_len (pdev, 3));
	writel (0x00000069, p);
	/* Dummy read to flush PCI posted writes */
	readl (p);
	/* Hold nCONFIG low for 40 us */
	udelay (40L);
	writel (0x00000078, p);
	/* Dummy read to flush PCI posted writes */
	readl (p);
	/* Wait for CONF_DONE to go high */
	msleep (200);
	while (!(readl (p) & 0x00000040)) {
		set_current_state (TASK_UNINTERRUPTIBLE);
		schedule_timeout (1);
	}
	/* Wait 14 us for INIT_DONE to go high */
	udelay (14L);
	iounmap (p);
	writel (0x000000ff, card->bridge_addr + GT64_BOOTCSLDA);
	writel (0x00000000, card->bridge_addr + GT64_BOOTCSHDA);

	/* Get the firmware version and flush PCI posted writes */
	card->version = readl (card->core.addr + DVBM_QI_FPGAID) & 0x0000ffff;
	printk (KERN_INFO "%s: %s detected, firmware version %u.%u (0x%04X)\n",
		dvbm_driver_name, card->name,
		card->version >> 8, card->version & 0x00ff, card->version);

	/* Store the pointer to the board info structure
	 * in the PCI info structure */
	pci_set_drvdata (pdev, card);

	/* Initialize the device */
	writel (0, card->core.addr + DVBM_QI_HL2CR);
	wmb ();
	writel (DVBM_QI_HL2CR_TRSTZ_N | DVBM_QI_HL2CR_RXLE |
		DVBM_QI_HL2CR_MIE,
		card->core.addr + DVBM_QI_HL2CR);

	/* Register a DVB Master device */
	if ((err = dvbm_register (card)) < 0) {
		goto NO_DEV;
	}

	/* Register the receive interfaces */
	cap = ASI_CAP_RX_SYNC | ASI_CAP_RX_MAKE188 |
		ASI_CAP_RX_BYTECOUNTER |
		ASI_CAP_RX_CD |
		ASI_CAP_RX_DSYNC | ASI_CAP_RX_DATA |
		ASI_CAP_RX_PIDFILTER | ASI_CAP_RX_PIDCOUNTER |
		ASI_CAP_RX_4PIDCOUNTER |
		ASI_CAP_RX_27COUNTER |
		ASI_CAP_RX_TIMESTAMPS;
	if (card->version >= 0x0103) {
		cap |= ASI_CAP_RX_PTIMESTAMPS;
	}
	if (card->version >= 0x0200) {
		cap &= ~ASI_CAP_RX_DSYNC;
		cap |= ASI_CAP_RX_NULLPACKETS;
	}
	for (i = 0; i < 4; i++) {
		if ((err = asi_register_iface (card,
			&gt64_dma_ops,
			0x1c000000 + DVBM_QI_FIFO(i),
			MASTER_DIRECTION_RX,
			&dvbm_qi_fops,
			&dvbm_qi_ops,
			cap,
			8,
			ASI_CTL_TRANSPORT_DVB_ASI)) < 0) {
			goto NO_IFACE;
		}
	}

	return 0;

NO_IFACE:
NO_DEV:
NO_MEM:
	dvbm_qi_pci_remove (pdev);
NO_PCI:
	return err;
}

/**
 * dvbm_qi_pci_remove - PCI removal handler for a DVB Master Q/i
 * @pdev: PCI device
 *
 * Handle the removal of a DVB Master Q/i.
 * This function may be called during PCI probe error handling,
 * so don't mark it as __devexit.
 **/
void
dvbm_qi_pci_remove (struct pci_dev *pdev)
{
	struct master_dev *card = pci_get_drvdata (pdev);

	if (card) {
		/* Unregister the device and all interfaces */
		dvbm_unregister_all (card);

		writel (0, card->core.addr + DVBM_QI_HL2CR);
		iounmap (card->core.addr);
		iounmap (card->bridge_addr);
		kfree (card);
	}
	dvbm_pci_remove_generic (pdev);
	return;
}

/**
 * dvbm_qi_irq_handler - DVB Master Q/i interrupt service routine
 * @irq: interrupt number
 * @dev_id: pointer to the device data structure
 * @regs: processor context
 **/
static irqreturn_t
IRQ_HANDLER(dvbm_qi_irq_handler,irq,dev_id,regs)
{
	struct master_dev *card = dev_id;
	struct list_head *p = &card->iface_list;
	struct master_iface *iface;
	struct master_dma *dma;
	unsigned int status, interrupting_iface = 0, i;

	for (i = 0; i < 4; i++) {
		p = p->next;

		/* Clear the interrupt source */
		spin_lock (&card->irq_lock);
		status = readl (card->core.addr + DVBM_QI_ICSR(i));
		if ((status & DVBM_QI_ICSR_ISMASK) == 0) {
			spin_unlock (&card->irq_lock);
			continue;
		}
		writel (status, card->core.addr + DVBM_QI_ICSR(i));
		spin_unlock (&card->irq_lock);

		iface = list_entry (p, struct master_iface, list);
		if (status & DVBM_QI_ICSR_DTIS) {
			/* Assume that the transfer has completed */

			dma = iface->dma;

			/* Increment the buffer pointer */
			mdma_advance (dma);

			if (mdma_rx_isempty (dma)) {
				set_bit (ASI_EVENT_RX_BUFFER_ORDER,
					&iface->events);
			}

			interrupting_iface |= (0x1 << i);
		}
		if (status & DVBM_QI_ICSR_CDIS) {
			set_bit (ASI_EVENT_RX_CARRIER_ORDER, &iface->events);
			interrupting_iface |= (0x1 << i);
		}
		if (status & DVBM_QI_ICSR_AOSIS) {
			set_bit (ASI_EVENT_RX_AOS_ORDER, &iface->events);
			interrupting_iface |= (0x1 << i);
		}
		if (status & DVBM_QI_ICSR_LOSIS) {
			set_bit (ASI_EVENT_RX_LOS_ORDER, &iface->events);
			interrupting_iface |= (0x1 << i);
		}
		if (status & DVBM_QI_ICSR_OIS) {
			set_bit (ASI_EVENT_RX_FIFO_ORDER, &iface->events);
			interrupting_iface |= (0x1 << i);
		}
		if (status & DVBM_QI_ICSR_DIS) {
			set_bit (ASI_EVENT_RX_DATA_ORDER, &iface->events);
			interrupting_iface |= (0x1 << i);
		}
		if (interrupting_iface & (0x1 << i)) {
			wake_up (&iface->queue);
		}
	}

	if (interrupting_iface) {
		/* Dummy read to flush PCI posted writes */
		readl (card->core.addr + DVBM_QI_ICSR(3));
		return IRQ_HANDLED;
	}
	return IRQ_NONE;
}

/**
 * dvbm_qi_init - Initialize the DVB Master Q/i receiver
 * @iface: interface
 **/
static void
dvbm_qi_init (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg, i;

	/* Enable the HOTLink II channel */
	spin_lock (&card->reg_lock);
	reg = readl (card->core.addr + DVBM_QI_HL2CR);
	writel (reg | DVBM_QI_HL2CR_BOE(channel),
		card->core.addr + DVBM_QI_HL2CR);
	spin_unlock (&card->reg_lock);
	/* Dummy read to flush PCI posted writes */
	readl (card->core.addr + DVBM_QI_HL2CR);
	/* Wait for 10 ms */
	msleep (10);

	reg = DVBM_QI_CSR_DSYNC | (iface->null_packets ? DVBM_QI_CSR_RNP : 0);
	switch (iface->timestamps) {
	default:
	case ASI_CTL_TSTAMP_NONE:
		reg |= 0;
		break;
	case ASI_CTL_TSTAMP_APPEND:
		reg |= DVBM_QI_CSR_TSE;
		break;
	case ASI_CTL_TSTAMP_PREPEND:
		reg |= DVBM_QI_CSR_PTSE;
		break;
	}
	switch (iface->mode) {
	default:
	case ASI_CTL_RX_MODE_RAW:
		reg |= 0;
		break;
	case ASI_CTL_RX_MODE_188:
		reg |= DVBM_QI_CSR_188 | DVBM_QI_CSR_PFE;
		break;
	case ASI_CTL_RX_MODE_204:
		reg |= DVBM_QI_CSR_204 | DVBM_QI_CSR_PFE;
		break;
	case ASI_CTL_RX_MODE_AUTO:
		reg |= DVBM_QI_CSR_AUTO | DVBM_QI_CSR_PFE;
		break;
	case ASI_CTL_RX_MODE_AUTOMAKE188:
		reg |= DVBM_QI_CSR_AUTOMAKE188 | DVBM_QI_CSR_PFE;
		break;
	case ASI_CTL_RX_MODE_204MAKE188:
		reg |= DVBM_QI_CSR_204MAKE188 | DVBM_QI_CSR_PFE;
		break;
	}
	/* There will be no races on CSR
	 * until this code returns, so we don't need to lock it */
	writel (reg | DVBM_QI_CSR_RST, card->core.addr + DVBM_QI_CSR(channel));
	/* Dummy read to flush PCI posted writes */
	readl (card->core.addr + DVBM_QI_CSR(channel));
	writel (reg, card->core.addr + DVBM_QI_CSR(channel));
	/* Dummy read to flush PCI posted writes */
	readl (card->core.addr + DVBM_QI_CSR(channel));
	writel (DVBM_QI_DMATL, card->core.addr + DVBM_QI_DMATLR(channel));

	/* Reset the byte counter */
	readl (card->core.addr + DVBM_QI_BCOUNTR(channel));

	/* Reset the PID filter.
	 * There will be no races on PFLUT
	 * until this code returns, so we don't need to lock it */
	for (i = 0; i < 256; i++) {
		writel (i, card->core.addr + DVBM_QI_PFLUTAR(channel));
		/* Dummy read to flush PCI posted writes */
		readl (card->core.addr + DVBM_QI_FPGAID);
		writel (0xffffffff, card->core.addr + DVBM_QI_PFLUTR(channel));
		/* Dummy read to flush PCI posted writes */
		readl (card->core.addr + DVBM_QI_FPGAID);
	}

	/* Clear PID registers */
	writel (0, card->core.addr + DVBM_QI_PIDR0(channel));
	writel (0, card->core.addr + DVBM_QI_PIDR1(channel));
	writel (0, card->core.addr + DVBM_QI_PIDR2(channel));
	writel (0, card->core.addr + DVBM_QI_PIDR3(channel));

	/* Reset PID counters */
	readl (card->core.addr + DVBM_QI_PIDCOUNTR0(channel));
	readl (card->core.addr + DVBM_QI_PIDCOUNTR1(channel));
	readl (card->core.addr + DVBM_QI_PIDCOUNTR2(channel));
	readl (card->core.addr + DVBM_QI_PIDCOUNTR3(channel));

	writel (iface->bufsize, card->core.addr + DVBM_QI_DTSR(channel));
	/* Dummy read to flush PCI posted writes */
	readl (card->core.addr + DVBM_QI_DTSR(channel));
	return;
}

/**
 * dvbm_qi_start - Activate the DVB Master Q/i receiver
 * @iface: interface
 **/
static void
dvbm_qi_start (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	/* Enable and start DMA */
	spin_lock (&card->reg_lock);
	reg = readl (card->core.addr + DVBM_QI_HL2CR);
	writel (reg | DVBM_QI_HL2CR_MDREQ,
		card->core.addr + DVBM_QI_HL2CR);
	/* Dummy read to flush PCI posted writes */
	readl (card->core.addr + DVBM_QI_HL2CR);
	udelay (10L);
	writel (gt64_head_desc_bus_addr (iface->dma),
		card->bridge_addr + GT64_DMANRP(channel));
	clear_bit (0, &iface->dma_done);
	wmb ();
	writel (GT64_DMACTL_RLP |
		GT64_DMACTL_DLP |
		GT64_DMACTL_MDREQ |
		GT64_DMACTL_FETNEXREC |
		GT64_DMACTL_CHANEN |
		GT64_DMACTL_INTMODE |
		GT64_DMACTL_DATTRANSLIM32 |
		GT64_DMACTL_DESTDIRINC |
		GT64_DMACTL_SRCDIRHOLD,
		card->bridge_addr + GT64_DMACTL(channel));
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + GT64_DMACTL(channel));
	writel (reg, card->core.addr + DVBM_QI_HL2CR);
	spin_unlock (&card->reg_lock);

	/* Enable interrupts */
	spin_lock_irq (&card->irq_lock);
	writel (DVBM_QI_ICSR_IEMASK, card->core.addr + DVBM_QI_ICSR(channel));
	spin_unlock_irq (&card->irq_lock);
	/* Dummy read to flush PCI posted writes */
	readl (card->core.addr + DVBM_QI_ICSR(channel));

	/* Enable the receiver */
	spin_lock (&card->reg_lock);
	reg = readl (card->core.addr + DVBM_QI_CSR(channel));
	writel (reg | DVBM_QI_CSR_EN, card->core.addr + DVBM_QI_CSR(channel));
	spin_unlock (&card->reg_lock);
	/* Dummy read to flush PCI posted writes */
	readl (card->core.addr + DVBM_QI_FPGAID);

	return;
}

/**
 * dvbm_qi_stop - Deactivate the DVB Master Q/i receiver
 * @iface: interface
 **/
static void
dvbm_qi_stop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	/* Disable the receiver */
	spin_lock (&card->reg_lock);
	reg = readl (card->core.addr + DVBM_QI_CSR(channel));
	writel (reg & ~DVBM_QI_CSR_EN, card->core.addr + DVBM_QI_CSR(channel));
	spin_unlock (&card->reg_lock);

	/* Disable interrupts */
	spin_lock_irq (&card->irq_lock);
	writel (DVBM_QI_ICSR_ISMASK, card->core.addr + DVBM_QI_ICSR(channel));
	spin_unlock_irq (&card->irq_lock);

	/* Abort and disable DMA */
	spin_lock (&card->reg_lock);
	reg = readl (card->core.addr + DVBM_QI_HL2CR);
	writel (reg | DVBM_QI_HL2CR_MDREQ,
		card->core.addr + DVBM_QI_HL2CR);
	/* Dummy read to flush PCI posted writes */
	readl (card->core.addr + DVBM_QI_HL2CR);
	udelay (10L);
	writel (GT64_DMACTL_RLP |
		GT64_DMACTL_DLP |
		GT64_DMACTL_MDREQ |
		GT64_DMACTL_INTMODE |
		GT64_DMACTL_DATTRANSLIM32 |
		GT64_DMACTL_DESTDIRINC |
		GT64_DMACTL_SRCDIRHOLD,
		card->bridge_addr + GT64_DMACTL(channel));
	/* Poll DMA status and flush PCI posted writes */
	while (readl (card->bridge_addr + GT64_DMACTL(channel)) &
		GT64_DMACTL_DMAACTST) {
	}
	writel (reg, card->core.addr + DVBM_QI_HL2CR);
	spin_unlock (&card->reg_lock);

	return;
}

/**
 * dvbm_qi_exit - Clean up the DVB Master Q/i receiver
 * @iface: interface
 **/
static void
dvbm_qi_exit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	/* Reset the receiver.
	 * There will be no races on CSR here,
	 * so we don't need to lock it */
	writel (DVBM_QI_CSR_RST, card->core.addr + DVBM_QI_CSR(channel));

	/* Disable the HOTLink II channel */
	spin_lock (&card->reg_lock);
	reg = readl (card->core.addr + DVBM_QI_HL2CR);
	writel (reg & ~DVBM_QI_HL2CR_BOE(channel),
		card->core.addr + DVBM_QI_HL2CR);
	spin_unlock (&card->reg_lock);

	return;
}

/**
 * dvbm_qi_unlocked_ioctl - DVB Master Q/i unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
dvbm_qi_unlocked_ioctl (struct file *filp,
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
		/* Atomic reads of CSR and ICSR, so we don't need to lock */
		reg = readl (card->core.addr + DVBM_QI_ICSR(channel));
		switch (readl (card->core.addr + DVBM_QI_CSR(channel)) &
			DVBM_QI_CSR_SYNCMASK) {
		case 0:
			val = 1;
			break;
		case DVBM_QI_CSR_188:
			val = (reg & DVBM_QI_ICSR_PASSING) ? 188 : 0;
			break;
		case DVBM_QI_CSR_204:
		case DVBM_QI_CSR_204MAKE188:
			val = (reg & DVBM_QI_ICSR_PASSING) ? 204 : 0;
			break;
		case DVBM_QI_CSR_AUTO:
		case DVBM_QI_CSR_AUTOMAKE188:
			if (reg & DVBM_QI_ICSR_PASSING) {
				val = (reg & DVBM_QI_ICSR_204) ? 204 : 188;
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
		if (put_user (
			readl (card->core.addr + DVBM_QI_BCOUNTR(channel)),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETINVSYNC:
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
	case ASI_IOC_RXGETCARRIER:
		/* Atomic read of ICSR, so we don't need to lock */
		if (put_user (
			(readl (card->core.addr + DVBM_QI_ICSR(channel)) &
			DVBM_QI_ICSR_CD) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETDSYNC:
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if (iface->capabilities & ASI_CAP_RX_DSYNC) {
			switch (val) {
			case 0:
				reg |= 0;
				break;
			case 1:
				reg |= DVBM_QI_CSR_DSYNC;
				break;
			default:
				return -EINVAL;
			}
			spin_lock (&card->reg_lock);
			writel ((readl (card->core.addr +
				DVBM_QI_CSR(channel)) &
				~DVBM_QI_CSR_DSYNC) | reg,
				card->core.addr + DVBM_QI_CSR(channel));
			spin_unlock (&card->reg_lock);
		} else {
			if (val) {
				return -EINVAL;
			}
		}
		break;
	case ASI_IOC_RXGETRXD:
		/* Atomic read of ICSR, so we don't need to lock */
		if (put_user (
			(readl (card->core.addr + DVBM_QI_ICSR(channel)) &
			DVBM_QI_ICSR_DATA) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPF:
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
			writel (i, card->core.addr + DVBM_QI_PFLUTAR(channel));
			/* Dummy read to flush PCI posted writes */
			readl (card->core.addr + DVBM_QI_FPGAID);
			writel (pflut[i],
				card->core.addr + DVBM_QI_PFLUTR(channel));
			/* Dummy read to flush PCI posted writes */
			readl (card->core.addr + DVBM_QI_FPGAID);
		}
		spin_unlock (&card->reg_lock);
		kfree (pflut);
		break;
	case ASI_IOC_RXSETPID0:
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		writel (val, card->core.addr + DVBM_QI_PIDR0(channel));
		/* Reset PID count */
		readl (card->core.addr + DVBM_QI_PIDCOUNTR0(channel));
		break;
	case ASI_IOC_RXGETPID0COUNT:
		if (put_user (
			readl (card->core.addr + DVBM_QI_PIDCOUNTR0(channel)),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPID1:
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		writel (val, card->core.addr + DVBM_QI_PIDR1(channel));
		/* Reset PID count */
		readl (card->core.addr + DVBM_QI_PIDCOUNTR1(channel));
		break;
	case ASI_IOC_RXGETPID1COUNT:
		if (put_user (
			readl (card->core.addr + DVBM_QI_PIDCOUNTR1(channel)),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPID2:
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		writel (val, card->core.addr + DVBM_QI_PIDR2(channel));
		/* Reset PID count */
		readl (card->core.addr + DVBM_QI_PIDCOUNTR2(channel));
		break;
	case ASI_IOC_RXGETPID2COUNT:
		if (put_user (
			readl (card->core.addr + DVBM_QI_PIDCOUNTR2(channel)),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXSETPID3:
		if (get_user (val, (int __user *)arg)) {
			return -EFAULT;
		}
		if ((val < 0) || (val > 0x00001fff)) {
			return -EINVAL;
		}
		writel (val, card->core.addr + DVBM_QI_PIDR3(channel));
		/* Reset PID count */
		readl (card->core.addr + DVBM_QI_PIDCOUNTR3(channel));
		break;
	case ASI_IOC_RXGETPID3COUNT:
		if (put_user (
			readl (card->core.addr + DVBM_QI_PIDCOUNTR3(channel)),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case ASI_IOC_RXGET27COUNT:
		if (put_user (readl (card->core.addr + DVBM_QI_27COUNTR),
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
 * dvbm_qi_fsync - DVB Master Q/i fsync() method
 * @filp: file to flush
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
FSYNC_HANDLER(dvbm_qi_fsync,filp,datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	mutex_lock (&iface->buf_mutex);

	/* Stop the receiver */
	dvbm_qi_stop (iface);

	/* Reset the onboard FIFO and driver buffers */
	spin_lock (&card->reg_lock);
	reg = readl (card->core.addr + DVBM_QI_CSR(channel));
	writel (reg | DVBM_QI_CSR_RST, card->core.addr + DVBM_QI_CSR(channel));
	writel (reg, card->core.addr + DVBM_QI_CSR(channel));
	spin_unlock (&card->reg_lock);
	iface->events = 0;
	gt64_reset (iface->dma);

	/* Start the receiver */
	dvbm_qi_start (iface);

	mutex_unlock (&iface->buf_mutex);
	return 0;
}

