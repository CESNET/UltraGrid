/* hdsdim_qie.c
 *
 * Linux driver functions for Linear Systems Ltd. QuadPort H/i.
 *
 * Copyright (C) 2009-2010 Linear Systems Ltd.
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
#include <linux/device.h> /* device_create_file () */
#include <linux/mutex.h> /* mutex_init () */

#include <asm/uaccess.h> /* put_user () */
#include <asm/bitops.h> /* set_bit () */

#include "sdiaudiocore.h"
#include "sdivideocore.h"
#include "../include/master.h"
#include "mdma.h"
#include "mdev.h"
#include "lsdma.h"
#include "hdsdim_qie.h"
#include "hdsdim.h"

static const char hdsdim_qie_name[] = HDSDIM_NAME_QIE;

/* Static function prototypes */
static ssize_t hdsdim_qie_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static irqreturn_t IRQ_HANDLER(hdsdim_qie_irq_handler,irq,dev_id,regs);
static void hdsdim_qie_init (struct master_iface *iface);
static void hdsdim_qie_start (struct master_iface *iface);
static void hdsdim_qie_stop (struct master_iface *iface);
static void hdsdim_qie_exit (struct master_iface *iface);
static long hdsdim_qie_unlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static long hdsdim_qie_unlocked_audioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int FSYNC_HANDLER(hdsdim_qie_fsync,filp,datasync);

static struct file_operations hdsdim_qie_vidfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.read = sdivideo_read,
	.poll = sdivideo_rxpoll,
	.unlocked_ioctl = hdsdim_qie_unlocked_ioctl,
	.compat_ioctl = hdsdim_qie_unlocked_ioctl,
	.mmap = sdivideo_mmap,
	.open = sdivideo_open,
	.release = sdivideo_release,
	.fsync = hdsdim_qie_fsync,
	.fasync = NULL
};

static struct file_operations hdsdim_qie_audfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.read = sdiaudio_read,
	.poll = sdiaudio_rxpoll,
	.unlocked_ioctl = hdsdim_qie_unlocked_audioctl,
	.compat_ioctl = hdsdim_qie_unlocked_audioctl,
	.mmap = sdiaudio_mmap,
	.open = sdiaudio_open,
	.release = sdiaudio_release,
	.fsync = hdsdim_qie_fsync,
	.fasync = NULL
};

static struct master_iface_operations hdsdim_qie_ops = { //debug: check if aud and vid can share these functions
	.init = hdsdim_qie_init,
	.start = hdsdim_qie_start,
	.stop = hdsdim_qie_stop,
	.exit = hdsdim_qie_exit
};

/**
 * hdsdim_qie_show_uid - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
static ssize_t
hdsdim_qie_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	return snprintf (buf, PAGE_SIZE, "0x%08X%08X\n",
		readl (card->core.addr + HDSDIM_QIE_UIDR_HI),
		readl (card->core.addr + HDSDIM_QIE_UIDR_LO));
}

static DEVICE_ATTR(uid,S_IRUGO,
	hdsdim_qie_show_uid,NULL);

/**
 * hdsdim_qie_pci_probe - PCI insertion handler for a QuadPort H/i
 * @pdev: PCI device
 *
 * Handle the insertion of a QuadPort H/i.
 * Returns a negative error code on failure and 0 on success.
 **/
int __devinit
hdsdim_qie_pci_probe (struct pci_dev *pdev)
{
	int err;
	unsigned int i, vidcap, audcap;
	struct master_dev *card;

	err = hdsdim_pci_probe_generic (pdev);
	if (err < 0) {
		goto NO_PCI;
	}

	/* Initialize the driver_data pointer so that hdsdim_qie_pci_remove()
	 * doesn't try to free it if an error occurs */
	pci_set_drvdata (pdev, NULL);

	/* Allocate and initialize a board info structure */
	if ((card = (struct master_dev *)
		kzalloc (sizeof (*card), GFP_KERNEL)) == NULL) {
		err = -ENOMEM;
		goto NO_MEM;
	}

	card->bridge_addr = ioremap_nocache (pci_resource_start (pdev, 2),
		pci_resource_len (pdev, 2));
	/* SDI Core */
	card->core.addr = ioremap_nocache (pci_resource_start (pdev, 0),
		pci_resource_len (pdev, 0));
	card->version = readl (card->core.addr + HDSDIM_QIE_FPGAID) & 0xffff;
	card->name = hdsdim_qie_name;
	card->id = pdev->device;
	card->irq = pdev->irq;
	card->irq_handler = hdsdim_qie_irq_handler;
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
		hdsdim_driver_name, card->name,
		card->version >> 8, card->version & 0x00ff, card->version);

	/* Store the pointer to the board info structure
	 * in the PCI info structure */
	pci_set_drvdata (pdev, card);

	/* Reset the FPGA */
	for (i = 0; i < 8; i++) {
		writel (HDSDIM_QIE_RCSR_RST, card->core.addr + HDSDIM_QIE_RCSR(i));
	}

	/* Setup the LS DMA controller
	 * Even channels for video, odd channels for audio
	 */
	writel (LSDMA_INTMSK_CH(0) | LSDMA_INTMSK_CH(1) |
		LSDMA_INTMSK_CH(2) | LSDMA_INTMSK_CH(3) |
		LSDMA_INTMSK_CH(4) | LSDMA_INTMSK_CH(5) |
		LSDMA_INTMSK_CH(6) | LSDMA_INTMSK_CH(7),
		card->bridge_addr + LSDMA_INTMSK);
	for (i = 0; i < 8; i++) {
		writel (LSDMA_CH_CSR_INTDONEENABLE |
			LSDMA_CH_CSR_INTSTOPENABLE |
			LSDMA_CH_CSR_64BIT |
			LSDMA_CH_CSR_DIRECTION,
			card->bridge_addr + LSDMA_CSR(i));
	}

	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);

	/* Register a VidPort device */
	if ((err = hdsdim_register (card)) < 0) {
		goto NO_DEV;
	}

	/* Add device attributes */
	if (card->capabilities & MASTER_CAP_UID) {
		if ((err = device_create_file (card->dev,
			&dev_attr_uid)) < 0) {
			printk (KERN_WARNING
				"%s: Unable to create file 'uid'\n",
				hdsdim_driver_name);
		}
	}

	/* Register video receiver and audio receiver interfaces */
	vidcap = SDIVIDEO_CAP_RX_DATA | SDIVIDEO_CAP_RX_CD |
			SDIVIDEO_CAP_RX_ERR_COUNT |
			SDIVIDEO_CAP_RX_RAWMODE | SDIVIDEO_CAP_RX_DEINTERLACING;

	audcap = SDIAUDIO_CAP_RX_CD | SDIAUDIO_CAP_RX_DATA |
		SDIAUDIO_CAP_RX_STATS | SDIAUDIO_CAP_RX_24BIT;

//	if (card->version >= 0x0003) {
//			vidcap |= SDIVIDEO_CAP_RX_VBI;
//		} //debug: temporarily disable this, not supported on current release

	for (i = 0; i < 4; i++) {
		/* Channel 0,2,4,6 for video */
		if ((err = sdivideo_register_iface (card,
			&lsdma_dma_ops,
			0,
			MASTER_DIRECTION_RX,
			&hdsdim_qie_vidfops,
			&hdsdim_qie_ops,
			vidcap,
			4)) < 0) {
			goto NO_IFACE;
		}
		/* Channel 1,3,5,7 for audio */
		if ((err = sdiaudio_register_iface (card,
			&lsdma_dma_ops,
			0,
			MASTER_DIRECTION_RX,
			&hdsdim_qie_audfops,
			&hdsdim_qie_ops,
			audcap,
			4)) < 0) {
			goto NO_IFACE;
		}
	}

	return 0;

NO_IFACE:
NO_DEV:
NO_MEM:
	hdsdim_qie_pci_remove (pdev);
NO_PCI:
	return err;
}

/**
 * hdsdim_qie_pci_remove - PCI removal handler for a QuadPort H/i.
 * @pdev: PCI device
 *
 * Handle the removal of a QuadPort H/i.
 * This function may be called during PCI probe error handling,
 * so don't mark it as __devexit.
 **/
void
hdsdim_qie_pci_remove (struct pci_dev *pdev)
{
	struct master_dev *card = pci_get_drvdata (pdev);

	if (card) {
		/* Unregister the device and all interfaces */
		hdsdim_unregister_all (card);

		iounmap (card->core.addr);
		iounmap (card->bridge_addr);
		kfree (card);
	}
	hdsdim_pci_remove_generic (pdev);
	return;
}

/**
 * hdsdim_qie_irq_handler - QuadPort H/i interrupt service routine
 * @irq: interrupt number
 * @dev_id: pointer to the device data structure
 * @regs: processor context
 **/
static irqreturn_t
IRQ_HANDLER(hdsdim_qie_irq_handler,irq,dev_id,regs)
{
	struct master_dev *card = dev_id;
	struct list_head *p = &card->iface_list;
	struct master_iface *iface;
	unsigned int dmaintsrc = readl (card->bridge_addr + LSDMA_INTSRC);
	unsigned int status, interrupting_iface = 0, i;

	/* Interrupt handler for audio and video channels */
	for (i = 0; i < 8; i++) {
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
					set_bit (SDIVIDEO_EVENT_RX_BUFFER_ORDER,
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
		status = readl (card->core.addr + HDSDIM_QIE_ICSR(i));
		writel (status, card->core.addr + HDSDIM_QIE_ICSR(i));
		spin_unlock (&card->irq_lock);
			if (status & HDSDIM_QIE_ICSR_CDIS) {
				set_bit (SDIVIDEO_EVENT_RX_CARRIER_ORDER,
					&iface->events);
				interrupting_iface |= (0x1 << i);
			}
			if (status & HDSDIM_QIE_ICSR_ROIS) {
				set_bit (SDIVIDEO_EVENT_RX_FIFO_ORDER,
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
 * hdsdim_qie_init - Initialize a QuadPort H/i receiver
 * @iface: interface
 **/
static void
hdsdim_qie_init (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg = 0, regvbi1 = 0, regvbi2 = 0;

	/* Settings for audio and video are different */
	if (channel % 2 == 0) { // even channel = video
		switch (iface->mode) {
		case SDIVIDEO_CTL_MODE_UYVY:
			reg |= HDSDIM_QIE_RCSR_MODE_UYVY_DEINTERLACE;
			break;
		case SDIVIDEO_CTL_MODE_V210:
			reg |= HDSDIM_QIE_RCSR_MODE_V210_SYNC;
			break;
		case SDIVIDEO_CTL_MODE_V210_DEINTERLACE:
			reg |= HDSDIM_QIE_RCSR_MODE_V210_DEINTERLACE;
			break;
		case SDIVIDEO_CTL_MODE_RAW:
			reg |= HDSDIM_QIE_RCSR_MODE_RAW;
			break;
		default:
			break;
		}
		/* If VBI supported */
		if (card->version >= 0x0003) {
			regvbi1 |= iface->vb1cnt << 16;
			regvbi1 |= iface->vb1ln1;
			regvbi2 |= iface->vb2cnt << 16;
			regvbi2 |= iface->vb2ln1;
		}

	} else { //odd channel = audio
		switch (iface->sample_size) {
		case SDIAUDIO_CTL_AUDSAMP_SZ_16:
			reg |= HDSDIM_QIE_RCSR_AUDSAMP_SZ_16BIT;
			break;
		case SDIAUDIO_CTL_AUDSAMP_SZ_24:
			reg |= HDSDIM_QIE_RCSR_AUDSAMP_SZ_24BIT;
			break;
		case SDIAUDIO_CTL_AUDSAMP_SZ_32:
			reg |= HDSDIM_QIE_RCSR_AUDSAMP_SZ_32BIT;
			break;
		default:
			break;
		}

		switch (iface->channels) {
		case SDIAUDIO_CTL_AUDCH_EN_2:
			reg |= HDSDIM_QIE_RCSR_AUDCH_EN_2;
			break;
		case SDIAUDIO_CTL_AUDCH_EN_4:
			reg |= HDSDIM_QIE_RCSR_AUDCH_EN_4;
			break;
		case SDIAUDIO_CTL_AUDCH_EN_6:
			reg |= HDSDIM_QIE_RCSR_AUDCH_EN_6;
			break;
		case SDIAUDIO_CTL_AUDCH_EN_8:
			reg |= HDSDIM_QIE_RCSR_AUDCH_EN_8;
			break;
		default:
			break;
		}
	}

	/* There will be no races on RCR
	 * until this code returns, so we don't need to lock it */
	writel (reg | HDSDIM_QIE_RCSR_RST,
		card->core.addr + HDSDIM_QIE_RCSR(channel));
	if (card->version >= 0x0003) {
		wmb ();
		writel (regvbi1, card->core.addr + HDSDIM_QIE_VB1R(channel));
		wmb ();
		writel (regvbi2, card->core.addr + HDSDIM_QIE_VB2R(channel));
	}
	wmb ();
	writel (reg, card->core.addr + HDSDIM_QIE_RCSR(channel));
	wmb ();
	writel(HDSDIM_QIE_RDMATL,
		card->core.addr + HDSDIM_QIE_RDMATLR(channel));

	return;
}

/**
 * hdsdim_qie_start - Activate the QuadPort H/i receiver
 * @iface: interface
 **/
static void
hdsdim_qie_start (struct master_iface *iface)
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
	wmb ();
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_64BIT | LSDMA_CH_CSR_DIRECTION |
		LSDMA_CH_CSR_ENABLE,
		card->bridge_addr + LSDMA_CSR(channel));
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);

	/* Enable receiver interrupts */
	spin_lock_irq (&card->irq_lock);
	writel (HDSDIM_QIE_ICSR_CDIE | HDSDIM_QIE_ICSR_ROIE |
		HDSDIM_QIE_ICSR_RXDIE,
		card->core.addr + HDSDIM_QIE_ICSR(channel));
	spin_unlock_irq (&card->irq_lock);

	/* Enable the receiver */
	spin_lock (&card->reg_lock);
	reg = readl (card->core.addr + HDSDIM_QIE_RCSR(channel));
	writel (reg | HDSDIM_QIE_RCSR_RXE,
		card->core.addr + HDSDIM_QIE_RCSR(channel));
	spin_unlock (&card->reg_lock);

	return;
}

/**
 * hdsdim_qie_stop - Deactivate the QuadPort H/i receiver
 * @iface: interface
 **/
static void
hdsdim_qie_stop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	/* Disable the receiver */
	spin_lock (&card->reg_lock);
	reg = readl (card->core.addr + HDSDIM_QIE_RCSR(channel));
	writel (reg & ~HDSDIM_QIE_RCSR_RXE,
		card->core.addr + HDSDIM_QIE_RCSR(channel));
	spin_unlock (&card->reg_lock);

	/* Disable receiver interrupts */
	spin_lock_irq (&card->irq_lock);
	writel (HDSDIM_QIE_ICSR_CDIS | HDSDIM_QIE_ICSR_ROIS |
		HDSDIM_QIE_ICSR_RXDIS,
		card->core.addr + HDSDIM_QIE_ICSR(channel));

	/* Disable and abort DMA */
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_64BIT | LSDMA_CH_CSR_DIRECTION,
		card->bridge_addr + LSDMA_CSR(channel));
	wmb ();
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_64BIT | LSDMA_CH_CSR_DIRECTION |
		LSDMA_CH_CSR_STOP,
		card->bridge_addr + LSDMA_CSR(channel));

	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);
	spin_unlock_irq (&card->irq_lock);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_64BIT | LSDMA_CH_CSR_DIRECTION,
		card->bridge_addr + LSDMA_CSR(channel));

	return;
}

/**
 * hdsdim_qie_exit - Clean up the QuadPort H/i receiver
 * @iface: interface
 **/
static void
hdsdim_qie_exit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);

	/* Reset the receiver.
	 * There will be no races on RCR here,
	 * so we don't need to lock it */
	writel (HDSDIM_QIE_RCSR_RST,
		card->core.addr + HDSDIM_QIE_RCSR(channel));

	return;
}

/**
 * hdsdim_qie_unlocked_ioctl - QuadPort H/i receiver unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
hdsdim_qie_unlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int statusreg, frmode;

	switch (cmd) {
	case SDIVIDEO_IOC_RXGETSTATUS:
		/* We don't lock since this should be an atomic read */
		if (put_user ((readl (card->core.addr + HDSDIM_QIE_ICSR(channel)) &
			HDSDIM_QIE_ICSR_RXPASSING) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIVIDEO_IOC_RXGETCARRIER:
		/* We don't lock since this should be an atomic read */
		if (put_user ((readl (card->core.addr + HDSDIM_QIE_ICSR(channel)) &
			HDSDIM_QIE_ICSR_CD) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIVIDEO_IOC_RXGETYCRCERROR:
		/* We don't lock since this should be an atomic read */
		if (put_user ((readl (card->core.addr + HDSDIM_QIE_YCER(channel))),
			(int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIVIDEO_IOC_RXGETCCRCERROR:
		/* We don't lock since this should be an atomic read */
		if (put_user ((readl (card->core.addr + HDSDIM_QIE_CCER(channel))),
			(int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIVIDEO_IOC_RXGETVIDSTATUS:
		statusreg = readl (card->core.addr + HDSDIM_QIE_RCSR(channel));
		if (statusreg & HDSDIM_QIE_RCSR_STD_LOCKED) {
			switch (statusreg & HDSDIM_QIE_RCSR_STD) {
			case HDSDIM_QIE_STD_260M_1035i:
				frmode = SDIVIDEO_CTL_SMPTE_260M_1035I_59_94HZ;
				break;
			case HDSDIM_QIE_STD_295M_1080i:
				frmode = SDIVIDEO_CTL_SMPTE_295M_1080I_50HZ;
				break;
			case HDSDIM_QIE_STD_274M_1080i_60HZ:
				frmode = SDIVIDEO_CTL_SMPTE_274M_1080I_59_94HZ;
				break;
			case HDSDIM_QIE_STD_274M_1080i_50HZ:
				frmode = SDIVIDEO_CTL_SMPTE_274M_1080I_50HZ;
				break;
			case HDSDIM_QIE_STD_274M_1080p_30HZ:
				frmode = SDIVIDEO_CTL_SMPTE_274M_1080P_29_97HZ;
				break;
			case HDSDIM_QIE_STD_274M_1080p_25HZ:
				frmode = SDIVIDEO_CTL_SMPTE_274M_1080P_25HZ;
				break;
			case HDSDIM_QIE_STD_274M_1080p_24HZ:
				frmode = SDIVIDEO_CTL_SMPTE_274M_1080P_23_98HZ;
				break;
			case HDSDIM_QIE_STD_296M_720p_60HZ:
				frmode = SDIVIDEO_CTL_SMPTE_296M_720P_59_94HZ;
				break;
			case HDSDIM_QIE_STD_274M_1080sf_24HZ:
				frmode = SDIVIDEO_CTL_SMPTE_274M_1080PSF_23_98HZ;
				break;
			case HDSDIM_QIE_STD_296M_720p_50HZ:
				frmode = SDIVIDEO_CTL_SMPTE_296M_720P_50HZ;
				break;
			case HDSDIM_QIE_STD_296M_720p_30HZ:
				frmode = SDIVIDEO_CTL_SMPTE_296M_720P_29_97HZ;
				break;
			case HDSDIM_QIE_STD_296M_720p_25HZ:
				frmode = SDIVIDEO_CTL_SMPTE_296M_720P_25HZ;
				break;
			case HDSDIM_QIE_STD_296M_720p_24HZ:
				frmode = SDIVIDEO_CTL_SMPTE_296M_720P_23_98HZ;
				break;
			default:
				frmode = SDIVIDEO_CTL_UNLOCKED;
				break;
			}
		} else {
			frmode = SDIVIDEO_CTL_UNLOCKED;
		}
		if (put_user (frmode, (unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return sdivideo_rxioctl (filp, cmd, arg);
	}

	return 0;
}

/**
 * hdsdim_qie_unlocked_audioctl - QuadPort H/i receiver audio unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
hdsdim_qie_unlocked_audioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);

	switch (cmd) {
	case SDIAUDIO_IOC_RXGETSTATUS:
		/* We don't lock since this should be an atomic read */
		if (put_user ((readl (card->core.addr + HDSDIM_QIE_ICSR(channel)) &
			HDSDIM_QIE_ICSR_RXPASSING) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIAUDIO_IOC_RXGETCARRIER:
		/* We don't lock since this should be an atomic read */
		if (put_user ((readl (card->core.addr + HDSDIM_QIE_ICSR(channel)) &
			HDSDIM_QIE_ICSR_CD) ? 1 : 0, (int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIAUDIO_IOC_RXGETAUDIOGR0ERROR:
		/* We don't lock since this should be an atomic read */
		if (put_user ((readl (card->core.addr + HDSDIM_QIE_AG0ERR(channel))),
			(int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIAUDIO_IOC_RXGETAUDIOGR0DELAYA:
		/* We don't lock since this should be an atomic read */
		if (put_user ((readl (card->core.addr + HDSDIM_QIE_AG0DRA(channel))),
			(int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIAUDIO_IOC_RXGETAUDIOGR0DELAYB:
		/* We don't lock since this should be an atomic read */
		if (put_user ((readl (card->core.addr + HDSDIM_QIE_AG0DRB(channel))),
			(int __user *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return sdiaudio_rxioctl (filp, cmd, arg);
	}

	return 0;
}

/**
 * hdsdim_qie_fsync - QuadPort H/i receiver fsync() method
 * @filp: file to flush
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
FSYNC_HANDLER(hdsdim_qie_fsync,filp,datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	const unsigned int channel = mdev_index (card, &iface->list);
	unsigned int reg;

	mutex_lock (&iface->buf_mutex);

	/* Stop the receiver */
	hdsdim_qie_stop (iface);

	/* Reset the onboard FIFO and driver buffers */
	spin_lock (&card->reg_lock);
	reg = readl (card->core.addr + HDSDIM_QIE_RCSR(channel));
	writel (reg | HDSDIM_QIE_RCSR_RST,
		card->core.addr + HDSDIM_QIE_RCSR(channel));
	wmb ();
	writel (reg, card->core.addr + HDSDIM_QIE_RCSR(channel));
	spin_unlock (&card->reg_lock);
	iface->events = 0;
	lsdma_reset (iface->dma);

	/* Start the receiver */
	hdsdim_qie_start (iface);

	mutex_unlock (&iface->buf_mutex);
	return 0;
}

