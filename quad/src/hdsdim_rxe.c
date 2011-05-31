/* hdsdim_rxe.c
 *
 * Linux driver functions for Linear Systems Ltd. VidPort SD/HD I.
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
#include "hdsdim_rxe.h"
#include "hdsdim.h"

static const char hdsdim_rxe_name[] = HDSDIM_NAME_RXE;

/* Static function prototypes */
static ssize_t hdsdim_rxe_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static irqreturn_t IRQ_HANDLER(hdsdim_rxe_irq_handler,irq,dev_id,regs);
static void hdsdim_rxe_init (struct master_iface *iface);
static void hdsdim_rxe_audinit (struct master_iface *iface);
static void hdsdim_rxe_start (struct master_iface *iface);
static void hdsdim_rxe_audstart (struct master_iface *iface);
static void hdsdim_rxe_stop (struct master_iface *iface);
static void hdsdim_rxe_audstop (struct master_iface *iface);
static void hdsdim_rxe_exit (struct master_iface *iface);
static long hdsdim_rxe_unlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static long hdsdim_rxe_unlocked_audioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg);
static int FSYNC_HANDLER(hdsdim_rxe_fsync,filp,datasync);

static struct file_operations hdsdim_rxe_vidfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.read = sdivideo_read,
	.poll = sdivideo_rxpoll,
	.unlocked_ioctl = hdsdim_rxe_unlocked_ioctl,
	.compat_ioctl = hdsdim_rxe_unlocked_ioctl,
	.mmap = sdivideo_mmap,
	.open = sdivideo_open,
	.release = sdivideo_release,
	.fsync = hdsdim_rxe_fsync,
	.fasync = NULL
};

static struct file_operations hdsdim_rxe_audfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.read = sdiaudio_read,
	.poll = sdiaudio_rxpoll,
	.unlocked_ioctl = hdsdim_rxe_unlocked_audioctl,
	.compat_ioctl = hdsdim_rxe_unlocked_audioctl,
	.mmap = sdiaudio_mmap,
	.open = sdiaudio_open,
	.release = sdiaudio_release,
	.fsync = hdsdim_rxe_fsync,
	.fasync = NULL
};

static struct master_iface_operations hdsdim_rxe_vidops = {
	.init = hdsdim_rxe_init,
	.start = hdsdim_rxe_start,
	.stop = hdsdim_rxe_stop,
	.exit = hdsdim_rxe_exit
};

static struct master_iface_operations hdsdim_rxe_audops = {
	.init = hdsdim_rxe_audinit,
	.start = hdsdim_rxe_audstart,
	.stop = hdsdim_rxe_audstop,
	.exit = NULL
};

/**
 * hdsdim_rxe_show_uid - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
static ssize_t
hdsdim_rxe_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	return snprintf (buf, PAGE_SIZE, "0x%08X%08X\n",
		readl (card->core.addr + HDSDIM_RXE_SSN_HI),
		readl (card->core.addr + HDSDIM_RXE_SSN_LO));
}

static DEVICE_ATTR(uid,S_IRUGO,
	hdsdim_rxe_show_uid,NULL);

/**
 * hdsdim_rxe_pci_probe - PCI insertion handler for a VidPort SD/HD I
 * @pdev: PCI device
 *
 * Handle the insertion of a VidPort SD/HD I.
 * Returns a negative error code on failure and 0 on success.
 **/
int __devinit
hdsdim_rxe_pci_probe (struct pci_dev *pdev)
{
	int err;
	unsigned int cap;
	struct master_dev *card;

	err = hdsdim_pci_probe_generic (pdev);
	if (err < 0) {
		goto NO_PCI;
	}

	/* Initialize the driver_data pointer so that hdsdim_txe_pci_remove()
	 * doesn't try to free it if an error occurs */
	pci_set_drvdata (pdev, NULL);

	/* Allocate and initialize a board info structure */
	if ((card = (struct master_dev *)
		kzalloc (sizeof (*card), GFP_KERNEL)) == NULL) {
		err = -ENOMEM;
		goto NO_MEM;
	}

	card->core.addr = ioremap_nocache (pci_resource_start (pdev, 0),
		pci_resource_len (pdev, 0));
	card->bridge_addr = card->core.addr + 0x20;
	card->version = readl(card->core.addr + HDSDIM_RXE_FPGAID) & 0xffff;
	card->name = hdsdim_rxe_name;
	card->id = pdev->device;
	card->irq = pdev->irq;
	card->irq_handler = hdsdim_rxe_irq_handler;
	INIT_LIST_HEAD(&card->iface_list);
	card->capabilities = MASTER_CAP_UID;
	spin_lock_init (&card->irq_lock); /* unused, but don't remove because someday we might use it */
	spin_lock_init (&card->reg_lock);
	mutex_init (&card->users_mutex);
	card->parent = &pdev->dev;

	/* Print the firmware version */
	printk (KERN_INFO "%s: %s detected, firmware version %u.%u (0x%04X)\n",
		hdsdim_driver_name, card->name, card->version >> 8,
		card->version & 0x00ff, card->version);

	/* Store the pointer to the board info structure
	 * in the PCI info structure */
	pci_set_drvdata (pdev, card);

	/* Reset the FPGA */
	writel (HDSDIM_RXE_CTRL_SWRST,
		card->core.addr + HDSDIM_RXE_CTRL);

	/* Dummy read to flush PCI posted writes */
	readl (card->core.addr + HDSDIM_RXE_CTRL);

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

	cap = 0;
	if (card->version >= 0x0102) {
		cap |= SDIVIDEO_CAP_RX_VANC;
	}
	/* Register a video receive interface */
	if ((err = sdivideo_register_iface (card,
		&lsdma_dma_ops,
		0,
		MASTER_DIRECTION_RX,
		&hdsdim_rxe_vidfops,
		&hdsdim_rxe_vidops,
		cap,
		4)) < 0) { //4 is the dma alignment, means dma transfer has to begin and end on 4 byte alignment
		goto NO_IFACE;
	}

	cap = SDIAUDIO_CAP_RX_NONAUDIO;
	/* Register an audio receive interface */
	if ((err = sdiaudio_register_iface (card,
		&lsdma_dma_ops,
		0,
		MASTER_DIRECTION_RX,
		&hdsdim_rxe_audfops,
		&hdsdim_rxe_audops,
		cap,
		4)) < 0) { //4 is the dma alignment, means dma transfer has to begin and end on 4 byte alignment
		goto NO_IFACE;
	}

	return 0;

NO_IFACE:
NO_DEV:
NO_MEM:
	hdsdim_rxe_pci_remove (pdev);
NO_PCI:
	return err;
}

/**
 * hdsdim_rxe_pci_remove - PCI removal handler for a VidPort SD/HD I.
 * @pdev: PCI device
 *
 * Handle the removal of a VidPort SD/HD I.
 * This function may be called during PCI probe error handling,
 * so don't mark it as __devexit.
 **/
void
hdsdim_rxe_pci_remove (struct pci_dev *pdev)
{
	struct master_dev *card = pci_get_drvdata (pdev);

	if (card) {
		/* Unregister the device and all interfaces */
		hdsdim_unregister_all (card);

		iounmap (card->core.addr);
		kfree (card);
	}
	hdsdim_pci_remove_generic (pdev);
	return;
}

/**
 * hdsdim_rxe_irq_handler - VidPort SD/HD I interrupt service routine
 * @irq: interrupt number
 * @dev_id: pointer to the device data structure
 * @regs: processor context
 **/
static irqreturn_t
IRQ_HANDLER(hdsdim_rxe_irq_handler,irq,dev_id,regs)
{
	struct master_dev *card = dev_id;
	unsigned int interrupting_iface = 0;
	struct master_iface *vid_iface = list_entry (card->iface_list.next,
		struct master_iface, list);
	struct master_iface *aud_iface = list_entry (card->iface_list.prev,
		struct master_iface, list);
	unsigned int dmacsr;
	unsigned int isr = readl (card->core.addr + HDSDIM_RXE_ISR);

	if (!isr) {
		return IRQ_NONE;
	}

	/* Clear the source of the interrupts */
	writel (isr, card->core.addr + HDSDIM_RXE_ISR);

	/* Video */
	if (isr & HDSDIM_RXE_INT_DMA3) {
		/* Read the interrupt type */
		dmacsr = readl (card->bridge_addr + LSDMA_CSR(3));

		/* Increment the buffer pointer */
		mdma_advance (vid_iface->dma);

		if (mdma_rx_isempty (vid_iface->dma)) {
			set_bit (SDIVIDEO_EVENT_RX_BUFFER_ORDER,
				&vid_iface->events);
		}

		/* Flag end-of-chain */ //stop also have to wait until the end of chain
		if (dmacsr & LSDMA_CH_CSR_DONE) {
			set_bit (0, &vid_iface->dma_done);
		}

		interrupting_iface |= 0x01;
	}

	if (isr & HDSDIM_RXE_INT_RVOI) {
		set_bit (SDIVIDEO_EVENT_RX_FIFO_ORDER, &vid_iface->events);
		interrupting_iface |= 0x1;
	}
	if (isr & HDSDIM_RXE_INT_RVDI) {
		set_bit (SDIVIDEO_EVENT_RX_DATA_ORDER, &vid_iface->events);
		interrupting_iface |= 0x1;
	}
	if (isr & HDSDIM_RXE_INT_RSTDI) {
		set_bit (SDIVIDEO_EVENT_RX_STD_ORDER, &vid_iface->events);
		interrupting_iface |= 0x1;
	}

	/* Audio */
	if (isr & HDSDIM_RXE_INT_DMA4) {
		/* Read the interrupt type */
		dmacsr = readl (card->bridge_addr + LSDMA_CSR(4));

		/* Increment the buffer pointer */
		mdma_advance (aud_iface->dma);

		if (mdma_rx_isempty (aud_iface->dma)) {
			set_bit (SDIAUDIO_EVENT_RX_BUFFER_ORDER,
				&aud_iface->events);
		}

		/* Flag end-of-chain */ //stop also have to wait until the end of chain
		if (dmacsr & LSDMA_CH_CSR_DONE) {
			set_bit (0, &aud_iface->dma_done);
		}

		interrupting_iface |= 0x02;
	}

	if (isr & HDSDIM_RXE_INT_RAOI) {
		set_bit (SDIAUDIO_EVENT_RX_FIFO_ORDER, &aud_iface->events);
		interrupting_iface |= 0x2;
	}
	if (isr & HDSDIM_RXE_INT_RADI) {
		set_bit (SDIAUDIO_EVENT_RX_DATA_ORDER, &aud_iface->events);
		interrupting_iface |= 0x2;
	}

	if (interrupting_iface) {
		/* Dummy read to flush PCI posted writes */
		readl (card->core.addr + HDSDIM_RXE_FPGAID);

		if (interrupting_iface & 0x1) {
			wake_up (&vid_iface->queue);
		}
		if (interrupting_iface & 0x2) {
			wake_up (&aud_iface->queue);
		}
		return IRQ_HANDLED;
	}

	return IRQ_NONE;
}

/**
 * hdsdim_rxe_audinit - Initialize the VidPort SD/HD I Audio
 * @iface: interface
 **/
static void
hdsdim_rxe_audinit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg = 0;

	switch (iface->channels) {
	case SDIAUDIO_CTL_AUDCH_EN_0:
		reg |= HDSDIM_RXE_AUDCTRL_CHAN_0;
		break;
	case SDIAUDIO_CTL_AUDCH_EN_2:
		reg |= HDSDIM_RXE_AUDCTRL_CHAN_2;
		break;
	case SDIAUDIO_CTL_AUDCH_EN_4:
		reg |= HDSDIM_RXE_AUDCTRL_CHAN_4;
		break;
	case SDIAUDIO_CTL_AUDCH_EN_6:
		reg |= HDSDIM_RXE_AUDCTRL_CHAN_6;
		break;
	case SDIAUDIO_CTL_AUDCH_EN_8:
		reg |= HDSDIM_RXE_AUDCTRL_CHAN_8;
		break;
	default:
		break;
	}

	switch (iface->sample_size) {
	case SDIAUDIO_CTL_AUDSAMP_SZ_16:
		reg |= HDSDIM_RXE_AUDCTRL_FOURCC_16;
		break;
	case SDIAUDIO_CTL_AUDSAMP_SZ_32:
		reg |= HDSDIM_RXE_AUDCTRL_FOURCC_32;
		break;
	default:
		break;
	}
	writel (reg, card->core.addr + HDSDIM_RXE_AUDCTRL);

	return;
}

/**
 * hdsdim_rxe_init - Initialize a VidPort SD/HD I Video
 * @iface: interface
 **/
static void
hdsdim_rxe_init (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg = 0;

	switch (iface->mode) {
	default:
	case SDIVIDEO_CTL_MODE_UYVY:
		reg |= 0;
		break;
	case SDIVIDEO_CTL_MODE_V210:
		reg |= HDSDIM_RXE_CTRL_FOURCC_V210;
		break;
	}
	if (iface->vanc) {
		reg |= HDSDIM_RXE_CTRL_VANC;
	}

	/* There will be no races on CSR
	 * until this code returns, so we don't need to lock it
	*/
	writel (reg | HDSDIM_RXE_CTRL_SWRST,
		card->core.addr + HDSDIM_RXE_CTRL);

	/* Dummy read to flush PCI posted writes */
	readl (card->core.addr + HDSDIM_RXE_CTRL);

	return;
}

/**
 * hdsdim_rxe_start - Activate the VidPort SD/HD I Video
 * @iface: interface
 **/
static void
hdsdim_rxe_start (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	struct master_dma *dma = iface->dma;
	unsigned int reg = 0;

	/* Enable receiver interrupts */
	writel(HDSDIM_RXE_INT_DMA3 | HDSDIM_RXE_INT_RVOI |
		HDSDIM_RXE_INT_RVDI | HDSDIM_RXE_INT_RSTDI,
		card->core.addr + HDSDIM_RXE_IMS);

	/* Enable and start DMA */
	writel (mdma_dma_to_desc_low (lsdma_head_desc_bus_addr (dma)),
		card->bridge_addr + LSDMA_DESC(3));
	writel (mdma_dma_to_desc_high (lsdma_head_desc_bus_addr (dma)),
		card->bridge_addr + LSDMA_DESC_H(3));
	clear_bit (0, &iface->dma_done);
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_64BIT | LSDMA_CH_CSR_DIRECTION |
		LSDMA_CH_CSR_ENABLE,
		card->bridge_addr + LSDMA_CSR(3));

	/* Enable the receiver
	 * Put the card out of reset
	 */
	reg = readl(card->core.addr + HDSDIM_RXE_CTRL);
	writel (reg & ~HDSDIM_RXE_CTRL_SWRST,
		card->core.addr + HDSDIM_RXE_CTRL);

	return;
}

/**
 * hdsdim_rxe_audstart - Activate the VidPort SD/HD I Audio
 * @iface: interface
 **/
static void
hdsdim_rxe_audstart (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	struct master_dma *dma = iface->dma;

	/* Enable receiver interrupts */
	writel(HDSDIM_RXE_INT_DMA4 | HDSDIM_RXE_INT_RAOI |
		HDSDIM_RXE_INT_RADI,
		card->core.addr + HDSDIM_RXE_IMS);

	/* Enable and start DMA */
	writel (mdma_dma_to_desc_low  (lsdma_head_desc_bus_addr (dma)),
		card->bridge_addr + LSDMA_DESC(4));
	writel (mdma_dma_to_desc_high (lsdma_head_desc_bus_addr (dma)),
		card->bridge_addr + LSDMA_DESC_H(4));
	clear_bit (0, &iface->dma_done);
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_64BIT | LSDMA_CH_CSR_DIRECTION |
		LSDMA_CH_CSR_ENABLE,
		card->bridge_addr + LSDMA_CSR(4));

	/* Dummy read to flush PCI posted writes */
	readl(card->core.addr + HDSDIM_RXE_FPGAID);

	return;
}

/**
 * hdsdim_rxe_stop - Deactivate the VidPort SD/HD I Video
 * @iface: interface
 **/
static void
hdsdim_rxe_stop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;

	/* Disable the receiver */
	reg = readl(card->core.addr + HDSDIM_RXE_CTRL);
	writel(reg | HDSDIM_RXE_CTRL_SWRST,
		card->core.addr + HDSDIM_RXE_CTRL);

	/* Disable receiver non-DMA interrupts */
	writel (HDSDIM_RXE_INT_RVOI |
		HDSDIM_RXE_INT_RVDI | HDSDIM_RXE_INT_RSTDI,
		card->core.addr + HDSDIM_RXE_IMC);

	/* Disable and abort DMA */
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_64BIT | LSDMA_CH_CSR_DIRECTION,
		card->bridge_addr + LSDMA_CSR(3));
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_64BIT | LSDMA_CH_CSR_DIRECTION |
		LSDMA_CH_CSR_STOP,
		card->bridge_addr + LSDMA_CSR(3));
	/* Dummy read to flush PCI posted writes */
	readl(card->core.addr + HDSDIM_RXE_FPGAID);

	wait_event (iface->queue, test_bit (0, &iface->dma_done));

	/* Disable receiver DMA interrupts */
	writel (HDSDIM_RXE_INT_DMA3, card->core.addr + HDSDIM_RXE_IMC);

	return;
}

/**
 * hdsdim_rxe_audstop - Deactivate the VidPort SD/HD I Audio
 * @iface: interface
 **/
static void
hdsdim_rxe_audstop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;

	/* Disable receiver non-DMA interrupts */
	writel (HDSDIM_RXE_INT_RAOI | HDSDIM_RXE_INT_RADI,
		card->core.addr + HDSDIM_RXE_IMC);

	/* Disable and abort DMA */
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_64BIT | LSDMA_CH_CSR_DIRECTION,
		card->bridge_addr + LSDMA_CSR(4));
	writel (LSDMA_CH_CSR_INTDONEENABLE | LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_64BIT | LSDMA_CH_CSR_DIRECTION |
		LSDMA_CH_CSR_STOP,
		card->bridge_addr + LSDMA_CSR(4));
	/* Dummy read to flush PCI posted writes */
	readl(card->core.addr + HDSDIM_RXE_FPGAID);

	wait_event (iface->queue, test_bit (0, &iface->dma_done));

	/* Disable receiver DMA interrupts */
	writel (HDSDIM_RXE_INT_DMA4, card->core.addr + HDSDIM_RXE_IMC);

	return;
}

/**
 * hdsdim_rxe_exit - Clean up the VidPort SD/HD I
 * @iface: interface
 **/
static void
hdsdim_rxe_exit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;

	/* Reset the receiver.
	 * There will be no races on CSR here,
	 * so we don't need to lock it */
	writel (HDSDIM_RXE_CTRL_SWRST, card->core.addr + HDSDIM_RXE_CTRL);

	return;
}

/**
 * hdsdim_rxe_unlocked_ioctl - VidPort SD/HD I Video unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
hdsdim_rxe_unlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	unsigned int statusreg, frmode;

	switch (cmd) {
	case SDIVIDEO_IOC_RXGETVIDSTATUS:
		statusreg = readl (card->core.addr + HDSDIM_RXE_STATUS);
		if (statusreg & HDSDIM_RXE_STATUS_RXSTD_LOCKED) {
			if (statusreg & HDSDIM_RXE_STATUS_SD) {
				switch (statusreg & HDSDIM_RXE_STATUS_RXSTD) {
				case HDSDIM_RXE_STATUS_RXSTD_125M_486i:
					frmode = SDIVIDEO_CTL_SMPTE_125M_486I_59_94HZ;
					break;
				case HDSDIM_RXE_STATUS_RXSTD_BT601_576i:
					frmode = SDIVIDEO_CTL_BT_601_576I_50HZ;
					break;
				default:
					frmode = SDIVIDEO_CTL_UNLOCKED;
					break;
				}
			} else {
				switch (statusreg & HDSDIM_RXE_STATUS_RXSTD) {
				case HDSDIM_RXE_STATUS_RXSTD_260M_1035i:
					if (statusreg & HDSDIM_RXE_STATUS_M) {
						frmode = SDIVIDEO_CTL_SMPTE_260M_1035I_59_94HZ;
					} else {
						frmode = SDIVIDEO_CTL_SMPTE_260M_1035I_60HZ;
					}
					break;
				case HDSDIM_RXE_STATUS_RXSTD_295M_1080i:
					frmode = SDIVIDEO_CTL_SMPTE_295M_1080I_50HZ;
					break;
				case HDSDIM_RXE_STATUS_RXSTD_274M_1080i_60:
					if (statusreg & HDSDIM_RXE_STATUS_M) {
						frmode = SDIVIDEO_CTL_SMPTE_274M_1080I_59_94HZ;
					} else {
						frmode = SDIVIDEO_CTL_SMPTE_274M_1080I_60HZ;
					}
					break;
				case HDSDIM_RXE_STATUS_RXSTD_274M_1080i_50:
					frmode = SDIVIDEO_CTL_SMPTE_274M_1080I_50HZ;
					break;
				case HDSDIM_RXE_STATUS_RXSTD_274M_1080p_30:
					if (statusreg & HDSDIM_RXE_STATUS_M) {
						frmode = SDIVIDEO_CTL_SMPTE_274M_1080P_29_97HZ;
					} else {
						frmode = SDIVIDEO_CTL_SMPTE_274M_1080P_30HZ;
					}
					break;
				case HDSDIM_RXE_STATUS_RXSTD_274M_1080p_25:
					frmode = SDIVIDEO_CTL_SMPTE_274M_1080P_25HZ;
					break;
				case HDSDIM_RXE_STATUS_RXSTD_274M_1080p_24:
					if (statusreg & HDSDIM_RXE_STATUS_M) {
						frmode = SDIVIDEO_CTL_SMPTE_274M_1080P_23_98HZ;
					} else {
						frmode = SDIVIDEO_CTL_SMPTE_274M_1080P_24HZ;
					}
					break;
				case HDSDIM_RXE_STATUS_RXSTD_296M_720p_60:
					if (statusreg & HDSDIM_RXE_STATUS_M) {
						frmode = SDIVIDEO_CTL_SMPTE_296M_720P_59_94HZ;
					} else {
						frmode = SDIVIDEO_CTL_SMPTE_296M_720P_60HZ;
					}
					break;
				case HDSDIM_RXE_STATUS_RXSTD_274M_1080sf_24:
					if (statusreg & HDSDIM_RXE_STATUS_M) {
						frmode = SDIVIDEO_CTL_SMPTE_274M_1080PSF_23_98HZ;
					} else {
						frmode = SDIVIDEO_CTL_SMPTE_274M_1080PSF_24HZ;
					}
					break;
				case HDSDIM_RXE_STATUS_RXSTD_296M_720p_50:
					frmode = SDIVIDEO_CTL_SMPTE_296M_720P_50HZ;
					break;
				case HDSDIM_RXE_STATUS_RXSTD_296M_720p_30:
					if (statusreg & HDSDIM_RXE_STATUS_M) {
						frmode = SDIVIDEO_CTL_SMPTE_296M_720P_29_97HZ;
					} else {
						frmode = SDIVIDEO_CTL_SMPTE_296M_720P_30HZ;
					}
					break;
				case HDSDIM_RXE_STATUS_RXSTD_296M_720p_25:
					frmode = SDIVIDEO_CTL_SMPTE_296M_720P_25HZ;
					break;
				case HDSDIM_RXE_STATUS_RXSTD_296M_720p_24:
					if (statusreg & HDSDIM_RXE_STATUS_M) {
						frmode = SDIVIDEO_CTL_SMPTE_296M_720P_23_98HZ;
					} else {
						frmode = SDIVIDEO_CTL_SMPTE_296M_720P_24HZ;
					}
					break;
				default:
					frmode = SDIVIDEO_CTL_UNLOCKED;
					break;
				}
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
 * hdsdim_rxe_unlocked_audioctl - VidPort SD/HD I Audio unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
hdsdim_rxe_unlocked_audioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	unsigned int i, val = 0, nonaudio = 0;

	switch (cmd) {
	case SDIAUDIO_IOC_RXGETNONAUDIO:
		for (i = 0; i < 8; i++) {
			val = readl(card->core.addr + HDSDIM_RXE_AUDCS(i+1));
			val = (val & HDSDIM_RXE_AUDCS_CS0_NONAUDIO) >> 0x1;
			nonaudio |= val << i;
		}
		if (put_user (nonaudio,
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIAUDIO_IOC_RXGETAUDSTAT:
		if (put_user ((readl (card->core.addr +
			HDSDIM_RXE_RXAUDSTAT) & 0x000000ff) ,
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIAUDIO_IOC_RXGETAUDRATE:
		if (put_user ((readl (card->core.addr +
			HDSDIM_RXE_RXAUDRATE) & 0x0000000f) ,
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return sdiaudio_rxioctl (filp, cmd, arg);
	}

	return 0;
}

/**
 * hdsdim_rxe_fsync - VidPort SD/HD I Video fsync() method
 * @filp: file to flush
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
FSYNC_HANDLER(hdsdim_rxe_fsync,filp,datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	unsigned int reg;

	mutex_lock (&iface->buf_mutex);

	/* Stop the receiver */
	iface->ops->stop (iface);

	/* Reset the onboard FIFO and driver buffers */
	spin_lock (&card->reg_lock);
	reg = readl (card->core.addr + HDSDIM_RXE_CTRL);
	writel (reg | HDSDIM_RXE_CTRL_SWRST, card->core.addr + HDSDIM_RXE_CTRL);
	writel (reg, card->core.addr + HDSDIM_RXE_CTRL);
	spin_unlock (&card->reg_lock);
	iface->events = 0;
	lsdma_reset (iface->dma);

	/* Start the receiver */
	iface->ops->start (iface);

	mutex_unlock (&iface->buf_mutex);
	return 0;
}

