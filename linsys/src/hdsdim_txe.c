/* hdsdim_txe.c
 *
 * Linux driver for Linear Systems Ltd. VidPort SD/HD O.
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
#include <linux/delay.h> /* msleep () */
#include <linux/interrupt.h> /* irqreturn_t */
#include <linux/device.h> /* device_create_file */
#include <linux/mutex.h> /* mutex_init () */

#include <asm/uaccess.h> /* put_user () */
#include <asm/bitops.h> /* set_bit () */

#include "sdivideocore.h"
#include "sdiaudiocore.h"
#include "../include/master.h"
#include "mdev.h"
#include "mdma.h"
#include "lsdma.h"
#include "hdsdim_txe.h"
#include "hdsdim.h"

static const char hdsdim_txe_name[] = HDSDIM_NAME_TXE;

/* Static function prototypes */
static ssize_t hdsdim_txe_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf);
static irqreturn_t IRQ_HANDLER (hdsdim_txe_irq_handler, irq, dev_id, regs);
static unsigned int gcd (unsigned int a, unsigned int b);
static void i2c_write (struct master_dev *card,
	u8 slave,
	u8 addr,
	void *buf,
	unsigned int count);
static void hdsdim_txe_init (struct master_iface *iface);
static void hdsdim_txe_audinit (struct master_iface *iface);
static void hdsdim_txe_start (struct master_iface *iface);
static void hdsdim_txe_audstart (struct master_iface *iface);
static void hdsdim_txe_stop (struct master_iface *iface);
static void hdsdim_txe_audstop (struct master_iface *iface);
static void hdsdim_txe_exit (struct master_iface *iface);
static void hdsdim_txe_start_tx_dma (struct master_iface *iface);
static long hdsdim_txe_unlocked_ioctl (struct file *filp,
		unsigned int cmd,
		unsigned long arg);
static long hdsdim_txe_unlocked_audioctl (struct file *filp,
		unsigned int cmd,
		unsigned long arg);
static int FSYNC_HANDLER(hdsdim_txe_fsync,filp,datasync);

static struct file_operations hdsdim_txe_vidfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.write = sdivideo_write,
	.poll = sdivideo_txpoll,
	.unlocked_ioctl = hdsdim_txe_unlocked_ioctl,
	.compat_ioctl = hdsdim_txe_unlocked_ioctl,
	.mmap = sdivideo_mmap,
	.open = sdivideo_open,
	.release = sdivideo_release,
	.fsync = hdsdim_txe_fsync,
	.fasync = NULL
};

static struct file_operations hdsdim_txe_audfops = {
	.owner = THIS_MODULE,
	.llseek = no_llseek,
	.write = sdiaudio_write,
	.poll = sdiaudio_txpoll,
	.unlocked_ioctl = hdsdim_txe_unlocked_audioctl,
	.compat_ioctl = hdsdim_txe_unlocked_audioctl,
	.mmap = sdiaudio_mmap,
	.open = sdiaudio_open,
	.release = sdiaudio_release,
	.fsync = hdsdim_txe_fsync,
	.fasync = NULL
};

static struct master_iface_operations hdsdim_txe_vidops = {
	.init = hdsdim_txe_init,
	.start = hdsdim_txe_start,
	.stop = hdsdim_txe_stop,
	.exit = hdsdim_txe_exit,
	.start_tx_dma = hdsdim_txe_start_tx_dma
};

static struct master_iface_operations hdsdim_txe_audops = {
	.init = hdsdim_txe_audinit,
	.start = hdsdim_txe_audstart,
	.stop = hdsdim_txe_audstop,
	.exit = NULL,
	.start_tx_dma = hdsdim_txe_start_tx_dma
};

struct video_format {
	unsigned int total_samples_per_line;
	unsigned int total_lines_per_frame;
	unsigned int frame_rate;
	unsigned int m;
};

static const struct video_format FMT_486_29 = {
	.total_samples_per_line = 1716,
	.total_lines_per_frame = 525,
	.frame_rate = 30,
	.m = 1
};

static const struct video_format FMT_576_25 = {
	.total_samples_per_line = 1728,
	.total_lines_per_frame = 625,
	.frame_rate = 25,
	.m = 0
};

static const struct video_format FMT_486_59 = {
	.total_samples_per_line = 1716,
	.total_lines_per_frame = 525,
	.frame_rate = 60,
	.m = 1
};

static const struct video_format FMT_576_50 = {
	.total_samples_per_line = 1728,
	.total_lines_per_frame = 625,
	.frame_rate = 50,
	.m = 0
};

static const struct video_format FMT_720_60 = {
	.total_samples_per_line = 1650,
	.total_lines_per_frame = 750,
	.frame_rate = 60,
	.m = 0
};

static const struct video_format FMT_720_59 = {
	.total_samples_per_line = 1650,
	.total_lines_per_frame = 750,
	.frame_rate = 60,
	.m = 1
};

static const struct video_format FMT_720_50 = {
	.total_samples_per_line = 1980,
	.total_lines_per_frame = 750,
	.frame_rate = 50,
	.m = 0
};

static const struct video_format FMT_720_30 = {
	.total_samples_per_line = 3300,
	.total_lines_per_frame = 750,
	.frame_rate = 30,
	.m = 0
};

static const struct video_format FMT_720_29 = {
	.total_samples_per_line = 3300,
	.total_lines_per_frame = 750,
	.frame_rate = 30,
	.m = 1
};

static const struct video_format FMT_720_25 = {
	.total_samples_per_line = 3960,
	.total_lines_per_frame = 750,
	.frame_rate = 25,
	.m = 0
};

static const struct video_format FMT_720_24 = {
	.total_samples_per_line = 4125,
	.total_lines_per_frame = 750,
	.frame_rate = 24,
	.m = 0
};

static const struct video_format FMT_720_23 = {
	.total_samples_per_line = 4125,
	.total_lines_per_frame = 750,
	.frame_rate = 24,
	.m = 1
};

static const struct video_format FMT_1080_60 = {
	.total_samples_per_line = 2200,
	.total_lines_per_frame = 1125,
	.frame_rate = 60,
	.m = 0
};

static const struct video_format FMT_1080_59 = {
	.total_samples_per_line = 2200,
	.total_lines_per_frame = 1125,
	.frame_rate = 60,
	.m = 1
};

static const struct video_format FMT_1080_50 = {
	.total_samples_per_line = 2200,
	.total_lines_per_frame = 1125,
	.frame_rate = 50,
	.m = 0
};

static const struct video_format FMT_1080_30 = {
	.total_samples_per_line = 2200,
	.total_lines_per_frame = 1125,
	.frame_rate = 30,
	.m = 0
};

static const struct video_format FMT_1080_29 = {
	.total_samples_per_line = 2200,
	.total_lines_per_frame = 1125,
	.frame_rate = 30,
	.m = 1
};

static const struct video_format FMT_1080_25 = {
	.total_samples_per_line = 2640,
	.total_lines_per_frame = 1125,
	.frame_rate = 25,
	.m = 0
};

static const struct video_format FMT_1080_24 = {
	.total_samples_per_line = 2750,
	.total_lines_per_frame = 1125,
	.frame_rate = 24,
	.m = 0
};

static const struct video_format FMT_1080_23 = {
	.total_samples_per_line = 2750,
	.total_lines_per_frame = 1125,
	.frame_rate = 24,
	.m = 1
};

/**
 * hdsdim_txe_show_uid - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
static ssize_t
hdsdim_txe_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct master_dev *card = dev_get_drvdata(dev);

	return snprintf (buf, PAGE_SIZE, "0x%08X%08X\n",
		readl (card->core.addr + HDSDIM_TXE_SSN_HI),
		readl (card->core.addr + HDSDIM_TXE_SSN_LO));
}

static DEVICE_ATTR(uid,S_IRUGO, hdsdim_txe_show_uid, NULL);

/**
 * hdsdim_txe_pci_probe - PCI insertion handler for a VidPort SD/HD O
 * @pdev: PCI device
 *
 * Handle the insertion of a VidPort SD/HD O.
 * Returns a negative error code on failure and 0 on success.
 **/
int __devinit
hdsdim_txe_pci_probe (struct pci_dev *pdev)
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
	card->version = readl(card->core.addr + HDSDIM_TXE_FPGAID) & 0xffff;
	card->name = hdsdim_txe_name;
	card->id = pdev->device;
	card->irq = pdev->irq;
	card->irq_handler = hdsdim_txe_irq_handler;
	card->capabilities = MASTER_CAP_UID;
	INIT_LIST_HEAD(&card->iface_list);
	spin_lock_init (&card->irq_lock);
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
	writel (HDSDIM_TXE_CTRL_SWRST |
		HDSDIM_TXE_CTRL_HL2RST |
		HDSDIM_TXE_CTRL_CLKRST,
		card->core.addr + HDSDIM_TXE_CTRL);

	/* Dummy read to flush PCI posted writes */
	readl (card->core.addr + HDSDIM_TXE_CTRL);

	/* Register a VidPort device */
	if ((err = hdsdim_register (card)) < 0) {
		goto NO_DEV;
	}

	/* Add device attributes */
	if (card->capabilities & MASTER_CAP_UID) {
		if ((err = device_create_file (card->dev,
			&dev_attr_uid)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'uid'\n",
				hdsdim_driver_name);
		}
	}

	/* Register a video transmit interface */
	cap = 0;
	if (card->version >= 0x0202) {
		cap |= SDIVIDEO_CAP_TX_VANC;
	}
	if ((err = sdivideo_register_iface (card,
		&lsdma_dma_ops,
		0,
		MASTER_DIRECTION_TX,
		&hdsdim_txe_vidfops,
		&hdsdim_txe_vidops,
		cap,
		4)) < 0) { //4 is the dma alignment, means dma transfer has to begin and end on 4 byte alignment
		goto NO_IFACE;
	}

	/* Register an audio transmit interface */
	if ((err = sdiaudio_register_iface (card,
		&lsdma_dma_ops,
		0,
		MASTER_DIRECTION_TX,
		&hdsdim_txe_audfops,
		&hdsdim_txe_audops,
		0,
		4)) < 0) { //4 is the dma alignment, means dma transfer has to begin and end on 4 byte alignment
		goto NO_IFACE;
	}

	return 0;

NO_IFACE:
NO_DEV:
NO_MEM:
	hdsdim_txe_pci_remove (pdev);
NO_PCI:
	return err;
}

/**
 * hdsdim_txe_pci_remove - PCI removal handler for a VidPort SD/HD O
 * @pdev: PCI device
 *
 * Handle the removal of a VidPort SD/HD O.
 * This function may be called during PCI probe error handling,
 * so don't mark it as __devexit.
 **/
void
hdsdim_txe_pci_remove (struct pci_dev *pdev)
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
 * hdsdim_txe_irq_handler - VidPort SD/HD O interrupt service routine
 * @irq: interrupt number
 * @dev_id: pointer to the device data structure
 * @regs: processor context
 **/
static irqreturn_t
IRQ_HANDLER (hdsdim_txe_irq_handler, irq, dev_id, regs)
{
	struct master_dev *card = dev_id;
	unsigned int interrupting_iface = 0;
	struct master_iface *vid_iface = list_entry (card->iface_list.next,
		struct master_iface, list);
	struct master_iface *aud_iface = list_entry (card->iface_list.prev,
		struct master_iface, list);
	unsigned int dmacsr;
	unsigned int isr = readl (card->core.addr + HDSDIM_TXE_ISR);

	if (!isr) {
		return IRQ_NONE;
	}

	/* Clear the source of the interrupts */
	writel (isr, card->core.addr + HDSDIM_TXE_ISR);

	if (isr & HDSDIM_TXE_INT_DMA0) {
		/* Read the interrupt type */
		dmacsr = readl (card->bridge_addr + LSDMA_CSR(0));

		/* Increment the buffer pointer */
		mdma_advance (vid_iface->dma);

		/* Flag end-of-chain */ //stop also have to wait until the end of chain
		if (dmacsr & LSDMA_CH_CSR_DONE) {
			set_bit (SDIVIDEO_EVENT_TX_BUFFER_ORDER,
				&vid_iface->events);
			set_bit (0, &vid_iface->dma_done);
		}

		interrupting_iface |= 0x01;
	}
	if (isr & HDSDIM_TXE_INT_TVUI) {
		set_bit (SDIVIDEO_EVENT_TX_FIFO_ORDER, &vid_iface->events);
		interrupting_iface |= 0x1;
	}
	if (isr & HDSDIM_TXE_INT_TVDI) {
		set_bit (SDIVIDEO_EVENT_TX_DATA_ORDER, &vid_iface->events);
		interrupting_iface |= 0x1;
	}
	if (isr & HDSDIM_TXE_INT_REFI) {
		set_bit (SDIVIDEO_EVENT_TX_REF_ORDER, &vid_iface->events);
		interrupting_iface |= 0x1;
	}

	if (isr & HDSDIM_TXE_INT_DMA1) {
		/* Read the interrupt type */
		dmacsr = readl (card->bridge_addr + LSDMA_CSR(1));

		/* Increment the buffer pointer */
		mdma_advance (aud_iface->dma);

		/* Flag end-of-chain */ //stop also have to wait until the end of chain
		if (dmacsr & LSDMA_CH_CSR_DONE) {
			set_bit (SDIAUDIO_EVENT_TX_BUFFER_ORDER,
				&aud_iface->events);
			set_bit (0, &aud_iface->dma_done);
		}

		interrupting_iface |= 0x02;
	}
	if (isr & HDSDIM_TXE_INT_TAUI) {
		set_bit (SDIAUDIO_EVENT_TX_FIFO_ORDER, &aud_iface->events);
		interrupting_iface |= 0x2;
	}
	if (isr & HDSDIM_TXE_INT_TADI) {
		set_bit (SDIAUDIO_EVENT_TX_DATA_ORDER, &aud_iface->events);
		interrupting_iface |= 0x2;
	}

	if (interrupting_iface) {
		/* Dummy read to flush PCI posted writes */
		readl (card->core.addr + HDSDIM_TXE_FPGAID);

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
 * gcd - Greatest common divisor
 *
 * Implementation of the Euclidean algorithm.
 **/
static unsigned int
gcd (unsigned int a, unsigned int b)
{
	unsigned int temp;

	while (b != 0) {
		temp = b;
		b = a % b;
		a = temp;
	}
	return a;
}

/**
 * i2c_write - I2C write sequence
 * @card: Master device
 * @slave: slave device
 * @addr: address
 * @buf: data buffer
 * @count: data count
 **/
static void
i2c_write (struct master_dev *card,
	u8 slave,
	u8 addr,
	void *buf,
	unsigned int count)
{
	unsigned char *p = buf;

	if (!count) {
		return;
	}
	/* Enable I2C */
	writeb (HDSDIM_TXE_I2CCTRL_EN, card->core.addr + HDSDIM_TXE_I2CCTRL);
	/* Pass the I2C slave address */
	writeb (slave, card->core.addr + HDSDIM_TXE_I2CD);
	/* Write and start */
	writeb (HDSDIM_TXE_I2CCMD_WR | HDSDIM_TXE_I2CCMD_STA,
		card->core.addr + HDSDIM_TXE_I2CCMD);
	/* Wait for ACK */
	do {
		msleep(1);
	} while (readb (card->core.addr + HDSDIM_TXE_I2CS) &
		HDSDIM_TXE_I2CS_TIP);
	/* Pass the address */
	writeb (addr, card->core.addr + HDSDIM_TXE_I2CD);
	/* Write */
	writeb (HDSDIM_TXE_I2CCMD_WR, card->core.addr + HDSDIM_TXE_I2CCMD);
	/* Wait for ACK */
	do {
		msleep(1);
	} while (readb (card->core.addr + HDSDIM_TXE_I2CS) &
		HDSDIM_TXE_I2CS_TIP);
	while (count > 1) {
		/* Pass the data */
		writeb (*p, card->core.addr + HDSDIM_TXE_I2CD);
		/* Write */
		writeb (HDSDIM_TXE_I2CCMD_WR,
			card->core.addr + HDSDIM_TXE_I2CCMD);
		/* Wait for ACK */
		do {
			msleep(1);
		} while (readb (card->core.addr + HDSDIM_TXE_I2CS) &
			HDSDIM_TXE_I2CS_TIP);
		count--;
		p++;
	}
	/* Pass the data */
	writeb (*p, card->core.addr + HDSDIM_TXE_I2CD);
	/* Write and stop */
	writeb (HDSDIM_TXE_I2CCMD_WR | HDSDIM_TXE_I2CCMD_STO,
		card->core.addr + HDSDIM_TXE_I2CCMD);
	/* Wait for ACK */
	do {
		msleep(1);
	} while (readb (card->core.addr + HDSDIM_TXE_I2CS) &
		HDSDIM_TXE_I2CS_TIP);
	/* Disable I2C */
	writeb (0x00, card->core.addr + HDSDIM_TXE_I2CCTRL);
	return;
}

#if 0
static u8 i2c_read (struct master_dev *card, u8 slave, u8 addr);

/**
 * i2c_read - I2C read sequence
 * @card: Master device
 * @slave: slave device
 * @addr: address
 **/
static u8
i2c_read (struct master_dev *card, u8 slave, u8 addr)
{
	u8 b;

	/* Enable I2C */
	writeb (HDSDIM_TXE_I2CCTRL_EN, card->core.addr + HDSDIM_TXE_I2CCTRL);
	/* Pass the I2C slave address */
	writeb (slave, card->core.addr + HDSDIM_TXE_I2CD);
	/* Write and start */
	writeb (HDSDIM_TXE_I2CCMD_WR | HDSDIM_TXE_I2CCMD_STA,
		card->core.addr + HDSDIM_TXE_I2CCMD);
	/* Wait for ACK */
	do {
		msleep(1);
	} while (readb (card->core.addr + HDSDIM_TXE_I2CS) &
		HDSDIM_TXE_I2CS_TIP);
	/* Pass the address */
	writeb (addr, card->core.addr + HDSDIM_TXE_I2CD);
	/* Write */
	writeb (HDSDIM_TXE_I2CCMD_WR, card->core.addr + HDSDIM_TXE_I2CCMD);
	/* Wait for ACK */
	do {
		msleep(1);
	} while (readb (card->core.addr + HDSDIM_TXE_I2CS) &
		HDSDIM_TXE_I2CS_TIP);
	/* Pass the I2C slave address and the read bit */
	writeb (slave | 0x01, card->core.addr + HDSDIM_TXE_I2CD);
	/* Write and start */
	writeb (HDSDIM_TXE_I2CCMD_WR | HDSDIM_TXE_I2CCMD_STA,
		card->core.addr + HDSDIM_TXE_I2CCMD);
	/* Wait for ACK */
	do {
		msleep(1);
	} while (readb (card->core.addr + HDSDIM_TXE_I2CS) &
		HDSDIM_TXE_I2CS_TIP);
	/* Read, stop, NACK */
	writeb (HDSDIM_TXE_I2CCMD_RD | HDSDIM_TXE_I2CCMD_STO |
		HDSDIM_TXE_I2CCMD_ACK,
		card->core.addr + HDSDIM_TXE_I2CCMD);
	/* Wait for ACK */
	do {
		msleep(1);
	} while (readb (card->core.addr + HDSDIM_TXE_I2CS) &
		HDSDIM_TXE_I2CS_TIP);
	/* Read data */
	b = readb (card->core.addr + HDSDIM_TXE_I2CD);
	/* Disable I2C */
	writeb (0x00, card->core.addr + HDSDIM_TXE_I2CCTRL);
	return b;
}
#endif

/**
 * hdsdim_txe_audinit - Initialize the VidPort SD/HD O audio
 * @iface: interface
 **/
static void
hdsdim_txe_audinit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int ctrlreg = 0, csreg = 1, csregfinal, i, mask;

	/* Configure Audio Control Register */
	switch (iface->channels) {
	case SDIAUDIO_CTL_AUDCH_EN_0:
		ctrlreg |= HDSDIM_TXE_AUDCTRL_CHAN_0;
		break;
	case SDIAUDIO_CTL_AUDCH_EN_2:
		ctrlreg |= HDSDIM_TXE_AUDCTRL_CHAN_2;
		break;
	case SDIAUDIO_CTL_AUDCH_EN_4:
		ctrlreg |= HDSDIM_TXE_AUDCTRL_CHAN_4;
		break;
	case SDIAUDIO_CTL_AUDCH_EN_6:
		ctrlreg |= HDSDIM_TXE_AUDCTRL_CHAN_6;
		break;
	case SDIAUDIO_CTL_AUDCH_EN_8:
		ctrlreg |= HDSDIM_TXE_AUDCTRL_CHAN_8;
		break;
	default:
		break;
	}

	switch (iface->sample_size) {
	case SDIAUDIO_CTL_AUDSAMP_SZ_16:
		ctrlreg |= HDSDIM_TXE_AUDCTRL_FOURCC_16;
		break;
	case SDIAUDIO_CTL_AUDSAMP_SZ_32:
		ctrlreg |= HDSDIM_TXE_AUDCTRL_FOURCC_32;
		break;
	default:
		break;
	}
	writel (ctrlreg, card->core.addr + HDSDIM_TXE_AUDCTRL);

	/* Configure Audio Channel Status Register */
	switch (iface->sample_rate) {
	case 32000:
		csreg |= HDSDIM_TXE_TXAUDCS_CS0_32KHZ;
		break;
	case 44100:
		csreg |= HDSDIM_TXE_TXAUDCS_CS0_44_1KHZ;
		break;
	case 48000:
		csreg |= HDSDIM_TXE_TXAUDCS_CS0_48KHZ;
		break;
	default:
		break;
	}

	/* Check the sample size, decide the max audio length */
	switch (iface->sample_size) {
	case SDIAUDIO_CTL_AUDSAMP_SZ_32: //32-bit
		csreg |= HDSDIM_TXE_TXAUDCS_CS2_MAXLENGTH_24BITS;
		break;
	case SDIAUDIO_CTL_AUDSAMP_SZ_16: //16-bit
		csreg |= HDSDIM_TXE_TXAUDCS_CS2_MAXLENGTH_20BITS;
		break;
	default:
		break;
	}

	/* Check Non-audio */
	mask = 0x0001;
	for (i = 0; i < 16; i++) {
		if (iface->nonaudio & mask) {
			csregfinal = csreg | HDSDIM_TXE_TXAUDCS_CS0_NONAUDIO;
		} else {
			csregfinal = csreg;
		}
		writel (csregfinal, card->core.addr + HDSDIM_TXE_TXAUDCS(i+1));
		mask <<= 1;
	}

	return;
}

/**
 * hdsdim_txe_init - Initialize the VidPort SD/HD O
 * @iface: interface
 **/
static void
hdsdim_txe_init (struct master_iface *iface)
{
	/* Set tof_27mhz = true to generate TOF using 27 MHz in HD
	 * as recommended in the LMH1982 data sheet */
	const unsigned int tof_27mhz = 1;

	struct master_dev *card = iface->card;
	unsigned int reg = 0;
	const struct video_format *out_fmt, *ref_fmt;
	unsigned int numer, denom, divisor;
	unsigned int tof_offset;
	unsigned int tof_rst;
	unsigned int tof_ppl;
	u8 tof_clk;
	unsigned int tof_lpfm;
	unsigned int ref_lpfm;
	unsigned int fb_div;
	unsigned int ref_div;
	u8 icp1;
	u8 lock_ctrl;
	u8 data[21], *p;
	unsigned int status;

	switch (iface->frmode) {
	case SDIVIDEO_CTL_SMPTE_125M_486I_59_94HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_125M_486i |
			HDSDIM_TXE_CTRL_SD |
			HDSDIM_TXE_CTRL_M;
		out_fmt = &FMT_486_29;
		break;
	case SDIVIDEO_CTL_BT_601_576I_50HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_BT601_576i |
			HDSDIM_TXE_CTRL_SD;
		out_fmt = &FMT_576_25;
		break;
	case SDIVIDEO_CTL_SMPTE_260M_1035I_60HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_260M_1035i;
		out_fmt = &FMT_1080_30;
		break;
	case SDIVIDEO_CTL_SMPTE_260M_1035I_59_94HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_260M_1035i |
			HDSDIM_TXE_CTRL_M;
		out_fmt = &FMT_1080_29;
		break;
	case SDIVIDEO_CTL_SMPTE_274M_1080I_60HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_274M_1080i_30;
		out_fmt = &FMT_1080_30;
		break;
	case SDIVIDEO_CTL_SMPTE_274M_1080PSF_30HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_274M_1080i_30 |
			HDSDIM_TXE_CTRL_PSF;
		out_fmt = &FMT_1080_30;
		break;
	case SDIVIDEO_CTL_SMPTE_274M_1080I_59_94HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_274M_1080i_30 |
			HDSDIM_TXE_CTRL_M;
		out_fmt = &FMT_1080_29;
		break;
	case SDIVIDEO_CTL_SMPTE_274M_1080PSF_29_97HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_274M_1080i_30 |
			HDSDIM_TXE_CTRL_M |
			HDSDIM_TXE_CTRL_PSF;
		out_fmt = &FMT_1080_29;
		break;
	case SDIVIDEO_CTL_SMPTE_274M_1080I_50HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_274M_1080i_25;
		out_fmt = &FMT_1080_25;
		break;
	case SDIVIDEO_CTL_SMPTE_274M_1080PSF_25HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_274M_1080i_25 |
			HDSDIM_TXE_CTRL_PSF;
		out_fmt = &FMT_1080_25;
		break;
	case SDIVIDEO_CTL_SMPTE_274M_1080PSF_24HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_274M_1080sf_24;
		out_fmt = &FMT_1080_24;
		break;
	case SDIVIDEO_CTL_SMPTE_274M_1080PSF_23_98HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_274M_1080sf_24 |
			HDSDIM_TXE_CTRL_M;
		out_fmt = &FMT_1080_23;
		break;
	case SDIVIDEO_CTL_SMPTE_274M_1080P_30HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_274M_1080p_30;
		out_fmt = &FMT_1080_30;
		break;
	case SDIVIDEO_CTL_SMPTE_274M_1080P_29_97HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_274M_1080p_30 |
			HDSDIM_TXE_CTRL_M;
		out_fmt = &FMT_1080_29;
		break;
	case SDIVIDEO_CTL_SMPTE_274M_1080P_25HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_274M_1080p_25;
		out_fmt = &FMT_1080_25;
		break;
	case SDIVIDEO_CTL_SMPTE_274M_1080P_24HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_274M_1080p_24;
		out_fmt = &FMT_1080_24;
		break;
	case SDIVIDEO_CTL_SMPTE_274M_1080P_23_98HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_274M_1080p_24 |
			HDSDIM_TXE_CTRL_M;
		out_fmt = &FMT_1080_23;
		break;
	case SDIVIDEO_CTL_SMPTE_296M_720P_60HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_296M_720p_60;
		out_fmt = &FMT_720_60;
		break;
	case SDIVIDEO_CTL_SMPTE_296M_720P_59_94HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_296M_720p_60 |
			HDSDIM_TXE_CTRL_M;
		out_fmt = &FMT_720_59;
		break;
	case SDIVIDEO_CTL_SMPTE_296M_720P_50HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_296M_720p_50;
		out_fmt = &FMT_720_50;
		break;
	case SDIVIDEO_CTL_SMPTE_296M_720P_30HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_296M_720p_30;
		out_fmt = &FMT_720_30;
		break;
	case SDIVIDEO_CTL_SMPTE_296M_720P_29_97HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_296M_720p_30 |
			HDSDIM_TXE_CTRL_M;
		out_fmt = &FMT_720_29;
		break;
	case SDIVIDEO_CTL_SMPTE_296M_720P_25HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_296M_720p_25;
		out_fmt = &FMT_720_25;
		break;
	case SDIVIDEO_CTL_SMPTE_296M_720P_24HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_296M_720p_24;
		out_fmt = &FMT_720_24;
		break;
	case SDIVIDEO_CTL_SMPTE_296M_720P_23_98HZ:
		reg |= HDSDIM_TXE_CTRL_TXSTD_296M_720p_24 |
			HDSDIM_TXE_CTRL_M;
		out_fmt = &FMT_720_23;
		break;
	default:
	case SDIVIDEO_CTL_UNLOCKED:
	case SDIVIDEO_CTL_SMPTE_295M_1080I_50HZ:
		out_fmt = &FMT_486_59;
		break;
	}
	switch (iface->clksrc) {
	default:
	case SDIVIDEO_CTL_TX_CLKSRC_ONBOARD:
	case SDIVIDEO_CTL_TX_CLKSRC_NTSC:
		ref_fmt = &FMT_486_29;
		/* From the LMH1981 data sheet,
		 * HSYNC and VSYNC are within delta_T_hv on line 267.
		 * Add 262 lines to generate TOF on line 4. */
		tof_offset = 262;
		break;
	case SDIVIDEO_CTL_TX_CLKSRC_PAL:
		ref_fmt = &FMT_576_25;
		/* From the LMH1982 data sheet,
		 * HSYNC and VSYNC are within delta_T_hv on line 313.
		 * Add 312 lines to generate TOF on line 1. */
		tof_offset = 312;
		break;
	case SDIVIDEO_CTL_TX_CLKSRC_525P:
		ref_fmt = &FMT_486_59;
		tof_offset = 524;
		break;
	case SDIVIDEO_CTL_TX_CLKSRC_625P:
		ref_fmt = &FMT_576_50;
		tof_offset = 624;
		break;
	case SDIVIDEO_CTL_TX_CLKSRC_720P_60:
		ref_fmt = &FMT_720_60;
		tof_offset = 749;
		break;
	case SDIVIDEO_CTL_TX_CLKSRC_720P_59_94:
		ref_fmt = &FMT_720_59;
		tof_offset = 749;
		break;
	case SDIVIDEO_CTL_TX_CLKSRC_720P_50:
		ref_fmt = &FMT_720_50;
		tof_offset = 749;
		break;
	case SDIVIDEO_CTL_TX_CLKSRC_720P_30:
		ref_fmt = &FMT_720_30;
		tof_offset = 749;
		break;
	case SDIVIDEO_CTL_TX_CLKSRC_720P_29_97:
		ref_fmt = &FMT_720_29;
		tof_offset = 749;
		break;
	case SDIVIDEO_CTL_TX_CLKSRC_720P_25:
		ref_fmt = &FMT_720_25;
		tof_offset = 749;
		break;
	case SDIVIDEO_CTL_TX_CLKSRC_720P_24:
		ref_fmt = &FMT_720_24;
		tof_offset = 749;
		break;
	case SDIVIDEO_CTL_TX_CLKSRC_720P_23_98:
		ref_fmt = &FMT_720_23;
		tof_offset = 749;
		break;
	case SDIVIDEO_CTL_TX_CLKSRC_1080P_60:
		ref_fmt = &FMT_1080_60;
		tof_offset = 1124;
		break;
	case SDIVIDEO_CTL_TX_CLKSRC_1080P_59_94:
		ref_fmt = &FMT_1080_59;
		tof_offset = 1124;
		break;
	case SDIVIDEO_CTL_TX_CLKSRC_1080P_50:
		ref_fmt = &FMT_1080_50;
		tof_offset = 1124;
		break;
	case SDIVIDEO_CTL_TX_CLKSRC_1080P_30:
		ref_fmt = &FMT_1080_30;
		tof_offset = 1124;
		break;
	case SDIVIDEO_CTL_TX_CLKSRC_1080P_29_97:
		ref_fmt = &FMT_1080_29;
		tof_offset = 1124;
		break;
	case SDIVIDEO_CTL_TX_CLKSRC_1080P_25:
		ref_fmt = &FMT_1080_25;
		tof_offset = 1124;
		break;
	case SDIVIDEO_CTL_TX_CLKSRC_1080P_24:
		ref_fmt = &FMT_1080_24;
		tof_offset = 1124;
		break;
	case SDIVIDEO_CTL_TX_CLKSRC_1080P_23_98:
		ref_fmt = &FMT_1080_23;
		tof_offset = 1124;
		break;
	case SDIVIDEO_CTL_TX_CLKSRC_1080I_60:
		ref_fmt = &FMT_1080_30;
		tof_offset = 562;
		break;
	case SDIVIDEO_CTL_TX_CLKSRC_1080I_59_94:
		ref_fmt = &FMT_1080_29;
		tof_offset = 562;
		break;
	case SDIVIDEO_CTL_TX_CLKSRC_1080I_50:
		ref_fmt = &FMT_1080_25;
		tof_offset = 562;
		break;
	}
	switch (iface->mode) {
	default:
	case SDIVIDEO_CTL_MODE_UYVY:
		reg |= 0;
		break;
	case SDIVIDEO_CTL_MODE_V210:
		reg |= HDSDIM_TXE_CTRL_FOURCC_V210;
		break;
	}
	if (iface->vanc) {
		reg |= HDSDIM_TXE_CTRL_VANC;
	}

	/* Enable the clock generator */
	writel (reg | HDSDIM_TXE_CTRL_SWRST | HDSDIM_TXE_CTRL_HL2RST,
		card->core.addr + HDSDIM_TXE_CTRL);

	/* Program the output clock frequency */
	p = data;
	if (out_fmt->m) {
		if (card->version < 0x0200) {
			*p++ = 0x04; /* HD_FREQ = 1 */
		} else {
			*p++ = 0x0c; /* HD_FREQ = 3 */
		}
	} else {
		if (card->version < 0x0200) {
			*p++ = 0x00; /* HD_FREQ = 0 */
		} else {
			*p++ = 0x08; /* HD_FREQ = 2 */
		}
	}
	i2c_write (card, 0xdc, 0x08, data, 1);

	/* 148.35 MHz PLL initialization sequence */
	fb_div = 1;
	p = data;
	*p++ = fb_div & 0xff;
	*p++ = fb_div >> 8;
	i2c_write (card, 0xdc, 0x04, data, 2);
	tof_rst = 1;
	p = data;
	*p++ = tof_rst & 0xff;
	*p++ = tof_rst >> 8;
	i2c_write (card, 0xdc, 0x09, data, 2);
	ref_lpfm = 1;
	p = data;
	*p++ = ref_lpfm & 0xff;
	*p++ = ref_lpfm >> 8;
	i2c_write (card, 0xdc, 0x0f, data, 2);
	p = data;
	*p = 0x80 | (tof_rst >> 8); /* EN_TOF_RST = 1 */
	i2c_write (card, 0xdc, 0x0a, data, 1);
	*p = tof_rst >> 8; /* EN_TOF_RST = 0 */
	i2c_write (card, 0xdc, 0x0a, data, 1);

	/* Program the output TOF timing */
	if (out_fmt->m && !ref_fmt->m) {
		numer = ref_fmt->frame_rate * 1001;
		denom = out_fmt->frame_rate * 1000;
	} else if (!out_fmt->m && ref_fmt->m) {
		numer = ref_fmt->frame_rate * 1000;
		denom = out_fmt->frame_rate * 1001;
	} else {
		numer = ref_fmt->frame_rate;
		denom = out_fmt->frame_rate;
	}
	tof_rst = numer / gcd (numer, denom);
	p = data;
	*p++ = tof_rst & 0xff;
	*p++ = tof_rst >> 8;
	if (tof_27mhz) {
		tof_clk = 0;
		if (out_fmt->m) {
			numer = 27027000;
		} else {
			numer = 27000000;
		}
		denom = out_fmt->total_lines_per_frame * out_fmt->frame_rate;
		tof_lpfm = (out_fmt->total_lines_per_frame *
			gcd (numer, denom)) / denom;
		tof_ppl = numer / (tof_lpfm * out_fmt->frame_rate);
	} else {
		tof_ppl = out_fmt->total_samples_per_line;
		tof_clk = (out_fmt->total_lines_per_frame < 750) ? 0 : 1;
		tof_lpfm = out_fmt->total_lines_per_frame;
	}
	*p++ = tof_ppl & 0xff;
	*p++ = (tof_clk << 5) | (tof_ppl >> 8);
	*p++ = tof_lpfm & 0xff;
	*p++ = tof_lpfm >> 8;
	ref_lpfm = ref_fmt->total_lines_per_frame;
	*p++ = ref_lpfm & 0xff;
	*p++ = ref_lpfm >> 8;
	*p++ = tof_offset & 0xff;
	*p++ = tof_offset >> 8;
	/* NB. Don't temporarily increase ICP1 to reduce lock time
	 * as suggested in the LMH1982 data sheet
	 * because changing ICP1 seems to cause loss of lock. */
	i2c_write (card, 0xdc, 0x09, data, 10);

	/* Program the PLL 1 dividers */
	/* FB_DIV / REF_DIV = f_vcxo / f_hsync */
	if (ref_fmt->m) {
		numer = 27027000;
	} else {
		numer = 27000000;
	}
	denom = ref_fmt->total_lines_per_frame * ref_fmt->frame_rate;
	divisor = gcd (numer, denom);
	fb_div = numer / divisor;
	ref_div = denom / divisor;
	/* Scale FB_DIV, REF_DIV, and ICP1 within their limits
	 * to keep the loop gain relatively constant
	 * across reference formats given
	 * FB_DIV = 1716 and ICP1 = 8 as a starting point */
	if (fb_div < 1716 / 2) {
		fb_div *= 5;
		ref_div *= 5;
	}
	icp1 = fb_div * 8 / 1716;
	if (icp1 < 3) {
		icp1 = 3;
	}
	if (icp1 > 31) {
		icp1 = 31;
	}
	p = data;
	switch (ref_div) {
	default:
	case 1:
		*p++ = 0x01; /* REF_DIV_SEL = 1 */
		break;
	case 2:
		*p++ = 0x00; /* REF_DIV_SEL = 0 */
		break;
	case 5:
		*p++ = 0x02; /* REF_DIV_SEL = 2 */
		break;
	}
	*p++ = fb_div & 0xff;
	*p++ = fb_div >> 8;
	i2c_write (card, 0xdc, 0x03, data, 3);
	p = data;
	*p = 0x80 | icp1;
	i2c_write (card, 0xdc, 0x13, data, 1);

	/* Enable Genlock mode and reduce the lock detector threshold */
	p = data;
	if (iface->clksrc == SDIVIDEO_CTL_TX_CLKSRC_ONBOARD) {
		/* Switch to the unused input to avoid
		 * spurious REF_LOST */
		*p++ = 0xb3; /* GNLK = 0, RSEL = 1 */
	} else {
		*p++ = 0xe3; /* GNLK = 1, RSEL = 0 */
	}
	lock_ctrl = 1;
	*p++ = lock_ctrl << 3;
	i2c_write (card, 0xdc, 0x00, data, 2);

	/* Program the output initialization sequence */
	p = data;
	*p = 0x80 | (tof_rst >> 8); /* EN_TOF_RST = 1 */
	i2c_write (card, 0xdc, 0x0a, data, 1);
	*p = 0xa0 | (tof_rst >> 8); /* TOF_INIT = 1 */
	i2c_write (card, 0xdc, 0x0a, data, 1);
	/* Direct communication with National indicates that
	 * TOF alignment can cause PLL1 to become unlocked,
	 * so we need to wait for lock before the end of the sequence
	 * instead of before the start of the sequence. */
	if (iface->clksrc != SDIVIDEO_CTL_TX_CLKSRC_ONBOARD) {
		/* Wait to acquire the reference */
		msleep (500);

		/* Wait to gain PLL1 lock or lose reference */
		do {
			msleep (50);
			status = readl (card->core.addr + HDSDIM_TXE_STATUS);
		} while ((status & HDSDIM_TXE_STATUS_LOCK_LOST) &&
			!(status & HDSDIM_TXE_STATUS_REF_LOST));
	}
	msleep (50);
	*p = 0x00 | (tof_rst >> 8); /* EN_TOF_RST = 0, TOF_INIT = 0 */
	i2c_write (card, 0xdc, 0x0a, data, 1);

	/* Enable the HOTLink and wait for the reset sequence */
	writel (reg | HDSDIM_TXE_CTRL_SWRST,
		card->core.addr + HDSDIM_TXE_CTRL);
	do {
		msleep (1);
	} while (!(readl (card->core.addr + HDSDIM_TXE_STATUS) &
		HDSDIM_TXE_STATUS_TXCLK_LOCKED));

	/* Dummy read to flush PCI posted writes */
	readl(card->core.addr + HDSDIM_TXE_FPGAID);

	return;
}

/**
 * hdsdim_txe_start - Activate the VidPort SD/HD O Video Transmitter
 * @iface: interface
 **/
static void
hdsdim_txe_start (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;

	/* Enable transmitter interrupts */
	writel(HDSDIM_TXE_INT_DMA0 | HDSDIM_TXE_INT_TVUI |
		HDSDIM_TXE_INT_TVDI | HDSDIM_TXE_INT_REFI,
		card->core.addr + HDSDIM_TXE_IMS);

	/* Enable the transmitter
	 * Put the card out of reset
	 */
	reg = readl(card->core.addr + HDSDIM_TXE_CTRL);
	writel (reg & ~HDSDIM_TXE_CTRL_SWRST,
		card->core.addr + HDSDIM_TXE_CTRL);

	/* Dummy read to flush PCI posted writes */
	readl(card->core.addr + HDSDIM_TXE_FPGAID);

	return;
}

/**
 * hdsdim_txe_audstart - Activate the VidPort SD/HD O Audio Transmitter
 * @iface: interface
 **/
static void
hdsdim_txe_audstart (struct master_iface *iface)
{
	struct master_dev *card = iface->card;

	/* Enable transmitter interrupts */
	writel(HDSDIM_TXE_INT_DMA1 | HDSDIM_TXE_INT_TAUI |
		HDSDIM_TXE_INT_TADI,
		card->core.addr + HDSDIM_TXE_IMS);

	/* Dummy read to flush PCI posted writes */
	readl(card->core.addr + HDSDIM_TXE_FPGAID);

	return;
}

/**
 * hdsdim_txe_stop - Deactivate the VidPort SD/HD O
 * @iface: interface
 **/
static void
hdsdim_txe_stop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	struct master_dma *dma = iface->dma;
	unsigned int reg;

	lsdma_tx_link_all (dma);
	wait_event (iface->queue, test_bit (0, &iface->dma_done));
	lsdma_reset (dma);

	/* Wait for the onboard FIFOs to empty */
	/*
	wait_event (iface->queue,
		!(readl (card->core.addr + HDSDIM_TXE_ISR) &
		SDIM_QOE_ICSR_TXD));
	*/
	msleep(200);

	/* Disable the Transmitter */
	reg = readl(card->core.addr + HDSDIM_TXE_CTRL);
	writel(reg | HDSDIM_TXE_CTRL_SWRST,
		card->core.addr + HDSDIM_TXE_CTRL);

	/* Disable transmitter interrupts */
	writel (HDSDIM_TXE_INT_DMA0 | HDSDIM_TXE_INT_TVUI |
		HDSDIM_TXE_INT_TVDI | HDSDIM_TXE_INT_REFI,
		card->core.addr + HDSDIM_TXE_IMC);

	/* Dummy read to flush PCI posted writes */
	readl(card->core.addr + HDSDIM_TXE_FPGAID);

	return;
}

/**
 * hdsdim_txe_audstop - Deactivate the VidPort SD/HD O
 * @iface: interface
 **/
static void
hdsdim_txe_audstop (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	struct master_dma *dma = iface->dma;

	/* If the firmware is enabled, wait for DMA done */
	if (!(readl(card->core.addr + HDSDIM_TXE_CTRL) &
		HDSDIM_TXE_CTRL_SWRST)) {
		lsdma_tx_link_all (dma);
		wait_event (iface->queue, test_bit (0, &iface->dma_done));
		lsdma_reset (dma);
	}

	/* Disable the audio */
	writel (HDSDIM_TXE_AUDCTRL_CHAN_0,
		card->core.addr + HDSDIM_TXE_AUDCTRL);

	/* Disable transmitter interrupts */
	writel (HDSDIM_TXE_INT_DMA1 | HDSDIM_TXE_INT_TAUI |
		HDSDIM_TXE_INT_TADI,
		card->core.addr + HDSDIM_TXE_IMC);

	/* Dummy read to flush PCI posted writes */
	readl(card->core.addr + HDSDIM_TXE_FPGAID);

	return;
}

/**
 * hdsdim_txe_exit - Clean up the VidPort SD/HD O
 * @iface: interface
 **/
static void
hdsdim_txe_exit (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	unsigned int reg;

	/* Disable the HOTLink and clock generator */
	reg = readl(card->core.addr + HDSDIM_TXE_CTRL);
	writel(reg | HDSDIM_TXE_CTRL_SWRST |
		HDSDIM_TXE_CTRL_HL2RST,
		card->core.addr + HDSDIM_TXE_CTRL);
	writel(reg | HDSDIM_TXE_CTRL_SWRST |
		HDSDIM_TXE_CTRL_HL2RST |
		HDSDIM_TXE_CTRL_CLKRST,
		card->core.addr + HDSDIM_TXE_CTRL);
	return;
}

/**
 * hdsdim_txe_start_tx_dma - start transmit DMA
 * @iface: interface
 **/
static void
hdsdim_txe_start_tx_dma (struct master_iface *iface)
{
	struct master_dev *card = iface->card;
	struct master_dma *dma = iface->dma;
	const unsigned int dma_channel = mdev_index (card, &iface->list);

	writel (LSDMA_CH_CSR_INTDONEENABLE |
		LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_64BIT,
		card->bridge_addr + LSDMA_CSR(dma_channel));
	writel (mdma_dma_to_desc_low (lsdma_head_desc_bus_addr (dma)),
		card->bridge_addr + LSDMA_DESC(dma_channel));
	writel (mdma_dma_to_desc_high (lsdma_head_desc_bus_addr (dma)),
		card->bridge_addr + LSDMA_DESC_H(dma_channel));
	clear_bit (0, &iface->dma_done);
	writel (LSDMA_CH_CSR_INTDONEENABLE |
		LSDMA_CH_CSR_INTSTOPENABLE |
		LSDMA_CH_CSR_64BIT |
		LSDMA_CH_CSR_ENABLE,
		card->bridge_addr + LSDMA_CSR(dma_channel));
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + LSDMA_INTMSK);
	return;
}


/**
 * hdsdim_txe_unlocked_ioctl - VidPort SD/HD O unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
hdsdim_txe_unlocked_ioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;
	unsigned int val;

	switch (cmd) {
	case SDIVIDEO_IOC_TXGETREF:
		val = readl (card->core.addr + HDSDIM_TXE_STATUS) &
			HDSDIM_TXE_STATUS_REF_LOST;
		if (put_user (val, (unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return sdivideo_txioctl (filp, cmd, arg);
	}

	return 0;
}

/**
 * hdsdim_txe_unlocked_audioctl - VidPort SD/HD O audio unlocked_ioctl() method
 * @filp: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static long
hdsdim_txe_unlocked_audioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	return sdiaudio_txioctl (filp, cmd, arg);
}

/**
 * hdsdim_txe_fsync - VidPort SD/HD O fsync() method
 * @filp: file to flush
 * @datasync: used by filesystems
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
FSYNC_HANDLER(hdsdim_txe_fsync,filp,datasync)
{
	struct master_iface *iface = filp->private_data;
	struct master_dma *dma = iface->dma;

	mutex_lock (&iface->buf_mutex);
	lsdma_tx_link_all(dma);
	wait_event(iface->queue, test_bit(0, &iface->dma_done));
	lsdma_reset(dma);

	//TO-DO:
	/* Wait for the onboard FIFOs to empty */
	/*
	wait_event(iface->queue,
		!(readl(card->core.addr +
			HDSDIM_TXE_ISR) & SDIM_QOE_ICSR_TXD));
	*/
	msleep(200);

	mutex_unlock (&iface->buf_mutex);

	return 0;
}

