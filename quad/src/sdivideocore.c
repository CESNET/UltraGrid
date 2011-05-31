/* sdivideocore.c
 *
 * Linear Systems Ltd. SDI video API
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

#include <linux/version.h> /* LINUX_VERSION_CODE */
#include <linux/kernel.h> /* snprintf () */
#include <linux/module.h> /* MODULE_LICENSE */
#include <linux/moduleparam.h> /* module_param () */

#include <linux/init.h> /* module_init () */
#include <linux/fs.h> /* register_chrdev_region () */
#include <linux/slab.h> /* kzalloc () */
#include <linux/errno.h> /* error codes */
#include <linux/list.h> /* list_add_tail () */
#include <linux/spinlock.h> /* spin_lock_init () */
#include <linux/wait.h> /* init_waitqueue_head () */
#include <linux/device.h> /* class_create () */
#include <linux/cdev.h> /* cdev_init () */
#include <linux/mutex.h> /* mutex_init () */

#include <asm/uaccess.h> /* put_user () */
#include <asm/bitops.h> /* test_and_clear_bit () */

#include "sdivideocore.h"
#include "../include/master.h"
#include "miface.h"
#include "mdma.h"

/* Static function prototypes */
static int sdivideo_ioctl (struct master_iface *iface,
	unsigned int cmd,
	unsigned long arg);
static int sdivideo_validate_buffers (struct master_iface *iface,
	unsigned long val);
static int sdivideo_validate_bufsize (struct master_iface *iface,
	unsigned long val);
static int sdivideo_validate_clksrc (struct master_iface *iface,
	unsigned long val);
static int sdivideo_validate_frmode (struct master_iface *iface,
	unsigned long val);
static int sdivideo_validate_mode (struct master_iface *iface,
	unsigned long val);
static int sdivideo_validate_vb1cnt (struct master_iface *iface,
	unsigned long val);
static int sdivideo_validate_vb1ln1 (struct master_iface *iface,
	unsigned long val);
static int sdivideo_validate_vb2cnt (struct master_iface *iface,
	unsigned long val);
static int sdivideo_validate_vb2ln1 (struct master_iface *iface,
	unsigned long val);
static ssize_t sdivideo_store_buffers (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static ssize_t sdivideo_store_bufsize (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static ssize_t sdivideo_store_clksrc (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static ssize_t sdivideo_store_frmode (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static ssize_t sdivideo_store_mode (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static ssize_t sdivideo_store_vanc (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static ssize_t sdivideo_store_vb1cnt (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static ssize_t sdivideo_store_vb1ln1 (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static ssize_t sdivideo_store_vb2cnt (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static ssize_t sdivideo_store_vb2ln1 (struct device *dev,
	struct device_attribute *attr,
	const char *buf, size_t count);
static int sdivideo_init_module (void) __init;
static void sdivideo_cleanup_module (void) __exit;

/* The major number can be overridden from the command line
 * when the module is loaded. */
static unsigned int major = SDIVIDEO_MAJOR;
module_param(major,uint,S_IRUGO);
MODULE_PARM_DESC(major,"Major number of the first device, or zero for dynamic");

static const unsigned int count = 1 << MASTER_MINORBITS;

MODULE_AUTHOR("Linear Systems Ltd.");
MODULE_DESCRIPTION("SMPTE 292M and SMPTE 259M-C video module");
MODULE_LICENSE("GPL");
MODULE_VERSION(MASTER_DRIVER_VERSION);

EXPORT_SYMBOL(sdivideo_open);
EXPORT_SYMBOL(sdivideo_write);
EXPORT_SYMBOL(sdivideo_read);
EXPORT_SYMBOL(sdivideo_txpoll);
EXPORT_SYMBOL(sdivideo_rxpoll);
EXPORT_SYMBOL(sdivideo_txioctl);
EXPORT_SYMBOL(sdivideo_rxioctl);
EXPORT_SYMBOL(sdivideo_mmap);
EXPORT_SYMBOL(sdivideo_release);
EXPORT_SYMBOL(sdivideo_register_iface);
EXPORT_SYMBOL(sdivideo_unregister_iface);

static char sdivideo_driver_name[] = SDIVIDEO_DRIVER_NAME;

static LIST_HEAD(sdivideo_iface_list);

static spinlock_t sdivideo_iface_lock;

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,27))
#define device_create(cls,parent,devt,drvdata,fmt,...) \
	device_create(cls,parent,devt,fmt,##__VA_ARGS__)
#endif

static struct class *sdivideo_class;
static CLASS_ATTR(version,S_IRUGO,
	miface_show_version,NULL);

/**
 * sdivideo_open - SMPTE 292M and SMPTE 259M-C video interface open() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
int
sdivideo_open (struct inode *inode, struct file *filp)
{
	return miface_open (inode, filp);
}

/**
 * sdivideo_write - SMPTE 292M and SMPTE 259M-C video interface write() method
 * @filp: file
 * @data: data
 * @length: size of data
 * @offset:
 *
 * Returns a negative error code on failure and
 * the number of bytes written on success.
 **/
ssize_t
sdivideo_write (struct file *filp,
	const char __user *data,
	size_t length,
	loff_t *offset)
{
	return miface_write (filp, data, length, offset);
}

/**
 * sdivideo_read - SMPTE 292M and SMPTE 259M-C video interface read() method
 * @filp: file
 * @data: read buffer
 * @length: size of data
 * @offset:
 *
 * Returns a negative error code on failure and
 * the number of bytes read on success.
 **/
ssize_t
sdivideo_read (struct file *filp,
	char __user *data,
	size_t length,
	loff_t *offset)
{
	return miface_read (filp, data, length, offset);
}

/**
 * sdivideo_txpoll - transmitter poll() method
 * @filp: file
 * @wait:
 **/
unsigned int
sdivideo_txpoll (struct file *filp, poll_table *wait)
{
	return miface_txpoll (filp, wait);
}

/**
 * sdivideo_rxpoll - receiver poll() method
 * @filp: file
 * @wait:
 **/
unsigned int
sdivideo_rxpoll (struct file *filp, poll_table *wait)
{
	return miface_rxpoll (filp, wait);
}

/**
 * sdivideo_ioctl - generic ioctl() method
 * @iface: interface
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
sdivideo_ioctl (struct master_iface *iface,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_dev *card = iface->card;

	switch (cmd) {
	case SDIVIDEO_IOC_GETID:
		if (put_user (card->id,
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIVIDEO_IOC_GETVERSION:
		if (put_user (card->version, (unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	default:
		return -ENOTTY;
	}
	return 0;
}

/**
 * sdivideo_txioctl - SMPTE 292M and SMPTE 259M-C video transmitter interface ioctl() method
 * @iface: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
long
sdivideo_txioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	unsigned int reg = 0, i;

	switch (cmd) {
	case SDIVIDEO_IOC_TXGETCAP:
		if (put_user (iface->capabilities,
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIVIDEO_IOC_TXGETEVENTS:
		for (i = 0; i < 8 * sizeof (reg); i++) {
			if (test_and_clear_bit (i, &iface->events)) {
				reg |= (1 << i);
			}
		}
		if (put_user (reg, (unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIVIDEO_IOC_TXGETBUFLEVEL:
		if (put_user (mdma_tx_buflevel (iface->dma),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIVIDEO_IOC_QBUF_DEPRECATED:
	case SDIVIDEO_IOC_QBUF:
		return miface_txqbuf (filp, arg);
	case SDIVIDEO_IOC_DQBUF_DEPRECATED:
	case SDIVIDEO_IOC_DQBUF:
		return miface_txdqbuf (filp, arg);
	default:
		return sdivideo_ioctl (iface, cmd, arg);
	}
	return 0;
}

/**
 * sdivideo_rxioctl - SMPTE 292M and SMPTE 259M-C video receiver interface ioctl() method
 * @iface: file
 * @cmd: ioctl command
 * @arg: ioctl argument
 *
 * Returns a negative error code on failure and 0 on success.
 **/
long
sdivideo_rxioctl (struct file *filp,
	unsigned int cmd,
	unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	unsigned int reg = 0, i;

	switch (cmd) {
	case SDIVIDEO_IOC_RXGETCAP:
		if (put_user (iface->capabilities,
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIVIDEO_IOC_RXGETEVENTS:
		for (i = 0; i < 8 * sizeof (reg); i++) {
			if (test_and_clear_bit (i, &iface->events)) {
				reg |= (1 << i);
			}
		}
		if (put_user (reg, (unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIVIDEO_IOC_RXGETBUFLEVEL:
		if (put_user (mdma_rx_buflevel (iface->dma),
			(unsigned int __user *)arg)) {
			return -EFAULT;
		}
		break;
	case SDIVIDEO_IOC_QBUF_DEPRECATED:
	case SDIVIDEO_IOC_QBUF:
		return miface_rxqbuf (filp, arg);
	case SDIVIDEO_IOC_DQBUF_DEPRECATED:
	case SDIVIDEO_IOC_DQBUF:
		return miface_rxdqbuf (filp, arg);
	default:
		return sdivideo_ioctl (iface, cmd, arg);
	}
	return 0;
}

/**
 * sdivideo_mmap - SMPTE 292M and SMPTE 259M-C video interface mmap() method
 * @filp: file
 * @vma: VMA
 **/
int
sdivideo_mmap (struct file *filp, struct vm_area_struct *vma)
{
	return miface_mmap (filp, vma);
}

/**
 * sdivideo_release - SMPTE 292M and SMPTE 259M-C video interface release() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
int
sdivideo_release (struct inode *inode, struct file *filp)
{
	return miface_release (inode, filp);
}

/**
 * sdivideo_validate_buffers - validate a buffers attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
sdivideo_validate_buffers (struct master_iface *iface,
	unsigned long val)
{
	const unsigned int min = (iface->direction == MASTER_DIRECTION_TX) ?
		SDIVIDEO_TX_BUFFERS_MIN : SDIVIDEO_RX_BUFFERS_MIN;
	const unsigned int max = SDIVIDEO_BUFFERS_MAX;

	if ((val < min) || (val > max)) {
		return -EINVAL;
	}
	return 0;
}

/**
 * sdivideo_validate_bufsize - validate a bufsize attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
sdivideo_validate_bufsize (struct master_iface *iface,
	unsigned long val)
{
	const unsigned int min = (iface->direction == MASTER_DIRECTION_TX) ?
		SDIVIDEO_TX_BUFSIZE_MIN : SDIVIDEO_RX_BUFSIZE_MIN;
	const unsigned int max = SDIVIDEO_BUFSIZE_MAX;
	const unsigned int mult = iface->granularity;

	if ((val < min) || (val > max) || (val % mult)) {
		return -EINVAL;
	}
	return 0;
}

/**
 * sdivideo_validate_clksrc - validate a clksrc attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
sdivideo_validate_clksrc (struct master_iface *iface,
	unsigned long val)
{
	const unsigned int min = SDIVIDEO_CTL_TX_CLKSRC_ONBOARD;
	const unsigned int max = SDIVIDEO_CTL_TX_CLKSRC_1080I_50;

	if ((val < min) || (val > max)) {
		return -EINVAL;
	}
	return 0;
}

/**
 * sdivideo_validate_frmode - validate a frmode attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
sdivideo_validate_frmode (struct master_iface *iface,
	unsigned long val)
{
	switch (val) {
	case SDIVIDEO_CTL_SMPTE_125M_486I_59_94HZ:
	case SDIVIDEO_CTL_BT_601_576I_50HZ:
	case SDIVIDEO_CTL_SMPTE_260M_1035I_60HZ:
	case SDIVIDEO_CTL_SMPTE_260M_1035I_59_94HZ:
	case SDIVIDEO_CTL_SMPTE_274M_1080I_60HZ:
	case SDIVIDEO_CTL_SMPTE_274M_1080PSF_30HZ:
	case SDIVIDEO_CTL_SMPTE_274M_1080I_59_94HZ:
	case SDIVIDEO_CTL_SMPTE_274M_1080PSF_29_97HZ:
	case SDIVIDEO_CTL_SMPTE_274M_1080I_50HZ:
	case SDIVIDEO_CTL_SMPTE_274M_1080PSF_25HZ:
	case SDIVIDEO_CTL_SMPTE_274M_1080PSF_24HZ:
	case SDIVIDEO_CTL_SMPTE_274M_1080PSF_23_98HZ:
	case SDIVIDEO_CTL_SMPTE_274M_1080P_30HZ:
	case SDIVIDEO_CTL_SMPTE_274M_1080P_29_97HZ:
	case SDIVIDEO_CTL_SMPTE_274M_1080P_25HZ:
	case SDIVIDEO_CTL_SMPTE_274M_1080P_24HZ:
	case SDIVIDEO_CTL_SMPTE_274M_1080P_23_98HZ:
	case SDIVIDEO_CTL_SMPTE_296M_720P_60HZ:
	case SDIVIDEO_CTL_SMPTE_296M_720P_59_94HZ:
	case SDIVIDEO_CTL_SMPTE_296M_720P_50HZ:
	case SDIVIDEO_CTL_SMPTE_296M_720P_30HZ:
	case SDIVIDEO_CTL_SMPTE_296M_720P_29_97HZ:
	case SDIVIDEO_CTL_SMPTE_296M_720P_25HZ:
	case SDIVIDEO_CTL_SMPTE_296M_720P_24HZ:
	case SDIVIDEO_CTL_SMPTE_296M_720P_23_98HZ:
		break;
	default:
	case SDIVIDEO_CTL_UNLOCKED:
	case SDIVIDEO_CTL_SMPTE_295M_1080I_50HZ:
		return -EINVAL;
	}
	return 0;
}

/**
 * sdivideo_validate_mode - validate a mode attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
sdivideo_validate_mode (struct master_iface *iface,
	unsigned long val)
{
	const unsigned int min = SDIVIDEO_CTL_MODE_UYVY;
	unsigned int max;

	if (iface->capabilities & SDIVIDEO_CAP_RX_RAWMODE) {
		max = SDIVIDEO_CTL_MODE_RAW;
	} else if (iface->capabilities & SDIVIDEO_CAP_RX_DEINTERLACING) {
		max = SDIVIDEO_CTL_MODE_V210_DEINTERLACE;
	} else {
		max = SDIVIDEO_CTL_MODE_V210;
	}
	if ((val < min) || (val > max)) {
		return -EINVAL;
	}
	return 0;
}

#define sdivideo_validate_vanc(iface,val) (0)

/**
 * sdivideo_validate_vb1cnt - validate a vb1cnt attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
sdivideo_validate_vb1cnt (struct master_iface *iface,
	unsigned long val)
{
	const unsigned int min = 0;
	const unsigned int max = 40; // maximum VBI lines for 1080i

	if ((val < min) || (val > max)) {
		return -EINVAL;
	}
	return 0;
}

/**
 * sdivideo_validate_vb1ln1 - validate a vb1ln1 attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
sdivideo_validate_vb1ln1 (struct master_iface *iface,
	unsigned long val)
{
	const unsigned int min = 1;
	const unsigned int max = 40;

	if ((val < min) || (val > max)) {
		return -EINVAL;
	}
	return 0;
}

/**
 * sdivideo_validate_vb2cnt - validate a vb2cnt attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
sdivideo_validate_vb2cnt (struct master_iface *iface,
	unsigned long val)
{
	const unsigned int min = 0;
	const unsigned int max = 45;

	if ((val < min) || (val > max)) {
		return -EINVAL;
	}
	return 0;
}

/**
 * sdivideo_validate_vb2ln1 - validate a vb2ln1 attribute value
 * @iface: interface being written
 * @val: new attribute value
 **/
static int
sdivideo_validate_vb2ln1 (struct master_iface *iface,
	unsigned long val)
{
	const unsigned int min = 558;
	const unsigned int max = 602;

	if ((val < min) || (val > max)) {
		return -EINVAL;
	}
	return 0;
}

/**
 * sdivideo_store_* - SMPTE 292M and SMPTE 259M-C interface attribute write handler
 * @dev: device being written
 * @attr: device attribute
 * @buf: input buffer
 * @count:
 **/
#define SDIVIDEO_STORE(var) \
	static ssize_t sdivideo_store_##var (struct device *dev, \
		struct device_attribute *attr, \
		const char *buf, \
		size_t count) \
	{ \
		struct master_iface *iface = dev_get_drvdata (dev); \
		char *endp; \
		unsigned long val = simple_strtoul (buf, &endp, 0); \
		ssize_t err; \
		if ((endp == buf) || \
			sdivideo_validate_##var (iface, val)) { \
			return -EINVAL; \
		} \
		err = miface_store (iface, &iface->var, val); \
		if (err) { \
			return err; \
		} \
		return count; \
	}
SDIVIDEO_STORE(buffers)
SDIVIDEO_STORE(bufsize)
SDIVIDEO_STORE(clksrc)
SDIVIDEO_STORE(frmode)
SDIVIDEO_STORE(mode)
SDIVIDEO_STORE(vanc)
SDIVIDEO_STORE(vb1cnt)
SDIVIDEO_STORE(vb1ln1)
SDIVIDEO_STORE(vb2cnt)
SDIVIDEO_STORE(vb2ln1)

static DEVICE_ATTR(buffers,S_IRUGO|S_IWUSR,
	miface_show_buffers,sdivideo_store_buffers);
static DEVICE_ATTR(bufsize,S_IRUGO|S_IWUSR,
	miface_show_bufsize,sdivideo_store_bufsize);
static DEVICE_ATTR(clock_source,S_IRUGO|S_IWUSR,
	miface_show_clksrc,sdivideo_store_clksrc);
static DEVICE_ATTR(frame_mode,S_IRUGO|S_IWUSR,
	miface_show_frmode,sdivideo_store_frmode);
static DEVICE_ATTR(mode,S_IRUGO|S_IWUSR,
	miface_show_mode,sdivideo_store_mode);
static DEVICE_ATTR(vanc,S_IRUGO|S_IWUSR,
	miface_show_vanc,sdivideo_store_vanc);
static DEVICE_ATTR(vb1_cnt,S_IRUGO|S_IWUSR,
	miface_show_vb1cnt,sdivideo_store_vb1cnt);
static DEVICE_ATTR(vb1_ln1,S_IRUGO|S_IWUSR,
	miface_show_vb1ln1,sdivideo_store_vb1ln1);
static DEVICE_ATTR(vb2_cnt,S_IRUGO|S_IWUSR,
	miface_show_vb2cnt,sdivideo_store_vb2cnt);
static DEVICE_ATTR(vb2_ln1,S_IRUGO|S_IWUSR,
	miface_show_vb2ln1,sdivideo_store_vb2ln1);

/**
 * sdivideo_register_iface - register an interface
 * @card: pointer to the board info structure
 * @dma_ops: pointer to DMA helper functions
 * @data_addr: local bus address of the FIFO
 * @direction: direction of data flow
 * @fops: file operations structure
 * @iface_ops: pointer to Master interface helper functions
 * @cap: capabilities flags
 * @granularity: buffer size granularity in bytes
 *
 * Allocate and initialize an interface information structure.
 * Assign the lowest unused minor number to this interface
 * and add it to the list of interfaces for this device
 * and the list of all interfaces.
 * Also initialize the device parameters.
 * Returns a negative error code on failure and 0 on success.
 **/
int
sdivideo_register_iface (struct master_dev *card,
	struct master_dma_operations *dma_ops,
	u32 data_addr,
	unsigned int direction,
	struct file_operations *fops,
	struct master_iface_operations *iface_ops,
	unsigned int cap,
	unsigned int granularity)
{
	struct master_iface *iface, *entry;
	const char *type;
	struct list_head *p;
	unsigned int minminor, minor, maxminor, found;
	int err;

	/* Allocate an interface info structure */
	iface = (struct master_iface *)kzalloc (sizeof (*iface), GFP_KERNEL);
	if (iface == NULL) {
		err = -ENOMEM;
		goto NO_IFACE;
	}

	/* Initialize an interface info structure */
	iface->direction = direction;
	cdev_init (&iface->cdev, fops);
	iface->cdev.owner = THIS_MODULE;
	iface->capabilities = cap;
	if (iface->direction == MASTER_DIRECTION_TX) {
		iface->buffers = SDIVIDEO_TX_BUFFERS;
		iface->bufsize = SDIVIDEO_TX_BUFSIZE;
		iface->clksrc = SDIVIDEO_CTL_TX_CLKSRC_ONBOARD;
		iface->frmode = SDIVIDEO_CTL_SMPTE_296M_720P_59_94HZ;
	} else {
		iface->buffers = SDIVIDEO_RX_BUFFERS;
		iface->bufsize = SDIVIDEO_RX_BUFSIZE;
		iface->vb1cnt = 0;
		iface->vb1ln1 = 1;
		iface->vb2cnt = 0;
		iface->vb2ln1 = 558;
	}
	iface->granularity = granularity;
	iface->mode = SDIVIDEO_CTL_MODE_UYVY;
	iface->vanc = 0;
	iface->ops = iface_ops;
	iface->dma_ops = dma_ops;
	iface->data_addr = data_addr;
	iface->dma_flags = MDMA_MMAP;
	init_waitqueue_head (&iface->queue);
	mutex_init (&iface->buf_mutex);
	iface->card = card;

	/* Assign the lowest unused minor number to this interface */
	switch (iface->direction) {
	case MASTER_DIRECTION_TX:
		type = "transmitter";
		minminor = minor = 0;
		break;
	case MASTER_DIRECTION_RX:
		type = "receiver";
		minminor = minor = 1 << (MASTER_MINORBITS - 1);
		break;
	default:
		err = -EINVAL;
		goto NO_MINOR;
	}
	maxminor = minor + (1 << (MASTER_MINORBITS - 1)) - 1;
	spin_lock (&sdivideo_iface_lock);
	while (minor <= maxminor) {
		found = 0;
		list_for_each (p, &sdivideo_iface_list) {
			entry = list_entry (p, struct master_iface, list_all);
			if (MINOR(entry->cdev.dev) == minor) {
				found = 1;
				break;
			}
		}
		if (!found) {
			break;
		}
		minor++;
	}
	spin_unlock (&sdivideo_iface_lock);
	if (minor > maxminor) {
		err = -ENOSPC;
		goto NO_MINOR;
	}

	/* Add this interface to the list of all interfaces */
	spin_lock (&sdivideo_iface_lock);
	list_add_tail (&iface->list_all, &sdivideo_iface_list);
	spin_unlock (&sdivideo_iface_lock);

	/* Add this interface to the list for this device */
	list_add_tail (&iface->list, &card->iface_list);

	/* Create the device */
	iface->dev = device_create (sdivideo_class,
		card->dev,
		MKDEV(major, minor),
		iface,
		"sdivideo%cx%i",
		type[0],
		minor & ((1 << (MASTER_MINORBITS - 1)) - 1));
	if (IS_ERR(iface->dev)) {
		printk (KERN_WARNING "%s: unable to register device\n",
			sdivideo_driver_name);
		err = PTR_ERR(iface->dev);
		goto NO_DEV;
	}
	dev_set_drvdata (iface->dev, iface);

	/* Add device attributes */
	if ((err = device_create_file (iface->dev,
		&dev_attr_buffers)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create file 'buffers'\n",
			sdivideo_driver_name);
	}
	if ((err = device_create_file (iface->dev,
		&dev_attr_bufsize)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create file 'bufsize'\n",
			sdivideo_driver_name);
	}
	if (iface->direction == MASTER_DIRECTION_TX) {
		if ((err = device_create_file (iface->dev,
			&dev_attr_clock_source)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'clock_source'\n",
				sdivideo_driver_name);
		}
		if ((err = device_create_file (iface->dev,
			&dev_attr_frame_mode)) < 0) {
			printk (KERN_WARNING
				"%s: unable to create file 'frame_mode'\n",
				sdivideo_driver_name);
		}
		if (iface->capabilities & SDIVIDEO_CAP_TX_VANC) {
			if ((err = device_create_file (iface->dev,
				&dev_attr_vanc)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'vanc'\n",
					sdivideo_driver_name);
			}
		}
	}
	if (iface->direction == MASTER_DIRECTION_RX) {
		if (iface->capabilities & SDIVIDEO_CAP_RX_VANC) {
			if ((err = device_create_file (iface->dev,
				&dev_attr_vanc)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'vanc'\n",
					sdivideo_driver_name);
			}
		}
		if (iface->capabilities & SDIVIDEO_CAP_RX_VBI) {
			if ((err = device_create_file (iface->dev,
				&dev_attr_vb1_cnt)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'vb1_cnt'\n",
					sdivideo_driver_name);
			}
			if ((err = device_create_file (iface->dev,
				&dev_attr_vb1_ln1)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'vb1_ln1'\n",
					sdivideo_driver_name);
			}
			if ((err = device_create_file (iface->dev,
				&dev_attr_vb2_cnt)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'vb2_cnt'\n",
					sdivideo_driver_name);
			}
			if ((err = device_create_file (iface->dev,
				&dev_attr_vb2_ln1)) < 0) {
				printk (KERN_WARNING
					"%s: unable to create file 'vb2_ln1'\n",
					sdivideo_driver_name);
			}
		}
	}
	if ((err = device_create_file (iface->dev,
		&dev_attr_mode)) < 0) {
		printk (KERN_WARNING
			"%s: unable to create file 'mode'\n",
			sdivideo_driver_name);
	}

	/* Activate the cdev */
	if ((err = cdev_add (&iface->cdev, MKDEV(major,minor), 1)) < 0) {
		printk (KERN_WARNING "%s: unable to add character device\n",
			sdivideo_driver_name);
		goto NO_CDEV;
	}

	printk (KERN_INFO "%s: registered %s %u\n",
		sdivideo_driver_name, type, minor - minminor);
	return 0;

NO_CDEV:
	device_destroy (sdivideo_class, MKDEV(major,minor));
NO_DEV:
	list_del (&iface->list);
	spin_lock (&sdivideo_iface_lock);
	list_del (&iface->list_all);
	spin_unlock (&sdivideo_iface_lock);
NO_MINOR:
	kfree (iface);
NO_IFACE:
	return err;
}

/**
 * sdivideo_unregister_iface - remove an interface from the list
 * @iface: interface
 **/
void
sdivideo_unregister_iface (struct master_iface *iface)
{
	cdev_del (&iface->cdev);
	device_destroy (sdivideo_class, iface->cdev.dev);
	list_del (&iface->list);
	spin_lock (&sdivideo_iface_lock);
	list_del (&iface->list_all);
	spin_unlock (&sdivideo_iface_lock);
	kfree (iface);
	return;
}

/**
 * sdivideo_init_module - initialize the module
 *
 * Register the module as a character PCI driver.
 * Returns a negative error code on failure and 0 on success.
 **/
static int __init
sdivideo_init_module (void)
{
	int err;
	dev_t dev = MKDEV(major,0);

	spin_lock_init (&sdivideo_iface_lock);

	/* Create a device class */
	sdivideo_class = class_create (THIS_MODULE, sdivideo_driver_name);
	if (IS_ERR(sdivideo_class)) {
		printk (KERN_WARNING "%s: unable to register device class\n",
			sdivideo_driver_name);
		err = PTR_ERR(sdivideo_class);
		goto NO_CLASS;
	}

	/* Add class attributes */
	if ((err = class_create_file (sdivideo_class, &class_attr_version)) < 0) {
		printk (KERN_WARNING "%s: unable to create file 'version' \n",
			sdivideo_driver_name);
	}

	/* Reserve a range of device numbers */
	if (major) {
		err = register_chrdev_region (dev,
			count,
			sdivideo_driver_name);
	} else {
		err = alloc_chrdev_region (&dev,
			0,
			count,
			sdivideo_driver_name);
	}
	if (err < 0) {
		printk (KERN_WARNING
			"%s: unable to reserve device number range\n",
			sdivideo_driver_name);
		goto NO_RANGE;
	}
	major = MAJOR(dev);

	return 0;

NO_RANGE:
	class_destroy (sdivideo_class);
NO_CLASS:
	return err;
}

/**
 * sdivideo_cleanup_module - cleanup the module
 *
 * Unregister the module as a character PCI driver.
 **/
static void __exit
sdivideo_cleanup_module (void)
{
	unregister_chrdev_region (MKDEV(major,0), count);
	class_destroy (sdivideo_class);
	return;
}

module_init (sdivideo_init_module);
module_exit (sdivideo_cleanup_module);

