/* mdev.h
 *
 * Definitions for Linear Systems Ltd. Master devices.
 *
 * Copyright (C) 2005-2010 Linear Systems Ltd.
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

#ifndef _MDEV_H
#define _MDEV_H

#include <linux/version.h> /* LINUX_VERSION_CODE */

#include <linux/spinlock.h> /* spinlock_t */
#include <linux/list.h> /* list_head */
#include <linux/interrupt.h> /* irqreturn_t */
#include <linux/device.h> /* device */
#include <linux/mutex.h> /* mutex */

#include <asm/io.h> /* inl () */

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,18))
typedef unsigned long resource_size_t;
#endif

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,19))
#define IRQ_HANDLER(name,irq,dev_id,regs) \
	name (int irq, void *dev_id, struct pt_regs *regs)
#if defined RHEL_RELEASE_VERSION
#if RHEL_RELEASE_CODE < RHEL_RELEASE_VERSION(4,5) || \
RHEL_RELEASE_CODE == RHEL_RELEASE_VERSION(5,0)
typedef irqreturn_t (*irq_handler_t)(int, void *, struct pt_regs *);
#endif
#else
typedef irqreturn_t (*irq_handler_t)(int, void *, struct pt_regs *);
#endif
#else
#define IRQ_HANDLER(name,irq,dev_id,regs) \
	name (int irq, void *dev_id)
#endif

/**
 * master_dev - generic Master device
 * @list: handle for linked list of Master devices
 * @bridge_addr: PCI bridge or DMA Controller base address
 * @core: device core base address
 * @name: marketing name of this device
 * @id: device ID
 * @version: firmware version
 * @irq: interrupt line
 * @irq_handler: pointer to interrupt handler
 * @iface_list: linked list of interfaces
 * @capabilities: capabilities flags
 * @dev: pointer to device structure
 * @irq_lock: lock for shared board registers accessed in interrupt context
 * @reg_lock: lock for all other shared board registers
 * @users_mutex: mutex for iface[].users
 * @parent: parent device
 **/
struct master_dev {
	struct list_head list;
	void __iomem *bridge_addr;
	union {
		void __iomem *addr;
		resource_size_t port;
	} core;
	const char *name;
	unsigned int id;
	unsigned int version;
	unsigned int irq;
	irq_handler_t irq_handler;
	struct list_head iface_list;
	unsigned int capabilities;
	struct device *dev;
	spinlock_t irq_lock;
	spinlock_t reg_lock;
	struct mutex users_mutex;
	struct device *parent;
};

/* External function prototypes */

unsigned int mdev_index (struct master_dev *card, struct list_head *list);
int mdev_register (struct master_dev *card,
	struct list_head *devlist,
	char *driver_name,
	struct class *cls);
void mdev_unregister (struct master_dev *card, struct class *cls);
struct class *mdev_init (char *name);
void mdev_cleanup (struct class *cls);

/* Inline functions */

#define master_inl(card,offset) \
	inl((card)->core.port+(offset))
#define master_outl(card,offset,val) \
	outl((val),(card)->core.port+(offset))

#endif

