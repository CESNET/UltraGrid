/* dvbm_rx.h
 *
 * Header file for the Linear Systems Ltd. DVB Master Receive.
 *
 * Copyright (C) 2001-2010 Linear Systems Ltd.
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

#ifndef _DVBM_RX_H
#define _DVBM_RX_H

#include <linux/pci.h> /* pci_dev */
#include <linux/init.h> /* __devinit */

#define DVBM_PCI_DEVICE_ID_LINSYS_DVBRX 0x7630
#define DVBM_NAME_RX "DVB Master Receive"

/* DVB Master Receive configuration */
#define DVBM_RX_PAE 0x003
#define DVBM_RX_PAF 0x1f0

/* Register addresses */
#define DVBM_RX_CFG	0x00
#define DVBM_RX_STATUS	0x00
#define DVBM_RX_FIFO	0x04
#define DVBM_RX_CFG2	0x08
#define DVBM_RX_STATUS2	0x08
#define DVBM_RX_COUNTR	0x0c

/* Configuration register bit locations */
#define DVBM_RX_CFG_NORMAL		0x80000000 /* Normal Mode */
#define DVBM_RX_CFG_PGMFIFO		0x40000000 /* Program the FIFO */
#define DVBM_RX_CFG_REFRAME		0x20000000 /* HotLink Reframe Mode */
#define DVBM_RX_CFG_ENABLE		0x10000000 /* Receiver En. */
#define DVBM_RX_CFG_OVERRUNINT		0x08000000 /* Overrun Int. En. */
#define DVBM_RX_CFG_CLOVERRUNINT	0x04000000 /* Clear Overrun Int. */

#define DVBM_RX_CFG_DEFAULT	(DVBM_RX_CFG_NORMAL | \
				(DVBM_RX_PAF << 11) | DVBM_RX_PAE)

/* Status register bit locations */
#define DVBM_RX_STATUS_NORMAL		0x80000000 /* Normal Mode */
#define DVBM_RX_STATUS_PGMFIFO		0x40000000 /* Programming FIFO */
#define DVBM_RX_STATUS_REFRAME		0x20000000 /* HotLink Reframe Mode */
#define DVBM_RX_STATUS_ENABLE		0x10000000 /* Receiver Enabled */
#define DVBM_RX_STATUS_TESTREGBUSY	0x08000000 /* Test Register Busy */
#define DVBM_RX_STATUS_OVERRUN		0x04000000 /* Overrun Int. Act. */
#define DVBM_RX_STATUS_AF		0x02000000 /* FIFO Almost Full */
#define DVBM_RX_STATUS_AE		0x01000000 /* FIFO Almost Empty */
#define DVBM_RX_STATUS_FULL		0x00800000 /* FIFO Full */
#define DVBM_RX_STATUS_EMPTY		0x00400000 /* FIFO Empty */

/* Configuration register 2 bit locations */
#define DVBM_RX_CFG2_FORCEDMA		0x00008000 /* Force DMA */
#define DVBM_RX_CFG2_27MHZ		0x00004000 /* 27 MHz Counter En. */
#define DVBM_RX_CFG2_DSYNC		0x00002000 /* Double Packet Sync En. */
#define DVBM_RX_CFG2_PSTARTEDRST	0x00001000 /* Rst. Pkt. Started Flag */
#define DVBM_RX_CFG2_INVSYNC		0x00000800 /* Inv. Sync Bytes En. */
#define DVBM_RX_CFG2_CDIE		0x00000200 /* Carrier Det. Int. En. */
#define DVBM_RX_CFG2_AOSINT		0x00000040 /* Acq. of Sync Int. En. */
#define DVBM_RX_CFG2_APS		0x00000010 /* Auto Packet Size En. */
#define DVBM_RX_CFG2_LOSINT		0x00000004 /* Loss of Sync Int. En. */
#define DVBM_RX_CFG2_204		0x00000002 /* 204-byte Packets */
#define DVBM_RX_CFG2_SYNC		0x00000001 /* Packet Sync En. */

/* Status register 2 bit locations */
#define DVBM_RX_STATUS2_PSTARTED	0x00001000 /* Packet Started */
#define DVBM_RX_STATUS2_CD		0x00000400 /* Carrier Detect */
#define DVBM_RX_STATUS2_CDIS		0x00000200 /* Carrier Det. Int. Act. */
#define DVBM_RX_STATUS2_AOSINT		0x00000040 /* Acq. of Sync Int. Act. */
#define DVBM_RX_STATUS2_PASSING		0x00000020 /* Passing Data */
#define DVBM_RX_STATUS2_APS		0x00000010 /* Auto Packet Size */
#define DVBM_RX_STATUS2_204		0x00000008 /* 204-byte Packets Found */
#define DVBM_RX_STATUS2_LOSINT		0x00000004 /* Loss of Sync Int. Act. */

/* External function prototypes */

int dvbm_rx_pci_probe (struct pci_dev *pdev) __devinit;
void dvbm_rx_pci_remove (struct pci_dev *pdev);

#endif

