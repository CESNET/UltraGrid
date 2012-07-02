/* dvbm_tx.h
 *
 * Header file for the Linear Systems Ltd. DVB Master Send.
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

#ifndef _DVBM_TX_H
#define _DVBM_TX_H

#include <linux/pci.h> /* pci_dev */
#include <linux/init.h> /* __devinit */

#define DVBM_PCI_DEVICE_ID_LINSYS_DVBTX 0x7629
#define DVBM_NAME_TX "DVB Master Send"

/* DVB Master Send configuration */
#define DVBM_TX_PAE 0x1f0
#define DVBM_TX_PAF 0x005

/* Register addresses */
#define DVBM_TX_CFG		0x00
#define DVBM_TX_STATUS		0x00
#define DVBM_TX_FIFO		0x04
#define DVBM_TX_STUFFING	0x08
#define DVBM_TX_FINETUNE	0x0c
#define DVBM_TX_COUNTR		0x0c

/* Configuration register bit locations */
#define DVBM_TX_CFG_FIFOLEVELS	0x00000001
#define DVBM_TX_CFG_ENABLE	0x00000002
#define DVBM_TX_CFG_204		0x00000004
#define DVBM_TX_CFG_27MHZ	0x00000008

/* Status register bit locations */
#define DVBM_TX_STATUS_FIFOLEVELS	0x00000001
#define DVBM_TX_STATUS_ENABLE		0x00000002
#define DVBM_TX_STATUS_204		0x00000004
#define DVBM_TX_STATUS_27MHZ		0x00000008

/* Status register bitmasks */
#define DVBM_TX_STATUS_VERSIONMASK	0xff000000

/* External function prototypes */

int dvbm_tx_pci_probe (struct pci_dev *pdev) __devinit;
void dvbm_tx_pci_remove (struct pci_dev *pdev);

#endif

