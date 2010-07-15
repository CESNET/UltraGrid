/* dvbm_fd.h
 *
 * Header file for the Linear Systems Ltd. DVB Master FD.
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

#ifndef _DVBM_FD_H
#define _DVBM_FD_H

#include <linux/pci.h> /* pci_dev */
#include <linux/init.h> /* __devinit */

#define DVBM_PCI_DEVICE_ID_LINSYS_DVBFD 0x7643
#define DVBM_NAME_FD "DVB Master FD"

/* DVB Master FD configuration */
#define DVBM_FD_RDMATL	0x010
#define DVBM_FD_TDMATL	0x1ef
#define DVBM_FD_TFSL	0x0c0

/* Register addresses */
#define DVBM_FD_CSR		0x00 /* Control/Status */
#define DVBM_FD_FIFO		0x04 /* FIFO */
#define DVBM_FD_STR		0x08 /* Stuffing */
#define DVBM_FD_FTR		0x0c /* Finetuning */
#define DVBM_FD_DMATLR		0x10 /* DMA Trigger Level */
#define DVBM_FD_FSR		0x14 /* FIFO Status */
#define DVBM_FD_ICSR		0x18 /* Interrupt Control/Status */
#define DVBM_FD_TFSLR		0x1c /* Transmit FIFO Start Level */
#define DVBM_FD_TXBCOUNTR	0x20 /* Transmit Byte Count */
#define DVBM_FD_RXBCOUNTR	0x24 /* Receive Byte Count */
#define DVBM_FD_PFLUTAR		0x28 /* PID Filter Lookup Table Address */
#define DVBM_FD_PFLUTR		0x2c /* PID Filter Lookup Table Data */
#define DVBM_FD_PIDR0		0x30 /* PID 0 */
#define DVBM_FD_PIDCOUNTR0	0x34 /* PID Count 0 */
#define DVBM_FD_PIDR1		0x38 /* PID 1 */
#define DVBM_FD_PIDCOUNTR1	0x3c /* PID Count 1 */
#define DVBM_FD_PIDR2		0x40 /* PID 2 */
#define DVBM_FD_PIDCOUNTR2	0x44 /* PID Count 2 */
#define DVBM_FD_PIDR3		0x48 /* PID 3 */
#define DVBM_FD_PIDCOUNTR3	0x4c /* PID Count 3 */
#define DVBM_FD_27COUNTR	0x50 /* 27 MHz Counter */

/* Control/Status Register bit locations */
#define DVBM_FD_CSR_RXDSYNC	0x00008000 /* Rx Double Packet Sync. */
#define DVBM_FD_CSR_RXRF	0x00004000 /* Rx Reframe */
#define DVBM_FD_CSR_RXRST	0x00002000 /* Rx Reset */
#define DVBM_FD_CSR_RXE		0x00001000 /* Rx Enable */
#define DVBM_FD_CSR_RXPFE	0x00000800 /* Rx PID Filter Enable */
#define DVBM_FD_CSR_TXRXCLK	0x00000080 /* Tx Recovered Rx Clock */
#define DVBM_FD_CSR_TXEXTCLK	0x00000040 /* Tx External Clock */
#define DVBM_FD_CSR_TXRST	0x00000020 /* Tx Reset */
#define DVBM_FD_CSR_TXE		0x00000010 /* Tx Enable */
#define DVBM_FD_CSR_TXMAKE204	0x00000002 /* Tx Make 204 */
#define DVBM_FD_CSR_TX204	0x00000001 /* Tx 204-byte packets */

/* Control/Status Register bitmasks */
#define DVBM_FD_CSR_TXMASK		0xff0000ff
#define DVBM_FD_CSR_RXMASK		0xff00ff00
#define DVBM_FD_CSR_TXLARGEIBMASK	0x00ff0000
#define DVBM_FD_CSR_RXSYNCMASK		0x00000700
#define DVBM_FD_CSR_TXCLKMASK		0x000000c0
#define DVBM_FD_CSR_TXMODEMASK		0x00000007

#define DVBM_FD_CSR_RX204MAKE188	0x00000500 /* Rx Sync. 204 Make 188 */
#define DVBM_FD_CSR_RXAUTOMAKE188	0x00000400 /* Rx Sync. Auto Make 188 */
#define DVBM_FD_CSR_RXAUTO		0x00000300 /* Rx Sync. Auto */
#define DVBM_FD_CSR_RX204		0x00000200 /* Rx Sync. 204 */
#define DVBM_FD_CSR_RX188		0x00000100 /* Rx Sync. 188 */

/* DMA Trigger Level Register bit locations */
#define DVBM_FD_DMATLR_RXTSE	0x80000000 /* Rx Timestamp Enable */
#define DVBM_FD_DMATLR_TXTSS	0x40000000 /* Tx Timestamp Strip */

/* DMA Trigger Level Register bitmasks */
#define DVBM_FD_DMATLR_TXMASK	0x400001ff
#define DVBM_FD_DMATLR_RXMASK	0x81ff0000

/* Interrupt Control/Status Register bit locations */
#define DVBM_FD_ICSR_RX204	0x01000000 /* Rx 204-byte packets */
#define DVBM_FD_ICSR_TXDIS	0x00400000 /* Tx Data Int. Status */
#define DVBM_FD_ICSR_RXDIS	0x00200000 /* Rx Data Int. Status */
#define DVBM_FD_ICSR_RXCDIS	0x00100000 /* Rx Carrier Detect Int. Status */
#define DVBM_FD_ICSR_RXAOSIS	0x00080000 /* Rx Acq. of Sync. Int. Status */
#define DVBM_FD_ICSR_RXLOSIS	0x00040000 /* Rx Loss of Sync. Int. Status */
#define DVBM_FD_ICSR_RXOIS	0x00020000 /* Rx FIFO Overrun Int. Status */
#define DVBM_FD_ICSR_TXUIS	0x00010000 /* Tx FIFO Underrun Int. Status */
#define DVBM_FD_ICSR_TXD	0x00004000 /* Tx Data */
#define DVBM_FD_ICSR_RXD	0x00002000 /* Rx Data */
#define DVBM_FD_ICSR_RXCD	0x00001000 /* Rx Carrier Detect Status */
#define DVBM_FD_ICSR_RXPASSING	0x00000400 /* Rx Passing Data Status */
#define DVBM_FD_ICSR_RXO	0x00000200 /* Rx FIFO Overrun Status */
#define DVBM_FD_ICSR_TXU	0x00000100 /* Tx FIFO Underrun Status */
#define DVBM_FD_ICSR_TXDIE	0x00000040 /* Tx Data Int. Enable */
#define DVBM_FD_ICSR_RXDIE	0x00000020 /* Rx Data Int. Enable */
#define DVBM_FD_ICSR_RXCDIE	0x00000010 /* Rx Carrier Detect Int. Enable */
#define DVBM_FD_ICSR_RXAOSIE	0x00000008 /* Rx Acq. of Sync. Int. Enable */
#define DVBM_FD_ICSR_RXLOSIE	0x00000004 /* Rx Loss of Sync. Int. Enable */
#define DVBM_FD_ICSR_RXOIE	0x00000002 /* Rx FIFO Overrun Int. Enable */
#define DVBM_FD_ICSR_TXUIE	0x00000001 /* Tx FIFO Underrun Int. Enable */

#define DVBM_FD_ICSR_TXCTRLMASK	0x00000041
#define DVBM_FD_ICSR_TXSTATMASK	0x00414100
#define DVBM_FD_ICSR_RXCTRLMASK	0x0000003e
#define DVBM_FD_ICSR_RXSTATMASK	0x013e3600

/* External function prototypes */

int dvbm_fd_pci_probe (struct pci_dev *pdev) __devinit;
void dvbm_fd_pci_remove (struct pci_dev *pdev);

#endif

