/* hdsdim_txe.h
 *
 * Header file for the Linear Systems Ltd. VidPort SD/HD O.
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

#ifndef _HDSDIM_TXE_H
#define _HDSDIM_TXE_H

#include <linux/pci.h> /* pci_dev */
#include <linux/init.h> /* __devinit */

#define HDSDIM_PCI_DEVICE_ID_LINSYS_HDSDITXE 0x00c1
#define HDSDIM_NAME_TXE		"VidPort SD/HD O"

/* Register addresses */
#define HDSDIM_TXE_FPGAID	0x000 /* FPGA ID */
#define HDSDIM_TXE_CTRL		0x004 /* Transmit/Receive Control */
#define HDSDIM_TXE_STATUS	0x008 /* Status */
#define HDSDIM_TXE_SSN_HI	0x00c /* Silicon serial number, High */
#define HDSDIM_TXE_SSN_LO	0x010 /* Silicon serial number, low */
#define HDSDIM_TXE_JTAGR	0x014 /* JTAG */
#define HDSDIM_TXE_I2CCTRL	0x018 /* I2C Control */
#define HDSDIM_TXE_I2CD		0x019 /* I2C Data */
#define HDSDIM_TXE_I2CCMD	0x01a /* I2C Command */
#define HDSDIM_TXE_I2CS		0x01b /* I2C Status */
#define HDSDIM_TXE_ISR		0x020 /* Interrupt Source Read/Clear */
#define HDSDIM_TXE_IMS		0x028 /* Interrupt Mask Set/Read */
#define HDSDIM_TXE_IMC		0x02c /* Interrupt Mask Clear */
#define HDSDIM_TXE_AUDCTRL	0x030 /* Audio Control */
#define HDSDIM_TXE_DMACSR0	0x040 /* DMA Channel 0 Control/Status */
#define HDSDIM_TXE_DMACSR1	0x060 /* DMA Channel 1 Control/Status */
#define HDSDIM_TXE_TXAUDCS(c)	(0x100+((c)-1)*0x18) /* Transmit Audio Channel Status */
#define HDSDIM_TXE_TXAUDCS1	0x100 /* Transmit Audio Channel 1 Status */
#define HDSDIM_TXE_TXAUDCS2	0x118 /* Transmit Audio Channel 2 Status */
#define HDSDIM_TXE_TXAUDCS3	0x130 /* Transmit Audio Channel 3 Status */
#define HDSDIM_TXE_TXAUDCS4	0x148 /* Transmit Audio Channel 4 Status */
#define HDSDIM_TXE_TXAUDCS5	0x160 /* Transmit Audio Channel 5 Status */
#define HDSDIM_TXE_TXAUDCS6	0x178 /* Transmit Audio Channel 6 Status */
#define HDSDIM_TXE_TXAUDCS7	0x190 /* Transmit Audio Channel 7 Status */
#define HDSDIM_TXE_TXAUDCS8	0x1a8 /* Transmit Audio Channel 8 Status */

/* Control Register (CTRL) bit locations */
#define HDSDIM_TXE_CTRL_SWRST		0x00000001 /* Reset */
#define HDSDIM_TXE_CTRL_HL2RST		0x00000004 /* HOTLink Reset */
#define HDSDIM_TXE_CTRL_CLKRST		0x00000008 /* Clock Reset */
#define HDSDIM_TXE_CTRL_GENLOCK		0x00000010 /* Genlock Mode */
#define HDSDIM_TXE_CTRL_TXSTD		0x00000f00 /* Video Standard */
#define HDSDIM_TXE_CTRL_SD		0x00001000 /* Standard Definition */
#define HDSDIM_TXE_CTRL_M		0x00002000 /* 74.176 MHz */
#define HDSDIM_TXE_CTRL_PSF		0x00004000 /* Progressive Segmented Frame */
#define HDSDIM_TXE_CTRL_FOURCC_V210	0x00010000 /* UYVY if not set */

#define HDSDIM_TXE_CTRL_TXSTD_260M_1035i	0x00000000 /* SMPTE 260M 1035i30 */
#define HDSDIM_TXE_CTRL_TXSTD_295M_1080i	0x00000100 /* SMPTE 295M 1080i25 */
#define HDSDIM_TXE_CTRL_TXSTD_274M_1080i_30	0x00000200 /* SMPTE 274M 1080i30 */
#define HDSDIM_TXE_CTRL_TXSTD_274M_1080i_25	0x00000300 /* SMPTE 274M 1080i25 */
#define HDSDIM_TXE_CTRL_TXSTD_274M_1080p_30	0x00000400 /* SMPTE 274M 1080p30 */
#define HDSDIM_TXE_CTRL_TXSTD_274M_1080p_25	0x00000500 /* SMPTE 274M 1080p25 */
#define HDSDIM_TXE_CTRL_TXSTD_274M_1080p_24	0x00000600 /* SMPTE 274M 1080p24 */
#define HDSDIM_TXE_CTRL_TXSTD_296M_720p_60	0x00000700 /* SMPTE 296M 720p60 */
#define HDSDIM_TXE_CTRL_TXSTD_274M_1080sf_24	0x00000800 /* SMPTE 274M 1080sf24 */
#define HDSDIM_TXE_CTRL_TXSTD_296M_720p_50	0x00000900 /* SMPTE 296M 720p50 */
#define HDSDIM_TXE_CTRL_TXSTD_296M_720p_30	0x00000a00 /* SMPTE 296M 720p30 */
#define HDSDIM_TXE_CTRL_TXSTD_296M_720p_25	0x00000b00 /* SMPTE 296M 720p25 */
#define HDSDIM_TXE_CTRL_TXSTD_296M_720p_24	0x00000c00 /* SMPTE 296M 720p24 */
#define HDSDIM_TXE_CTRL_TXSTD_125M_486i		0x00000000 /* SMPTE 125M 486i29.97 */
#define HDSDIM_TXE_CTRL_TXSTD_BT601_576i	0x00000400 /* ITU-R BT.601 720x576i25 */
#define HDSDIM_TXE_CTRL_VPID	0x00100000 /* Video Payload Identifier */
#define HDSDIM_TXE_CTRL_VANC	0x01000000 /* Vertical Ancillary Space */

/* Status Register (STATUS) bit locations */
#define HDSDIM_TXE_STATUS_REF_LOST	0x00000001 /* Reference Lost */
#define HDSDIM_TXE_STATUS_LOCK_LOST	0x00000002 /* Lock Lost */
#define HDSDIM_TXE_STATUS_TXCLK_LOCKED	0x00000004 /* Transmit Clock Locked */

/* Interrupt (ISR, IMS, IMC) bit locations */
#define HDSDIM_TXE_INT_I2CI	0x00000001 /* I2C Interrupt */
#define HDSDIM_TXE_INT_TVUI	0x00000100 /* Transmit Video Underrun Interrupt */
#define HDSDIM_TXE_INT_TVDI	0x00000200 /* Transmit Video Data Interrupt */
#define HDSDIM_TXE_INT_REFI	0x00000400 /* Transmit Reference Interrupt */
#define HDSDIM_TXE_INT_TAUI	0x00001000 /* Transmit Audio Underrun Interrupt */
#define HDSDIM_TXE_INT_TADI	0x00002000 /* Transmit Audio Data Interrupt */
#define HDSDIM_TXE_INT_DMA0	0x01000000 /* DMA Channel 0 Interrupt */
#define HDSDIM_TXE_INT_DMA1	0x02000000 /* DMA Channel 1 Interrupt */

/* I2C Control (I2CCTRL) bit locations */
#define HDSDIM_TXE_I2CCTRL_EN	0x80 /* I2C Core Enable */

/* I2C Command (I2CCMD) bit locations */
#define HDSDIM_TXE_I2CCMD_ACK	0x08 /* Acknowledge */
#define HDSDIM_TXE_I2CCMD_WR	0x10 /* Write to slave */
#define HDSDIM_TXE_I2CCMD_RD	0x20 /* Read from slave */
#define HDSDIM_TXE_I2CCMD_STO	0x40 /* Generate stop condition */
#define HDSDIM_TXE_I2CCMD_STA	0x80 /* Generate (repeated) start condition */

/* I2C Status (I2CS) bit locations */
#define HDSDIM_TXE_I2CS_TIP	0x02 /* Transfer In Progress */
#define HDSDIM_TXE_I2CS_AL	0x20 /* Arbitration Lost */
#define HDSDIM_TXE_I2CS_BUSY	0x40 /* Bus Busy */
#define HDSDIM_TXE_I2CS_RXACK	0x80 /* Received Acknowledge */

/* Audio Control (AUDCTRL) bit locations */
#define HDSDIM_TXE_AUDCTRL_CHAN_0	0x00000000 /* Audio disabled */
#define HDSDIM_TXE_AUDCTRL_CHAN_2	0x00000003 /* Two Channels */
#define HDSDIM_TXE_AUDCTRL_CHAN_4	0x0000000f /* Four Channels */
#define HDSDIM_TXE_AUDCTRL_CHAN_6	0x0000003f /* Six Channels */
#define HDSDIM_TXE_AUDCTRL_CHAN_8	0x000000ff /* Eight Channels */

#define HDSDIM_TXE_AUDCTRL_FOURCC_32	0x00010000 /* 32-bit Audio Sample Format */
#define HDSDIM_TXE_AUDCTRL_FOURCC_16	0x00000000 /* 16-bit Audio Sample Format */

/* Audio Channel Status (AUDCS) bit locations */
#define HDSDIM_TXE_TXAUDCS_CS0_44_1KHZ	0x00000040 /* Sampling Rate 44.1 kHz */
#define HDSDIM_TXE_TXAUDCS_CS0_48KHZ	0x00000080 /* Sampling Rate 48 kHz */
#define HDSDIM_TXE_TXAUDCS_CS0_32KHZ	0x000000c0 /* Sampling Rate 32 kHz */
#define HDSDIM_TXE_TXAUDCS_CS0_NONAUDIO	0x00000002 /* Non-audio bit mask */

#define HDSDIM_TXE_TXAUDCS_CS2_MAXLENGTH_24BITS		0x00040000 /* Maximum audio sample word length 24 bits */
#define HDSDIM_TXE_TXAUDCS_CS2_MAXLENGTH_20BITS		0x00000000 /* Maximum audio sample word length 20 bits */

/* External function prototypes */

int hdsdim_txe_pci_probe (struct pci_dev *pdev) __devinit;
void hdsdim_txe_pci_remove (struct pci_dev *pdev);

#endif

