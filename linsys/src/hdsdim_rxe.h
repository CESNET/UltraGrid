/* hdsdim_rxe.h
 *
 * Header file for the Linear Systems Ltd. VidPort SD/HD I.
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

#ifndef _HDSDIM_RXE_H
#define _HDSDIM_RXE_H

#include <linux/pci.h> /* pci_dev */
#include <linux/init.h> /* __devinit */

#define HDSDIM_PCI_DEVICE_ID_LINSYS_HDSDIRXE 0x00c8
#define HDSDIM_NAME_RXE		"VidPort SD/HD I"

/* Register addresses */
#define HDSDIM_RXE_FPGAID	0x000 /* FPGA ID */
#define HDSDIM_RXE_CTRL		0x004 /* Transmit/Receive Control */
#define HDSDIM_RXE_STATUS	0x008 /* Status */
#define HDSDIM_RXE_SSN_HI	0x00c /* Silicon serial number, High */
#define HDSDIM_RXE_SSN_LO	0x010 /* Silicon serial number, low */
#define HDSDIM_RXE_JTAGR	0x014 /* JTAG */
#define HDSDIM_RXE_I2CCTRL	0x018 /* I2C Control */
#define HDSDIM_RXE_I2CD		0x019 /* I2C Data */
#define HDSDIM_RXE_I2CCMD	0x01a /* I2C Command */
#define HDSDIM_RXE_I2CS		0x01b /* I2C Status */
#define HDSDIM_RXE_ISR		0x020 /* Interrupt Source Read/Clear */
#define HDSDIM_RXE_IMS		0x028 /* Interrupt Mask Set/Read */
#define HDSDIM_RXE_IMC		0x02c /* Interrupt Mask Clear */
#define HDSDIM_RXE_AUDCTRL	0x030 /* Audio Control */
#define HDSDIM_RXE_RXAUDSTAT	0x038 /* Receive Audio Status */
#define HDSDIM_RXE_RXAUDRATE	0x03c /* Receive Audio Rate */
#define HDSDIM_RXE_DMACSR3	0x0a0 /* DMA Channel 3 Control/Status */
#define HDSDIM_RXE_DMACSR4	0x0c0 /* DMA Channel 4 Control/Status */
#define HDSDIM_RXE_AUDCS(c)	(0x280+((c)-1)*0x18) /* Receive Audio Channel Status */
#define HDSDIM_RXE_AUDCS1	0x280 /* Receive Audio Channel 1 Status */
#define HDSDIM_RXE_AUDCS2	0x298 /* Receive Audio Channel 2 Status */
#define HDSDIM_RXE_AUDCS3	0x2b0 /* Receive Audio Channel 3 Status */
#define HDSDIM_RXE_AUDCS4	0x2c8 /* Receive Audio Channel 4 Status */
#define HDSDIM_RXE_AUDCS5	0x2e0 /* Receive Audio Channel 5 Status */
#define HDSDIM_RXE_AUDCS6	0x2f8 /* Receive Audio Channel 6 Status */
#define HDSDIM_RXE_AUDCS7	0x310 /* Receive Audio Channel 7 Status */
#define HDSDIM_RXE_AUDCS8	0x328 /* Receive Audio Channel 8 Status */

/* Control Register (CTRL) bit locations */
#define HDSDIM_RXE_CTRL_SWRST		0x00000001 /* Reset */
#define HDSDIM_RXE_CTRL_CLKRST		0x00000008 /* Clock Reset */
#define HDSDIM_RXE_CTRL_FOURCC_V210	0x00010000 /* UYVY if not set */
#define HDSDIM_RXE_CTRL_VANC		0x01000000 /* Vertical Ancillary Space */

/* Status Register (STATUS) bit locations */
#define HDSDIM_RXE_STATUS_RXSTD		0x000f0000 /* Receive Video Standard */
#define HDSDIM_RXE_STATUS_RXSTD_LOCKED	0x00100000 /* Receive Video Standard Locked */
#define HDSDIM_RXE_STATUS_NLFI		0x00200000 /* No Link Fault Indicator */
#define HDSDIM_RXE_STATUS_RXCLK_LOCKED	0x00400000 /* Receive Clock Locked */
#define HDSDIM_RXE_STATUS_SD		0x00800000 /* Standard Definition */
#define HDSDIM_RXE_STATUS_M		0x01000000 /* 74.176 MHz */

#define HDSDIM_RXE_STATUS_RXSTD_260M_1035i	0x00000000 /* SMPTE 260M 1035i60 */
#define HDSDIM_RXE_STATUS_RXSTD_295M_1080i	0x00010000 /* SMPTE 295M 1080i50 */
#define HDSDIM_RXE_STATUS_RXSTD_274M_1080i_60	0x00020000 /* SMPTE 274M 1080i60 */
#define HDSDIM_RXE_STATUS_RXSTD_274M_1080i_50	0x00030000 /* SMPTE 274M 1080i50 */
#define HDSDIM_RXE_STATUS_RXSTD_274M_1080p_30	0x00040000 /* SMPTE 274M 1080p30 */
#define HDSDIM_RXE_STATUS_RXSTD_274M_1080p_25	0x00050000 /* SMPTE 274M 1080p25 */
#define HDSDIM_RXE_STATUS_RXSTD_274M_1080p_24	0x00060000 /* SMPTE 274M 1080p24 */
#define HDSDIM_RXE_STATUS_RXSTD_296M_720p_60	0x00070000 /* SMPTE 296M 720p60 */
#define HDSDIM_RXE_STATUS_RXSTD_274M_1080sf_24	0x00080000 /* SMPTE 274M 1080sf24 */
#define HDSDIM_RXE_STATUS_RXSTD_296M_720p_50	0x00090000 /* SMPTE 296M 720p50 */
#define HDSDIM_RXE_STATUS_RXSTD_296M_720p_30	0x000a0000 /* SMPTE 296M 720p30 */
#define HDSDIM_RXE_STATUS_RXSTD_296M_720p_25	0x000b0000 /* SMPTE 296M 720p25 */
#define HDSDIM_RXE_STATUS_RXSTD_296M_720p_24	0x000c0000 /* SMPTE 296M 720p24 */
#define HDSDIM_RXE_STATUS_RXSTD_125M_486i	0x00000000 /* SMPTE 125M 486i59.94 */
#define HDSDIM_RXE_STATUS_RXSTD_BT601_576i	0x00040000 /* ITU-R BT.601 720x576i50 */

/* Interrupt (ISR, IMS, IMC) bit locations */
#define HDSDIM_RXE_INT_I2CI	0x00000001 /* I2C Interrupt */
#define HDSDIM_RXE_INT_RVOI	0x00010000 /* Receive Video Overrun Interrupt */
#define HDSDIM_RXE_INT_RVDI	0x00020000 /* Receive Video Data Interrupt */
#define HDSDIM_RXE_INT_RSTDI	0x00040000 /* Receive Video Standard Interrupt */
#define HDSDIM_RXE_INT_RAOI	0x00100000 /* Receive Audio Overrun Interrupt */
#define HDSDIM_RXE_INT_RADI	0x00200000 /* Receive Audio Data Interrupt */
#define HDSDIM_RXE_INT_DMA3	0x08000000 /* DMA Channel 3 Interrupt */
#define HDSDIM_RXE_INT_DMA4	0x10000000 /* DMA Channel 4 Interrupt */

/* I2C Control (I2CCTRL) bit locations */
#define HDSDIM_RXE_I2CCTRL_EN	0x80 /* I2C Core Enable */

/* I2C Command (I2CCMD) bit locations */
#define HDSDIM_RXE_I2CCMD_ACK	0x08 /* Acknowledge */
#define HDSDIM_RXE_I2CCMD_WR	0x10 /* Write to slave */
#define HDSDIM_RXE_I2CCMD_RD	0x20 /* Read from slave */
#define HDSDIM_RXE_I2CCMD_STO	0x40 /* Generate stop condition */
#define HDSDIM_RXE_I2CCMD_STA	0x80 /* Generate (repeated) start condition */

/* I2C Status (I2CS) bit locations */
#define HDSDIM_RXE_I2CS_TIP	0x02 /* Transfer In Progress */
#define HDSDIM_RXE_I2CS_AL	0x20 /* Arbitration Lost */
#define HDSDIM_RXE_I2CS_BUSY	0x40 /* Bus Busy */
#define HDSDIM_RXE_I2CS_RXACK	0x80 /* Received Acknowledge */

/* Audio Control (AUDCTRL) bit locations */
#define HDSDIM_RXE_AUDCTRL_CHAN_0	0x00000000 /* Audio disabled */
#define HDSDIM_RXE_AUDCTRL_CHAN_2	0x00000003 /* Two Channels */
#define HDSDIM_RXE_AUDCTRL_CHAN_4	0x0000000f /* Four Channels */
#define HDSDIM_RXE_AUDCTRL_CHAN_6	0x0000003f /* Six Channels */
#define HDSDIM_RXE_AUDCTRL_CHAN_8	0x000000ff /* Eight Channels */

#define HDSDIM_RXE_AUDCTRL_FOURCC_16	0x00000000 /* 16-bit Audio Sample Format */
#define HDSDIM_RXE_AUDCTRL_FOURCC_32	0x00010000 /* 32-bit Audio Sample Format */

/* Audio Channel Status (AUDCS) bit locations */
#define HDSDIM_RXE_AUDCS_CS0_NONAUDIO	0x00000002 /* Non-audio bit mask */

#define HDSDIM_RXE_AUDCS_CS2_MAXAUDLENGTH_24BITS	0x00010000 /* Maximum audio sample word length 24 bits */

/* External function prototypes */

int hdsdim_rxe_pci_probe (struct pci_dev *pdev) __devinit;
void hdsdim_rxe_pci_remove (struct pci_dev *pdev);

#endif

