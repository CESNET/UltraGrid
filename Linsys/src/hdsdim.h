/* hdsdim.h
 *
 * Header file for the Linux driver for
 * Linear Systems Ltd. VidPort SMPTE 292M and SMPTE 259M-C interface boards.
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

#ifndef _HDSDIM_H
#define _HDSDIM_H

#include <linux/pci.h> /* pci_dev */
#include <linux/init.h> /* __devinit */

#include "mdev.h"

/* External variables */

extern char hdsdim_driver_name[];

/* External function prototypes */

int hdsdim_pci_probe_generic (struct pci_dev *pdev) __devinit;
void hdsdim_pci_remove_generic (struct pci_dev *pdev);
int hdsdim_register (struct master_dev *card);
void hdsdim_unregister_all (struct master_dev *card);

#endif

