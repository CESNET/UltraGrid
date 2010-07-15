/* dvbm.h
 *
 * Header file for the Linux driver for
 * Linear Systems Ltd. DVB Master ASI interface boards.
 *
 * Copyright (C) 2004-2010 Linear Systems Ltd.
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

#ifndef _DVBM_H
#define _DVBM_H

#include <linux/pci.h> /* pci_dev */
#include <linux/init.h> /* __devinit */

#include "mdev.h"

/* External variables */

extern char dvbm_driver_name[];

/* External function prototypes */

int dvbm_pci_probe_generic (struct pci_dev *pdev) __devinit;
void dvbm_pci_remove_generic (struct pci_dev *pdev);
int dvbm_register (struct master_dev *card);
void dvbm_unregister_all (struct master_dev *card);

#endif

