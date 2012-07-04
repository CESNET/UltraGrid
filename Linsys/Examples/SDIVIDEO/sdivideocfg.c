/* sdivideocfg.c
 *
 * SDI video configuration program.
 *
 * Copyright (C) 2009-2010 Linear Systems Ltd. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   1. Redistributions of source code must retain the above copyright notice,
 *      this list of conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *   3. Neither the name of Linear Systems Ltd. nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LINEAR SYSTEMS LTD. "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL LINEAR SYSTEMS LTD. OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Linear Systems can be contacted at <http://www.linsys.ca/>.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>

#include "sdivideo.h"
#include "master.h"
#include "../util.h"

#define MAXLEN 256
#define BUFFERS_FLAG	0x00000001
#define BUFSIZE_FLAG	0x00000002
#define CLKSRC_FLAG	0x00000004
#define MODE_FLAG	0x00000008
#define FRMODE_FLAG	0x00000010
#define VANC_FLAG	0x00000020

static const char progname[] = "sdivideocfg";

int
main (int argc, char **argv)
{
	const char fmt[] = "/sys/class/sdivideo/sdivideo%cx%i/%s";
	int opt;
	unsigned int write_flags;
	struct stat buf;
	int num;
	char type, name[MAXLEN], data[MAXLEN];
	unsigned long int buffers, bufsize, clksrc, mode, frmode, vanc;
	int retcode;
	char *endptr;

	/* Parse the command line */
	write_flags = 0;
	buffers = 0;
	bufsize = 0;
	clksrc = 0;
	mode = 0;
	frmode = 0;
	vanc = 0;
	while ((opt = getopt (argc, argv, "aAb:f:hm:s:Vx:")) != -1) {
		switch (opt) {
		case 'a':
			write_flags |= VANC_FLAG;
			vanc = 1;
			break;
		case 'A':
			write_flags |= VANC_FLAG;
			vanc = 0;
			break;
		case 'b':
			write_flags |= BUFFERS_FLAG;
			buffers = strtoul (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid number of buffers: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'f':
			write_flags |= FRMODE_FLAG;
			frmode = strtoul (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid frame mode: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'h':
			printf ("Usage: %s [OPTION]... DEVICE_FILE\n",
				argv[0]);
			printf ("Configure an SDI video interface.\n\n");
			printf ("  -a\t\tenable vertical ancillary space\n");
			printf ("  -A\t\tdisable vertical ancillary space\n");
			printf ("  -b BUFFERS\tset the number of buffers\n");
			printf ("  -f FRMODE\tset the frame mode "
				"(transmitters only)\n");
			printf ("  -h\t\tdisplay this help and exit\n");
			printf ("  -m MODE\tset the operating mode\n");
			printf ("  -s BUFSIZE\tset the buffer size\n");
			printf ("  -V\t\toutput version information "
				"and exit\n");
			printf ("  -x CLKSRC\tset the clock source "
				"(transmitters only)\n");
			printf ("\nIf no options are specified, "
				"the current configuration is displayed.\n");
			printf ("\nBUFFERS must be two or more.\n");
			printf ("\nCLKSRC may be:\n"
				"\t0 (onboard oscillator)\n"
				"\t1 (external 525i or NTSC reference)\n"
				"\t2 (external 625i or PAL reference)\n"
				"\t3 (external 525p reference)\n"
				"\t4 (external 625p reference)\n"
				"\t5 (external 720p/60 reference)\n"
				"\t6 (external 720p/59.94 reference)\n"
				"\t7 (external 720p/50 reference)\n"
				"\t8 (external 720p/30 reference)\n"
				"\t9 (external 720p/29.97 reference)\n"
				"\t10 (external 720p/25 reference)\n"
				"\t11 (external 720p/24 reference)\n"
				"\t12 (external 720p/23.98 reference)\n"
				"\t13 (external 1080p/60 reference)\n"
				"\t14 (external 1080p/59.94 reference)\n"
				"\t15 (external 1080p/50 reference)\n"
				"\t16 (external 1080p/30 reference)\n"
				"\t17 (external 1080p/29.97 reference)\n"
				"\t18 (external 1080p/25 reference)\n"
				"\t19 (external 1080p/24 reference)\n"
				"\t20 (external 1080p/23.98 reference)\n"
				"\t21 (external 1080i/60 reference)\n"
				"\t22 (external 1080i/59.94 reference)\n"
				"\t23 (external 1080i/50 reference)\n");
			printf ("\nFRMODE may be:\n"
				"\t1 (SMPTE 125M 486i 59.94 Hz)\n"
				"\t2 (ITU-R BT.601 720x576i 50 Hz)\n"
				"\t5 (SMPTE 260M 1035i 60 Hz)\n"
				"\t6 (SMPTE 260M 1035i 59.94 Hz)\n"
				"\t8 (SMPTE 274M 1080i 60 Hz)\n"
				"\t9 (SMPTE 274M 1080psf 30 Hz)\n"
				"\t10 (SMPTE 274M 1080i 59.94 Hz)\n"
				"\t11 (SMPTE 274M 1080psf 29.97 Hz)\n"
				"\t12 (SMPTE 274M 1080i 50 Hz)\n"
				"\t13 (SMPTE 274M 1080psf 25 Hz)\n"
				"\t14 (SMPTE 274M 1080psf 24 Hz)\n"
				"\t15 (SMPTE 274M 1080psf 23.98 Hz)\n"
				"\t16 (SMPTE 274M 1080p 30 Hz)\n"
				"\t17 (SMPTE 274M 1080p 29.97 Hz)\n"
				"\t18 (SMPTE 274M 1080p 25 Hz)\n"
				"\t19 (SMPTE 274M 1080p 24 Hz)\n"
				"\t20 (SMPTE 274M 1080p 23.98 Hz)\n"
				"\t21 (SMPTE 296M 720p 60 Hz)\n"
				"\t22 (SMPTE 296M 720p 59.94 Hz)\n"
				"\t23 (SMPTE 296M 720p 50 Hz)\n"
				"\t24 (SMPTE 296M 720p 30 Hz)\n"
				"\t25 (SMPTE 296M 720p 29.97 Hz)\n"
				"\t26 (SMPTE 296M 720p 25 Hz)\n"
				"\t27 (SMPTE 296M 720p 24 Hz)\n"
				"\t28 (SMPTE 296M 720p 23.98 Hz)\n");
			printf ("\nMODE may be:\n"
				"\t0 (uyvy 8-bit mode) \n"
				"\t1 (v210 10-bit mode)\n"
				"\t2 (v210 10-bit deinterlaced mode) \n"
				"\t3 (raw) \n");
			printf ("\nBUFSIZE must be "
				"a positive multiple of four,\n"
				"and at least 1024 bytes for transmitters.\n");
			printf ("\nReport bugs to <support@linsys.ca>.\n");
			return 0;
		case 'm':
			write_flags |= MODE_FLAG;
			mode = strtoul (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid mode: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 's':
			write_flags |= BUFSIZE_FLAG;
			bufsize = strtoul (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid buffer size: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'V':
			printf ("%s from master-%s (%s)\n", progname,
				MASTER_DRIVER_VERSION,
				MASTER_DRIVER_DATE);
			printf ("\nCopyright (C) 2009-2010 "
				"Linear Systems Ltd.\n"
				"This is free software; "
				"see the source for copying conditions.  "
				"There is NO\n"
				"warranty; not even for MERCHANTABILITY "
				"or FITNESS FOR A PARTICULAR PURPOSE.\n");
			return 0;
		case 'x':
			write_flags |= CLKSRC_FLAG;
			clksrc = strtoul (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid clock source: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case '?':
			goto USAGE;
		}
	}

	/* Check the number of arguments */
	if ((argc - optind) < 1) {
		fprintf (stderr, "%s: missing arguments\n", argv[0]);
		goto USAGE;
	} else if ((argc - optind) > 1) {
		fprintf (stderr, "%s: extra operand\n", argv[0]);
		goto USAGE;
	}

	/* Get the sysfs info */
	memset (&buf, 0, sizeof (buf));
	/* Stat the file, fills the structure with info about the file
	 * Get the major number from device node
	 */
	if (stat (argv[optind], &buf) < 0) {
		fprintf (stderr, "%s: ", argv[0]);
		perror ("unable to get the file status");
		return -1;
	}
	/* Check if it is a character device or not */
	if (!S_ISCHR (buf.st_mode)) {
		fprintf (stderr, "%s: not a character device\n", argv[0]);
		return -1;
	}
	/* Check the minor number to determine if it is a receive or transmit device */
	type = (buf.st_rdev & 0x0080) ? 'r' : 't';
	/* Get the receiver or transmitter number */
	num = buf.st_rdev & 0x007f;
	/* Build the path to sysfs file */
	snprintf (name, sizeof (name), fmt, type, num, "dev");
	memset (data, 0, sizeof (data));
	/* Read sysfs file (dev) */
	if (util_read (name, data, sizeof (data)) < 0) {
		fprintf (stderr, "%s: error reading %s: ", argv[0], name);
		perror (NULL);
		return -1;
	}
	/* Compare the major number taken from sysfs file to the one taken from device node */
	if (strtoul (data, &endptr, 0) != (buf.st_rdev >> 8)) {
		fprintf (stderr, "%s: not an SDI video device\n", argv[0]);
		return -1;
	}
	if (*endptr != ':') {
		fprintf (stderr, "%s: error reading %s\n", argv[0], name);
		return -1;
	}

	retcode = 0;
	printf ("%s:\n", argv[optind]);
	if (write_flags) {
		if (write_flags & BUFFERS_FLAG) {
			snprintf (name, sizeof (name),
				fmt, type, num, "buffers");
			snprintf (data, sizeof (data), "%lu\n", buffers);
			if (util_write (name, data, sizeof (data)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to set "
					"the number of buffers");
				return -1;
			}
			printf ("\tSet number of buffers = %lu.\n", buffers);
		}
		if (write_flags & BUFSIZE_FLAG) {
			snprintf (name, sizeof (name),
				fmt, type, num, "bufsize");
			snprintf (data, sizeof (data), "%lu\n", bufsize);
			if (util_write (name, data, sizeof (data)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to set "
					"the buffer size");
				return -1;
			}
			printf ("\tSet buffer size = %lu bytes.\n", bufsize);
		}
		if (write_flags & CLKSRC_FLAG) {
			if (type == 'r') {
				fprintf (stderr, "%s: "
					"unable to set the clock source: "
					"Not a transmitter\n", argv[0]);
				return -1;
			}
			snprintf (name, sizeof (name),
				fmt, type, num, "clock_source");
			snprintf (data, sizeof (data), "%lu\n", clksrc);
			if (util_write (name, data, sizeof (data)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to set "
					"the clock source");
				return -1;
			}

			switch (clksrc) {
			case SDIVIDEO_CTL_TX_CLKSRC_ONBOARD:
				printf ("\tUsing onboard oscillator.\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_NTSC:
				printf ("\tUsing external 525i or NTSC reference.\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_PAL:
				printf ("\tUsing external 625i or PAL reference.\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_525P:
				printf ("\tUsing external 525p reference.\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_625P:
				printf ("\tUsing external 625p reference.\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_720P_60:
				printf ("\tUsing external 720p/60 reference.\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_720P_59_94:
				printf ("\tUsing external 720p/59.94 reference.\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_720P_50:
				printf ("\tUsing external 720p/50 reference.\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_720P_30:
				printf ("\tUsing external 720p/30 reference.\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_720P_29_97:
				printf ("\tUsing external 720p/29.97 reference.\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_720P_25:
				printf ("\tUsing external 720p/25 reference.\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_720P_24:
				printf ("\tUsing external 720p/24 reference.\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_720P_23_98:
				printf ("\tUsing external 720p/23.98 reference.\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_1080P_60:
				printf ("\tUsing external 1080p/60 reference.\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_1080P_59_94:
				printf ("\tUsing external 1080p/59.94 reference.\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_1080P_50:
				printf ("\tUsing external 1080p/50 reference.\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_1080P_30:
				printf ("\tUsing external 1080p/30 reference.\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_1080P_29_97:
				printf ("\tUsing external 1080p/29.97 reference.\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_1080P_25:
				printf ("\tUsing external 1080p/25 reference.\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_1080P_24:
				printf ("\tUsing external 1080p/24 reference.\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_1080P_23_98:
				printf ("\tUsing external 1080p/23.98 reference.\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_1080I_60:
				printf ("\tUsing external 1080i/60 reference.\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_1080I_59_94:
				printf ("\tUsing external 1080i/59.94 reference.\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_1080I_50:
				printf ("\tUsing external 1080i/50 reference.\n");
				break;
			default:
				printf ("\tSet clock source = %lu.\n", mode);
				break;
			}
		}
		if (write_flags & MODE_FLAG) {
			snprintf (name, sizeof (name),
				fmt, type, num, "mode");
			snprintf (data, sizeof (data), "%lu\n", mode);
			if (util_write (name, data, sizeof (data)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to set "
					"the interface operating mode");
				return -1;
			}

			switch (mode) {
			case SDIVIDEO_CTL_MODE_UYVY:
				printf ("\tAssuming 8-bit uyvy "
					"data.\n");
				break;
			case SDIVIDEO_CTL_MODE_V210:
				printf ("\tAssuming 10-bit v210 "
					"synchronized data.\n");
				break;
			case SDIVIDEO_CTL_MODE_V210_DEINTERLACE:
				printf ("\tAssuming 10-bit v210 "
					"deinterlaced data.\n");
				break;
			case SDIVIDEO_CTL_MODE_RAW:
				printf ("\tAssuming "
					"raw data.\n");
				break;
			default:
				printf ("\tSet mode = %lu.\n", mode);
				break;
			}
		}
		if (write_flags & FRMODE_FLAG) {
			if (type == 'r') {
				fprintf (stderr, "%s: "
					"unable to set the interface frame mode: "
					"Not a transmitter\n", argv[0]);
				return -1;
			}
			snprintf (name, sizeof (name),
				fmt, type, num, "frame_mode");
			snprintf (data, sizeof (data), "%lu\n", frmode);
			if (util_write (name, data, sizeof (data)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to set "
					"the interface frame mode");
				return -1;
			}

			switch (frmode) {
			case SDIVIDEO_CTL_SMPTE_125M_486I_59_94HZ:
				printf ("\tAssuming "
					"SMPTE 125M 486i 59.94 Hz.\n");
				break;
			case SDIVIDEO_CTL_BT_601_576I_50HZ:
				printf ("\tAssuming "
					"ITU-R BT.601 720x576i 50 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_260M_1035I_60HZ:
				printf ("\tAssuming "
					"SMPTE 260M 1035i 60 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_260M_1035I_59_94HZ:
				printf ("\tAssuming "
					"SMPTE 260M 1035i 59.94 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_295M_1080I_50HZ:
				printf ("\tAssuming "
					"SMPTE 295M 1080i 50 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080I_60HZ:
				printf ("\tAssuming "
					"SMPTE 274M 1080i 60 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080PSF_30HZ:
				printf ("\tAssuming "
					"SMPTE 274M 1080psf 30 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080I_59_94HZ:
				printf ("\tAssuming "
					"SMPTE 274M 1080i 59.94 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080PSF_29_97HZ:
				printf ("\tAssuming "
					"SMPTE 274M 1080psf 29.97 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080I_50HZ:
				printf ("\tAssuming "
					"SMPTE 274M 1080i 50 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080PSF_25HZ:
				printf ("\tAssuming "
					"SMPTE 274M 1080psf 25 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080PSF_24HZ:
				printf ("\tAssuming "
					"SMPTE 274M 1080psf 24 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080PSF_23_98HZ:
				printf ("\tAssuming "
					"SMPTE 274M 1080psf 23.98 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080P_30HZ:
				printf ("\tAssuming "
					"SMPTE 274M 1080p 30 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080P_29_97HZ:
				printf ("\tAssuming "
					"SMPTE 274M 1080p 29.97 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080P_25HZ:
				printf ("\tAssuming "
					"SMPTE 274M 1080p 25 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080P_24HZ:
				printf ("\tAssuming "
					"SMPTE 274M 1080p 24 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080P_23_98HZ:
				printf ("\tAssuming "
					"SMPTE 274M 1080p 23.98 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_296M_720P_60HZ:
				printf ("\tAssuming "
					"SMPTE 296M 720p 60 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_296M_720P_59_94HZ:
				printf ("\tAssuming "
					"SMPTE 296M 720p 59.94 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_296M_720P_50HZ:
				printf ("\tAssuming "
					"SMPTE 296M 720p 50 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_296M_720P_30HZ:
				printf ("\tAssuming "
					"SMPTE 296M 720p 30 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_296M_720P_29_97HZ:
				printf ("\tAssuming "
					"SMPTE 296M 720p 29.97 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_296M_720P_25HZ:
				printf ("\tAssuming "
					"SMPTE 296M 720p 25 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_296M_720P_24HZ:
				printf ("\tAssuming "
					"SMPTE 296M 720p 24 Hz.\n");
				break;
			case SDIVIDEO_CTL_SMPTE_296M_720P_23_98HZ:
				printf ("\tAssuming "
					"SMPTE 296M 720p 23.98 Hz.\n");
				break;
			default:
				printf ("\tSet frame mode = %lu.\n", frmode);
				break;
			}
		}
		if (write_flags & VANC_FLAG) {
			snprintf (name, sizeof (name),
				fmt, type, num, "vanc");
			snprintf (data, sizeof (data), "%lu\n", vanc);
			if (util_write (name, data, sizeof (data)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				if (vanc) {
					perror ("unable to enable "
						"vertical ancillary space");
				} else {
					perror ("unable to disable "
						"vertical ancillary space");
				}
				return -1;
			}
			printf ("\t%sabled vertical ancillary space.\n",
				vanc ? "En" : "Dis");
		}
	} else {
		snprintf (name, sizeof (name),
			fmt, type, num, "buffers");
		if (util_strtoul (name, &buffers) < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to get the number of buffers");
			retcode = -1;
		}

		snprintf (name, sizeof (name),
			fmt, type, num, "bufsize");
		if (util_strtoul (name, &bufsize) < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to get the buffer size");
			retcode = -1;
		}

		printf ("\t%lu x %lu-byte buffers\n",
			buffers, bufsize);

		snprintf (name, sizeof (name),
				fmt, type, num, "mode");
		if (util_strtoul (name, &mode) < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to get the pixel mode");
			retcode = -1;
		}
		printf ("\tMode: %lu ", mode);
		switch (mode) {
		case SDIVIDEO_CTL_MODE_UYVY:
			printf ("(assume 8-bit uyvy data)\n");
			break;
		case SDIVIDEO_CTL_MODE_V210:
			printf ("(assume 10-bit v210 synchronized data)\n");
			break;
		case SDIVIDEO_CTL_MODE_V210_DEINTERLACE:
			printf ("(assume 10-bit v210 deinterlaced data)\n");
			break;
		case SDIVIDEO_CTL_MODE_RAW:
			printf ("(assume raw data)\n");
			break;
		default:
			printf ("(unknown)\n");
			break;
		}

		if (type == 't') {
			snprintf (name, sizeof (name),
				fmt, type, num, "frame_mode");
			if (util_strtoul (name, &frmode) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to get the frame mode");
				retcode = -1;
			}
			printf ("\tFrame mode: %lu ", frmode);
			switch (frmode) {
			case SDIVIDEO_CTL_SMPTE_125M_486I_59_94HZ:
				printf ("(assume "
					"SMPTE 125M 486i 59.94 Hz)\n");
				break;
			case SDIVIDEO_CTL_BT_601_576I_50HZ:
				printf ("(assume "
					"ITU-R BT.601 720x576i 50 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_260M_1035I_60HZ:
				printf ("(assume "
					"SMPTE 260M 1035i 60 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_260M_1035I_59_94HZ:
				printf ("(assume "
					"SMPTE 260M 1035i 59.94 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_295M_1080I_50HZ:
				printf ("(assume "
					"SMPTE 295M 1080i 50 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080I_60HZ:
				printf ("(assume "
					"SMPTE 274M 1080i 60 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080PSF_30HZ:
				printf ("(assume "
					"SMPTE 274M 1080psf 30 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080I_59_94HZ:
				printf ("(assume "
					"SMPTE 274M 1080i 59.94 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080PSF_29_97HZ:
				printf ("(assume "
					"SMPTE 274M 1080psf 29.97 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080I_50HZ:
				printf ("(assume "
					"SMPTE 274M 1080i 50 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080PSF_25HZ:
				printf ("(assume "
					"SMPTE 274M 1080psf 25 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080PSF_24HZ:
				printf ("(assume "
					"SMPTE 274M 1080psf 24 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080PSF_23_98HZ:
				printf ("(assume "
					"SMPTE 274M 1080psf 23.98 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080P_30HZ:
				printf ("(assume "
					"SMPTE 274M 1080p 30 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080P_29_97HZ:
				printf ("(assume "
					"SMPTE 274M 1080p 29.97 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080P_25HZ:
				printf ("(assume "
					"SMPTE 274M 1080p 25 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080P_24HZ:
				printf ("(assume "
					"SMPTE 274M 1080p 24 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_274M_1080P_23_98HZ:
				printf ("(assume "
					"SMPTE 274M 1080p 23.98 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_296M_720P_60HZ:
				printf ("(assume "
					"SMPTE 296M 720p 60 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_296M_720P_59_94HZ:
				printf ("(assume "
					"SMPTE 296M 720p 59.94 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_296M_720P_50HZ:
				printf ("(assume "
					"SMPTE 296M 720p 50 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_296M_720P_30HZ:
				printf ("(assume "
					"SMPTE 296M 720p 30 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_296M_720P_29_97HZ:
				printf ("(assume "
					"SMPTE 296M 720p 29.97 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_296M_720P_25HZ:
				printf ("(assume "
					"SMPTE 296M 720p 25 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_296M_720P_24HZ:
				printf ("(assume "
					"SMPTE 296M 720p 24 Hz)\n");
				break;
			case SDIVIDEO_CTL_SMPTE_296M_720P_23_98HZ:
				printf ("(assume "
					"SMPTE 296M 720p 23.98 Hz)\n");
				break;
			default:
				printf ("(unknown)\n");
				break;
			}
		}

		if (type == 't') {
			snprintf (name, sizeof (name),
				fmt, type, num, "clock_source");
			if (util_strtoul (name, &clksrc) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to get the clock source");
				retcode = -1;
			}
			printf ("\tClock source: %lu ", clksrc);
			switch (clksrc) {
			case SDIVIDEO_CTL_TX_CLKSRC_ONBOARD:
				printf ("(onboard oscillator)\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_NTSC:
				printf ("(external 525i or NTSC reference)\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_PAL:
				printf ("(external 625i or PAL reference)\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_525P:
				printf ("(external 525p reference)\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_625P:
				printf ("(external 625p reference)\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_720P_60:
				printf ("(external 720p/60 reference)\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_720P_59_94:
				printf ("(external 720p/59.94 reference)\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_720P_50:
				printf ("(external 720p/50 reference)\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_720P_30:
				printf ("(external 720p/30 reference)\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_720P_29_97:
				printf ("(external 720p/29.97 reference)\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_720P_25:
				printf ("(external 720p/25 reference)\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_720P_24:
				printf ("(external 720p/24 reference)\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_720P_23_98:
				printf ("(external 720p/23.98 reference)\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_1080P_60:
				printf ("(external 1080p/60 reference)\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_1080P_59_94:
				printf ("(external 1080p/59.94 reference)\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_1080P_50:
				printf ("(external 1080p/50 reference)\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_1080P_30:
				printf ("(external 1080p/30 reference)\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_1080P_29_97:
				printf ("(external 1080p/29.97 reference)\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_1080P_25:
				printf ("(external 1080p/25 reference)\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_1080P_24:
				printf ("(external 1080p/24 reference)\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_1080P_23_98:
				printf ("(external 1080p/23.98 reference)\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_1080I_60:
				printf ("(external 1080i/60 reference)\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_1080I_59_94:
				printf ("(external 1080i/59.94 reference)\n");
				break;
			case SDIVIDEO_CTL_TX_CLKSRC_1080I_50:
				printf ("(external 1080i/50 reference)\n");
				break;
			default:
				printf ("(unknown)\n");
				break;
			}
		}

		snprintf (name, sizeof (name),
			fmt, type, num, "vanc");
		if (util_strtoul (name, &vanc) > 0) {
			/* Don't complain on an error,
			 * since this parameter may not exist. */
			printf ("\tVertical ancillary space: %sabled\n",
				vanc ? "en" : "dis");
		}
	}
	return retcode;

USAGE:
	fprintf (stderr, "Try '%s -h' for more information.\n", argv[0]);
	return -1;
}

