/* rp155.c
 *
 * SMPTE RP 155 tone generator for Linear Systems Ltd. SMPTE 292M boards.
 *
 * Copyright (C) 2009 Linear Systems Ltd. All rights reserved.
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
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "master.h"

#define MAX_CHANNELS 8
#define MAX_SAMPLING_RATE 48000
#define MAX_SAMPLE_SIZE 32
#define PI 3.141592654

static const char progname[] = "rp155";

int
main (int argc, char **argv)
{
	int opt, seconds;
	char *endptr;
	unsigned int channels, sampling_rate, sample_size, i, j;
	uint8_t data[MAX_CHANNELS * MAX_SAMPLING_RATE * MAX_SAMPLE_SIZE];
	int32_t *p32 = (int32_t *)data;
	int16_t *p16 = (int16_t *)data;
	size_t wavelength, bytes;
	int ret;

	channels = 2;
	seconds = -1; /* Loop forever */
	sampling_rate = 48000;
	sample_size = 16;
	while ((opt = getopt (argc, argv, "c:hn:r:s:V")) != -1) {
		switch (opt) {
		case 'c':
			channels = strtoul (optarg, &endptr, 0);
			if ((*endptr != '\0') ||
				(channels > 8) ||
				(channels % 2)) {
				fprintf (stderr,
					"%s: invalid number of channels: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'h':
			printf ("Usage: %s [OPTION]...\n", argv[0]);
			printf ("Output a SMPTE RP 155 audio tone "
				"for transmission by\n"
				"a Linear Systems Ltd. "
				"SMPTE 292M board.\n\n");
			printf ("  -c CHANNELS\tnumber of channels\n");
			printf ("  -h\t\tdisplay this help and exit\n");
			printf ("  -n NUM\tstop after NUM seconds\n");
			printf ("  -r RATE\tsampling rate in Hz\n");
			printf ("  -s SIZE\tsample size in bits\n");
			printf ("  -V\t\toutput version information "
				"and exit\n");
			printf ("\nCHANNELS may be:\n"
				"\t2 (default), 4, 6, or 8\n");
			printf ("\nRATE may be:\n"
				"\t48000 (default), 44100, or 32000\n");
			printf ("\nSIZE may be:\n"
				"\t16 (default), or 32\n");
			printf ("\nReport bugs to <support@linsys.ca>.\n");
			return 0;
		case 'n':
			seconds = strtol (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid number of seconds: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'r':
			sampling_rate = strtoul (optarg, &endptr, 0);
			if ((*endptr != '\0') ||
				!((sampling_rate == 48000) ||
				(sampling_rate == 44100) ||
				(sampling_rate == 32000))) {
				fprintf (stderr,
					"%s: invalid sampling rate: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 's':
			sample_size = strtoul (optarg, &endptr, 0);
			if ((*endptr != '\0') ||
				!((sample_size == 16) ||
				(sample_size == 32))) {
				fprintf (stderr,
					"%s: invalid sample size: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'V':
			printf ("%s from master-%s (%s)\n", progname,
				MASTER_DRIVER_VERSION,
				MASTER_DRIVER_DATE);
			printf ("\nCopyright (C) 2009 "
				"Linear Systems Ltd.\n"
				"This is free software; "
				"see the source for copying conditions.  "
				"There is NO\n"
				"warranty; not even for MERCHANTABILITY "
				"or FITNESS FOR A PARTICULAR PURPOSE.\n");
			return 0;
		case '?':
			goto USAGE;
		}
	}

	/* Check the number of arguments */
	if ((argc - optind) > 0) {
		fprintf (stderr, "%s: extra operand\n", argv[0]);
		goto USAGE;
	}

	/* Generate one second */
	if (sample_size == 16) {
		for (i = 0; i < sampling_rate; i++) {
			for (j = 0; j < channels; j++) {
				*p16++ = (int)(0x0ccd * sin (2 * PI * i / sampling_rate * 1000) + 0.5);
			}
		}
		wavelength = channels * sampling_rate * sizeof (*p16);
	} else {
		for (i = 0; i < sampling_rate; i++) {
			for (j = 0; j < channels; j++) {
				*p32++ = (int)(0x0cccd * sin (2 * PI * i / sampling_rate * 1000) + 0.5) << 12;
			}
		}
		wavelength = channels * sampling_rate * sizeof (*p32);
	}

	while (seconds) {
		/* Output one second */
		bytes = 0;
		while (bytes < wavelength) {
			if ((ret = write (STDOUT_FILENO,
				data + bytes, wavelength - bytes)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to write");
				return -1;
			}
			bytes += ret;
		}
		if (seconds > 0) {
			seconds--;
		}
	}
	return 0;

USAGE:
	fprintf (stderr, "Try '%s -h' for more information.\n", argv[0]);
	return -1;
}

