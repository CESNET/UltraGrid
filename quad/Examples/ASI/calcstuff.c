/* calcstuff.c
 * 
 * Calculate DVB ASI stuffing parameters
 * for a given bitrate and packet size.
 *
 * Copyright (C) 2000-2004 Linear Systems Ltd. All rights reserved.
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
#include <math.h>

#include "master.h"

static const char progname[] = "calcstuff";

/* Convert packet size and stuffing parameters to bitrate */
static double
br (int packetsize, int ip_int, int normal_ip, int big_ip)
{
	if ((normal_ip == 0) || (big_ip == 0)) {
		normal_ip = 1;
		big_ip = 0;
	}
	return 270000000 * 0.8 * packetsize /
		(packetsize + ip_int +
		 (double)big_ip / (normal_ip + big_ip) + 2);
}

/* Return the error in parts per million */
static double
ppm (double testrate, double bitrate) {
	return 1000000 * (testrate - bitrate) / bitrate;
}

/* Simulate a finetuning cycle with interleaving,
 * while measuring the differences between the
 * ideal packet transmission times and the actual
 * packet transmission times.
 * Return the maximum range of these differences in seconds.
 */
static double
jitter (int ft0, int ft1, int il0, int il1, double frac)
{
	int i, j, il_cycles;
	double time = 0, il_time = 0;
	double jit = 0, min_jitter = 0, max_jitter = 0;

	/* The number of interleaved finetuning cycles to perform */
	if ((ft0 / il0) < (ft1 / il1)) {
		il_cycles = ft0 / il0;
	} else {
		il_cycles = ft1 / il1;
	}

	/* Interleaved finetuning cycles */
	for (i = 0; i < il_cycles; i++) {

		/* Transmit IL0 packets */
		for (j = 0; j < il0; j++) {
			time += frac;
			jit = il_time - time;
			if (jit < min_jitter) {
				min_jitter = jit;
			} else if (jit > max_jitter) {
				max_jitter = jit;
			}
//			printf ("\t%i\t%f\n", il_time, time);
		}

		/* Transmit IL1 packets */
		for (j = 0; j < il1; j++) {
			il_time++;
			time += frac;
			jit = il_time - time;
			if (jit < min_jitter) {
				min_jitter = jit;
			} else if (jit > max_jitter) {
				max_jitter = jit;
			}
//			printf ("\t%i\t%f\n", il_time, time);
		}
	}

	/* The remainder of the finetuning cycle */
	for (i = 0; i < (ft0 - il0 * il_cycles); i++) {
		time += frac;
		jit = il_time - time;
		if (jit < min_jitter) {
			min_jitter = jit;
		} else if (jit > max_jitter) {
			max_jitter = jit;
		}
//		printf ("\t%i\t%f\n", il_time, time);
	}
	for (i = 0; i < (ft1 - il1 * il_cycles); i++) {
		il_time++;
		time += frac;
		jit = il_time - time;
		if (jit < min_jitter) {
			min_jitter = jit;
		} else if (jit > max_jitter) {
			max_jitter = jit;
		}
//		printf ("\t%i\t%f\n", il_time, time);
	}

	return (max_jitter - min_jitter) / 27000000;
}

/* Return the interpacket stuffing which remains after smoothing
 * with the given interbyte stuffing */
static int
smooth_ip (int ip_int, int ib, int packetsize)
{
	return (ip_int - ib * (packetsize - 1));
}

int
main (int argc, char **argv)
{
	char *endptr;
	int opt, packetsize, ip_int, coarse_ip, normal_ip, big_ip, ib;
	int il_normal, il_big;
	int bufsize, buffers;
	double bitrate, ip, ip_frac, ft_jitter, il_jitter;
	double testrate, tolerance;

	tolerance = 0;
	while ((opt = getopt (argc, argv, "ht:V")) != -1) {
		switch (opt) {
		case 'h':
			printf ("Usage: %s [OPTION]... BITRATE PACKETSIZE\n",
				argv[0]);
			printf ("Calculate DVB ASI stuffing parameters "
				"for a given BITRATE and PACKETSIZE.\n\n");
			printf ("  -h\t\tdisplay this help and exit\n");
			printf ("  -t TOLERANCE\tstop optimizing the bitrate "
				"at an error of\n"
				"\t\tTOLERANCE parts per million "
				"(default 0)\n");
			printf ("  -V\t\toutput version information "
				"and exit\n\n");
			printf ("Report bugs to <support@linsys.ca>.\n");
			return 0;
		case 't':
			tolerance = strtod (optarg, &endptr);
			if (*endptr != '\0' || tolerance > 30) {
				fprintf (stderr,
					"%s: invalid bitrate tolerance: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'V':
			printf ("%s from master-%s (%s)\n", progname,
				MASTER_DRIVER_VERSION,
				MASTER_DRIVER_DATE);
			printf ("\nCopyright (C) 2000-2004 "
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
	if ((argc - optind) < 2) {
		fprintf (stderr, "%s: missing arguments\n", argv[0]);
		goto USAGE;
	} else if ((argc - optind) > 2) {
		fprintf (stderr, "%s: extra operand\n", argv[0]);
		goto USAGE;
	}

	/* Read the packet size */
	packetsize = strtol (argv[optind + 1], &endptr, 0);
	if ((*endptr != '\0') ||
		((packetsize != 188) && (packetsize != 204))) {
		fprintf (stderr, "%s: invalid packet size: %s\n",
			argv[0], argv[optind + 1]);
		return -1;
	}

	/* Read the bitrate */
	bitrate = strtod (argv[optind], &endptr);
	if ((*endptr != '\0') || (bitrate > br (packetsize, 0, 1, 0)) ||
		(bitrate < br (packetsize, 16777215, 1, 255))) {
		fprintf (stderr, "%s: invalid bitrate: %s\n",
			argv[0], argv[optind]);
		return -1;
	}

	/* Assume no interbyte stuffing.
	 * This will allow bitrates down to about 2400 bps,
	 * which is probably good enough! */
	ip = 270000000 * 0.8 * packetsize / bitrate - packetsize - 2;
	printf ("\n%f bytes of stuffing required per packet.\n", ip);
	ip_int = ip;
	ip_frac = ip - ip_int;
	if (ip_frac > 0.5) {
		coarse_ip = ip_int + 1;
	} else {
		coarse_ip = ip_int;
	}
	testrate = br (packetsize, coarse_ip, 1, 0);
	printf ("ib = 0, ip = %i  =>  bitrate = %f bps, err = %.0f ppm\n",
		coarse_ip, testrate, ppm (testrate, bitrate));
	normal_ip = 0;
	big_ip = 0;
	il_normal = 0;
	il_big = 0;
	if ((ip_frac > (double)1 / 256 / 2) &&
		(ip_frac < (1 - (double)1 / 256 / 2)) &&
		(fabs (ppm (testrate, bitrate)) > tolerance)) {
		int n = 1, b = 1, il0, il1, il_normal_max, il_big_max;
		double best_error = 1.0, error, jit = 0.1;

		/* Find the finetuning parameters which
		 * best approximate the desired bitrate.
		 * Break at 1 us p-p network jitter or
		 * the desired bitrate accuracy, whichever
		 * gives less network jitter */
		printf ("\nFinetuning ib = 0, ip = %i:\n", ip_int);
		while ((b < 256) && (n < 256) &&
			(((double)(n * b) / (n + b)) <= 27)) {
			error = (double)b / (n + b) - ip_frac;
			if (fabs (error) < fabs (best_error)) {
				best_error = error;
				normal_ip = n;
				big_ip = b;
				testrate = br (packetsize, ip_int, n, b);
				printf ("normal_ip = %i, big_ip = %i"
					"  =>  bitrate = %f bps, "
					"err = %.0f ppm\n",
					n, b,
					testrate, ppm (testrate, bitrate));
				if (fabs (ppm (testrate, bitrate)) < tolerance) {
					break;
				}
			}
			if (error < 0) {
				b++;
			} else {
				n++;
			}
		}

		/* Calculate the network jitter produced by finetuning */
		ft_jitter = jitter (normal_ip, big_ip,
			normal_ip, big_ip,
			(double)big_ip / (normal_ip + big_ip));

		/* Find the interleaving parameters which
		 * produce the least network jitter */
		if ((normal_ip == 1) || (big_ip == 1)) {
			il_jitter = jitter (normal_ip, big_ip,
				normal_ip, big_ip,
				(double)big_ip / (normal_ip + big_ip));
		} else {
			il_jitter = 0.1;
			il_normal_max = (normal_ip > 14) ? 14 : normal_ip;
			for (il0 = 1; il0 <= il_normal_max; il0++) {
				il_big_max = (big_ip > 14) ? 14 : big_ip;
				il1 = 1;
				while ((il1 <= il_big_max) &&
					(il0 + il1 <= 15)) {
					jit = jitter (normal_ip, big_ip,
						il0, il1,
						(double)big_ip / (normal_ip + big_ip));
					if (jit < il_jitter) {
						il_jitter = jit;
						il_normal = il0;
						il_big = il1;
					}
					il1++;
				}
			}
		}
	} else {
		ip_int = coarse_ip;
		ft_jitter = 0;
		il_jitter = 0;
	}

	/* Print the best stuffing parameters
	 * for transmitters with interleaved finetuning */
	printf ("\nRecommended stuffing parameters with "
		"interleaved finetuning:\n");
	printf ("ib = 0, ip = %i, normal_ip = %i, big_ip = %i,\n"
		"\til_normal = %i, il_big = %i (burst mode) OR\n",
		ip_int, normal_ip, big_ip, il_normal, il_big);
	ib = ip_int / (packetsize - 1);
	while ((ib - smooth_ip (ip_int, ib, packetsize)) >
		(packetsize + 2) / 2) {
		ib--;
	}
	if (ib > 255) {
		if (ib > 65535) {
			ib = 65535;
		}
		printf ("ib = %i, ip = %i, normal_ip = %i, big_ip = %i,\n"
			"\til_normal = %i, il_big = %i (large ib) OR\n",
			ib, smooth_ip (ip_int, ib, packetsize),
			normal_ip, big_ip, il_normal, il_big);
		ib = 255;
	}
	printf ("ib = %i, ip = %i, normal_ip = %i, big_ip = %i,\n"
		"\til_normal = %i, il_big = %i\n",
		ib, smooth_ip (ip_int, ib, packetsize),
		normal_ip, big_ip, il_normal, il_big);
	testrate = br (packetsize, ip_int, normal_ip, big_ip);
	printf ("Bitrate = %f bps; "
		"err = %.0f ppm, network jitter = %.0f ns p-p\n",
		testrate, ppm (testrate, bitrate), il_jitter * 1E+09);

	/* Print the best stuffing parameters
	 * for transmitters with ordinary finetuning */
	printf ("\nRecommended stuffing parameters with finetuning:\n");
	printf ("ib = 0, ip = %i, "
		"normal_ip = %i, big_ip = %i (burst mode) OR\n",
		ip_int, normal_ip, big_ip);
	ib = ip_int / (packetsize - 1);
	while ((ib - smooth_ip (ip_int, ib, packetsize)) >
		(packetsize + 2) / 2) {
		ib--;
	}
	if (ib > 255) {
		if (ib > 65535) {
			ib = 65535;
		}
		printf ("ib = %i, ip = %i, "
			"normal_ip = %i, big_ip = %i (large ib) OR\n",
			ib, smooth_ip (ip_int, ib, packetsize),
			normal_ip, big_ip);
		ib = 255;
	}
	printf ("ib = %i, ip = %i, normal_ip = %i, big_ip = %i\n",
		ib, smooth_ip (ip_int, ib, packetsize),
		normal_ip, big_ip);
	testrate = br (packetsize, ip_int, normal_ip, big_ip);
	printf ("Bitrate = %f bps; "
		"err = %.0f ppm, network jitter = %.0f ns p-p\n",
		testrate, ppm (testrate, bitrate), ft_jitter * 1E+09);

	/* Print the best stuffing parameters
	 * for transmitters without finetuning */
	printf ("\nRecommended stuffing parameters without finetuning:\n");
	printf ("ib = 0, ip = %i (burst mode) OR\n", coarse_ip);
	ib = coarse_ip / (packetsize - 1);
	while ((ib - smooth_ip (coarse_ip, ib, packetsize)) >
		(packetsize + 2) / 2) {
		ib--;
	}
	if (ib > 255) {
		ib = 255;
	}
	printf ("ib = %i, ip = %i\n",
		ib, smooth_ip (coarse_ip, ib, packetsize));
	testrate = br (packetsize, coarse_ip, 1, 0);
	printf ("Bitrate = %f bps; err = %.0f ppm, network jitter = 0 ns p-p\n",
		testrate, ppm (testrate, bitrate));

	/* Print a sample calculation of the interrupt rate and buffer time
	 * based on a reasonable number and size of driver buffers
	 * and the bitrate */
	bufsize = bitrate / (8 * 500);
	if (bufsize < 4) {
		bufsize = 4;
	}
	bufsize = (bufsize / 19176 + ((bufsize % 19176) ? 1 : 0)) * 19176;
	buffers = bitrate * 0.25 / (8 * bufsize);
	if (buffers < 2) {
		buffers = 2;
	}
	printf ("\nSample buffer parameter calculation:\n");
	printf ("At %f bps, %i x %i-byte buffers gives\n",
		bitrate, buffers, bufsize);
	printf ("%.0f interrupt(s) per second "
		"and up to %.3f seconds of buffering.\n",
		bitrate / (8 * bufsize), 8 * buffers * bufsize / bitrate);
	return 0;

USAGE:
	fprintf (stderr, "Try '%s -h' for more information.\n", argv[0]);
	return -1;
}

