#!/usr/bin/python3

import re
import sys

def makeline(line, ending):
	if not line:
		return ending
	if line[-1] == '\n':
		line = line[:-1]

	if not line:
		return ending
	if line[-1] == '\r':
		line = line[:-1]

	return line + ending

def main(argv):
	markerexpr = re.compile("^@@\s[-]([0-9]+)((,([0-9]+))?)\s[+]([0-9]+)((,([0-9]+))?)\s@@")
	newfileexpr = re.compile("^[+]{3}\s")
	oldfileexpr = re.compile("^[-]{3}\s")

	newlineexpr = re.compile("^[+]")
	oldlineexpr = re.compile("^[-]")

	filename = argv[1]

	patching = 0
	patchlen = 0

	oldpos, newpos = 0, 0
	oldend, newend = 0, 0

	with open(filename, "r") as filetoread:
		while True:
			line = filetoread.readline()
			if not line:
				break

			print(makeline(line, "\n" if not patching else "\r\n"), end="")

			if patching:
				if not oldlineexpr.match(line):
					newpos += 1
				if not newlineexpr.match(line):
					oldpos += 1
				
				patching &= not (oldpos >= oldend and newpos >= newend)
				continue

			# not patching, so...
			marker = markerexpr.match(line)
			if marker:
				oldfrom, oldto, newfrom, newto = (marker.group(i) for i in [1, 4, 5, 8])
				oldfrom, oldto, newfrom, newto = (int(i) if i else None for i in [oldfrom, oldto, newfrom, newto])

				oldpos, newpos = oldfrom, newfrom
				oldend = oldpos + (oldto if oldto else 0)
				newend = newpos + (newto if newto else 0)
				patching = True
				continue

	filetoread.close()

if __name__ == "__main__":
	main(sys.argv)

