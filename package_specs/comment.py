#!/usr/bin/python3
import re, sys

MODE_ON  = "on"
MODE_OFF = "off"

def usage(arg):
    print("Usage: "+arg+" on|off pattern")
    print("\tcomment out code between following markers:")
    print("####################")
    print("# > pattern (optional note)")
    print("####################")
    print("code to comment out")
    print("####################")
    print("# < pattern (optional note)")
    print("####################")

def checkArgs(argv):
    if (len(argv) != 3):
        print("Error, expected two arguments, yet", len(argv)-1, "given")
        return False

    if (argv[1] != MODE_ON and argv[1] != MODE_OFF):
        print("Mode must be either 'on' or 'off'")
        return False

    return True

def comment_line(line, mode):
    expr_skip = re.compile("^#{2,}$")

    # rpmspec commenting
    expr_spec_on = re.compile("^\s*%.*")
    expr_spec_off   = re.compile("^#\s*%%.*")

    if (expr_skip.match(line)):
        print(line, end="")
    elif(mode == MODE_ON):
        if (expr_spec_on.match(line)):
            line = line.replace("%", "%%")
        print("#"+line, end="")
    elif(mode == MODE_OFF and len(line) > 1 and line[0] == "#"):
        if (expr_spec_off.match(line)):
            line = line.replace("%%", "%")
        print(line[1:], end="")
    else:
        print(line, end="")


def main(argv):
    if (not checkArgs(argv)):
        usage(argv[0])
        sys.exit(-1)

    mode    = argv[1]
    pattern = argv[2]

    expr_on   = re.compile("^# > "+pattern+"( .*)?$")
    expr_off  = re.compile("^# < "+pattern+"( .*)?$")

    commentingMode = False

    lineIdx = 0
    for line in sys.stdin:
        if (expr_on.match(line)):
            if (commentingMode):
                print("Error on line", lineIdx, ": unexpected comment marker start while in commenting mode")
                sys.exit(-2)
            commentingMode = not commentingMode
            print(line, end="")
        elif (expr_off.match(line)):
            if (not commentingMode):
                print("Error on line", lineIdx, ": unexpected comment marker end while outside commenting mode")
                sys.exit(-2)
            commentingMode = not commentingMode
            print(line, end="")
        elif (commentingMode):
            comment_line(line, mode)
        else:
            print(line, end="")
        lineIdx += 1




if __name__ == "__main__":
    main(sys.argv)
