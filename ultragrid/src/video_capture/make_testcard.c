#include <stdio.h>

int main(int argc, char *argv[])
{
        FILE *inf, *outf;
        int c, i = 0;

        if (argc != 3) {
                printf("Usage: make_testcard <infile> <outfile>\n");
                return 1;
        }

        outf = fopen(argv[2], "w");
        fprintf(outf, "/* Automatically generated: DO NOT EDIT! */\n");
        fprintf(outf, "unsigned char testcard_image[] = \"");

        inf = fopen(argv[1], "r");
        while (!feof(inf)) {
                c = fgetc(inf);
                if (c != EOF)
                        fprintf(outf, "\\x%02x", c);
        }
        fclose(inf);

        fprintf(outf, "\";\n");
        fclose(outf);

        return 0;
}
