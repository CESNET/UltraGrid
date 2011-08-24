
#include <string.h>

typedef struct {
  unsigned char red;
  unsigned char green;
  unsigned char blue;
  unsigned char pad;
} analyzer_color;

typedef struct {
  unsigned char * pbuffer;
  int             xsize;
  int             ysize;
} analyzer_buffer;


#define ANALYZER_NOP        0
#define ANALYZER_HISTOGRAM  1
#define ANALYZER_RGBPARADE  2
typedef struct {
  int operation;
  int lineselect;
  int transparent;
} analyzer_options;


int analyzer(analyzer_options * panalyzer, analyzer_buffer * pdest, analyzer_buffer * psource);


