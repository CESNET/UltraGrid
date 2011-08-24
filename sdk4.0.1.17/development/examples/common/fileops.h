#ifndef DVSSDK_FILEOPS_H
#define DVSSDK_FILEOPS_H

#include <stdio.h>
#include <fcntl.h>

#include "../../header/dvs_setup.h"
#include "../../header/dvs_version.h"

void * file_open(char * filename, int mode, int creatmode, int buffering);
void   file_close(void * fd);
int    file_read(void * fd, char * buffer, int size);
int    file_write(void * fd, char * buffer, int size);
int64  file_lseek(void * fd, int64 position, int mode);
int    file_errno();

#endif

