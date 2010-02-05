/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    fileops - Shows the use of the fifoapi to do display and record of images
//            directly to a file.
//
*/


#include "fileops.h"

#ifdef WIN32
#include <stdio.h>
#include <windows.h>
#include <io.h>
#else
#ifdef linux
#define __USE_LARGEFILE64
#endif
#include <unistd.h>
#include <errno.h>
#endif

#ifndef donttag
# define donttag
#endif

#ifndef TRUE
# define TRUE 1
#endif

#ifndef FALSE
# define FALSE 0
#endif

/*** internal prototypes */
#ifdef WIN32
static void * win32_open(char * filename, int mode, int createmode, int buffering);
static void win32_close(void * fd);
static int win32_rw(void * fd, int bwrite, char * buffer, int size);
static int64 win32_lseek(void * fd, int64 position, int mode);
static int  win32_errno();
#else
static void * unix_open(char * filename, int mode, int createmode, int buffering);
static void   unix_close(void * fd);
static int    unix_read(void * fd, char * buffer, int size, int * pcookie);
static int    unix_write(void * fd, char * buffer, int size, int * pcookie);
static int64  unix_lseek(void * fd, int64 position, int mode);
static int    unix_errno(void);
#endif


/*** generic file functions */
void * file_open(char * filename, int mode, int creatmode, int buffering)
{
#ifdef WIN32
  return win32_open(filename, mode, creatmode, buffering);
#else
  return unix_open(filename, mode, creatmode, buffering);
#endif
}

void file_close(void * fd)
{
#ifdef WIN32
  win32_close(fd);
#else
  unix_close(fd);
#endif
}


int file_read(void * fd, char * buffer, int size)
{
#ifdef WIN32
  return win32_rw(fd, FALSE, buffer, size);
#else
  return unix_read(fd, buffer, size, NULL);
#endif
}


int file_write(void * fd, char * buffer, int size)
{
#ifdef WIN32
  return win32_rw(fd, TRUE, buffer, size);
#else
  return unix_write(fd, buffer, size, NULL);
#endif
}


int64 file_lseek(void * fd, int64 position, int mode)
{
#ifdef WIN32
  return win32_lseek(fd, position, mode);
#else
  return unix_lseek(fd, position, mode);
#endif
}


int file_errno()
{
#ifdef WIN32
  return win32_errno();
#else
  return unix_errno();
#endif
}


/*** Win32 specific functions */
#ifdef WIN32



static void * win32_open(char * filename, int mode, int createmode, int buffering)
{
  HANDLE file;
  int filemode;
  int fileopen;
  int flags;

  if(mode & O_CREAT) {
    filemode = GENERIC_WRITE;
    fileopen = OPEN_ALWAYS;
  } else {
    filemode = GENERIC_READ;
    fileopen = OPEN_EXISTING;
  }

  /*
  // For fastest access it is important to remove the buffering that NT does.
  */
  flags = buffering ? 0 : FILE_FLAG_NO_BUFFERING;	  
  flags |= FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN;

  file = CreateFile(filename, 
                    filemode, 
                    0, 
                    NULL, 
                    fileopen, 
                    flags,
                    NULL);

  if(file == INVALID_HANDLE_VALUE) {
    return NULL;
  } 

  return file;
}

static void win32_close(void * file)
{
  CloseHandle(file);
}



static int win32_rw(void * file, int bwrite, char * buffer, int size)
{
  DWORD result = 0;
  int   res;

  if(bwrite) {
    res = WriteFile(file, buffer, size, &result, NULL);
  } else {
    res = ReadFile(file, buffer, size, &result, NULL);
  }
  if(!res) {
    if(GetLastError() != ERROR_IO_PENDING) {
      printf("file_read %d\n", GetLastError());
    }
  }

  return result;
}


static int64 win32_lseek(void * file, int64 position, int mode)
{
  int seekmode;
  LONG position_low;
  LONG position_high;

  if (mode == SEEK_SET) {
    seekmode = FILE_BEGIN;
  } else if (mode == SEEK_CUR) {
    seekmode = FILE_CURRENT;
  } else {
    seekmode = FILE_END;
  }

  position_low  = position;
  position_high = position >> 32;

  position_low = SetFilePointer(file, position_low, &position_high, seekmode);

  if(position_low == INVALID_SET_FILE_POINTER) {
    int error = GetLastError();
    if(error != NO_ERROR) {
      printf("seek returned %d\n", error);
      return -1;
    }
  }

  return (position_low) | ((int64)position_high<<32);
}

static int win32_errno()
{
  return GetLastError();
}
#endif


/*** Linux specific functions */
#ifndef WIN32
static void * unix_open(char *filename, int mode, int createmode, int buffering)
{
  int fd = open(filename, mode | O_NDELAY, createmode);

#if defined(macintosh)
  if(fd) {
    int ret;
    /* Set no cache bit */
    ret = fcntl(fd, F_NOCACHE, TRUE);
    if(ret < 0) {
      printf("unix_open: fcntl(fd, F_NOCACHE, TRUE) failed, errno=%d\n", errno);
      close(fd);
      return (void*)-1;
    }
    /* Clear read ahead bit */
    ret = fcntl(fd, F_RDAHEAD, FALSE);
    if(ret < 0) {
      printf("unix_open: fcntl(fd, F_NOCACHE, TRUE) failed, errno=%d\n", errno);
      close(fd);
      return (void*)-1;
    }
  }
#endif

  /*
  //  fd == 0 is stdout, lets ignore this.
  */

  if(fd < 0) {
    return NULL;
  }

  return (void *)fd;
}

static void unix_close(void * file)
{
  close((int)file);
}

static int unix_read(void * file, char *buffer, int size, int * pcookie)
{
  if(pcookie) {
    return -1;
  }

  return read((int)file, buffer, size);
}

static int unix_write(void * file, char *buffer, int size, int * pcookie)
{
  if(pcookie) {
    return -1;
  }
  return write((int)file, buffer, size);
}


static int64 unix_lseek(void * file, int64 position, int mode)
{
#ifdef linux
  return lseek64((int)file, position, mode);
#else
  return lseek((int)file, position, mode);
#endif
}

static int unix_errno(void)
{
  return errno;
}
#endif
