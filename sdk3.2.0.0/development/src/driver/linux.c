/*
//  Copyright (c) 1996-2008 DVS Digital Video Systems AG
//
//  DVSDriver:  Linux kernel interface
//
*/

/** \file */

/*** DVS INCLUDES */
#include "../../header/dvs_setup.h"
#include "../../header/dvs_errors.h"
#include "../../header/dvs_version.h"
#include "ps_common.h"
#include "ps_debug.h"

#include "../../header/dvs_compile.h"


/*** DEBUG DEFINES */
#define DEBUG_LINUX_HAL   0
#define DEBUG_LINUX_INIT  1
#define DEBUG_LINUX_FREE  1
#define DEBUG_LINUX_MMAP  0
#define DEBUG_LINUX_POLL  0
#define DEBUG_LINUX_IOCTL 0

#if defined(DEBUG) || defined(_DEBUG)
# define dprintk(x...)  printk(KERN_ALERT x)
#else
# define dprintk(x...)
#endif

#define DBG_HAL   if (DEBUG_LINUX_HAL)  dprintk
#define DBG_INIT  if (DEBUG_LINUX_INIT) dprintk
#define DBG_FREE  if (DEBUG_LINUX_FREE) dprintk
#define DBG_POLL  if (DEBUG_LINUX_POLL) DPF
#define DBG_IOCTL if (DEBUG_LINUX_IOCTL)  DPF_CMD
#define DBG_MMAP  if (DEBUG_LINUX_MMAP) DPF


/*** INCLUDES */
#ifndef MAKEDEPEND
# define uint uint_xyzzy
# undef DEBUG
# undef memcpy
# undef memset
# undef min
# define DEBUG 0

# include "linux.h"

# include <asm/io.h>
# include <linux/spinlock.h>

# include <linux/device.h>
# include <linux/errno.h>
# include <linux/interrupt.h>
# include <linux/kernel.h>
# include <linux/major.h>
# include <linux/slab.h>
# include <linux/unistd.h>
# include <linux/wait.h>
# include <linux/sched.h>
# include <linux/signal.h>
# include <linux/mm.h>
# include <linux/pci.h>
# include <linux/pci_ids.h>
# include <linux/delay.h>
# include <linux/module.h>
# include <linux/poll.h>
# if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,4)
#  include <linux/wrapper.h>
# endif
# include <linux/vmalloc.h>
# include <linux/proc_fs.h>
# include <linux/blkdev.h>
# include <linux/moduleparam.h>
# include <linux/cdev.h>
# include <linux/pagemap.h>
# if LINUX_VERSION_CODE <= KERNEL_VERSION(2,6,13)
#  if defined(__x86_64__) && defined(CONFIG_COMPAT)
#   include <asm/ioctl32.h>
#  endif
# endif
# include <asm/uaccess.h>
# include <linux/nmi.h>

# undef DEBUG
# undef uint
#endif


/*** DEFINES */
#define DVS_MAXDEVICES 8
#define DVS_DRIVERNAME "dvsdriver"

#define DVSDEVICE_EXISTS  0x01

#ifndef KERNEL_VERSION
# define KERNEL_VERSION(a,b,c) ((a)*256*256+(b)*256+(c))
#endif

#ifndef VM_SHM
# define VM_SHM 0
#endif

#ifdef COMPILE_REGPARM
/* Newer 2.6 kernels use the -mregparm compiler-switch.
   Take care how to call internal and kernel functions,
   otherwise we have a catastrophy when intermixing wrong
   stack layouts.
*/
# define INTERNAL __attribute__((regparm(0)))
#else
# define INTERNAL
#endif

/*** CHECKS */
#if ((LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,0)) && (LINUX_VERSION_CODE < KERNEL_VERSION(2,7,0)))
# define DVS_KERNEL_26
#endif
#if !defined(DVS_KERNEL_26)
# error Kernel version is not supported by dvsdriver!
#endif
#if !defined(__i386__) && !defined(__alpha__) && !defined(__ia64__) && !defined(__x86_64__)
# error Unsupported architecture (supported: i386, ia64, x86_64, alpha)
#endif


/*** TYPEDEFS */
typedef void ps_rec;

typedef struct {
  struct pci_dev *  dev;    /* PCI device structure   */
  int               mask;   /* availability bit mask  */

  struct {
    wait_queue_head_t wait;
    int               res;
    atomic_t          cnt;
    atomic_t          wakeup;
  } vsync[JACK_COUNT];

  struct {
    wait_queue_head_t wait;
    atomic_t          cnt;
  } fifo[JACK_COUNT];

  wait_queue_head_t asyncwait;
  wait_queue_head_t userdma;
  atomic_t          userdma_cnt;

  struct semaphore vma_mutex;

  spinlock_t    irq_lock;
  unsigned long flags;
  int           spinlock_init;
  int           major;
  int           minor;
  int           serial;
  char          name[16];
} dvs_device_t;

typedef struct {
  int nkiobuf;
  struct kiobuf ** kiovec;
} dvs_linux_mdl;


/*** PROTOTYPES */
INTERNAL void dvs_procfs_register(void);
INTERNAL void dvs_procfs_unregister(void);
int  dvsdriver_pci_probe( struct pci_dev *pDev, const struct pci_device_id *pId );
void dvsdriver_pci_remove( struct pci_dev *pDev );

/*** PROTOTYPES in precompiled driver */
INTERNAL void linuxmid_init(void);
INTERNAL void linuxmid_free(void);
INTERNAL int  linuxmid_set_dvsdriver_rec(int card);
INTERNAL void linuxmid_free_dvsdriver_rec(int card);

INTERNAL void linuxmid_device_setup(int card);
INTERNAL int  linuxmid_device_init(int card);
INTERNAL void linuxmid_device_reset(int card);
INTERNAL void linuxmid_device_exit(int card);
INTERNAL void linuxmid_device_open(int card, void * file);
INTERNAL void linuxmid_device_close(int card, void * file);
INTERNAL int  linuxmid_device_irq(void * handle);
INTERNAL int  linuxmid_device_ioctl(int card, void * arg, void * file);
INTERNAL int  linuxmid_device_rw(int card, int bcard2host, void * buffer, uint32 count, uint64 offset);

INTERNAL int  linuxmid_aio_ready(int card, int * magic);
INTERNAL int  linuxmid_aio_isactive(int card, int magic);
INTERNAL void linuxmid_aio_dma_dpc(void * handle, void * pirp, int bcard2host);
INTERNAL int  linuxmid_aio_cancel(int card, int magic);

INTERNAL void linuxmid_set_hwpath(int card, const char * path);
INTERNAL void linuxmid_set_pcimapall(int card, int value);
INTERNAL void linuxmid_set_dma64bit_enable(int card, int value);
INTERNAL void linuxmid_set_dmapageshift(int card, int value);
INTERNAL void linuxmid_set_relay(int card, int value);
INTERNAL void linuxmid_set_device_index(int card);
INTERNAL void linuxmid_set_cardtype(int card, int cardtype);

INTERNAL void * linuxmid_get_handle(int card);
INTERNAL int  linuxmid_get_device_index(void * handle);
INTERNAL int  linuxmid_get_tick(void * handle);
INTERNAL int  linuxmid_get_ncards(void);
INTERNAL int  linuxmid_get_epld_load_error(int card);
INTERNAL int  linuxmid_get_load_error(int card);
INTERNAL void linuxmid_get_addr_dram(int card, uintptr * virt, uintphysaddr * phys, unsigned long * size);
INTERNAL void linuxmid_get_addr_status(int card, uintptr * virt, uintphysaddr * phys, unsigned long * size);
INTERNAL int  linuxmid_get_cardtype(int card);
INTERNAL int  linuxmid_get_fps(void * handle, int jack);
INTERNAL int  linuxmid_get_serial(int card);

INTERNAL int  linuxmid_is_rg_loaded(int card);
INTERNAL int  linuxmid_is_epld_loaded(int card);
INTERNAL int  linuxmid_is_nv_bridge(int card, int mybus);
INTERNAL int  linuxmid_is_device_present(int card, unsigned int vendor, unsigned int deviceid);
INTERNAL int  linuxmid_check_version(int major, int minor, int patch);

INTERNAL void linuxmid_pci_set_ids(int card, int bus, int device, int function);
INTERNAL void linuxmid_pci_set_badr(int card, int index, uintphysaddr busaddr, uintphysaddr physaddr, uint32 length, uint32 mapped, uintptr virtaddr);
INTERNAL void linuxmid_pci_get_badr(int card, int index, uintphysaddr * busaddr, uintphysaddr * physaddr, uint32 * length, uint32 * mapped, uintptr * virtaddr);
INTERNAL void linuxmid_pci_limit_mapping(int card, int index, uint32 * mapped);
INTERNAL int  linuxmid_pci_maxbadr(int card);
INTERNAL void linuxmid_pci_set_irq(int card, uint32 level, uint32 vector);
INTERNAL void linuxmid_pci_get_irq(int card, uint32 * level, uint32 * vector);

INTERNAL int  linuxmid_debug_read(int card, char * buffer, int size, int bkernel);
INTERNAL void linuxmid_debug_psconf(int card, const char * node);
INTERNAL int linuxmid_setdebug(int card, char * word);
INTERNAL void linuxmid_set_driverdma(void * handle, int iochannel, int jack, void * pio);
INTERNAL void DPF_printf(char * s, ...);
INTERNAL void hugo_gettick(void * handle, uint64 * ptick);
INTERNAL void iris_gettick(void * handle, uint64 * ptick);
INTERNAL int  hal_TableRead(char*, void *, int);


/*** module description */
MODULE_AUTHOR("info@dvs.de");
MODULE_DESCRIPTION("DVS Video Capture Card Driver");
MODULE_LICENSE("Proprietary");

/*** command line parameters */
static char *hwpath = NULL;   /* epld-path default value */
static int allow_shirq = 1;   /* shared irq's enabled by default */
static int pcimapall = 0;   /* map full pci memory into kernel */
static int no_irq = 0;      /* enable irq's by default */
static int dmapageshift = 12;  /* Sets the dma block size */
static int dma64bit = 1;                /* perform 64bit addressing DMA */
static int relay = 0;         /* startup status of SDI relays */

module_param(hwpath, charp, 0);
module_param(allow_shirq, int, 0);
module_param(pcimapall, int, 0);
# ifdef _DEBUG
module_param(no_irq, int, 0);
# endif
module_param(dmapageshift,int,0);
module_param(dma64bit, int, 0);
module_param(relay, int, 0);

/*** global variables */
extern ps_rec * global_pps[];
dvs_device_t dvs_device[DVS_MAXDEVICES];

static int mMajor;
static int mCards;
static int mProc;
static int mTranslation[DVS_MAXDEVICES] = {0};
#if LINUX_VERSION_CODE <= KERNEL_VERSION(2,6,13)
# if defined(__x86_64__) && defined(CONFIG_COMPAT)
static int bioctl32 = TRUE;
# endif
#endif

#define MINOR_TO_INTERNAL(minor) ((minor) < DVS_MAXDEVICES ?  mTranslation[(minor)] : 0)

INTERNAL void linux_sort_cards(void)
{
  int card            = 0;
  int pos             = 0;
  int internal_number = 0;
  int own_serial      = 0;
  int tmp_serial      = 0;

  for( card = 0; card <= mCards; card ++ ) {
    internal_number = 0;
    own_serial = dvs_device[card].serial;
    for( pos = 0; pos < DVS_MAXDEVICES; pos++) {
      tmp_serial = dvs_device[pos].serial;
      if( tmp_serial != 0 ) {
        if( tmp_serial < own_serial ) {
          internal_number++;
        }
      }
      mTranslation[card] = internal_number;
    }
  }
}

/*** wrapper functions for kernel compatibility */
INTERNAL int dvs_remap_page_range(struct vm_area_struct *vma, unsigned long from, unsigned long phys_addr, unsigned long size, pgprot_t prot)
{
#ifdef io_remap_pfn_range
  return io_remap_pfn_range(vma, from, phys_addr >> PAGE_SHIFT, size, prot);
#else
  return io_remap_page_range(vma, from, phys_addr, size, prot);
#endif
}

#define dvs_wait_event(wq, condition, before)                           \
({                                                                      \
        int __ret = 0;                                                  \
        do {                                                            \
                wait_queue_t __wait;                                    \
                init_waitqueue_entry(&__wait, current);                 \
                add_wait_queue_exclusive(&(wq), &__wait);               \
                for (;;) {                                              \
                        set_current_state(TASK_UNINTERRUPTIBLE);        \
                        if ((before) && (condition))                    \
                                break;                                  \
                        schedule();                                     \
                        if (!(before) && (condition))                   \
                                break;                                  \
                }                                                       \
                current->state = TASK_RUNNING;                          \
                remove_wait_queue(&(wq), &__wait);                      \
        } while (0);                                                    \
        __ret;                                                          \
})

#define dvs_wait_event_interruptible(wq, condition, before)             \
({                                                                      \
        int __ret = 0;                                                  \
        do {                                                            \
                wait_queue_t __wait;                                    \
                init_waitqueue_entry(&__wait, current);                 \
                add_wait_queue_exclusive(&(wq), &__wait);               \
                for (;;) {                                              \
                        set_current_state(TASK_INTERRUPTIBLE);          \
                        if ((before) && (condition))                    \
                                break;                                  \
                        if (!signal_pending(current)) {                 \
                                schedule();                             \
                                if (!(before) && (condition))           \
                                        break;                          \
                                continue;                               \
                        }                                               \
                        __ret = -ERESTARTSYS;                           \
                        break;                                          \
                }                                                       \
                current->state = TASK_RUNNING;                          \
                remove_wait_queue(&(wq), &__wait);                      \
        } while (0);                                                    \
        __ret;                                                          \
})

#define dvs_wait_event_timeout(wq, condition, before, timeout)          \
({                                                                      \
        long __ret = timeout;                                           \
        do {                                                            \
                wait_queue_t __wait;                                    \
                init_waitqueue_entry(&__wait, current);                 \
                add_wait_queue_exclusive(&(wq), &__wait);               \
                for (;;) {                                              \
                        set_current_state(TASK_UNINTERRUPTIBLE);        \
                        if ((before) && (condition))                    \
                                break;                                  \
                        __ret = schedule_timeout(__ret);                \
                        if (!__ret)                                     \
                                break;                                  \
                        if (!(before) && (condition))                   \
                                break;                                  \
                }                                                       \
                current->state = TASK_RUNNING;                          \
                remove_wait_queue(&(wq), &__wait);                      \
        } while (0);                                                    \
        __ret;                                                          \
})

#define dvs_wait_event_interruptible_timeout(wq, condition, before, timeout)    \
({                                                                      \
        long __ret = timeout;                                           \
        do {                                                            \
                wait_queue_t __wait;                                    \
                init_waitqueue_entry(&__wait, current);                 \
                add_wait_queue_exclusive(&(wq), &__wait);               \
                for (;;) {                                              \
                        set_current_state(TASK_INTERRUPTIBLE);          \
                        if ((before) && (condition))                    \
                                break;                                  \
                        if (!signal_pending(current)) {                 \
                                __ret = schedule_timeout(__ret);        \
                                if (!__ret)                             \
                                        break;                          \
                                if (!(before) && (condition))           \
                                        break;                          \
                                continue;                               \
                        }                                               \
                        __ret = -ERESTARTSYS;                           \
                        break;                                          \
                }                                                       \
                current->state = TASK_RUNNING;                          \
                remove_wait_queue(&(wq), &__wait);                      \
        } while (0);                                                    \
        __ret;                                                          \
})

#ifndef mem_map_reserve
# define mem_map_reserve(p)    set_bit(PG_reserved, &((p)->flags))
#endif
#ifndef mem_map_unreserve
# define mem_map_unreserve(p)  clear_bit(PG_reserved, &((p)->flags))
#endif

/*** HAL functions */
INTERNAL static int dvsdriver_helper_getorder(int size)
{
  int aligned_size = ((size - 1) & PAGE_MASK) + PAGE_SIZE;
  int pages = aligned_size >> PAGE_SHIFT;
  int order;
  int hang_over = FALSE;

  for (order = -1; pages > 0; order++, pages >>= 1) {
    if (pages > 1 && (pages & 0x1) && !hang_over) {
      hang_over = TRUE;
      order++;
    }
  }

  return order;
}

INTERNAL donttag void * hal_Malloc(void * ps, int size)
{
  void * res;

  if (size <= 0x20000) {
    /* slab cache of kmalloc only supports up to 0x20000 bytes */
    res = kmalloc(size, GFP_KERNEL);
  } else {
    int order = dvsdriver_helper_getorder(size);
    Assert(order >= 0);
    res = NULL;
    if (order >= 0) {
      res = (void *) __get_free_pages(GFP_KERNEL, order);
    }
  }

  return res;
}

INTERNAL donttag void hal_Free(void * ps, void * addr, int size)
{
  if (size <= 0x20000) {
    /* slab cache of kmalloc only supports up to 0x20000 bytes */
    kfree(addr);
  } else {
    int order = dvsdriver_helper_getorder(size);
    Assert(order >= 0);
    if (order >= 0) {
      free_pages((unsigned long) addr, order);
    }
  }
}

INTERNAL donttag void * hal_MallocVirtual(void * ps, int size)
{
  return vmalloc(size);
}

INTERNAL donttag void hal_FreeVirtual(void * ps, void * addr, int size)
{
  vfree(addr);
}

INTERNAL donttag void * hal_MallocDMATable(void * ps, int size, uintphysaddr * physaddr)
{
  struct pci_dev * dev = dvs_device[linuxmid_get_device_index(ps)].dev;
  dma_addr_t pci_busaddr;
  void * ptr;

  /* have to be sure, the DMA table definitely has a good address, doing it
     this way also gives a correct _bus_ address, independent from platform */
  ptr = pci_alloc_consistent(dev, size, &pci_busaddr);
  if (ptr && physaddr) {
    *physaddr = (uintphysaddr) pci_busaddr;
  }

  DBG_HAL("hal_MallocDMATable: ptr=%p size=%d busaddr=0x%016Lx\n", ptr, size, (uintphysaddr)pci_busaddr);

  return ptr;
}

INTERNAL donttag void hal_FreeDMATable(void * ps, void * ptr, int size, uintphysaddr physaddr)
{
  struct pci_dev * dev = dvs_device[linuxmid_get_device_index(ps)].dev;

  pci_free_consistent(dev, size, ptr, physaddr);
}

INTERNAL donttag void hal_InitLock(void * ps, ps_lock * lock)
{
  int card = linuxmid_get_device_index(ps);
  if (!dvs_device[card].spinlock_init) {
    spin_lock_init(&dvs_device[card].irq_lock);
    dvs_device[card].spinlock_init = TRUE;
  }
}

INTERNAL donttag void hal_FreeLock(void * ps, ps_lock * lock)
{
  int card = linuxmid_get_device_index(ps);
  dvs_device[card].spinlock_init = FALSE;
}

INTERNAL donttag void hal_Lock(void * ps, ps_lock * lock)
{
  int card = linuxmid_get_device_index(ps);

  if (!in_interrupt()) {
    // Disable irqs on local CPU (avoids deadlock) and lock against other CPUs.
    // When in irq, the lock is already held.
    spin_lock_irqsave(&dvs_device[card].irq_lock, dvs_device[card].flags);
  }
}

INTERNAL donttag void hal_Unlock(void * ps, ps_lock * lock)
{
  int card = linuxmid_get_device_index(ps);

  if (!in_interrupt()) {
    spin_unlock_irqrestore(&dvs_device[card].irq_lock, dvs_device[card].flags);
  }
}

INTERNAL donttag void hal_Wait(int us)
{
  udelay(us);
}

INTERNAL donttag void hal_Sleep(int us)
{
  DECLARE_WAIT_QUEUE_HEAD(sleep);

  /* as the shortest time for sleep is 1/HZ seconds
     (HZ=100 on intel / HZ=1024 on alpha), 'us' is rounded up */
  dvs_wait_event_interruptible_timeout(sleep, 0, FALSE, (us*HZ + 1000000 - HZ) / 1000000);
}

INTERNAL donttag void hal_WatchdogTouch()
{
#ifdef CONFIG_DETECT_SOFTLOCKUP
  touch_softlockup_watchdog();
#endif
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,18)
  touch_nmi_watchdog();
#endif
}

INTERNAL donttag uintphysaddr hal_GetPhysicalAddress(void * ps, void * addr)
{
  uintphysaddr paddr = 0;
  unsigned long offset;
  pgd_t * pgd;
  pmd_t * pmd;
  pte_t * pte;
  pte_t * ptep;
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,11)
  pud_t * pud;
#endif

  if (((uintptr)addr >= PAGE_OFFSET) && ((uintptr)addr < (uintptr) high_memory)) {
    paddr = (uintphysaddr)virt_to_phys(addr);
  } else {
    /* highmem handling - the physical address for a kmapped virtual address
       can only be reached by walking the page tables, as the arithmetical
       conversion does not work for pages not contained in the mem_map array */

    offset = (uintptr)addr & (PAGE_SIZE - 1);   // preserve offset
    addr = (void *)((uintptr)addr & ~(PAGE_SIZE - 1));

    // first level - page global directory
    pgd = pgd_offset_k((uintptr) addr);
    if (!pgd) {
      DPF_WARN("pgd_offset_k failed\n");
    } else if (pgd_none(*pgd)) {
      DPF_WARN("no PGD\n");
    } else if (pgd_bad(*pgd)) {
      DPF_WARN("bad PGD\n");
    } else {
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,11)
      pud = pud_offset(pgd, (uintptr) addr);
      if (pud_none(*pud)) {
        DPF_WARN("no PUD\n");
        pmd = NULL;
      } else {
        pmd = pmd_offset(pud, (uintptr) addr);
      }
#else
      // second level - page middle directory
      pmd = pmd_offset(pgd, (uintptr) addr);
#endif
      if (pmd_none(*pmd)) {
        DPF_WARN("no PMD\n");
      } else if (!pmd_present(*pmd)) {
        DPF_WARN("PMD not present\n");
      } else if (pmd_bad(*pmd)) {
        DPF_WARN("bad PMD\n");
      } else {
        // third level - page table entry
        ptep = pte = pte_offset_map(pmd, (uintptr) addr);
        if (!ptep) {
          DPF_WARN("pte map failed\n");
        } else {
          pte_unmap(ptep);
        }
        if (pte_none(*pte)) {
          DPF_WARN("no PTE\n");
        } else if (!pte_present(*pte)) {
          DPF_WARN("PTE not present\n");
        } else if (!pte_write(*pte)) {
          DPF_WARN("PTE not writable\n");
        } else {
          // here it is - the physical address
          // only valid for x86 in this way
          paddr = (uintphysaddr) pte_val(*pte) & ~((uintphysaddr)PAGE_SIZE - 1);
#if (defined(__i386__) || defined(__x86_64)) && defined(_PAGE_NX)
          // mask out noexec flag - _PAGE_NX is 0 when feature is not activated
          paddr &= (pte_val(*pte) & _PAGE_NX) ? ~(1ULL << 63) : ~0ULL;
#endif
        }
      }
    }

    paddr += offset;  // restore offset into physical address
  }

  //DBG_HAL("hal_GetPhysicalAddress: addr=%p paddr=0x%016Lx\n", addr, paddr);

  return paddr;
}

INTERNAL donttag uintphysaddr hal_GetBusAddress(void * ps, void * addr)
{
  uintphysaddr baddr = 0;

  if (((uintptr)addr >= PAGE_OFFSET) && ((uintptr)addr < (uintptr) high_memory)) {
    baddr = (uintphysaddr) virt_to_bus(addr);
  } else {
    /* highmem handling - there seems to be no portable way to get the busaddr
       for a kmapped virtual address */
#if defined(__i386__) || defined(__x86_64__)
    /* here it's easy as bus:phys is 1:1 */
    baddr = hal_GetPhysicalAddress(ps, addr);
#elif defined(__ia64__)
    /* sure, we cannot easily do this on ia64 - but who cares about highmem on a 64bit machine */
    PE();
#else
    PE();
# warning Need to implement for this platform!
#endif
  }

  DBG_HAL("hal_GetBusAddress: addr=%p baddr=0x%016Lx\n", addr, baddr);

  return baddr;
}


INTERNAL donttag ps_alenlist * hal_MallocDmaMemory(void * ps, int size)
{
  ps_alenlist * palenlist = NULL;
  char * addr;
  int pages;
  uintptr in, out;
  int i;
  unsigned int dma_page_shift = dmapageshift;
  unsigned int dma_page_size  = 1UL << dma_page_shift;
  unsigned int dma_page_mask  = ~(dma_page_size-1);

  size = (size + dma_page_size - 1) & ~(dma_page_size - 1);

  addr = vmalloc(size);
  if (addr) {
    in = (uintptr)addr & dma_page_mask;
    out = (((uintptr)addr + (size - 1)) & dma_page_mask) + dma_page_size;
    pages = (out - in) >> dma_page_shift;

    palenlist = vmalloc(sizeof_alenlist(pages));
    if (palenlist) {
      memset(palenlist, 0, sizeof_alenlist(pages));
      palenlist->addr      = addr;
      palenlist->size      = size;
      palenlist->count     = pages;
      palenlist->allocated = pages;
      palenlist->flags     = 0;

      for (i = 0; i < pages; i++) {
        palenlist->alen[i].physaddr = hal_GetBusAddress(ps, addr + (dma_page_size * i));
        palenlist->alen[i].size     = dma_page_size;
      }
    } else {
      vfree(addr);
    }
  }

  DBG_HAL("hal_MallocDmaMemory: addr=%p size=%d palenlist=%p\n", addr, size, palenlist);

  return palenlist;
}

INTERNAL donttag void hal_FreeDmaMemory(ps_rec * ps, ps_alenlist * palenlist)
{
  DBG_HAL("hal_FreeDmaMemory: addr=%p palenlist=%p\n", palenlist->addr, palenlist);

  vfree(palenlist->addr);
  vfree(palenlist);
}


INTERNAL donttag int hal_GetPageSize(void * ps)
{
  /* page size is different: intel - 0x1000; alpha - 0x2000 */
  return PAGE_SIZE;
}

INTERNAL donttag static void hal_pages2alenlist(void * ps, ps_alenlist * alenlist, int npages, int bcard2host, int * bhigh)
{
#if defined(__ia64__) || defined(__x86_64__)
  int card = linuxmid_get_device_index(ps);
  dvs_device_t * device = &dvs_device[card];
#endif
  uintptr virttemp; /* temporary virtual address */
  uintptr addr = (uintptr) alenlist->addr;
  uint32 size = alenlist->size;
  int i;
  int is_high = *bhigh;
  struct page ** pages = alenlist->mdl;

  {
    if (PageHighMem(pages[0])) {
      /* kmap should not be used for ioremapped pages */
      virttemp  = (uintptr) kmap(pages[0]);
    } else {
      virttemp  = (uintptr) page_address(pages[0]);
    }
    virttemp += addr & (PAGE_SIZE-1);
#if defined(__ia64__) || defined(__x86_64__)
    alenlist->alen[0].physaddr = pci_map_page(device->dev,
      pages[0],         // page
      addr & (PAGE_SIZE-1),     // offset
      PAGE_SIZE - (addr & (PAGE_SIZE-1)), // size
      bcard2host ? PCI_DMA_FROMDEVICE : PCI_DMA_TODEVICE);
#else
    alenlist->alen[0].physaddr = hal_GetBusAddress(ps, (void *) virttemp);
#endif
    if (PageHighMem(pages[0])) {
      kunmap(pages[0]);
    }
  }
  alenlist->alen[0].size     = PAGE_SIZE - (addr & (PAGE_SIZE-1));
  is_high = is_high ? is_high : alenlist->alen[0].physaddr >> 32;

  DPF_DMA_VERBOSE("hal_pages2alenlist: i=%d virt=%p phys=0x%08x%08x size=0x%x\n", 0, virttemp, (uint32)(alenlist->alen[0].physaddr>>32), (uint32)alenlist->alen[0].physaddr, alenlist->alen[0].size);

  for (i = 1; i < npages; i++) {
    {
      if (PageHighMem(pages[i])) {
        virttemp = (uintptr) kmap(pages[i]);
      } else {
        virttemp = (uintptr) page_address(pages[i]);
      }
#if defined(__ia64__) || defined(__x86_64__)
    alenlist->alen[i].physaddr = pci_map_page(device->dev,
      pages[i],         // page
      0,          // offset
      PAGE_SIZE,        // size
      bcard2host ? PCI_DMA_FROMDEVICE : PCI_DMA_TODEVICE);
#else
      alenlist->alen[i].physaddr = hal_GetBusAddress(ps, (void *) virttemp);
#endif
      if (PageHighMem(pages[i])) {
        kunmap(pages[i]);
      }
    }
    alenlist->alen[i].size     = PAGE_SIZE;
    is_high = is_high ? is_high : alenlist->alen[i].physaddr >> 32;

    DPF_DMA_VERBOSE("hal_pages2alenlist: i=%d virt=%p phys=0x%08x%08x size=0x%x\n", i, virttemp, (uint32)(alenlist->alen[i].physaddr>>32), (uint32)alenlist->alen[i].physaddr, alenlist->alen[i].size);
  }

  size = (addr + size) & (PAGE_SIZE-1);

  if (size) {
    /* subtract non-used part of last page */
    alenlist->alen[i-1].size -= PAGE_SIZE - size;
  }

  DPF_DMA_VERBOSE("hal_pages2alenlist: i=%d virt=%p phys=0x%08x%08x size=0x%x\n", i-1, virttemp, (uint32)(alenlist->alen[i-1].physaddr>>32), (uint32)alenlist->alen[i-1].physaddr, alenlist->alen[i-1].size);

  alenlist->count = i;

  *bhigh = is_high;

  return;
}

INTERNAL donttag ps_alenlist * hal_LockMemory(void * ps, int bcard2host, void * addr, uint32 size, int extrapages, int * error)
{
  int npages;     /* number of pages */
  int totalpages;   /* number of pages incl. extrapages */
  ps_alenlist * alenlist;
  uintptr in, out;    /* adress in- and outpoints */
  int result;
  struct page ** pages;   /* pages from get_user_pages */
  void * mdl;     /* stores pages array */
  int errorcode = SV_ERROR_MALLOC;
  int is_high = 0;
  unsigned int dma_page_shift = dmapageshift;
  unsigned int dma_page_size = 1UL << dma_page_shift;
  unsigned int dma_page_mask = ~(dma_page_size-1);

  DPF_DMA_SETUP("hal_LockMemory(ps=%p, bcard2host=%d, addr=%p, size=0x%08x, extrapages=%d)\n", ps, bcard2host, addr, size, extrapages);

  if (size == 0) {
    DPF_WARN("hal_LockMemory: size == 0\n");
    goto out;
  }

  /*** calculate number of pages to lock */
  /* in => address of first page */
  in = (uintptr)addr & PAGE_MASK;
  /* out => address of first page not belonging to range*/
  out = (((uintptr)addr + (size - 1)) & PAGE_MASK) + PAGE_SIZE;
  npages = (out - in) >> PAGE_SHIFT;
  totalpages = npages + extrapages;

  DPF_DMA_SETUP("hal_LockMemory: in:%p out:%p npages=%d totalpages=%d\n", in, out, npages, totalpages);

  /*** alloc alenlist */
  alenlist = (ps_alenlist *) hal_Malloc(ps, sizeof_alenlist(totalpages));
  if (!alenlist) {
    DPF_WARN("hal_LockMemory: allocation of alenlist failed (%d)\n", sizeof_alenlist(totalpages));
    goto out;
  }

  /*** alloc pages array */
  pages = (struct page **) hal_Malloc(ps, sizeof(struct page *) * npages);
  if (!pages) {
    DPF_WARN("hal_LockMemory: allocation of page array failed (%d)\n", sizeof(struct page *) * npages);
    goto out_free_alenlist;
  }

  /*** make all pages present */
  down_read(&current->mm->mmap_sem);
  result = get_user_pages(current, current->mm, (unsigned long) addr, npages, bcard2host, 0 /* force */, pages, NULL);
  up_read(&current->mm->mmap_sem);
  if (result != npages) {
    DPF_WARN("hal_LockMemory: get_user_pages failed (%d)\n", result);
    goto out_free_pages;
  }

  /*** assign pages array as mdl */
  mdl = pages;

  /*** build alen table */
  alenlist->allocated = totalpages;
  alenlist->addr      = addr;
  alenlist->size      = size;
  alenlist->flags     = bcard2host ? PS_ALENLIST_READ : PS_ALENLIST_WRITE;
  alenlist->mdl       = mdl;

  /*** translate kiovec or pages into alenlist */
  hal_pages2alenlist(ps, alenlist, npages, bcard2host, &is_high);

  /*** perform corrections for maximum dma pagesize (e.g. ia64) */

if( PAGE_SIZE != dma_page_size )
  {
    ps_alenlist * alenlist2;
    uintphysaddr paddr;
    int length;
    uintptr dma_in, dma_out;
    int dma_npages, dma_totalpages;
    int i, j;

    dma_in = (uintptr)addr & dma_page_mask;
    dma_out = (((uintptr)addr + (size - 1)) & dma_page_mask) + dma_page_size;
    dma_npages = (dma_out - dma_in) >> dma_page_shift;
    dma_totalpages = dma_npages + (extrapages << PAGE_SHIFT >> dma_page_shift);

    DPF_DMA_SETUP("hal_LockMemory: dma_in:%p dma_out:%p dma_npages=%d dma_totalpages=%d\n", dma_in, dma_out, dma_npages, dma_totalpages);

    alenlist2 = (ps_alenlist *) hal_Malloc(ps, sizeof_alenlist(dma_totalpages));
    if (!alenlist2) {
      DPF_WARN("hal_LockMemory: allocation of secondary alenlist failed (%d)\n", sizeof_alenlist(dma_totalpages));
      goto out_free_pages;
    }

    for (i = j = 0; i < npages; i++) {
      paddr  = alenlist->alen[i].physaddr;
      length = alenlist->alen[i].size;

      while (length > 0) {
        if (length >= dma_page_size) {
          alenlist2->alen[j].physaddr = paddr;
          alenlist2->alen[j].size = dma_page_size;
        } else {
          alenlist2->alen[j].physaddr = paddr;
          alenlist2->alen[j].size = length;
        }
        length -= dma_page_size;
        paddr  += dma_page_size;
        j++;
      }
    }

    alenlist2->allocated = dma_totalpages;
    alenlist2->count     = j;
    alenlist2->addr      = addr;
    alenlist2->size      = size;
    alenlist2->flags     = alenlist->flags;
    alenlist2->mdl       = alenlist->mdl;

    // free original alenlist and keep newly calculated one
    hal_Free(ps, alenlist, sizeof_alenlist(alenlist->allocated));
    alenlist = alenlist2;
    totalpages = dma_totalpages;
  }

  *error = SV_OK;
  return alenlist;

out_free_pages:
  hal_Free(ps, pages, sizeof(struct page *) * npages);
out_free_alenlist:
  hal_Free(ps, alenlist, sizeof_alenlist(totalpages));
out:
  *error = errorcode;
  return NULL;
}

INTERNAL donttag void hal_FlushMemory(void * ps, ps_alenlist * alenlist, int read)
{
  PE();
}

INTERNAL donttag void hal_UnlockMemory(void * ps, ps_alenlist * alenlist)
{
#if defined(__ia64__) || defined(__x86_64__)
  int card = linuxmid_get_device_index(ps);
  dvs_device_t * device = &dvs_device[card];
#endif
  int nelem = 0;
  int i;
  struct page ** pages;
  uintptr in, out;

  if (alenlist) {
#if defined(__ia64__) || defined(__x86_64__)
    /*** unmap pci hardware */
    for (i = 0; i < alenlist->count; i++) {
      pci_unmap_page(device->dev,
        alenlist->alen[i].physaddr,
        alenlist->alen[i].size,
        alenlist->flags & PS_ALENLIST_READ ? PCI_DMA_FROMDEVICE : PCI_DMA_TODEVICE);
    }
#endif

    in = (uintptr)alenlist->addr & PAGE_MASK;
    out = (((uintptr)alenlist->addr + (alenlist->size - 1)) & PAGE_MASK) + PAGE_SIZE;
    nelem = (out - in) >> PAGE_SHIFT;

    pages = alenlist->mdl;

    for (i = 0; i < nelem; i++) {
      if (!PageReserved(pages[i])) {
        if (alenlist->flags & PS_ALENLIST_READ) {
          SetPageDirty(pages[i]);
        }
        page_cache_release(pages[i]);
      }
    }

    if (alenlist->mdl) {
      hal_Free(ps, alenlist->mdl, sizeof(struct page *) * nelem);
    }

    /* free memory of alen table */
    hal_Free(ps, alenlist, sizeof_alenlist(alenlist->allocated));
  } else {
    DPF_WARN("hal_UnlockMemory: alenlist == NULL\n");
  }
}

INTERNAL donttag void hal_QueueDPC(void * ps, int dpctype)
{
  int card = linuxmid_get_device_index(ps);
  dvs_device_t * device = &dvs_device[card];
  int count;
  int iochannel;
  int jack = JACK_DEFAULT;
  int bvsync;
  int bfifo;

  for(iochannel = 0; iochannel < SCHED_COUNT; iochannel++) {
    bvsync = bfifo = FALSE;

    if(dpctype & PS_QUEUE_TYPE_VSYNC_RECORD(iochannel)) {
      jack = JACK_INPUT(iochannel);
      bvsync = TRUE;
    }
    if(dpctype & PS_QUEUE_TYPE_VSYNC_DISPLAY(iochannel)) {
      jack = JACK_OUTPUT(iochannel);
      bvsync = TRUE;
    }
    if(dpctype & PS_QUEUE_TYPE_FIFO_RECORD(iochannel)) {
      jack = JACK_INPUT(iochannel);
      bfifo = TRUE;
    }
    if(dpctype & PS_QUEUE_TYPE_FIFO_DISPLAY(iochannel)) {
      jack = JACK_OUTPUT(iochannel);
      bfifo = TRUE;
    }

    if(bvsync) {
      // wake all, that are sleeping at the moment
      device->vsync[jack].res = 1;
      for(count = atomic_read(&device->vsync[jack].cnt); count > 0; count--) {
        if (waitqueue_active(&device->vsync[jack].wait)) {
          atomic_set(&device->vsync[jack].wakeup, atomic_read(&device->vsync[jack].cnt));
          wake_up(&device->vsync[jack].wait);
        }
      }
    }
  
    if(bfifo) {
      // wake one, in case there is one
      if (atomic_read(&device->fifo[jack].cnt) > 0) {
        atomic_dec(&device->fifo[jack].cnt);
        if (waitqueue_active(&device->fifo[jack].wait)) {
          wake_up(&device->fifo[jack].wait);
        }
      }
    }
  }
}

INTERNAL donttag int hal_AsyncQueue(void * ps, void * pirp)
{
  return SV_ACTIVE;
}

INTERNAL donttag void hal_AsyncReady(void * ps)
{
  int card = linuxmid_get_device_index(ps);
  dvs_device_t * device = &dvs_device[card];
  int magic;

  if (linuxmid_aio_ready(card, &magic)) {
    if (waitqueue_active(&device->asyncwait)) {
      wake_up_all(&device->asyncwait);
    }
  }
}

INTERNAL int hal_VSyncWaitCancel(void * ps, int iochannel)
{
  int card = linuxmid_get_device_index(ps);
  dvs_device_t * device = &dvs_device[card];
  int rec, dis;
  int jack_input  = JACK_INPUT(iochannel);
  int jack_output = JACK_OUTPUT(iochannel);

  rec = (int)waitqueue_active(&device->vsync[jack_input].wait);
  dis = (int)waitqueue_active(&device->vsync[jack_output].wait);

  if (dis || rec) {
    /* wake up all processes waiting for a vsync */
    if(rec) {
      atomic_set(&device->vsync[jack_input].wakeup, atomic_read(&device->vsync[jack_input].cnt));
      device->vsync[jack_input].res = 2;
      wake_up_all(&device->vsync[jack_input].wait);
    }
    if(dis) {
      atomic_set(&device->vsync[jack_output].wakeup, atomic_read(&device->vsync[jack_output].cnt));
      device->vsync[jack_output].res = 2;
      wake_up_all(&device->vsync[jack_output].wait);
    }
  }

  return SV_OK;
}

INTERNAL int hal_VSyncWait(void * ps, int jack)
{
  int card = linuxmid_get_device_index(ps);
  dvs_device_t * device = &dvs_device[card];
  int timeout, jiffies;
  int res = SV_OK;
  int ret;

  DBG_HAL("hal_VSyncWait: called for jack:%d\n", jack);

  if(linuxmid_get_fps(ps, jack) < 24) {
    jiffies = 250 * HZ / 1000;
  } else {
    jiffies = 100 * HZ / 1000;
  }

  atomic_inc(&device->vsync[jack].cnt);

  timeout = dvs_wait_event_timeout(device->vsync[jack].wait, atomic_read(&device->vsync[jack].wakeup) > 0, FALSE, jiffies);
  ret = device->vsync[jack].res;

  if (atomic_read(&device->vsync[jack].wakeup) > 0) {
    atomic_dec(&device->vsync[jack].wakeup);
  }

  if (atomic_read(&device->vsync[jack].cnt) > 0) {
    atomic_dec(&device->vsync[jack].cnt);
  } else {
    device->vsync[jack].res = 0;
  }

  switch (ret) {
  case 1:
    res = SV_OK;
    break;
  case 2:
    res = SV_ERROR_CANCELED; // canceled by hal_VSyncWaitCancel
    break;
  }

  if ((res == SV_OK) && (timeout == 0)) {
    res = SV_ERROR_TIMEOUT;
  }

  return res;
}

INTERNAL int hal_FifoWait(void * ps, int jack, ps_lock * plock)
{
  int card = linuxmid_get_device_index(ps);
  dvs_device_t * device = &dvs_device[card];
  int timeout, jiffies;

  if(linuxmid_get_fps(ps, jack) < 24) {
    jiffies = 500 * HZ / 1000;
  } else {
    jiffies = 250 * HZ / 1000;
  }

  atomic_inc(&device->fifo[jack].cnt);

  if(plock) {
    hal_Unlock(ps, plock);
  }

  DBG_HAL("hal_FifoWait: called for jack:%d\n", jack);

  timeout = dvs_wait_event_timeout(device->fifo[jack].wait, atomic_read(&device->fifo[jack].cnt) <= 0, TRUE, jiffies);

  if(plock) {
    hal_Lock(ps, plock);
  }

  if (atomic_read(&device->fifo[jack].cnt) > 0) {
    atomic_dec(&device->fifo[jack].cnt);
  }

  if (timeout == 0) {
    return SV_ERROR_TIMEOUT;
  }

  return SV_OK;
}

INTERNAL donttag int hal_LockDriverDma(void * ps, int iochannel, int jack, void * pio)
{
  int card = linuxmid_get_device_index(ps);
  dvs_device_t * device = &dvs_device[card];
  int jiffies = HZ;  // 1000 ms timeout
  int timeout = HZ;
  int res = SV_OK;

  if(!atomic_dec_and_test(&device->userdma_cnt)) {
    timeout = dvs_wait_event_timeout(device->userdma, TRUE, TRUE, jiffies);
  }

  if(timeout == 0) {
    atomic_inc(&device->userdma_cnt);
    res = SV_ERROR_TIMEOUT;
  } else {
    linuxmid_set_driverdma(ps, iochannel, jack, pio);
  }

  return res;
}

INTERNAL donttag void hal_UnlockDriverDma(void * ps)
{
  int card = linuxmid_get_device_index(ps);
  dvs_device_t * device = &dvs_device[card];

  if(waitqueue_active(&device->userdma)) {
    wake_up(&device->userdma);
  }

  atomic_inc(&device->userdma_cnt);
}

INTERNAL donttag int hal_DMAStart(void * ps, void * pirp, int bcard2host)
{
  linuxmid_aio_dma_dpc(ps, pirp, bcard2host);

  return SV_ACTIVE;
}

INTERNAL donttag void hal_DMAReady(void * ps, void * pirp)
{
  linuxmid_aio_dma_dpc(ps, pirp, -1);
}


INTERNAL donttag void hal_GetTime(uint64 * ptick)
{
  struct timeval tv;

  /* xtime is only updated every 1/HZ second - so request exact time */
  do_gettimeofday(&tv);
  *ptick = (uint64) tv.tv_sec * 1000000 + tv.tv_usec;
}


INTERNAL donttag void hal_GetTick(void * ps, uint64 * ptick)
{
  switch (linuxmid_get_cardtype(linuxmid_get_device_index(ps))) {
  case PS_CARDTYPE_HUGO:
    hugo_gettick(ps, ptick);
    break;
  case PS_CARDTYPE_IRIS:
    iris_gettick(ps, ptick);
    break;
  default:
    hal_GetTime(ptick);
  }
}


INTERNAL int hal_FileRead (void * ps, char * pathname, char * filename, uint8 * buffer, int buffersize, int * pcount, int bfallback)
{
  char *fname = NULL;
  struct file * file = NULL;
  int ret = SV_OK;
  int bytes_read;
  mm_segment_t old_fs;

  *pcount = 0;
  if (pathname != NULL){
    fname = vmalloc (strlen (filename) + strlen (pathname) + 18);
    if (fname != NULL ) {
      sprintf (fname, "%s/%s", pathname, filename);
    } else {
      ret = SV_ERROR_MALLOC;
    }
  } else {
    fname = filename;
  }

  // Try to open file
  if( ret == SV_OK )
  {
    DBG_INIT("%s: loading file %s\n", DVS_DRIVERNAME, fname);

    file = filp_open(fname, O_RDONLY, S_IRUSR);
    if (file == NULL || IS_ERR(file))
    {
      ret = SV_ERROR_FILEOPEN;
      if (strcmp(filename, "userdef.ref") != 0) {
        // loading of userdef.ref may fail, because not always present
        printk("%s: loading file '%s' failed (open returned %p)\n", DVS_DRIVERNAME, fname, file);
      }
    }
  }

  // Try to read file
  if( ret == SV_OK )
  {
    if (file->f_op && file->f_op->read) {
      old_fs = get_fs();
      set_fs(KERNEL_DS);
      bytes_read = file->f_op->read(file, (char *)buffer, buffersize, &file->f_pos);
      set_fs(old_fs);

      if (bytes_read < 0) {
        ret = SV_ERROR_FILEREAD;
        printk("%s: loading file '%s' failed (read returned %d)\n", DVS_DRIVERNAME, fname, bytes_read);
      }
      *pcount = bytes_read;
    }
  } else if( (ret==SV_ERROR_FILEOPEN) && bfallback ) {
    DPF_WARN("Using fallback for file '%s'\n", filename);
    bytes_read = hal_TableRead( filename, buffer, buffersize);

    if(bytes_read <= 0) {
      DPF_DEBUG("hal_FileRead: TableRead failed\n");
      ret = SV_ERROR_FILEREAD;
      *pcount = 0;
    } else {
      ret = SV_OK;
      *pcount = bytes_read;
    }
  }

  // Close file handle
  if( file != NULL ) {
    if( !IS_ERR(file) ) {
      filp_close(file, current->files);
    }
  }

  // Free buffer
  if (fname != filename) {
    if( fname != 0 ) {
      vfree (fname);
    }
  }

  // Log
  if( ret == SV_OK ) {
    DBG_INIT("%s: file loaded\n", DVS_DRIVERNAME );
  }

  return ret;
}

INTERNAL void * hal_memcpy(void * ps, void * to , const void * from, int size, int mode)
{
  unsigned long nbytes = size;

  switch (mode) {
  case HAL_MEMCPY_U2K:
    nbytes = copy_from_user(to, from, size);
    break;
  case HAL_MEMCPY_K2U:
    nbytes = copy_to_user(to, from, size);
    break;
  case HAL_MEMCPY_K2K:
  case HAL_MEMCPY_U2U:
    memcpy(to, from, size);
    break;
  default:
    PE();
  }

  return to;
}

INTERNAL void * hal_memset(void * ps, void * addr , uint32 value, int count)
{
  return memset(addr, value, count);
}

INTERNAL char * hal_strcpy(void * ps, void * dest , const void * src)
{
  return strcpy(dest, src);
}

INTERNAL char * hal_strncpy(void * ps, void * dest , const void * src, uint32 n)
{
  return strncpy(dest, src, n);
}

INTERNAL char * hal_strcat(void * ps, void * dest , const void * src)
{
  return strcat(dest, src);
}

INTERNAL char * hal_strncat(void * ps, void * dest , const void * src, uint32 n)
{
  return strncat(dest, src, n);
}

INTERNAL int hal_strcmp(void * ps, const void * s1 , const void * s2)
{
  return strcmp(s1, s2);
}

INTERNAL int hal_strncmp(void * ps, const void * s1 , const void * s2, uint32 n)
{
  return strncmp(s1, s2, n);
}

INTERNAL uint32 hal_strlen(void * ps, const void * s)
{
  return strlen(s);
}

INTERNAL donttag int hal_PciFindDeviceId(void * ps, int vendor, int device, int index, int * pbus, int * pdev, int * pfunction)
{
  struct pci_dev * dev = NULL;
  int cnt = 0;

  DPF_INIT("hal_PciFindDeviceId(vendor=0x%04x,device0x=%04x,index=%d)\n", vendor, device, index);

  while ((dev = pci_find_device (vendor, device, dev))) {
    if (cnt++ == index) {
      *pbus = dev->bus->number;
      *pdev = PCI_SLOT(dev->devfn);
      *pfunction = PCI_FUNC(dev->devfn);

      return 0;
    }
  }

  return -1;
}

INTERNAL donttag int hal_PciConfigReadByte(void * ps, int bus, int dev, int function, int address, uint32 * value)
{
  struct pci_dev * device = pci_find_slot(bus, PCI_DEVFN(dev, function));
  uint8 tmp8;
  uint32 res;

  *value = 0;

  if (!device) {
    return SV_ERROR_FILEREAD;
  }

  res = pci_read_config_byte(device, address, &tmp8);

  if (res == 0) {
    *value = tmp8;
  }

  DPF_INIT("hal_PciConfigReadByte(bus=%d,dev=%d/%d,adr=%p) = %x\n", bus, dev, function, address, *value);

  return (res == 0) ? SV_OK : SV_ERROR_FILEREAD;
}

INTERNAL donttag int hal_PciConfigReadWord(void * ps, int bus, int dev, int function, int address, uint32 * value)
{
  struct pci_dev * device = pci_find_slot(bus, PCI_DEVFN(dev, function));
  uint16 tmp16;
  uint32 res;

  *value = 0;

  if (!device) {
    return SV_ERROR_FILEREAD;
  }

  res = pci_read_config_word(device, address, &tmp16);

  if (res == 0) {
    *value = tmp16;
  }

  DPF_INIT("hal_PciConfigReadWord(bus=%d,dev=%d/%d,adr=%p) = %x\n", bus, dev, function, address, *value);

  return (res == 0) ? SV_OK : SV_ERROR_FILEREAD;
}

INTERNAL donttag int hal_PciConfigReadLong(void * ps, int bus, int dev, int function, int address, uint32 * value)
{
  struct pci_dev * device = pci_find_slot(bus, PCI_DEVFN(dev, function));
  uint32 tmp32;
  uint32 res;

  *value = 0;

  if (!device) {
    return SV_ERROR_FILEREAD;
  }

  res = pci_read_config_dword(device, address, &tmp32);

  if (res == 0) {
    *value = tmp32;
  }

  DPF_INIT("hal_PciConfigReadLong(bus=%d,dev=%d/%d,adr=%p) = %x\n", bus, dev, function, address, *value);

  return (res == 0) ? SV_OK : SV_ERROR_FILEREAD;
}

INTERNAL donttag int hal_PciConfigWriteByte(void * ps, int bus, int dev, int function, int address, uint32 value)
{
  struct pci_dev * device = pci_find_slot(bus, PCI_DEVFN(dev, function));
  uint32 res;

  if (!device) {
    return SV_ERROR_FILEWRITE;
  }

  DPF_INIT("hal_PciConfigWriteByte(bus=%d,dev=%d/%d,adr=%p) = %x\n", bus, dev, function, address, value);

  res = pci_write_config_byte(device, address, (uint8)value);

  return (res == 0) ? SV_OK : SV_ERROR_FILEWRITE;
}

INTERNAL donttag int hal_PciConfigWriteWord(void * ps, int bus, int dev, int function, int address, uint32 value)
{
  struct pci_dev * device = pci_find_slot(bus, PCI_DEVFN(dev, function));
  uint32 res;

  if (!device) {
    return SV_ERROR_FILEWRITE;
  }

  DPF_INIT("hal_PciConfigWriteWord(bus=%d,dev=%d/%d,adr=%p) = %x\n", bus, dev, function, address, value);

  res = pci_write_config_word(device, address, (uint16)value);

  return (res == 0) ? SV_OK : SV_ERROR_FILEWRITE;
}

INTERNAL donttag int hal_PciConfigWriteLong(void * ps, int bus, int dev, int function, int address, uint32 value)
{
  struct pci_dev * device = pci_find_slot(bus, PCI_DEVFN(dev, function));
  uint32 res;

  if (!device) {
    return SV_ERROR_FILEWRITE;
  }

  DPF_INIT("hal_PciConfigWriteLong(bus=%d,dev=%d/%d,adr=%p) = %x\n", bus, dev, function, address, value);

  res = pci_write_config_dword(device, address, (uint32)value);

  return (res == 0) ? SV_OK : SV_ERROR_FILEWRITE;
}

INTERNAL donttag int hal_WritePciByteDirect(void * ps, int bus, int dev, int function, int address, uint32 value)
{
  return hal_PciConfigWriteByte(ps, bus, dev, function, address, value);
}

INTERNAL void hal_write(void * ps, uint32 * addr, uint32 value)
{
  /* write a value into io memory */
  writel(value, addr);
}

INTERNAL uint32 hal_read(void * ps, uint32 * addr)
{
  /* read a value from io memory */
  return readl(addr);
}

INTERNAL void hal_memset_io(void * ps, void * addr, uint32 value, uint32 count)
{
  memset_io(addr, value, count);
}

INTERNAL void hal_memcpy_toio(void * ps, void * to, void * from, uint32 count)
{
  memcpy_toio(to, from, count);
}

INTERNAL void hal_memcpy_fromio(void * ps, void * to, void * from, uint32 count)
{
  memcpy_fromio(to, from, count);
}


INTERNAL uintptr hal_GetCurrentProcess(void * ps)
{
  return (uintptr)current->pid;
}

INTERNAL uintptr hal_GetCurrentThread(void * ps)
{
  return (uintptr)current->pid;
}

INTERNAL donttag int hal_PermanentOption(void * ps, int bwrite, int code, int * value)
{
  return SV_ERROR_NOTIMPLEMENTED;
}


#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,21)
irqreturn_t linux_device_irq(int irq, void * handle)
#else
irqreturn_t linux_device_irq(int irq, void * handle, struct pt_regs * regs)
#endif
{
  int handled = 0;

#if defined(CONFIG_SMP)
  int card = linuxmid_get_device_index(handle);
  unsigned long flags = 0;
  spin_lock_irqsave(&dvs_device[card].irq_lock, flags);
  handled = linuxmid_device_irq(handle) ? 1 : 0;
  spin_unlock_irqrestore(&dvs_device[card].irq_lock, flags);
#else
  handled = linuxmid_device_irq(handle) ? 1 : 0;
#endif

  return IRQ_RETVAL(handled);
}


/**
//  \ingroup linux
*/
INTERNAL void dvs_linux_layout(unsigned long * ts, unsigned long * te, unsigned long * ds, unsigned long * de, unsigned long * brk)
{
  *ts = init_mm.start_code;
  *te = init_mm.end_code;
  *ds = init_mm.start_data;
  *de = init_mm.end_data;
  *brk = init_mm.brk;
}


/**
//  \ingroup linux
*/
INTERNAL int dvs_linux_device_free(int card)
{
  uintphysaddr busaddr;
  uintphysaddr physaddr;
  uint32       length;
  uint32       mapped;
  uintptr      virtaddr;
  uint32       level;
  uint32       vector;
  int i;

  printk(KERN_INFO "%s: free card %d\n", DVS_DRIVERNAME, card);

  for (i = 0; i < linuxmid_pci_maxbadr(card); i++) {
    linuxmid_pci_get_badr(card, i, &busaddr, &physaddr, &length, &mapped, &virtaddr);

    if (virtaddr) {
      DBG_FREE("%s: unmap badr%d: %p\n", DVS_DRIVERNAME, i, (void *)virtaddr);

      iounmap((void *)virtaddr);
    }

    if (physaddr) {
      DBG_FREE("%s: release badr%d: 0x%016Lx/%d\n", DVS_DRIVERNAME, i, physaddr, mapped);
      release_mem_region(physaddr, mapped);
    }

    linuxmid_pci_set_badr(card, i, (uintphysaddr)NULL, (uintphysaddr)NULL, 0, 0, (uintptr)NULL);
  }

  linuxmid_pci_get_irq(card, &level, &vector);

  if (vector) {
    DBG_FREE("%s: free irq %d\n", DVS_DRIVERNAME, level);
    free_irq(level, linuxmid_get_handle(card));
  }

  linuxmid_pci_set_irq(card, 0, 0);

  return TRUE;
}


/**
//  \ingroup linux
*/
INTERNAL donttag int dvs_linux_asyncready(int card, int magic, int timeout, int binterruptible)
{
  int running = TRUE;
  int res = 0;
  int state;

  DPF_QUEUE("dvs_linux_asyncready(card=%d, magic=%d, timeout=%d, binterruptible=%d)\n", card, magic, timeout, binterruptible);

  do {
    if (magic == 0) {
      /* take the first one that's ready */
      linuxmid_aio_ready(card, &magic);
    }

    /* irp still active? */
    state = linuxmid_aio_isactive(card, magic);

    if ((state == SV_ACTIVE && magic) ||
        (magic == 0)) {
      switch (timeout) {
      case -1:
        DPF_QUEUE("dvs_linux_asyncready: sleep on magic:%d\n", magic);
        dvs_wait_event(dvs_device[card].asyncwait, linuxmid_aio_isactive(card, magic) == SV_OK, TRUE);
        break;
      case 0:
        /* no sleep */
        running = FALSE;
        res = -EAGAIN;
        break;
      default:
        DPF_QUEUE("dvs_linux_asyncready: sleep on magic:%d for %dms\n", magic, timeout);
        if (0 >= dvs_wait_event_timeout(dvs_device[card].asyncwait, linuxmid_aio_isactive(card, magic) == SV_OK, TRUE, timeout * HZ / 1000)) {
          DPF_QUEUE("dvs_linux_asyncready: sleep on magic:%d for %dms timed out\n", magic, timeout);
          running = FALSE;
          res = -EAGAIN;
        }
        break;
      }
    }

  } while (((state == SV_ACTIVE && magic) || (magic == 0)) && running);

  DPF_QUEUE("dvs_linux_asyncready: running=%d res=%d magic:%d\n", running, res, magic);

  return res;
}

/**
//  \ingroup linux
*/
INTERNAL unsigned long * dvs_linux_timer(void)
{
  return (unsigned long *)(&xtime.tv_sec);
}

/*** driver file operations functions */
static ssize_t dvsdriver_read(struct file * file, char * buffer, size_t count, loff_t * ppos)
{
  int   minor = MINOR(file->f_dentry->d_inode->i_rdev);
  int   card  = MINOR_TO_INTERNAL( minor );
  int   res = -EIO;
  int           magic;

  /*** special read - return first irpmagic that is ready */
  if (*ppos == 0 && count == sizeof(int)) {
    DPF_QUEUE("%s_read(file=%p, buffer=%p, count=0x%x, *ppos=0x%x)\n", DVS_DRIVERNAME, file, buffer, count, (uint32)*ppos);

    linuxmid_aio_ready(card, &magic);
    put_user(magic, (int *)buffer);    /* next ready irp - 0 if none ready */
    res = sizeof(magic);
    *ppos += res;

    DPF_QUEUE("%s_read: special - next irpmagic:%d\n", DVS_DRIVERNAME, magic);
    return res;
  }

  /*** regular device read call */
  res = linuxmid_device_rw(card, TRUE, buffer, count, *ppos);

  if (res == SV_OK) {
    *ppos += count;
    res = count;
  } else {
    res = -EIO;
  }

  return res;
}


/**
//  \ingroup linux
*/
static ssize_t dvsdriver_write(struct file * file, const char * buffer, size_t count, loff_t * ppos)
{
  int   minor = MINOR(file->f_dentry->d_inode->i_rdev);
  int   card  = MINOR_TO_INTERNAL( minor );
  int   res = -EIO;
  int           timeout;
  int           magic;

  /*** special write - buffer contains irpmagic to wait for until ready */
  if (*ppos == 0 && count == 2 * sizeof(int)) {
    DPF_QUEUE("%s_write(file=%p, buffer=%p, count=0x%x, *ppos=0x%x)\n", DVS_DRIVERNAME, file, buffer, count, (uint32)*ppos);

    get_user(magic, (int *)buffer);             /* irp to wait for -
                                                    0 means any irp
                                                   !0 means special irp */
    get_user(timeout, (int *)buffer + 1);       /* timeout for wait -
                                                   -1 means infinite
                                                    0 means no wait
                                                   >0 timeout in ms */

    DPF_QUEUE("%s_write: special - wait for magic:%d timeout:%d\n", DVS_DRIVERNAME, magic, timeout);
    res = dvs_linux_asyncready(card, magic, timeout, TRUE);

    if (res == 0) {
      DPF_QUEUE("%s_write: special - irpmagic:%d ready\n", DVS_DRIVERNAME, magic);
      res = 2 * sizeof(int);
      *ppos += res;
    } else if (res == -EAGAIN) {
      DPF_QUEUE("%s_write: special - irpmagic:%d timeout\n", DVS_DRIVERNAME, magic);
    } else {
      DPF_QUEUE("%s_write: special - irpmagic:%d interrupted\n", DVS_DRIVERNAME, magic);

      linuxmid_aio_cancel(card, magic);
    }

    return res;
  }

  /*** regular device write call */
  res = linuxmid_device_rw(card, FALSE, (void *)buffer, count, *ppos);

  if (res == SV_OK) {
    *ppos += count;
    res = count;
  } else {
    res = -EIO;
  }

  return res;
}


/**
//  \ingroup linux
*/
static int dvsdriver_ioctl(struct inode *inode, struct file * file, unsigned int cmd, unsigned long arg)
{
  int minor = MINOR(inode->i_rdev);
  int card  = MINOR_TO_INTERNAL( minor );
  int res = SV_OK;

  DBG_IOCTL("%s_ioctl: card%d ioctl: cmd:0x%x, arg:0x%x\n", DVS_DRIVERNAME, card, cmd, (int) arg);

  if (!arg) {
    return -EINVAL;
  }

  res = linuxmid_device_ioctl(card, (void *)arg, file);

  DBG_IOCTL("%s_ioctl: card%d ioctl: cmd:0x%x, arg:0x%x res:0x%x\n", DVS_DRIVERNAME, card, cmd, (int) arg, res);

  return 0;
}


#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,16)
static long dvsdriver_compat_ioctl(struct file * file, unsigned int cmd, unsigned long arg)
{
  int minor = MINOR(file->f_dentry->d_inode->i_rdev);
  int res = SV_OK;

  DBG_IOCTL("dvsdriver_compat_ioctl: card%d ioctl: cmd:0x%x, arg:0x%x\n", minor, cmd, (int) arg);

  if (!arg) {
    return -EINVAL;
  }

  res = linuxmid_device_ioctl(minor, (void *)arg, file);

  DBG_IOCTL("dvsdriver_compat_ioctl: card%d ioctl: cmd:0x%x, arg:0x%x res:0x%x\n", minor, cmd, (int) arg, res);

  return 0;
}
#endif


static void dvsdriver_vma_open(struct vm_area_struct * vma)
{
  DBG_MMAP("%s_vma_open(vma=%p)\n", DVS_DRIVERNAME, vma);
}


static void dvsdriver_vma_close(struct vm_area_struct * vma)
{
  DBG_MMAP("%s_vma_close(vma=%p)\n", DVS_DRIVERNAME, vma);
}


static struct page * dvsdriver_vma_nopage(struct vm_area_struct * vma, unsigned long addr, int * type)
{
  unsigned long offset = (vma->vm_pgoff << PAGE_SHIFT) & 0x0fffffff;
  int area             = ((vma->vm_pgoff << PAGE_SHIFT) >> 28) & 0xf;
  dvs_device_t * device = vma->vm_private_data;
  int card              = ((uintptr)device - (uintptr)&dvs_device[0]) / sizeof(dvs_device_t);
  struct page * page    = NOPAGE_SIGBUS;
  int ok                = TRUE;
  uintptr virtaddr      = 0;
  uintphysaddr physaddr = 0;
  unsigned long max_offset = 0;

  linuxmid_get_addr_status(card, &virtaddr, &physaddr, &max_offset);

  offset += addr - vma->vm_start;

  DBG_MMAP("%s_vma_nopage(vma=%p,addr=%08x,type=%p) card:%d offset:%08x max_offset:%08x\n", DVS_DRIVERNAME, vma, addr, type, card, offset, max_offset);

  if(area != 2) {
    DBG_MMAP("%s_vma_nopage: wrong area\n", DVS_DRIVERNAME);
    ok = FALSE;
  }

  if(offset + PAGE_SIZE > max_offset) {
    DBG_MMAP("%s_vma_nopage: offset to high (0x%x)\n", DVS_DRIVERNAME, (unsigned int)max_offset);
    ok = FALSE;
  }

  if(ok) {
    down(&device->vma_mutex);

    page = vmalloc_to_page((void *)(virtaddr + offset));
    get_page(page);
    if(type) {
      *type = VM_FAULT_MINOR;
    }

    up(&device->vma_mutex);
  }

  return page;
}


static struct vm_operations_struct dvsdriver_vm_ops = {
  .open   = dvsdriver_vma_open,
  .close  = dvsdriver_vma_close,
  .nopage = dvsdriver_vma_nopage,
};


/**
//  \ingroup linux
*/
static int dvsdriver_mmap(struct file * file, struct vm_area_struct * vma)
{
  int minor            = MINOR(file->f_dentry->d_inode->i_rdev);
  int card             = MINOR_TO_INTERNAL( minor );
  unsigned long offset = (vma->vm_pgoff << PAGE_SHIFT) & 0x0fffffff;
  int area             = ((vma->vm_pgoff << PAGE_SHIFT) >> 28) & 0xf;
  int size             = vma->vm_end - vma->vm_start;
  char * vm_offset;
  unsigned long vm_flags;
  unsigned long max_offset;
  uintptr virtaddr;
  uintphysaddr physaddr;

  DBG_MMAP("%s_mmap(file=%p, vma=%p)\n", DVS_DRIVERNAME,  file, vma);
  DBG_MMAP("%s_mmap: vm_start=%p vm_end=%p\n", DVS_DRIVERNAME, vma->vm_start, vma->vm_end);
  DBG_MMAP("%s_mmap: card=%d area=%d offset=%p size=0x%x\n", DVS_DRIVERNAME, card, area, offset, size);

  if(offset & ~PAGE_MASK) {
    DBG_MMAP("%s_mmap: offset %p not aligned\n", DVS_DRIVERNAME, (int)offset);
    return -EINVAL;
  }

  if (size <= 0) {
    DBG_MMAP("%s_mmap: vm_end-vm_start %p <= 0\n", DVS_DRIVERNAME, (int)size);
    return -EINVAL;
  }

  switch(area) {
  case 1:   /* sram memory */
    return -ENOMEDIUM;
    break;
  case 0:   /* dram memory */
  case 3:   /* dram bus address memory */
    linuxmid_get_addr_dram(card, &virtaddr, &physaddr, &max_offset);
#ifdef __alpha__
    /* this is tricky, but the only way */
    vm_offset = ((char *)(virtaddr)) + offset;
#else
    vm_offset = ((char *) (uintptr) (physaddr)) + offset;
#endif
    vm_flags = (VM_SHM | VM_SHARED | VM_IO);
    break;
  case 2:   /* status memory */
    linuxmid_get_addr_status(card, &virtaddr, &physaddr, &max_offset);
    vm_offset = (char *) (uintptr) physaddr;
    vm_flags = (VM_SHM | VM_SHARED | VM_RESERVED);
    break;
  default:
    DBG_MMAP("%s_mmap: invalid memory area %d\n", DVS_DRIVERNAME, area);
    return -EINVAL;
  }

  /* check the range of the area to be mapped */
  if (offset > max_offset) {
    DBG_MMAP("%s_mmap: offset to high (0x%x)\n", DVS_DRIVERNAME, (unsigned int)max_offset);
    return -EINVAL;
  }
  if (offset + size > max_offset) {
    DBG_MMAP("%s_mmap: offset+size to high (0x%x)\n", DVS_DRIVERNAME, (unsigned int)max_offset);
    return -EINVAL;
  }

  vma->vm_flags |= vm_flags;

  if (area != 2) {
    /*
    // Remapping virtual memory is not possible this way.
    // This is done via vm_ops->nopage.
    */
    if (dvs_remap_page_range(vma, vma->vm_start, (unsigned long)vm_offset, vma->vm_end-vma->vm_start, vma->vm_page_prot)) {
      DBG_MMAP("%s_mmap: remap_page_range failed (%p)\n", DVS_DRIVERNAME, vm_offset);
      return -EAGAIN;
    }
  }

  vma->vm_ops = &dvsdriver_vm_ops;
  vma->vm_private_data = &dvs_device[card];
  dvsdriver_vma_open(vma);

  DBG_MMAP("%s_mmap: mapped physical address %p\n", DVS_DRIVERNAME, vm_offset);

  return 0;
}


/**
//  \ingroup linux
*/
static int dvsdriver_open(struct inode * inode, struct file * file)
{
  unsigned int minor = MINOR(inode->i_rdev);
  unsigned int card  = MINOR_TO_INTERNAL( minor );

  if(minor >= DVS_MAXDEVICES) {
    return -ENODEV;
  }

  if(!(dvs_device[minor].mask & DVSDEVICE_EXISTS)) {
    return -ENODEV;
  }

  linuxmid_device_open(card, file);

  return 0;
}


/**
//  \ingroup linux
*/
static int dvsdriver_release(struct inode * inode, struct file * file)
{
  int minor = MINOR(file->f_dentry->d_inode->i_rdev);
  int card  = MINOR_TO_INTERNAL( minor );

  linuxmid_device_close(card, file);

  return 0;
}


/*** driver file operations list */
static struct file_operations dvsdriver_fops = {
  .read   = dvsdriver_read,
  .write    = dvsdriver_write,
  .ioctl    = dvsdriver_ioctl,
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,16)
  .compat_ioctl   = dvsdriver_compat_ioctl,
#endif
  .mmap   = dvsdriver_mmap,
  .open   = dvsdriver_open,
  .release  = dvsdriver_release,
  .owner    = THIS_MODULE,
};

static struct cdev dvsdriver_cdev = {
        .kobj   =       {.name = DVS_DRIVERNAME, },
        .owner  =       THIS_MODULE,
};

static struct pci_device_id dvs_pci_tbl[] = {
#ifdef COMPILE_SDIO
#error SDIO not supported by this driver
#endif
#ifdef COMPILE_IRIS
  {PCI_DEVICE(PCI_VENDORID_XILINX, PCI_DEVICEID_XILINX_IRIS  )},
  {PCI_DEVICE(PCI_VENDORID_DVS   , PCI_DEVICEID_DVS_IRIS     )},
  {PCI_DEVICE(PCI_VENDORID_DVS   , PCI_DEVICEID_DVS_IRIS_LUCY)},
  {PCI_DEVICE(PCI_VENDORID_DVS   , PCI_DEVICEID_DVS_IRIS_LUCYLT)},
#endif
#if defined(COMPILE_HUGO) && defined(_DEBUG)
  {PCI_DEVICE(PCI_VENDORID_XILINX, PCI_DEVICEID_XILINX_HUGO  )},
  {PCI_DEVICE(PCI_VENDORID_DVS   , PCI_DEVICEID_DVS_HUGO     )},
#endif
  {0,}
};
MODULE_DEVICE_TABLE( pci, dvs_pci_tbl );

static struct pci_driver dvsdriver_pci = {
  .name     = DVS_DRIVERNAME,
  .id_table = dvs_pci_tbl,
  .probe    = dvsdriver_pci_probe,
  .remove   = dvsdriver_pci_remove,
};


int init_module_checks(void)
{
#if defined(__i386__)
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,20)
# if !defined(COMPILE_REGPARM)
  printk("%s: CONFIG_REGPARM=0 but it is enabled in 2.6.20 by default.\n", DVS_DRIVERNAME );
  return -ENODEV;
# endif
#else
# if ((CONFIG_REGPARM == 1) && !defined(COMPILE_REGPARM))
  printk("%s: CONFIG_REGPARM=1 but COMPILE_REGPARM not set.\n", DVS_DRIVERNAME );
  return -ENODEV;
# endif
# if ((CONFIG_REGPARM == 0) &&  defined(COMPILE_REGPARM))
  printk("%s: CONFIG_REGPARM=0 but COMPILE_REGPARM is set.\n", DVS_DRIVERNAME );
  return -ENODEV;
# endif
#endif
#endif

  if (!linuxmid_check_version(DVS_VERSION_MAJOR, DVS_VERSION_MINOR, DVS_VERSION_PATCH)) {
    printk("%s: run-time check: version of precompiled driver binary and post-compile files do not match!\n", DVS_DRIVERNAME );
    return -ENODEV;
  }

  return 0;
}


/* driver module entry points */
int init_module(void)
{
  dev_t tmp_major;

  //Init device structure
  memset(&dvs_device, 0, sizeof(dvs_device_t) * DVS_MAXDEVICES);
  memset(&mTranslation,0 , sizeof(int) * DVS_MAXDEVICES);

  mCards = 0;
  mMajor = 0;
  mProc  = 0;

#ifdef _DEBUG
  printk("%s: DEBUG driver compiled %s %s\n", DVS_DRIVERNAME, __DATE__, __TIME__);
#endif

  //Init middle layer
  linuxmid_init();

  //Security checks
  if( init_module_checks() ) {
    return -ENODEV;
  }

  //Get major number dynamically
  if (alloc_chrdev_region( &tmp_major, 0, DVS_MAXDEVICES, DVS_DRIVERNAME )) {
    printk("%s: unable to get major number\n", DVS_DRIVERNAME);
    return -EIO;
  } else {
    mMajor = MAJOR(tmp_major);
  }

  // Init character device driver
  cdev_init(&dvsdriver_cdev, &dvsdriver_fops);
  if ( cdev_add(&dvsdriver_cdev, MKDEV( mMajor, 0), DVS_MAXDEVICES))
  {
    printk("%s: unable to add character device\n", DVS_DRIVERNAME );
    unregister_chrdev_region( MKDEV( mMajor, 0) , DVS_MAXDEVICES);
    return -EIO;
  }

  // Register driver into the pci bus
  if( pci_register_driver( &dvsdriver_pci ) < 0 )
  {
    printk("%s: unable to register pci driver \n", DVS_DRIVERNAME );

    pci_unregister_driver( &dvsdriver_pci );
    cdev_del( &dvsdriver_cdev );
    unregister_chrdev_region( MKDEV( mMajor, 0) , DVS_MAXDEVICES);

    return -EIO;
  }

#if LINUX_VERSION_CODE <= KERNEL_VERSION(2,6,13)
# if defined(__x86_64__) && defined(CONFIG_COMPAT)
  if(register_ioctl32_conversion(0, NULL) == -EINVAL) {
    printk("%s: Overriding 'duplicated ioctl32 handler' kernel warning. This has no influence on driver functionality.\n", DVS_DRIVERNAME);
    bioctl32 = FALSE;
  }
# endif
#endif

  return 0;
}

void cleanup_module(void)
{
#if LINUX_VERSION_CODE <= KERNEL_VERSION(2,6,13)
# if defined(__x86_64__) && defined(CONFIG_COMPAT)
  if(bioctl32) {
    unregister_ioctl32_conversion(0);
  }
# endif
#endif

  dvs_procfs_unregister();

  pci_unregister_driver( &dvsdriver_pci );
  cdev_del(&dvsdriver_cdev);
  unregister_chrdev_region(MKDEV( mMajor, 0), DVS_MAXDEVICES);

  linuxmid_free();
}


int linux_map_memory( int card, struct pci_dev *pDev )
{
  int i = 0;
  int error = 0;
  uintphysaddr addr;
  unsigned long size;

  for (i = 0; i < linuxmid_pci_maxbadr(card); i++) {
    uintphysaddr busaddr;
    uintphysaddr physaddr;
    uint32 length;
    uint32 mapped;
    uintptr virtaddr = (uintptr) NULL;

    busaddr  = pci_resource_start(pDev, i);
    physaddr = busaddr;
    length   = pci_resource_len(pDev, i);
    mapped   = length;

    linuxmid_pci_limit_mapping(card, i, &mapped);

    if (physaddr) {
      DBG_INIT("%s: phys%d: 0x%016Lx (length:%08x/mapped:%08x)\n", DVS_DRIVERNAME, i, physaddr, length, mapped);

      if (!request_mem_region(busaddr, mapped, DVS_DRIVERNAME )) {
        printk("%s: could not request memory region 0x%016Lx (%08x)\n", DVS_DRIVERNAME, busaddr, length);
        error = -1;
      }

      if ( error == 0 ) {
        if (physaddr && mapped) {
          addr = (physaddr & ~(PAGE_SIZE-1));
          size = (physaddr &  (PAGE_SIZE-1)) + mapped;

          virtaddr = (uintptr) ioremap_nocache(addr, size);

          if (virtaddr == (uintptr) NULL) {
            printk("%s: could not map virtaddr %d\n", DVS_DRIVERNAME, i);
            error = -1;
          }
          DBG_INIT("%s: badr%d: %p 0x%016Lx:%d\n", DVS_DRIVERNAME, i, (void *)virtaddr, addr, (unsigned int)size);
        }
      }

      if ( error == 0 ) {
        virtaddr += (physaddr & (PAGE_SIZE-1));
      }

      linuxmid_pci_set_badr(card, i, busaddr, physaddr, length, mapped, virtaddr);
    }
  }

  return error;
}

/** linux_unmap_memory
* This function unmap the memory from the dvsdevice.
* You can call this function without mapped memory,
* so you do not have to check if map memory was sucessfull.
*/
void linux_unmap_memory( int card, struct pci_dev *pDev )
{
  uintphysaddr busaddr;
  uintphysaddr physaddr;
  uint32       length;
  uint32       mapped;
  uintptr      virtaddr;
  int i;

  for (i = 0; i < linuxmid_pci_maxbadr(card); i++) {
    linuxmid_pci_get_badr(card, i, &busaddr, &physaddr, &length, &mapped, &virtaddr);

    if (virtaddr) {
      DBG_FREE("%s: unmap badr%d: %p\n", DVS_DRIVERNAME, i, (void *)virtaddr);

      iounmap((void *)virtaddr);
    }

    if (physaddr) {
      DBG_FREE("%s: release badr%d: 0x%016Lx/%d\n", DVS_DRIVERNAME, i, physaddr, mapped);
      release_mem_region(physaddr, mapped);
    }

    linuxmid_pci_set_badr(card, i, (uintphysaddr)NULL, (uintphysaddr)NULL, 0, 0, (uintptr)NULL);
  }
}


int linux_request_irq( int card, struct pci_dev *pDev )
{
  int irq = 0;
  int result = 0;
  uint32 vector;

  if( !pDev->irq ) {
    printk("%s: could not find irq line\n", DVS_DRIVERNAME);
    result = -1;
  } else {
    irq = pDev->irq;
  }

  if( (result==0) && !no_irq ) {
    DBG_INIT("%s: requesting irq:%d (allow_shirq=%d)\n", DVS_DRIVERNAME, irq, allow_shirq);

    vector = !request_irq( irq, linux_device_irq,
                           SA_INTERRUPT | (allow_shirq ? SA_SHIRQ : 0),
                           DVS_DRIVERNAME, linuxmid_get_handle(card));
    if( !vector ) {
      printk("%s: could not map irq %d\n", DVS_DRIVERNAME, irq );
      result = -1;
    }

    if( result == 0 ) {
      linuxmid_pci_set_irq( card, irq, vector );
    }

    DBG_INIT("%s: irq:%d vector:%d\n", DVS_DRIVERNAME, irq, vector );
  }

  return result;
}

/** linux_unrequest_irq
* This function free the hired irq.
* You can call this function without a hired irq,
* so you do not have to check if request irq was sucessfull.
*/
void linux_unrequest_irq( int card, struct pci_dev *pDev )
{
  uint32 level  = 0;
  uint32 vector = 0;

  //Free irq
  linuxmid_pci_get_irq( card, &level, &vector );
  if( vector ) {
    free_irq( pDev->irq, linuxmid_get_handle( card ) );
    linuxmid_pci_set_irq( card, 0, 0 );
  }
}


void init_dvs_device_struct( int card, struct pci_dev *pDev, char *name )
{
  int jack;

  dvs_device[card].mask |= DVSDEVICE_EXISTS;
  dvs_device[card].major = mMajor;
  dvs_device[card].minor = mCards;
  dvs_device[card].dev   = pDev;
  sprintf( dvs_device[card].name, name, card );
  init_waitqueue_head(&dvs_device[card].asyncwait);
  for(jack = 0; jack < JACK_COUNT; jack++) {
    init_waitqueue_head(&dvs_device[card].vsync[jack].wait);
    atomic_set(&dvs_device[card].vsync[jack].cnt, 0);
    atomic_set(&dvs_device[card].vsync[jack].wakeup, 0);
    dvs_device[card].vsync[jack].res = 0;

    init_waitqueue_head(&dvs_device[card].fifo[jack].wait);
    atomic_set(&dvs_device[card].fifo[jack].cnt, 0);
  }
  init_waitqueue_head(&dvs_device[card].userdma);
  atomic_set(&dvs_device[card].userdma_cnt, 1);
  init_MUTEX(&dvs_device[card].vma_mutex);
  dvs_device[card].spinlock_init = TRUE;
  spin_lock_init(&dvs_device[card].irq_lock);
}


void free_dvs_device_struct( int card )
{
  dvs_device[card].mask  = 0;
  dvs_device[card].major = 0;
  dvs_device[card].minor = 0;
  dvs_device[card].dev   = 0;
}


void init_midlayer( int card, struct pci_dev *pDev, int cardtype )
{
  linuxmid_set_cardtype( card, cardtype );
  linuxmid_set_hwpath( card, hwpath );
  linuxmid_set_pcimapall( card, pcimapall );
  linuxmid_set_relay( card, relay );
  linuxmid_set_dvsdriver_rec( card );
  linuxmid_set_device_index( card );
  linuxmid_pci_set_ids( card, pDev->bus->number, PCI_SLOT(pDev->devfn), PCI_FUNC(pDev->devfn));
}


void free_midlayer( int card )
{
  linuxmid_free_dvsdriver_rec( card );
}

void linux_set_dma_mask(struct pci_dev *pDev, int cardtype)
{
  switch (cardtype) {
  default:
    if(dma64bit) {
      if (!pci_set_dma_mask(pDev, (u64) 0xffffffffffffffffULL)) {
#ifdef __ia64__
        if (pci_set_consistent_dma_mask(pDev, (u64) 0xffffffffffffffff)) {
          DBG_INIT("Unable to obtain 64 bit DMA for consistent allocations\n");
        }
#endif
      } else {
        if (pci_set_dma_mask(pDev, 0xffffffffULL)) {
          DBG_INIT("No usable DMA configuration, aborting.\n");
        }
      }
    } else {
      DBG_INIT("%s: Limiting DMA addressing to 32bit.\n", DVS_DRIVERNAME);
      if (pci_set_dma_mask(pDev, 0xffffffffULL)) {
        DBG_INIT("No usable DMA configuration, aborting.\n");
      }
    }
    break;
  }
}


int dvsdriver_pci_probe( struct pci_dev *pDev, const struct pci_device_id *pId )
{
  int result       = 0;
  int cardtype     = 0;
  char* deviceName = 0;
  int card         = mCards;

  DBG_INIT("%s: pci_probe was called.\n", DVS_DRIVERNAME );

  switch(pDev->vendor)
  {
  case PCI_VENDORID_XILINX:
    switch(pDev->device)
    {
    case PCI_DEVICEID_XILINX_IRIS:
      deviceName = "iris%d";
      cardtype   = PS_CARDTYPE_IRIS;
      break;
    case PCI_DEVICEID_XILINX_HUGO:
      deviceName = "hugo%d";
      cardtype   = PS_CARDTYPE_HUGO;
      break;
    default:
      result = -1;
      break;
    }
    break;
  case PCI_VENDORID_DVS:
    switch(pDev->device & PCI_DEVICEID_DVS_MASK) {
    case PCI_DEVICEID_DVS_IRIS:
      deviceName = "iris%d";
      cardtype   = PS_CARDTYPE_IRIS;
      break;
    case PCI_DEVICEID_DVS_HUGO:
      deviceName = "hugo%d";
      cardtype   = PS_CARDTYPE_HUGO;
      break;
    default:
      result = -1;
      break;
    }
    break;
  default:
    result = -1;
  }

  if(result == -1) {
    printk("%s: pci_probe found invalid card (vendor:%04x device:%04x).\n", DVS_DRIVERNAME, pDev->vendor, pDev->device);
  }

  if( result == 0 ) {
    DBG_INIT("%s: pci_probe found card 0x%x.\n", DVS_DRIVERNAME, pDev->device );

    //Set middle layer information
    init_midlayer( card, pDev, cardtype ); //Need free_midlayer() function

    //Set dma size
    if( (dmapageshift < 9) || (dmapageshift > 12) )
    {
      DPF_WARN("%s: Wrong dmapageshift \"%d\" was set, now default value will be used.\n", DVS_DRIVERNAME, dmapageshift);
      dmapageshift = 12;
    }
    linuxmid_set_dmapageshift( card, dmapageshift );

    //Say the kernel that we are able to create own dma transfers
    pci_set_master( pDev );

    result = pci_enable_device(pDev);
    if( result ) {
      printk("%s: card %d pci_enable_device failed (result:%d)\n", DVS_DRIVERNAME, card, result );
    }
  }

  if( result == 0 ) {
    //Set DmaMask into pci core
    linux_set_dma_mask(pDev, cardtype);

    //Map memory
    result = linux_map_memory( card, pDev );  //Need linux_unmap_memory() function
  }

  //Init dvs_device_struct
  if( result == 0 ) {
    init_dvs_device_struct( card, pDev, deviceName ); //Need free_dvs_device_struct()
  }

  //Request irq
  if( result == 0 ) {
    result = linux_request_irq( card, pDev ); //Need linux_unrequest_irq(0 function
  }

  //Init real hardware
  ////////////////////
  if( result == 0 ) {
    linuxmid_device_setup( card );

    if (linuxmid_get_load_error(card)) {
      printk("%s: unable to setup card (%d)\n", DVS_DRIVERNAME, linuxmid_get_load_error(card));
      result = -EIO;
    }
  }

  if( result == 0 ) {
    if (!linuxmid_device_init(card)) {
      printk("%s: error initializing card %d\n", DVS_DRIVERNAME, card);
      result = -EIO;
    }
  }

  if( result == 0 ) {
    if (!linuxmid_is_rg_loaded(card)) {
      printk("%s: unable to load raster files\n", DVS_DRIVERNAME );
      result = -EIO;
    }
  }

  if (result == 0 ) {
    DBG_INIT("%s: resetting card %d\n", DVS_DRIVERNAME, card);
    if (!no_irq) {
      linuxmid_device_reset(card);
    }

    switch (linuxmid_get_load_error(card)) {
      case SV_OK:
      case SV_ERROR_NOLICENCE:
      case SV_ERROR_EPLD_VERSION:
        break;
      default:
        printk("%s: unable to init card %d (%d)\n", DVS_DRIVERNAME, card, linuxmid_get_load_error(card));
        result = -EIO;;
    }
  }

  if( result == 0 ) {
    if (!linuxmid_is_epld_loaded(card)) {
      printk("%s: unable to load epld files (%d)\n", DVS_DRIVERNAME, linuxmid_get_epld_load_error(card));
      result = -EIO;;
    }
  }

  if( result == 0 ) {
    dvs_device[card].serial = linuxmid_get_serial( card );
    //Sort
    linux_sort_cards();
  }

  if( result == 0 ) {
    mCards++;
  
    //Create proc filesystem
    dvs_procfs_register();
  }

  //Check bad devices
  if(linuxmid_is_device_present(card, 0x1022, 0x7450)) {
    DPF_WARN("%s: DVS_WARNING: Faulty bridge detected 0x%x:0x%x.\n", DVS_DRIVERNAME, 0x1022, 0x7450);
    if(dmapageshift>10) {
      printk(  "%s: DVS_WARNING: Faulty bridge detected 0x%x:0x%x.\n", DVS_DRIVERNAME, 0x1022, 0x7450);
    }
  }
  
  if( result == 0 ) {
    DBG_INIT("%s: probe succeed, type:0x%x card:%d\n", DVS_DRIVERNAME, pDev -> device, card );
  } else {
    printk("%s: probe failed, type:0x%x card:%d\n", DVS_DRIVERNAME, pDev -> device, card );

    //Cleanup
    dvsdriver_pci_remove( pDev );
  }

  return result;
}


int get_dvsdevice_pointer( struct pci_dev *pDev )
{
  int pos;

  for( pos = 0; pos < DVS_MAXDEVICES; pos++ ) {
    if( dvs_device[pos].dev == pDev ) {
      return pos;
    }
  }

  return -1;
}


void dvsdriver_pci_remove( struct pci_dev *pDev )
{
  int card = get_dvsdevice_pointer( pDev );

  if( card >= 0 ) {
    //Cleanup real hardware
    linuxmid_device_exit( card );

    //Cleanup internal struct
    free_dvs_device_struct( card );

    //Free irq
    linux_unrequest_irq( card, pDev );

    //Unmap memory
    linux_unmap_memory( card, pDev );

    free_midlayer( card );

    mCards--;

    DBG_INIT("%s: pci_remove succeed, type:0x%x card:%d\n", DVS_DRIVERNAME, pDev -> device, card );
  } else {
    printk("%s: pci_remove failed, type:0x%x card:%d\n", DVS_DRIVERNAME, pDev -> device, card );
  }

  // Remove proc
  if( mCards == 0 ) {
    dvs_procfs_unregister();
  }
}


/**
//  \ingroup linux
//
//  proc-fs functions
*/
donttag int dvs_debug_read_proc(char *buf, char **start, off_t offset, int len, int *eof, void *data)
{
  int limit;
  int bytes;

  if (len > PAGE_SIZE) {
    limit = PAGE_SIZE;
  } else {
    limit = len;
  }

  len = 0;

  bytes = linuxmid_debug_read(0, buf + len, limit, TRUE);
  len += bytes;
  limit -= bytes;

  if (limit > 0) {
    *eof = 1;
  }

  *start = buf;

  return len;
}

#ifdef _DEBUG
/**
//  \ingroup linux
*/
donttag int dvs_debug_write_proc(struct file *file, const char *buf, unsigned long len, void *data)
{
  int minor = MINOR(file->f_dentry->d_inode->i_rdev);
  int card  = MINOR_TO_INTERNAL( minor );
  char word[256];

  if (len > 256) {
    len = 256;
  }

  hal_memcpy(global_pps[0], word, buf, len, HAL_MEMCPY_U2K);
  word[len] = '\0';             /* len is without null-byte */
  if (word[len-1] == '\n') {    /* skip possible linefeed */
    word[len-1] = '\0';
  }

  linuxmid_setdebug(card, word);

  return len;
}
#endif  /* _DEBUG */



/**
//  \ingroup linux
*/
donttag int dvs_config_read_proc(char *buf, char **start, off_t offset, int len, int *eof, void *data)
{
  if (offset == 0) {
    linuxmid_debug_psconf(0, "config");
  }

  return dvs_debug_read_proc(buf, start, offset, len, eof, data);
}


/**
//  \ingroup linux
*/
donttag int dvs_driverinfo_read_proc(char *buf, char **start, off_t offset, int len, int *eof, void *data)
{
  if (offset == 0) {
    linuxmid_debug_psconf(0, "driverinfo");
  }

  return dvs_debug_read_proc(buf, start, offset, len, eof, data);
}


/**
//  \ingroup linux
*/
donttag int dvs_hwinfo_read_proc(char *buf, char **start, off_t offset, int len, int *eof, void *data)
{
  if (offset == 0) {
    linuxmid_debug_psconf(0, "hwinfo");
  }

  return dvs_debug_read_proc(buf, start, offset, len, eof, data);
}


/**
//  \ingroup linux
*/
donttag int dvs_irqinfo_read_proc(char *buf, char **start, off_t offset, int len, int *eof, void *data)
{
  if (offset == 0) {
    linuxmid_debug_psconf(0, "irqinfo");
  }

  return dvs_debug_read_proc(buf, start, offset, len, eof, data);
}


/**
//  \ingroup linux
*/
donttag int dvs_pciinfo_read_proc(char *buf, char **start, off_t offset, int len, int *eof, void *data)
{
  if (offset == 0) {
    linuxmid_debug_psconf(0, "pciinfo");
  }

  return dvs_debug_read_proc(buf, start, offset, len, eof, data);
}


/**
//  \ingroup linux
*/
donttag int dvs_setup_read_proc(char *buf, char **start, off_t offset, int len, int *eof, void *data)
{
  if (offset == 0) {
    linuxmid_debug_psconf(0, "setup");
  }

  return dvs_debug_read_proc(buf, start, offset, len, eof, data);
}


/**
//  \ingroup linux
*/
donttag int dvs_version_read_proc(char *buf, char **start, off_t offset, int len, int *eof, void *data)
{
  if (offset == 0) {
    linuxmid_debug_psconf(0, "version");
  }

  return dvs_debug_read_proc(buf, start, offset, len, eof, data);
}


/**
//  \ingroup linux
*/
donttag int dvs_cards_read_proc(char *buf, char **start, off_t offset, int len, int *eof, void *data)
{
  int card = 0;
  len = 0;
	
  for(card = 0; card < DVS_MAXDEVICES; card++){
    if((dvs_device[card].mask & DVSDEVICE_EXISTS) == DVSDEVICE_EXISTS) {
      len += sprintf( buf + len, dvs_device[card].name);
      len += sprintf( buf + len, " %d %d\n", dvs_device[card].major, dvs_device[card].minor);
    }
  }
  *eof =1;

  return len;
}


/**
//  \ingroup linux
*/
INTERNAL donttag void dvs_procfs_register()
{
  struct proc_dir_entry *dvs_dir;
  struct proc_dir_entry *dvs_entry;

  if( !mProc )
  {
    dvs_dir = create_proc_entry( DVS_DRIVERNAME, S_IFDIR, NULL);

    dvs_entry = create_proc_entry("config", S_IFREG | S_IRUSR, dvs_dir);
    dvs_entry->read_proc = &dvs_config_read_proc;

#ifdef _DEBUG
    dvs_entry = create_proc_entry("debug", S_IFREG | S_IRUSR, dvs_dir);
    dvs_entry->read_proc = &dvs_debug_read_proc;
    dvs_entry->write_proc = &dvs_debug_write_proc;
#endif  /* _DEBUG */

    dvs_entry = create_proc_entry("driverinfo", S_IFREG | S_IRUSR, dvs_dir);
    dvs_entry->read_proc = &dvs_driverinfo_read_proc;

    dvs_entry = create_proc_entry("hwinfo", S_IFREG | S_IRUSR, dvs_dir);
    dvs_entry->read_proc = &dvs_hwinfo_read_proc;

    dvs_entry = create_proc_entry("irqinfo", S_IFREG | S_IRUSR, dvs_dir);
    dvs_entry->read_proc = &dvs_irqinfo_read_proc;

    dvs_entry = create_proc_entry("pciinfo", S_IFREG | S_IRUSR, dvs_dir);
    dvs_entry->read_proc = &dvs_pciinfo_read_proc;

    dvs_entry = create_proc_entry("setup", S_IFREG | S_IRUSR, dvs_dir);
    dvs_entry->read_proc = &dvs_setup_read_proc;

    dvs_entry = create_proc_entry("version", S_IFREG | S_IRUSR, dvs_dir);
    dvs_entry->read_proc = &dvs_version_read_proc;

    dvs_entry = create_proc_entry("cards", S_IFREG | S_IRUSR, dvs_dir);
    dvs_entry->read_proc = &dvs_cards_read_proc;
    
    mProc = 1;
  }
}  


/**
//  \ingroup linux
*/
INTERNAL donttag void dvs_procfs_unregister()
{
  if( mProc )
  {
    remove_proc_entry( DVS_DRIVERNAME "/version", NULL);
    remove_proc_entry( DVS_DRIVERNAME "/setup", NULL);
    remove_proc_entry( DVS_DRIVERNAME "/pciinfo", NULL);
    remove_proc_entry( DVS_DRIVERNAME "/irqinfo", NULL);
    remove_proc_entry( DVS_DRIVERNAME "/hwinfo", NULL);
    remove_proc_entry( DVS_DRIVERNAME "/driverinfo", NULL);
    remove_proc_entry( DVS_DRIVERNAME "/cards", NULL);
#ifdef _DEBUG
    remove_proc_entry( DVS_DRIVERNAME "/debug", NULL);
#endif  /* _DEBUG */
    remove_proc_entry( DVS_DRIVERNAME "/config", NULL);
    remove_proc_entry( DVS_DRIVERNAME, NULL);

    mProc = 0;
  }
}

