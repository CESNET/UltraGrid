#include <linux/autoconf.h>
#include <linux/version.h>

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,15)
# include <linux/stringify.h>
# define KBUILD_BASENAME __stringify(dvsdriver)
# define KBUILD_MODNAME __stringify(dvsdriver)
#else
# define KBUILD_BASENAME dvsdriver
# define KBUILD_MODNAME dvsdriver
#endif
