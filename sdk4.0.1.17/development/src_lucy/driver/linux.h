#include <linux/version.h>

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,15)
# include <linux/stringify.h>
# undef KBUILD_BASENAME
# undef KBUILD_MODNAME
# define KBUILD_BASENAME __stringify(dvsdriver_lucy)
# define KBUILD_MODNAME __stringify(dvsdriver_lucy)
#else
# define KBUILD_BASENAME dvsdriver_lucy
# define KBUILD_MODNAME dvsdriver_lucy
#endif
