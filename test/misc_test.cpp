#include <climits>
#include <cmath>           // for abs
#include <cstring>         // for strcmp
#include <list>
#include <sstream>
#include <string>          // for allocator, basic_string, operator+, string

#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <cstdlib>
#include <io.h>
#else
#include <unistd.h>
#endif

#include "color_space.h"
#include "types.h"
#include "utils/fs.h"      // for NULL_FILE
#include "utils/misc.h"
#include "utils/net.h"
#include "utils/string.h"
#include "unit_common.h"
#include "video.h"
#include "video_frame.h"

extern "C" {
int misc_test_color_coeff_range();
int misc_test_net_getsockaddr();
int misc_test_net_sockaddr_compare_v4_mapped();
int misc_test_replace_all();
int misc_test_video_desc_io_op_symmetry();
int misc_test_unit_evaluate();
}

using namespace std;

/**
 * check that scaled coefficient for minimal values match approximately minimal
 * value of nominal range (== there is not significant shift)
 */
int
misc_test_color_coeff_range()
{
        const int depths[] = { 8, 10, 12, 16 };

        for (unsigned i = 0; i < sizeof depths / sizeof depths[0]; ++i) {
                const int d        = depths[i];
                const int d_max    = (1 << d) - 1;
                const int max_diff = 1 << (d - 8);
                const struct color_coeffs *cfs = get_color_coeffs(CS_DFL, d);

                // Y
                ASSERT_LE_MESSAGE(
                    "min Y diverges from nominal range min", max_diff,
                    abs((RGB_TO_Y(*cfs, 0, 0, 0) >> COMP_BASE) + LIMIT_LO(d)) -
                        LIMIT_LO(d));
                ASSERT_LE_MESSAGE(
                    "max Y diverges from nominal range max", max_diff,
                    abs((RGB_TO_Y(*cfs, d_max, d_max, d_max) >> COMP_BASE) +
                        LIMIT_LO(d) - LIMIT_HI_Y(d)));
                // Cb
                ASSERT_LE_MESSAGE(
                    "min Cb diverges from nominal range min", max_diff,
                    abs((RGB_TO_CB(*cfs, d_max, d_max, 0) >> COMP_BASE) +
                        (1 << (d - 1)) - LIMIT_LO(d)));
                ASSERT_LE_MESSAGE(
                    "max Cb diverges from nominal range max", max_diff,
                    abs((RGB_TO_CB(*cfs, 0, 0, d_max) >> COMP_BASE) +
                        (1 << (d - 1)) - LIMIT_HI_CBCR(d)));
                // Cr
                ASSERT_LE_MESSAGE(
                    "min Cr diverges from nominal range min", max_diff,
                    abs((RGB_TO_CR(*cfs, 0, d_max, d_max) >> COMP_BASE) +
                        (1 << (d - 1)) - LIMIT_LO(d)));
                ASSERT_LE_MESSAGE(
                    "max Cr diverges from nominal range max", max_diff,
                    abs((RGB_TO_CR(*cfs, d_max, 0, 0) >> COMP_BASE) +
                        (1 << (d - 1)) - LIMIT_HI_CBCR(d)));
        }

        return 0;
}

int
misc_test_net_getsockaddr()
{
        struct {
                const char *str;
                int ip_mode;
        } test_cases[] = {
                { "10.0.0.1:10",          4 },
                { "[100::1:abcd]:10",     6 },
#if defined   __APPLE__
                { "[fe80::123%lo0]:1000", 6 },
#elif defined __linux__
                { "[fe80::123%lo]:1000",  6 },
#elif defined _WIN32
                { "[fe80::123%1]:1000",   6 },
#endif
        };
        for (unsigned i = 0; i < sizeof test_cases / sizeof test_cases[0];
             ++i) {
                struct sockaddr_storage ss =
                    get_sockaddr(test_cases[i].str, test_cases[i].ip_mode);
                char buf[ADDR_STR_BUF_LEN] = "";
                get_sockaddr_str((struct sockaddr *) &ss, sizeof ss, buf,
                                 sizeof buf);
                char msg[2048];
                snprintf(msg, sizeof msg,
                         "get_sockaddr_str conversion of '%s' failed",
                         test_cases[i].str);
                ASSERT_MESSAGE(msg, ss.ss_family != AF_UNSPEC);
                snprintf(
                    msg, sizeof msg,
                    "sockaddr output string '%s' doesn't match the input '%s'",
                    buf, test_cases[i].str);

                ASSERT_MESSAGE(msg, strcmp(test_cases[i].str, buf) == 0);
        }

        return 0;
}

int misc_test_net_sockaddr_compare_v4_mapped() {
        struct sockaddr_storage v4 = get_sockaddr("10.0.0.1:10", 4);
        struct sockaddr_storage v4mapped = get_sockaddr("10.0.0.1:10", 0);
        ASSERT_MESSAGE("V4 mapped address doesn't equal plain V4",
                       sockaddr_compare((struct sockaddr *) &v4,
                                        (struct sockaddr *) &v4mapped) == 0);
        return 0;
}

#ifdef __clang__
#pragma clang diagnostic ignored "-Wstring-concatenation"
#endif
int misc_test_replace_all()
{
        char test[][20] =         { DELDEL DELDEL DELDEL, DELDEL DELDEL,               "XYZX" DELDEL, "XXXyX" };
        const char *repl_from[] = { DELDEL              , DELDEL,                      "X",           "X" };
        const char *repl_to[] =   { ":"                 , ESCAPED_COLON,               "x",           "" };
        const char *res[] =       { ":::"               , ESCAPED_COLON ESCAPED_COLON, "xYZx" DELDEL, "y" };
        for (unsigned i = 0; i < sizeof test / sizeof test[0]; ++i) {
                replace_all(test[i], repl_from[i], repl_to[i]);
                ASSERT(strcmp(test[i], res[i]) == 0);
        }
        return 0;
}

int misc_test_unit_evaluate()
{
        struct {
                const char *str;
                long long exp_val;
                const char *exp_endstr; // value, not ptr
        } test_cases[] = {
                { "1",     1,                 ""   },
                { "1M",    1000 * 1000,       ""   },
                { "1.2M",  1200 * 1000,       ""   },
                { "0.5Gi", 512 * 1024 * 1024, ""   },
                // partially parsed
                { "1x",    1,                 "x"  },
                // errors
                { "x",     LLONG_MIN,         "x"  },
                { "Gi",    LLONG_MIN,         "Gi" },
        };

        // temporarily redirect output to null file
        int stdout_save = dup(STDOUT_FILENO);
        int null_fd = open(NULL_FILE, O_WRONLY);
        dup2(null_fd, STDOUT_FILENO);
        close(null_fd);

        for (unsigned i = 0; i < sizeof test_cases / sizeof test_cases[0];
             ++i) {
              const char *endptr = nullptr;
              long long val = unit_evaluate(test_cases[i].str, &endptr);
              ASSERT_EQUAL(test_cases[i].exp_val, val);
              ASSERT_EQUAL_STR(test_cases[i].exp_endstr, endptr);
        }
        // restore output
        dup2(stdout_save, STDOUT_FILENO);
        close(stdout_save);

        return 0;
}

int misc_test_video_desc_io_op_symmetry()
{
        const std::list<video_desc> test_desc = {
                {1920, 1080, DXT5, 60, PROGRESSIVE, 4},
                {640, 480, H264, 15, PROGRESSIVE, 1}};
        for (const auto & i : test_desc) {
                video_desc tmp;

                ostringstream oss;
                oss << i;
                istringstream iss(oss.str());
                iss >> tmp;

                // Check
                ostringstream oss2;
                oss2 << tmp;
                string err_elem = oss.str() + " vs " +  oss2.str();
                ASSERT_MESSAGE(err_elem, video_desc_eq(tmp, i));
                ASSERT_EQUAL_MESSAGE(err_elem, tmp, i);
        }
        return 0;
}
