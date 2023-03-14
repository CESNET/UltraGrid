#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <list>
#include <sstream>

#include "types.h"
#include "utils/string.h"
#include "unit_common.h"
#include "video.h"
#include "video_frame.h"

extern "C" {
        int misc_test_replace_all();
        int misc_test_video_desc_io_op_symmetry();
}

using namespace std;

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
