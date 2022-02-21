#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#ifdef HAVE_CPPUNIT

#include <cppunit/config/SourcePrefix.h>
#include "misc_test.hpp"
#include "utils/misc.h"

using std::string;
using std::to_string;

// Registers the fixture into the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( misc_test );

misc_test::misc_test()
{
}

misc_test::~misc_test()
{
}

void
misc_test::setUp()
{
}


void
misc_test::tearDown()
{
}

void
misc_test::test_replace_all()
{
        char test[][20] =         { DELDEL DELDEL DELDEL, DELDEL DELDEL,               "XYZX" DELDEL, "XXXyX" };
        const char *repl_from[] = { DELDEL              , DELDEL,                      "X",           "X" };
        const char *repl_to[] =   { ":"                 , ESCAPED_COLON,               "x",           "" };
        const char *res[] =       { ":::"               , ESCAPED_COLON ESCAPED_COLON, "xYZx" DELDEL, "y" };
        for (unsigned i = 0; i < sizeof test / sizeof test[0]; ++i) {
                replace_all(test[i], repl_from[i], repl_to[i]);
                CPPUNIT_ASSERT(strcmp(test[i], res[i]) == 0);
        }
}

#endif // defined HAVE_CPPUNIT
