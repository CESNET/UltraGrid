#ifndef GPUJPEG_TEST_H_96F45515_1EFA_4C80_A802_D520ECE5A079
#define GPUJPEG_TEST_H_96F45515_1EFA_4C80_A802_D520ECE5A079

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#ifdef HAVE_GPUJPEG
#include <cppunit/extensions/HelperMacros.h>
#include <list>

#include "utils/misc.h"

struct compress_state;
struct state_decompress;

class gpujpeg_test : public CPPUNIT_NS::TestFixture
{
  CPPUNIT_TEST_SUITE( gpujpeg_test );
  CPPUNIT_TEST( test_simple );
  CPPUNIT_TEST_SUITE_END();

public:
  gpujpeg_test() = default;
  ~gpujpeg_test() = default;
  void setUp();
  void tearDown();

  void test_simple();

private:
  bool m_skip{false};
  compress_state *m_compress{nullptr};
  state_decompress *m_decompress{nullptr};
};

#endif // defined HAVE_GPUJPEG
#endif // defined GPUJPEG_TEST_H_96F45515_1EFA_4C80_A802_D520ECE5A079
