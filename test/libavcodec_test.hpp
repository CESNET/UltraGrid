#ifndef LIBAVCODEC_TEST_HPP_5452263A_93AB_4E74_8922_F2AB006FC351
#define LIBAVCODEC_TEST_HPP_5452263A_93AB_4E74_8922_F2AB006FC351

#include "config.h"

#ifdef HAVE_LAVC

#include <cppunit/extensions/HelperMacros.h>

class libavcodec_test : public CPPUNIT_NS::TestFixture
{
  CPPUNIT_TEST_SUITE(libavcodec_test);
  CPPUNIT_TEST(test_get_decoder_from_uv_to_uv);
  CPPUNIT_TEST_SUITE_END();

public:
  libavcodec_test() = default;
  ~libavcodec_test() override = default;
  void setUp() override;
  void tearDown() override;

  void test_get_decoder_from_uv_to_uv();
};

#endif // defined HAVE_LAVC

#endif //defined  LIBAVCODEC_TEST_HPP_5452263A_93AB_4E74_8922_F2AB006FC351
