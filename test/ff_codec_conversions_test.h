#ifndef FF_CODEC_CONVERSIONS_TEST_H_277D34B0_7056_45BF_9A47_EA2AD1DEA846
#define FF_CODEC_CONVERSIONS_TEST_H_277D34B0_7056_45BF_9A47_EA2AD1DEA846

#include "config.h"

#ifdef HAVE_LAVC

#include <cppunit/extensions/HelperMacros.h>

class ff_codec_conversions_test : public CPPUNIT_NS::TestFixture
{
  CPPUNIT_TEST_SUITE( ff_codec_conversions_test );
  CPPUNIT_TEST( test_yuv444p16le_from_to_r10k );
  CPPUNIT_TEST( test_yuv444p16le_from_to_r12l );
  CPPUNIT_TEST_SUITE_END();

public:
  ff_codec_conversions_test();
  ~ff_codec_conversions_test();
  void setUp();
  void tearDown();

  void test_yuv444p16le_from_to_r10k();
  void test_yuv444p16le_from_to_r12l();
};

#endif // defined HAVE_LAVC

#endif // defined FF_CODEC_CONVERSIONS_TEST_H_277D34B0_7056_45BF_9A47_EA2AD1DEA846
