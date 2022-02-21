#ifndef MISC_TEST_HPP_85A3153C_9322_11EC_B6F6_F0DEF1A0ACC9
#define MISC_TEST_HPP_85A3153C_9322_11EC_B6F6_F0DEF1A0ACC9

#include <cppunit/extensions/HelperMacros.h>

class misc_test : public CPPUNIT_NS::TestFixture
{
  CPPUNIT_TEST_SUITE( misc_test );
  CPPUNIT_TEST( test_replace_all );
  CPPUNIT_TEST_SUITE_END();

public:
  misc_test();
  ~misc_test();
  void setUp();
  void tearDown();

  void test_replace_all();
};

#endif // !defined MISC_TEST_HPP_85A3153C_9322_11EC_B6F6_F0DEF1A0ACC9
