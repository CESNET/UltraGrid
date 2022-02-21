#ifndef GET_FRAMERATE_TEST_HPP
#define GET_FRAMERATE_TEST_HPP

#include <cppunit/extensions/HelperMacros.h>
#include <list>

#include "utils/misc.h"

class get_framerate_test : public CPPUNIT_NS::TestFixture
{
  CPPUNIT_TEST_SUITE( get_framerate_test );
  CPPUNIT_TEST( test_2997 );
  CPPUNIT_TEST( test_3000 );
  CPPUNIT_TEST( test_free );
  CPPUNIT_TEST_SUITE_END();

public:
  get_framerate_test();
  ~get_framerate_test();
  void setUp();
  void tearDown();

  void test_2997();
  void test_3000();
  void test_free();
};

#endif // !defined GET_FRAMERATE_TEST_HPP
