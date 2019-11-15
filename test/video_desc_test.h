#ifndef VIDEO_DESC_TEST_H
#define VIDEO_DESC_TEST_H

#include <cppunit/extensions/HelperMacros.h>
#include <list>

#include "types.h"

class video_desc_test : public CPPUNIT_NS::TestFixture
{
  CPPUNIT_TEST_SUITE( video_desc_test );
  CPPUNIT_TEST( testIOOperatorSymetry );
  CPPUNIT_TEST_SUITE_END();

public:
  video_desc_test();
  ~video_desc_test();
  void setUp();
  void tearDown();

  void testIOOperatorSymetry();
private:
  const std::list<video_desc> m_test_desc; // tested desc
};

#endif //  VIDEO_DESC_TEST_H
