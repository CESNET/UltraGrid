#ifndef DEBUG_HPP_db8549e3cb58
#define DEBUG_HPP_db8549e3cb58

#include <cassert>

#ifdef DEBUG
#	define DEBUG_ASSERT(x) assert(x)
#else
#	define DEBUG_ASSERT(x) do{  }while(0)
#endif

#define DEBUG_FAIL(x) DEBUG_ASSERT(x && false)

#endif //DEBUG_HPP_db8549e3cb58
