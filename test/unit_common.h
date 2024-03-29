#include <iostream>

#define ASSERT(expr) if (!(expr)) { \
        std::cerr << "Assertion " << #expr << " failed!" << "\n"; \
        return -1; \
}

#define ASSERT_MESSAGE(msg, expr) if (!(expr)) { \
        std::cerr << "Assertion " << #expr << " failed: " << (msg) << "\n"; \
        return -1; \
}

#define ASSERT_EQUAL(expected, actual) if ((expected) != (actual)) { \
        std::cerr << "Assertion failed - expected " << (expected) << ", actual : " << (actual) << "\n"; \
        return -1; \
}

#define ASSERT_EQUAL_MESSAGE(msg, expected, actual) if ((expected) != (actual)) { \
        std::cerr << "Assertion failed - expected " << (expected) << ", actual : " << (actual) << ": " << (msg) << "\n"; \
        return -1; \
}

