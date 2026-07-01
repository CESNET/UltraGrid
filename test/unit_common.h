#ifndef TEST_UNIT_COMMON_H_7A471D89_C7E4_470A_A330_74F4BD85BBAC
#define TEST_UNIT_COMMON_H_7A471D89_C7E4_470A_A330_74F4BD85BBAC

#ifdef __cplusplus
#include <cinttypes>
#include <cstdio>
#include <iostream>
#include <string>
#else
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#endif

#define ASSERT(expr)                                                           \
        if (!(expr)) {                                                         \
                (void) fprintf(stderr,                                         \
                               "%s:%d: %s: Assertion " #expr " failed!\n",     \
                               __FILE__, __LINE__, __func__);                  \
                return -1;                                                     \
        }

#ifdef __cplusplus
#define ASSERT_MESSAGE(msg, expr) \
        if (!(expr)) { \
                std::cerr << "Assertion " << #expr << " failed: " << (msg) \
                          << "\n"; \
                return -1; \
        }
#else
#define ASSERT_MESSAGE(msg, expr) \
        if (!(expr)) { \
                (void) fprintf(stderr, "Assertion " #expr " failed: %s\n", (msg)); \
                return -1; \
        }
#endif

#ifdef __cplusplus
#define ASSERT_EQUAL(expected, actual) \
        if ((expected) != (actual)) { \
                std::cerr << "Assertion failed - expected " << (expected) \
                          << ", actual : " << (actual) << "\n"; \
                return -1; \
        }
/// compares cstr or std::string value
#define ASSERT_EQUAL_STR(expected, actual) \
        if (std::string(expected) != std::string(actual)) { \
                std::cerr << "Assertion failed - expected \"" << (expected) \
                          << "\", actual : \"" << (actual) << "\"\n"; \
                return -1; \
        }
#else
#define ASSERT_EQUAL(expected, actual) \
        if ((expected) != (actual)) { \
                (void) fprintf(stderr, \
                        "%s:%d: %s: Assertion failed - expected %" PRIdMAX \
                        ", actual %" PRIdMAX "\n", \
                        __FILE__, __LINE__, __func__, \
                        (intmax_t) (expected), (intmax_t) (actual)); \
                return -1; \
        }
/// compares cstr value
#define ASSERT_EQUAL_STR(expected, actual) \
        if (strcmp((expected), (actual)) != 0) { \
                (void) fprintf(stderr, \
                        "Assertion failed - expected \"%s\"" \
                        ", actual \"%s\"\n", \
                        (expected), (actual)); \
                return -1; \
        }
#endif

#define ASSERT_OP(op1, op, op2)                                                \
        if (!((op1) op (op2))) {                                               \
                (void) fprintf(                                                \
                    stderr,                                                    \
                    "%s:%d: %s: Assertion '%s %s %s' failed: %" PRIdMAX        \
                    " %s %" PRIdMAX "\n",                                      \
                    __FILE__, __LINE__, __func__, #op1, #op, #op2,             \
                    (intmax_t) (op1), #op, (intmax_t) (op2));                  \
                return -1;                                                     \
        }

#ifdef __cplusplus
#define ASSERT_EQUAL_MESSAGE(msg, expected, actual) \
        if ((expected) != (actual)) { \
                std::cerr << "Assertion failed - expected " << (expected) \
                          << ", actual " << (actual) << ": " << (msg) << "\n"; \
                return -1; \
        }
#else
#define ASSERT_EQUAL_MESSAGE(msg, expected, actual) \
        if ((expected) != (actual)) { \
                (void) fprintf(stderr, \
                        "Assertion failed - expected %" PRIdMAX " (%#" PRIxMAX \
                        "), actual %" PRIdMAX " (%#" PRIxMAX "): %s\n", \
                        (intmax_t) (expected), (uintmax_t) (expected), \
                        (intmax_t) (actual), (uintmax_t) (actual), (msg)); \
                return -1; \
        }
#endif

#define ASSERT_GE_MESSAGE(msg, expected, actual) \
        if ((actual) < (expected)) { \
                (void) fprintf(stderr, \
                        "Assertion failed - expected >=%" PRIdMAX \
                        ", got %" PRIdMAX ": %s\n", \
                        (intmax_t) (expected), (intmax_t) (actual), (msg)); \
                return -1; \
        }

#define ASSERT_LE_MESSAGE(msg, expected, actual) \
        if ((actual) > (expected)) { \
                (void) fprintf(stderr, \
                        "Assertion failed - expected >=%" PRIdMAX \
                        ", got %" PRIdMAX ": %s\n", \
                        (intmax_t) (expected), (intmax_t) (actual), (msg)); \
                return -1; \
        }

#endif // defined TEST_UNIT_COMMON_H_7A471D89_C7E4_470A_A330_74F4BD85BBAC
