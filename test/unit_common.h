#ifndef TEST_UNIT_COMMON_H_7A471D89_C7E4_470A_A330_74F4BD85BBAC
#define TEST_UNIT_COMMON_H_7A471D89_C7E4_470A_A330_74F4BD85BBAC

#ifdef __cplusplus
#include <cinttypes>
#include <cstdio>
#include <iostream>
#else
#include <inttypes.h>
#include <stdio.h>
#endif

#define ASSERT(expr) \
        if (!(expr)) { \
                fprintf(stderr, "Assertion " #expr " failed!\n"); \
                return -1; \
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
                fprintf(stderr, "Assertion " #expr " failed: %s\n", (msg)); \
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
#else
#define ASSERT_EQUAL(expected, actual) \
        if ((expected) != (actual)) { \
                fprintf(stderr, \
                        "Assertion failed - expected %" PRIdMAX \
                        ", actual %" PRIdMAX "\n", \
                        (intmax_t) (expected), (intmax_t) (actulal)); \
                return -1; \
        }
#endif

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
                fprintf(stderr, \
                        "Assertion failed - expected %" PRIdMAX \
                        ", actual %" PRIdMAX ": %s\n", \
                        (intmax_t) (expected), (intmax_t) (actual), (msg)); \
                return -1; \
        }
#endif

#endif // defined TEST_UNIT_COMMON_H_7A471D89_C7E4_470A_A330_74F4BD85BBAC
