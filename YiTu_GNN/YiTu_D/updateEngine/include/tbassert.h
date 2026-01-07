/**
 * Copyright (c) 2014 MIT License by Tao B. Schardl
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 **/

#ifndef _TBASSERT_H_
#define _TBASSERT_H_ 1

#ifndef NDEBUG

/**************************************************************************
 * Library of debugging macros.
 *************************************************************************/

#include <stdio.h>
#include <stdlib.h>

// Print a message to STREAM when DEBUG = 1.  This macro takes the
// same arguments as FPRINTF().
#define DEBUG_FPRINTF(STREAM, ...)                           \
  do {                                                       \
    fprintf(STREAM, "%s:%d (%s) ",                           \
            __FILE__, __LINE__, __PRETTY_FUNCTION__);        \
    fprintf(STREAM, __VA_ARGS__);                            \
  } while (0)

// Print a message to STDERR when DEBUG = 1.  This macro takes the
// same arguments as PRINTF().
#define DEBUG_EPRINTF(...) DEBUG_FPRINTF(stderr, __VA_ARGS__);

// If PREDICATE is true, do nothing.  Otherwise, print an error with
// the specified message to STDERR.  This macro only operates when
// DEBUG = 1.  This macro takes a PREDICATE to evaluate followed by
// the standard arguments to PRINTF().
#define DEBUG_ASSERT(PREDICATE, ...)                                    \
    do {                                                                \
      if (!(PREDICATE)) {                                               \
        fprintf(stderr, "%s:%d (%s) Assertion " #PREDICATE " failed: ", \
                __FILE__, __LINE__, __PRETTY_FUNCTION__);               \
        fprintf(stderr, __VA_ARGS__);                                   \
        abort();                                                        \
      }                                                                 \
    } while (0)

#define tbassert DEBUG_ASSERT

#else

#define DEBUG_PRINTF(...)  // Nothing.
#define DEBUG_EPRINTF(...)  // Nothing.
#define DEBUG_ASSERT(...)  // Nothing.
#define tbassert(...)  // Nothing.

#endif

#endif  // _TBASSERT_H_
