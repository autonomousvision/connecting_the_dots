#ifndef COMMON_H
#define COMMON_H

#include <cmath>
#include <algorithm>

#if defined(_OPENMP)
#include <omp.h>
#endif


#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&) = delete;\
  classname& operator=(const classname&) = delete;


#endif
