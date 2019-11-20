// Bandhav Veluri
// 11/19/2019

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "pybtloader.h"

using namespace pybtloader;

cbp2004::cbp2004() {
  cbptr = cbp_trace_reader_c(
