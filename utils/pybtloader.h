// Bandhav Veluri
// 11/19/2019

#ifndef PYBTLOADER_H
#define PYBTLOADER_H

#include <inttypes.h>
#include "tread.h"

namespace pybtloader {

  typedef struct item_t {
    uint32_t pc;
    bool is_cond; /* is condition branch? */
  };

  /* CBP 2004 branch trace loader class */
  class cbp2004 {
    cbp_trace_reader_c cbptr;
    branch_record_c br;

    public:
      item_t get_item();
      bool predict(bool);
  }

} // namespace pybtloader
