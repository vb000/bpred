// Usage: trace_reader <trace_filename_without_extension> <output_filename>

#include <cstdio>
#include <cstdlib>
#include "tread.h"

// include and define the predictor
#include "predictor.h"
PREDICTOR predictor;

using namespace std;

int main(int argc, char* argv[]) {
  if (argc != 3) {
      printf("usage: %s <trace_filename_without_extension> <output_filename>\n", argv[0]);
      exit(EXIT_FAILURE);
  }

  cbp_trace_reader_c cbptr = cbp_trace_reader_c(argv[1]);
  branch_record_c br;

  /* open a text file for writing */
  FILE* trace_out = fopen(argv[2], "w");

  // read the trace, one branch at a time, placing the branch info in br
  while (cbptr.get_branch_record(&br)) {
    // get_prediction() returns the prediction your predictor would like to make
    bool predicted_taken = predictor.get_prediction(&br, cbptr.osptr);

    // predict_branch() tells the trace reader how you have predicted the branch
    bool actual_taken    = cbptr.predict_branch(predicted_taken);
        
    if(br.is_conditional) {
      uint total_insts = cbptr.get_num_insts();
      fprintf(trace_out, "%d %d %d\n", br.instruction_addr, actual_taken, total_insts);
    }

    // finally, update_predictor() is used to update your predictor with the
    // correct branch result
    predictor.update_predictor(&br, cbptr.osptr, actual_taken);
  }
}
