export OMP_NUM_THREADS = 1

TRACE_DIR = $(shell git rev-parse --show-toplevel)/data/cbp2004/outputs
TRACE_LIST = $(shell find $(TRACE_DIR) -name *.bt)

all:
	time python bpred.py $(TRACE_LIST)

rnn:
	time python bpred_rnn.py $(TRACE_LIST)
