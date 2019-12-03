TRACE_DIR = $(shell git rev-parse --show-toplevel)/data/cbp2004/outputs
TRACE_LIST = $(shell find $(TRACE_DIR) -name *.bt)

all:
	python bpred.py $(TRACE_LIST)
