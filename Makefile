TRACE_DIR = $(shell git rev-parse --show-toplevel)/data/cbp2004/outputs
TRACE_LIST = $(shell find $(TRACE_DIR) -name *.bt)

all: $(foreach trace, $(TRACE_LIST), $(trace).run)

.PHONY : $(TARCE_LIST)
%.run:
	python bpred.py $*
