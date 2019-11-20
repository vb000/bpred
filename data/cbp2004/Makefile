CBP_DIR = $(shell git rev-parse --show-toplevel)/cbp2004
TRACE_DIR = $(CBP_DIR)/traces/without-values
TRACE_LIST = $(shell find $(TRACE_DIR) -name '*.bz2')
OUTPUTS = $(foreach trace, $(TRACE_LIST), outputs/$(notdir $(trace:.bz2=.bt)))

CFLAGS = -g -Wall
CXXFLAGS = -g -Wall -I$(CBP_DIR)

vpath %.cc $(CBP_DIR)

.PHONY : all
all: $(OUTPUTS)

objects = trace_reader.o cbp_inst.o op_state.o predictor.o tread.o

outputs/%.bt: $(TRACE_DIR)/%.bz2 trace_reader
	./trace_reader $(TRACE_DIR)/$* $@ > /dev/null

trace_reader : $(objects)
	$(CXX) -o $@ $^

%.o: %.cc
	$(CXX) -MMD -o $@ -c $< $(CXXFLAGS)

%.o: %.c
	$(CC) -MMD -o $@ -c $< $(CFLAGS)

.PHONY : clean
clean :
	rm -f trace_reader $(objects) $(subst .o,.d,$(objects)) outputs/*.bt
