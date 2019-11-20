CBP_DIR = ../cbp2004

CFLAGS = -g -Wall
CXXFLAGS = -g -Wall -I$(CBP_DIR)

vpath %.cc $(CBP_DIR)

.PHONY : help
help:
	$(info Usage: make <trace_path>/<trace_filename_without_extension>.data)
	$(info Above command outputs <trace_filename_without_extension>)

objects = trace_reader.o cbp_inst.o op_state.o predictor.o tread.o

%.data: %.bz2 trace_reader
	./trace_reader $* $(notdir $@)

trace_reader : $(objects)
	$(CXX) -o $@ $^

%.o: %.cc
	$(CXX) -MMD -o $@ -c $< $(CXXFLAGS)

%.o: %.c
	$(CC) -MMD -o $@ -c $< $(CFLAGS)

.PHONY : clean
clean :
	rm -f trace_reader $(objects) $(subst .o,.d,$(objects)) *.data
