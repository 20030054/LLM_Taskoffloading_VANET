CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
all: run
test_core: test_core.cc ../src/core/Core.cc ../src/core/Core.h
	$(CXX) $(CXXFLAGS) -o $@ test_core.cc ../src/core/Core.cc
run: test_core
	./test_core
clean:
	rm -f test_core
.PHONY: all run clean
