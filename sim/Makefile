# Builds the LiteOff OMNeT++ project as a shared library (src/libliteoff.so) linked against Veins.
VEINS ?= $(HOME)/sim/veins-veins-5.3.1
MODE ?= release
JOBS ?= $(shell nproc)

# clang may select a GCC installation without libstdc++ headers (e.g. gcc-16 runtime only).
# Detect that case and point clang to the newest GCC that has them.
CXXPROBE := $(shell echo '\#include <deque>' | $${CXX:-clang++} -x c++ -fsyntax-only - >/dev/null 2>&1 && echo ok)
ifneq ($(CXXPROBE),ok)
GCCDIR := $(shell for d in $$(ls -d /usr/lib/gcc/x86_64-linux-gnu/* 2>/dev/null | sort -V -r); do v=$$(basename $$d); [ -d /usr/include/c++/$$v ] && [ -d /usr/include/x86_64-linux-gnu/c++/$$v ] && echo $$d && break; done)
ifneq ($(GCCDIR),)
export CFLAGS_EXTRA += --gcc-install-dir=$(GCCDIR)
export LDFLAGS_EXTRA += --gcc-install-dir=$(GCCDIR)
endif
endif

all: src/Makefile
	$(MAKE) -C src MODE=$(MODE) -j$(JOBS)

src/Makefile: Makefile
	cd src && opp_makemake -f --deep --make-so -o liteoff -O out \
	    -I$(VEINS)/src -L$(VEINS)/src -lveins -DVEINS_IMPORT

test:
	$(MAKE) -C tests CXX="$${CXX:-g++}"

info:
	@echo "CXXPROBE=$(CXXPROBE) GCCDIR=$(GCCDIR) CFLAGS_EXTRA=$(CFLAGS_EXTRA)"

clean:
	-$(MAKE) -C src clean
	rm -f src/Makefile
	$(MAKE) -C tests clean

.PHONY: all test clean
