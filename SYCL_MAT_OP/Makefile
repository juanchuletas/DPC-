include make.defs.oneapi

ifdef VERBOSE
    CPPFLAGS += -DVERBOSE
endif

SYCLFLAGS = $(SYCLFLAG)

.PHONY: all

all: matmul_sycl

matmul_sycl: matmul_sycl.cpp
	$(SYCLCXX) $(CPPFLAGS) $(SYCLFLAGS) $< -o $@
