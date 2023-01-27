include make.defs.oneapi

ifdef VERBOSE
    CPPFLAGS += -DVERBOSE
endif

SYCLFLAGS = $(SYCLFLAG)

.PHONY: all

all: sycl_ludecomp main 

sycl: sycl_ludecomp.cpp 
	$(SYCLCXX) $(CPPFLAGS) $(SYCLFLAGS) $< -o $@
