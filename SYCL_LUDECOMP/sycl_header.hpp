#if !defined(_SYCL_HEADER_DEFS_LU)
#define _SYCL_HEADER_DEFS_LU
#include<CL/sycl.hpp>
#include<CL/sycl/atomic.hpp>
#include<iostream>
#include<cmath>
using namespace sycl; 

typedef sycl::accessor<double,1,cl::sycl::access::mode::read_write,cl::sycl::access::target::global_buffer> syclAccRW;
typedef sycl::accessor<double,1,cl::sycl::access::mode::read,cl::sycl::access::target::global_buffer> syclAccR;
typedef sycl::accessor<double,1,cl::sycl::access::mode::write,cl::sycl::access::target::global_buffer> syclAccW;
typedef sycl::accessor<double,1,cl::sycl::access::mode::atomic,cl::sycl::access::target::global_buffer> syclAccAtomic;


/*The arguments and access::mode::discard_write for the result. discard_write can be used whenever we write to the whole buffer and do not care about its previous contents. 
Since it will be overwritten entirely, we can discard whatever was there before.*/
typedef sycl::accessor<double,1,cl::sycl::access::mode::discard_write,cl::sycl::access::target::global_buffer> syclAccDW;





#endif // _SYCL_HEADER_DEFS_LU
