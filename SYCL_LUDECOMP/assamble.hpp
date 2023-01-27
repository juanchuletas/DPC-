#if !defined(_ASSAMBLE_H_)
#define _ASSAMBLE_H_

// SYCL HEADER AND STUFF
#include "sycl_header.hpp"

class assamble{

    int rows, cols;
    syclAccR matA; //syclAccX (X=R,W,RW) is a typedef accessor: see "sycl_header.hpp" for more information
    syclAccW matB;
    syclAccR bvec;

    public:
        SYCL_EXTERNAL assamble(syclAccR matIn, syclAccW matOut, syclAccR bvec, int rows, int cols);
        SYCL_EXTERNAL void operator()(sycl::nd_item<2> it) const;



};



#endif // _ASSAMBLE_H_
