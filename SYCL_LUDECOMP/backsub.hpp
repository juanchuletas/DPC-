#if !defined(_SYCL_BACK_SUB_H_)
#define _SYCL_BACK_SUB_H_
#include "sycl_header.hpp"
class backsub{


    syclAccRW xvec;
    syclAccR mat;
    syclAccDW store;
    int rows, cols, k; 

    public:
        SYCL_EXTERNAL backsub(int k, syclAccRW vecAcc,syclAccDW auxAcc, syclAccR matAcc, int _ros, int _cols);
        SYCL_EXTERNAL void operator()(cl::sycl::item<1> it)const;


};

#endif // _SYCL_BACK_SUB_H_
