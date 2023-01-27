#if !defined(_DECOMPOSE_H_)
#define _DECOMPOSE_H_
#include "sycl_header.hpp"
class decompose{

    syclAccRW mat;
    int rows, cols;
    int k;  

    public:
        SYCL_EXTERNAL decompose(int index, syclAccRW inputAcc, int _rows, int _cols);
        SYCL_EXTERNAL void operator()(cl::sycl::nd_item<2> it)const;


};



#endif // _DECOMPOSE_H_
