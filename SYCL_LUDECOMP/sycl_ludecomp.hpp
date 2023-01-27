#if !defined(_SYCL_LUDECOMP)
#define _SYCL_LUDECOMP
#include "decompose.hpp"
#include "backsub.hpp"
#include "assamble.hpp"
namespace SYCL_ENABLE{
    namespace GaussLU{

        void solve(sycl::queue Q, double *matIn,int rows, int cols);
        void getXvec(sycl::queue Q, double *matIn,double *xvec,int rows, int cols);
        void buildLinearSystem(sycl::queue Q, double *matIn,double *matTot,double *sourceVec,int rows, int cols); 


    }
}


#endif // _SYCL_LUDECOMP
