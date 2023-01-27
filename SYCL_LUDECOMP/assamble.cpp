#include "assamble.hpp"


assamble::assamble(syclAccR matIn, syclAccW matOut, syclAccR _bvec, int _rows, int _cols)
: matA{matIn}, matB{matOut},bvec{_bvec}, rows{_rows}, cols{_cols+1}{

}

void assamble::operator()(sycl::nd_item<2> it)const {

    int i = it.get_global_id(0);
    int j = it.get_global_id(1);

    //FILL THE MATRIX
    matB[i*cols + j] = matA[i*rows + j];
    matB[i*cols + rows] = bvec[i];

}