#include"decompose.hpp"

//*********** DECOMPOSE CLASS **********************************************
decompose::decompose(int index, syclAccRW inAcc, int _rows, int _cols)
: k{index},mat{inAcc}, rows{_rows}, cols{_cols+1}{

}
void decompose::operator()(cl::sycl::nd_item<2> it)const{
    int i = it.get_global_id(0);
    int j = it.get_global_id(1);
    double l_ik = mat[(i + (k+1))*cols + k]/mat[k*cols + k];;
    mat[(i + (k+1))*cols + (j+k)] = mat[(i + (k+1))*cols + (j+k)] - l_ik*mat[k*cols + (j+k)];

    mat[(i + (k+1))*cols + rows] = mat[(i + (k+1))*cols + rows] - l_ik*mat[k*cols + rows];
}
//*********************************************************************************************