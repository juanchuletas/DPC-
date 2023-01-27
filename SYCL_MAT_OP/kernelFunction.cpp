#include<CL/sycl.hpp>
#include<iostream>
#include<cmath>
using namespace sycl;


class Add{
    
    accessor<int> data_acc;
    public:
        Add(accessor<int> acc) : data_acc{acc}{

        }
        void operator()(id<1> i) const{
            data_acc[i] = data_acc[i] + 1;
        }
};

int main (){
   
    constexpr size_t size = 16;
    int array[16];
    for(int i=0; i<size; i++){
        array[i] = 0;
    }
    {
        auto Q = cl::sycl::queue{cl::sycl::gpu_selector{}};
        std::cout << "Chosen device: "  << Q.get_device().get_info<cl::sycl::info::device::name>()<<std::endl;
        std::cout<< "Max Work Group Size: "<< Q.get_device().get_info<cl::sycl::info::device::max_work_group_size>()<<std::endl;

        buffer<int, 1> buffc{array,cl::sycl::range<1>{static_cast<size_t>(size)}};

        Q.submit([&] (handler &h){

            accessor<int> data_acc{buffc,h};
            h.parallel_for<Add>(size, Add(data_acc));

        });

    }
    for(int i=0; i<size; i++){
        printf("%d\n", array[i]);
    }
    


    return 0;
}