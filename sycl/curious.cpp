#include <CL/sycl.hpp>
#include <iostream>


using namespace sycl;
int main()
{
	for(auto const& this_platform : platform::get_platforms() ){
		std::cout<<"Found platform: "
			<<this_platform.get_info<info::platform::name>()<<"\n";
		for(auto const& this_device : this_platform.get_devices() ){
		
			std::cout<<" Device: "
				<<this_device.get_info<info::device::name>()<<"\n";
		
		}
		std::cout<<"\n";

	}

	return 0;
		
}

