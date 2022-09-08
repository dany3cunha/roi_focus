# roi_focus
Dynamic focus using the setting of AEC_AGC_ROI by Stereolabs for ZED2 camera 

### Prerequisites

- Ubuntu 18.04
- [ZED SDK **≥ 3.7**](https://www.stereolabs.com/developers/) and its dependency [CUDA](https://developer.nvidia.com/cuda-downloads)

#### Instalation

Open a terminal, clone the repository, update the dependencies and build the packages:

    $ git clone https://github.com/dany3cunha/roi_focus
    $ cd roi_focus
    $ mkdir build
    $ cd build
    $ cmake ../
    $ make

### Run the Application

To run the application:

    $ ./squareFocus