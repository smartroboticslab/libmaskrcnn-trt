# libmaskrcnn-trt

A C++17 library for easily running
[Mask R-CNN](https://arxiv.org/abs/1703.06870v3) using
[NVIDIA TensorRT](https://developer.nvidia.com/tensorrt).


## Build

Requirements:

- An NVIDIA GPU (tested on an NVIDIA Jetson Xavier NX).
- A CUDA installation (tested on CUDA 10.2).
- A TensorRT installation (tested on TensorRT 7.1.3).
- A C++17 compatible compiler (tested on gcc 7.5.0 on Ubuntu 18.04).
- OpenCV 4
- CMake
- Make (optional, for convenience)

To install CUDA and TensorRT follow the
[instructions from NVIDIA](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).
To install the other dependencies on Debian/Ubuntu run

``` sh
sudo apt install g++ make cmake libopencv-dev
```

Then run `make` to build the library and the example executables.


## Usage

To use the library you'll need to first download the Mask R-CNN UFF model
`mrcnn_nchw.uff` by running `make download-model` (requires `wget`). This model
was generated using
[these instructions](https://github.com/NVIDIA/TensorRT/tree/master/samples/opensource/sampleUffMaskRCNN#generating-uff-model).

### Using the library in your code

The recommended way is to incorporate this repository into your project (e.g. as
a git submodule) and then add it as a subdirectory in your CMakeLists.txt, e.g.:

``` cmake
add_subdirectory(libmaskrcnn-trt)

# Build a CMake target with libmaskrcnn-trt.
target_include_directories(PUBLIC some-target libmaskrcnn-trt/include)
target_link_libraries(some-target maskrcnn-trt)

# Make sure to change the path to the libmaskrcnn-trt directory in
# add_subdirectory() and target_include_directories() if needed.
```

### Example executable

An example executable is compiled along with the library. To run inference on
two images you can run

``` sh
./build/release/maskrcnn-trt-example mrcnn_nchw.uff /PATH/TO/IMAGE2 /PATH/TO/IMAGE2
```

Information about the detected objects will be shown on standard output and
visualizations will be created in `/PATH/TO/IMAGE1.detections.png` and
`/PATH/TO/IMAGE2.detections.png`. The images can be in any format supported by
OpenCV's
[`cv::imread()`](https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56).

### Notes

- The first time the Uff model is loaded it will be converted into a
  device-specific format. This process can take a few minutes. To speed-up
  subsequent runs it is possible to serialize the device-specific format to the
  disk by setting `mr::MaskRCNNConfig::serialized_model_filename` to the name of
  a non-existent file. The serialized version will be loaded on subsequent runs
  instead of doing the conversion each time.
- The first inference can take up to 2x more time than subsequent inferences.
- On newer versions of TensorRT some of the functions used in libmaskrcnn-trt
  have been deprecated. The code was retained as is for compatibility with
  TensorRT 7 which is the only version currently officially supported on the
  Jetson Xavier NX. To suppress the deprecation warnings you can pass the
  `-Wno-deprecated-declarations` option to the compiler.


## License

Copyright 2019 NVIDIA Corporation<br>
Copyright 2021 Smart Robotics Lab, Imperial College London<br>
Copyright 2021 Sotiris Papatheodorou, Imperial College London<br>

Distributed under the [Apache License Version 2.0](LICENSES/Apache-2.0.txt)

The library uses code from
[this sample](https://github.com/NVIDIA/TensorRT/tree/master/samples/opensource/sampleUffMaskRCNN)
in the TensorRT repository.

Many thanks to ivanhc for their help in
[this issue](https://github.com/NVIDIA/TensorRT/issues/490).
