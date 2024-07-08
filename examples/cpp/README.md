
# Create a custom C++ application using Intel NPU acceleration Library

The example demonstrates how to create a custom C++ application using the Intel NPU acceleration Library. It showcases the usage of the library's features and functionalities for accelerating neural network inference on Intel NPUs. The provided code snippet shows the build process using CMake, where the project is configured and built in the Release configuration.

## Build

To build the custom C++ application using the Intel NPU acceleration Library, follow these steps:

1. Run the following commands to configure and build the project in the Release configuration:
    ```
    cmake -S . -B build
    cmake --build build --config Release
    ```
2. Once the build process is complete, you can find the executable file at `build\Release\intel_npu_acceleration_library_example.exe` (on windows)

Make sure you have the necessary dependencies, compiler and libraries installed before building the application.
