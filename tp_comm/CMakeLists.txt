cmake_minimum_required(VERSION 3.24)
project(inkling_comm VERSION 0.1 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_path(IBVERBS_INCLUDE_DIR infiniband/verbs.h REQUIRED)
find_library(IBVERBS_LIBRARY ibverbs REQUIRED)

set(INKLING_COMM_CUDA_ARCHITECTURES "121a" CACHE STRING "CUDA architectures for libinkling_comm")
option(INKLING_COMM_BUILD_TOOLS "Build standalone M1 qualification tools" ${PROJECT_IS_TOP_LEVEL})

add_library(inkling_comm STATIC
    src/comm_backend.cpp
    src/roce_uma_backend.cu
    src/roce_uma_rdma.cpp
)
add_library(inkling_comm::inkling_comm ALIAS inkling_comm)
target_include_directories(inkling_comm PUBLIC include PRIVATE "${IBVERBS_INCLUDE_DIR}")
target_compile_features(inkling_comm PUBLIC cxx_std_17 PRIVATE cuda_std_17)
target_link_libraries(inkling_comm PUBLIC CUDA::cudart PRIVATE "${IBVERBS_LIBRARY}" Threads::Threads)
set_target_properties(inkling_comm PROPERTIES
    CUDA_ARCHITECTURES OFF
    POSITION_INDEPENDENT_CODE ON
)
# CMake 3.28's CUDA_ARCHITECTURES parser does not accept the 'a' suffix.
foreach(arch IN LISTS INKLING_COMM_CUDA_ARCHITECTURES)
    target_compile_options(inkling_comm PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_${arch},code=sm_${arch}>")
endforeach()

if(INKLING_COMM_BUILD_TOOLS)
    target_compile_definitions(inkling_comm PRIVATE INKLING_COMM_TEST_HOOKS)
    foreach(tool IN ITEMS test_uma_visibility bench_roce_uma_graph test_comm_backend)
        add_executable(${tool} tools/${tool}.cu)
        target_link_libraries(${tool} PRIVATE inkling_comm::inkling_comm)
        # These targets contain only portable CUDA test kernels.
        set_target_properties(${tool} PROPERTIES CUDA_ARCHITECTURES 121)
        target_compile_features(${tool} PRIVATE cuda_std_17)
    endforeach()
endif()
