set_source_files_properties(test_pico12.cpp PROPERTIES LANGUAGE CUDA)
ingine_add_test(NAME kernels_pico12 SOURCES test_pico12.cpp LIBS ingine_kernels_dense ingine_kernels_batch GPU TIMEOUT 300)
ingine_cuda_target(kernels_pico12)
add_test(NAME kernels_pico12_host COMMAND kernels_pico12 --host-only)
