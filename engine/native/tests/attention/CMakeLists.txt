set_source_files_properties(test_kv12.cpp PROPERTIES LANGUAGE CUDA)
ingine_add_test(NAME kv12_attention SOURCES test_kv12.cpp LIBS ingine_kernels_attn GPU TIMEOUT 300)
ingine_cuda_target(kv12_attention)
add_test(NAME kv12_codec COMMAND kv12_attention --host)
