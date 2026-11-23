find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
include(GNUInstallDirs)

add_library(ink2-engine STATIC
    ink2_engine.cu
    ink2_cpu.cpp
    ink2_image.cpp
    ink2_embedded.cpp)
target_include_directories(ink2-engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ink2-engine PUBLIC CUDA::cublas CUDA::cudart Threads::Threads)
target_compile_options(ink2-engine PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wextra>)
set_target_properties(ink2-engine PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

add_executable(llama-ink2 ink2_main.cpp)
set_target_properties(llama-ink2 PROPERTIES OUTPUT_NAME bel-inkling2 CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_link_libraries(llama-ink2 PRIVATE ink2-engine)

add_executable(llama-ink2ref ink2ref.cu)
target_include_directories(llama-ink2ref PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(llama-ink2ref PRIVATE CUDA::cudart)
set_target_properties(llama-ink2ref PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

install(TARGETS llama-ink2 llama-ink2ref RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
