cmake_minimum_required(VERSION 3.16)
project(cnn_mpi LANGUAGES CXX)

find_package(MPI REQUIRED COMPONENTS CXX)

file(GLOB_RECURSE CNN_LIBRARY_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(cnn_core STATIC ${CNN_LIBRARY_SOURCES})
target_compile_features(cnn_core PUBLIC cxx_std_20)
target_include_directories(cnn_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(cnn_core PUBLIC MPI::MPI_CXX)

execute_process(
    COMMAND git rev-parse --short=12 HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    OUTPUT_VARIABLE CNN_SOURCE_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT CNN_SOURCE_REVISION)
    set(CNN_SOURCE_REVISION "unknown")
endif()
target_compile_definitions(
    cnn_core PRIVATE CNN_SOURCE_REVISION="${CNN_SOURCE_REVISION}")

add_executable(cnn_executable main.cpp)
target_link_libraries(cnn_executable PRIVATE cnn_core)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/experiments/optimizer_comparison/main.cpp")
    add_executable(optimizer_comparison experiments/optimizer_comparison/main.cpp)
    target_link_libraries(optimizer_comparison PRIVATE cnn_core)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/experiments/regularization_tuning/main.cpp")
    add_executable(regularization_tuning experiments/regularization_tuning/main.cpp)
    target_link_libraries(regularization_tuning PRIVATE cnn_core)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(cnn_tests tests/test_cross_validation.cpp)
    target_link_libraries(cnn_tests PRIVATE cnn_core)
    add_test(NAME cnn_tests_serial COMMAND cnn_tests)
    if(MPIEXEC_EXECUTABLE)
        add_test(
            NAME cnn_tests_mpi_2
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:cnn_tests>
                    ${MPIEXEC_POSTFLAGS})
    endif()
endif()
