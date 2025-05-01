cmake_minimum_required(VERSION 3.19)
include(FetchContent)

project(Optiz)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OPTIZ_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB SRC_FILES src/*.cpp src/NewtonSolver/*.cpp src/Linear/*.cpp
     src/Common/*.cpp)
add_library(${PROJECT_NAME} ${SRC_FILES})
target_include_directories(
  ${PROJECT_NAME}
  PRIVATE ${OPTIZ_INCLUDE_DIR}/Optiz ${OPTIZ_INCLUDE_DIR}/Optiz/NewtonSolver
          ${OPTIZ_INCLUDE_DIR}/Optiz/Linear ${OPTIZ_INCLUDE_DIR}/Optiz/Autodiff
          ${OPTIZ_INCLUDE_DIR}/Optiz/Common)

# Search for TBB
set(TBB_TEST OFF)
set(TBB_OUTPUT_DIR ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
find_package(TBB QUIET)
if(NOT TBB_FOUND)
  # TBB not found, fetch it from GitHub
  FetchContent_Declare(
    TBB
    GIT_REPOSITORY https://github.com/uxlfoundation/oneTBB.git
    GIT_TAG v2022.1.0
  )
  FetchContent_MakeAvailable(TBB)
  set(TBB_FOUND TRUE)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE TBB::tbb)

# Search for Eigen.
find_package(Eigen3 QUIET)
if(NOT TARGET Eigen3::Eigen)
  # Eigen not found, fetch it from gitlab.
  FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    SOURCE_SUBDIR cmake # Hack to avoid Eigen's CMakeLists.txt
  )
  FetchContent_MakeAvailable(Eigen3)
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                                 ${eigen3_SOURCE_DIR})
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
set_target_properties(${PROJECT_NAME} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                                 "${OPTIZ_INCLUDE_DIR}")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # using Clang
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_definitions(-Wno-non-template-friend -Wno-psabi)
endif()

if(EMSCRIPTEN)
  target_compile_options(${PROJECT_NAME} PRIVATE -msimd128 -msse)
  target_link_options(
    ${PROJECT_NAME}
    PRIVATE
    -pthread
    -sPTHREAD_POOL_SIZE=navigator.hardwareConcurrency
    -sWASM=1
    -sALLOW_MEMORY_GROWTH=1
    -sNO_EXIT_RUNTIME=0
    -sASSERTIONS=1)
endif()

if(APPLE)
  # SET(CMAKE_OSX_DEPLOYMENT_TARGET 14.0)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang")
  endif()
endif()

# Testing
if(PROJECT_IS_TOP_LEVEL)
  include(testing)
endif()


add_executable(test_optiz main.cpp)
target_link_libraries(test_optiz PRIVATE Optiz TBB::tbb)
