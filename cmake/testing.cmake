include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0)
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

file(GLOB TEST_SRC_FILES tests/*.cpp)
add_executable(tests ${TEST_SRC_FILES})
target_link_libraries(tests GTest::gtest_main Optiz Eigen3::Eigen)
# Add -L /opt/homebrew/opt/llvm/lib/c++/ to the linker flags
target_link_options(tests PRIVATE -L /opt/homebrew/opt/llvm/lib/c++/)
include(GoogleTest)
gtest_discover_tests(tests)
