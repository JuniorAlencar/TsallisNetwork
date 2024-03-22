cmake_minimum_required(VERSION 3.10)
project(exe1 VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message(STATUS "${CMAKE_CXX_STANDARD}")

set(CMAKE_BUILD_TYPE Release)

# configure_file(src/SamuraIConfig.hpp.in src/SamuraIConfig.hpp)

file(GLOB SOURCES src/*.cpp src/*.hpp)

set(Boost_USE_MULTITHREADED ON)
set(Boost_STATIC_LIBS ON)
find_package(Boost REQUIRED COMPONENTS log)
if (NOT BOOST_FOUND)
  message(FATAL_ERROR "exe1: Unable to find the VTK build folder.")
endif()
list(APPEND EXTRA_LIBS ${Boost_LIBRARIES})

# add the library jsoncpp to read json files (changer for cluster)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
link_libraries(${JSONCPP_LIBRARIES})


list(APPEND EXTRA_LIBS "${/usr/lib/x86_64-linux-gnu/libz.so.1}" "${/usr/lib/x86_64-linux-gnu/libz.so}" "/usr/lib/x86_64-linux-gnu/libz.a")

#find_package(ZLIB)
#list(APPEND EXTRA_LIBS "/usr/lib/x86_64-linux-gnu/libz.so")s


# to use ninja instead of make
# set(CMAKE_MAKE_PROGRAM ninja)
# message(STATUS "using .......... ${CMAKE_MAKE_PROGRAM}")


add_executable(exe1 ${SOURCES})

target_link_libraries(exe1 PUBLIC ${EXTRA_LIBS})

target_include_directories( exe1 PUBLIC
                           "${PROJECT_BINARY_DIR}"
                            ${EXTRA_INCLUDES}
                          )

target_compile_options(exe1 PUBLIC -Wall -Wextra -Werror)

target_compile_definitions(exe1 PRIVATE
    $<$<CONFIG:Release>:RELEASE_BUILD>
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
