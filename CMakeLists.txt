cmake_minimum_required(VERSION 2.8.12)
project(HelmholtzTransmissionProblem)


###########################################################################
#
# INCLUDE MODULES AND FUNCTIONS FOR CMAKE
#
###########################################################################
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
include(${PROJECT_SOURCE_DIR}/cmake/functions/add_all_subdirectories.cmake)

###########################################################################
#
# COMPILATION SETTINGS
#
###########################################################################

# STORE BINARIES IN ./bin
# (else CMake cannot find files to link for automatic targets)
###########################################################################
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# SET DEFAULT BUILD TYPE
###########################################################################
if( NOT CMAKE_BUILD_TYPE )
    message(STATUS "No build type selected. Default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

# SET COMMAND LINE OUPUT IF REQUESTED
###########################################################################
option(CMDL "Command line output requested." OFF)
if (CMDL)
    add_definitions(-DCMDL)
endif()
unset(CMDL CACHE)

# COMPILATION FLAGS
###########################################################################
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_COMPILER_ID Clang)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -mavx2 -mfma")

# IGNORE WARNINGS
###########################################################################

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    message(STATUS "Using Clang compiler")
    add_definitions(-Wno-deprecated-register -Wno-return-type-c-linkage -Wno-keyword-macro -Wno-missing-braces)
endif()

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    message(STATUS "Using GNU compiler")
    add_definitions(-Wno-deprecated-declarations -Wno-ignored-attributes -Wno-misleading-indentation -Wno-unknown-pragmas)
endif()

###########################################################################
#
# FIND DEPENDENCIES
#
###########################################################################

# FIND EIGEN & COMPLEX BESSEL LIBRARY
###########################################################################

include(${PROJECT_SOURCE_DIR}/cmake/functions/GetEIGEN3.cmake)
add_definitions(-DEIGEN3_ACTIVATED)
#include(${PROJECT_SOURCE_DIR}/cmake/functions/Getcomplex_bessel.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/functions/Getarpackpp.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/functions/GetH2Lib.cmake)

# ADD SUBDIRECTORIES
add_subdirectory(examples)
add_subdirectory(source)
add_subdirectory(test)

# DOCUMENTATION

# Check for Doxygen
find_package(Doxygen)

if (DOXYGEN_FOUND)
    # Set the output directory for Doxygen documentation
    set(DOXYGEN_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/docs)

    # Create Doxyfile content
    set(DOXYGEN_INPUT "${PROJECT_SOURCE_DIR}/include")
    set(DOXYGEN_RECURSIVE "YES")
    set(DOXYGEN_GENERATE_HTML "YES")
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "YES")
    set(DOXYGEN_ENABLE_LATEX "YES")

    # Create the Doxyfile using configure_file
    configure_file(
        ${PROJECT_SOURCE_DIR}/Doxyfile.in
        ${PROJECT_SOURCE_DIR}/Doxyfile
        @ONLY
    )

    # Add a custom target to run Doxygen
    add_custom_target(
        docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
else()
    message(WARNING "Doxygen not found. Documentation will not be generated.")
endif()
