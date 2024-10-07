# Define variables for the external library
set(LIBRARY_NAME "H2Lib")
set(LIBRARY_REPO "https://github.com/H2Lib/H2Lib.git")
set(LIBRARY_SOURCE_DIR "${PROJECT_SOURCE_DIR}/H2Lib")

if(NOT EXISTS "${LIBRARY_SOURCE_DIR}")
    message(STATUS "H2Lib library not found, installing locally.")
    execute_process(COMMAND git clone "${LIBRARY_REPO}"
                    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    file(WRITE "${LIBRARY_SOURCE_DIR}/options.inc"
         "BRIEF_OUTPUT=1\nOPT=1\nUSE_BLAS=1\nUSE_COMPLEX=1\nHARITH_RKMATRIX_QUICK_EXIT=1\n")
    execute_process(COMMAND make
                    WORKING_DIRECTORY "${LIBRARY_SOURCE_DIR}")
    execute_process(COMMAND rm -rf tmp src
                    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endif()

include_directories(${LIBRARY_SOURCE_DIR}/Library)

# Define the paths to the headers and the static library (.a file)
set(LIBRARY_INCLUDE_DIR "${LIBRARY_SOURCE_DIR}/Library")
set(LIBRARY_STATIC "${LIBRARY_SOURCE_DIR}/libh2.a")  # Adjust the name

# Create an IMPORTED target for the external static library
add_library(h2 STATIC IMPORTED)

# Set the properties of the imported library (e.g., where the .a file is located)
set_target_properties(h2 PROPERTIES
    IMPORTED_LOCATION "${LIBRARY_STATIC}"
    INTERFACE_INCLUDE_DIRECTORIES "${LIBRARY_INCLUDE_DIR}"
)
