include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(parametrizations)

add_subdirectory(quadrature)

add_subdirectory(utilities)

add_library(single_layer STATIC single_layer.cpp)
target_link_libraries(single_layer parametrizations)
add_library(single_layer_der STATIC single_layer_der.cpp)
target_link_libraries(single_layer_der parametrizations)
add_library(single_layer_der2 STATIC single_layer_der2.cpp)
target_link_libraries(single_layer_der2 parametrizations)

add_library(double_layer STATIC double_layer.cpp)
target_link_libraries(double_layer parametrizations)
add_library(double_layer_der STATIC double_layer_der.cpp)
target_link_libraries(double_layer_der parametrizations)
add_library(double_layer_der2 STATIC double_layer_der2.cpp)
target_link_libraries(double_layer_der2 parametrizations)

add_library(hypersingular STATIC hypersingular.cpp)
target_link_libraries(hypersingular parametrizations)
add_library(hypersingular_der STATIC hypersingular_der.cpp)
target_link_libraries(hypersingular_der parametrizations)
add_library(hypersingular_der2 STATIC hypersingular_der2.cpp)
target_link_libraries(hypersingular_der2 parametrizations)

add_library(mass_matrix STATIC mass_matrix.cpp)
target_link_libraries(mass_matrix parametrizations tbb)
add_library(sol_op STATIC mass_matrix single_layer double_layer hypersingular)
target_link_libraries(sol_op complex_bessel tbb)
add_library(sol_op_der STATIC single_layer_der double_layer_der hypersingular_der)
target_link_libraries(sol_op_der complex_bessel tbb)
add_library(sol_op_der2 STATIC single_layer_der2 double_layer_der2 hypersingular_der2)
target_link_libraries(sol_op_der2 complex_bessel tbb)

add_library(galerkin_builder STATIC galerkin_builder.cpp)
target_link_libraries(galerkin_builder sol_op complex_bessel parametrizations tbb)
