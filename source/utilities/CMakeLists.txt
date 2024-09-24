include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(compute_SV_der STATIC singular_values.cpp)
target_link_libraries(compute_SV_der quadrature)

add_library(compute_SV_der_arnoldi STATIC singular_values_arnoldi.cpp)
target_link_libraries(compute_SV_der_arnoldi arpp_eig_interface)

add_library(find_roots STATIC find_roots.cpp)
target_link_libraries(find_roots gsl tbb)

add_library(arpp_eig_interface STATIC arpp_eig_interface.cpp)

add_library(generate_solution STATIC gen_sol.cpp)

add_library(complex_bessel STATIC cbessel.cpp)

add_library(randomized_svd STATIC randsvd.cpp)

add_library(incoming STATIC incoming.cpp)

add_library(hierarchical_matrix STATIC hmatrix.cpp)
target_link_libraries(hierarchical_matrix parametrizations tbb)

add_library(scatterer STATIC scatterer.cpp)
target_link_libraries(scatterer parametrizations complex_spline quadrature find_roots tbb)

add_library(complex_spline STATIC cspline.cpp)
target_link_libraries(complex_spline gsl)

add_library(gen_sol_op STATIC gen_sol_op.cpp)
target_link_libraries(gen_sol_op galerkin_builder sol_op sol_op_der sol_op_der2 hierarchical_matrix)

add_library(solvers STATIC solvers.cpp)
target_link_libraries(solvers quadrature galerkin_builder complex_spline scatterer gen_sol_op lapack gsl tbb)
