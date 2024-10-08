# Helmholtz Transmission Problem BEM with randomized SVD
## Introduction
This is a fork of the project [HelmholtzTransmissionProblemBEM](https://github.com/DiegoRenner/HelmholtzTransmissionProblemBEM/tree/master) written by Diego Renner [[1]](#1), with the following improvements:

- The code is upgraded to the C++17 standard. The typedef <tt>data</tt> was replaced by <tt>grid_data</tt> because it clashed with <tt>std::data</tt>.
- The new dependencies are the Intel <tt>tbb</tt> library for parallelization, which is supported by GCC, and the <tt>gsl</tt> library which provides the spline interpolation routines used in trace approximation.
- Linking to [complex_bessel](https://github.com/joeydumont/complex_bessel) library is not needed anymore. The routines for computing Bessel functions are implemented from scratch in C++ using the theoretical background presented in two papers by Donald E. Amos (1983).
- Significant speedups were achieved in the routine for assembling solution matrices and their derivatives, mostly by vectorization and by removing duplicate computations.
- A randomized SVD algorithm was implemented following the ideas in [this paper](https://arxiv.org/abs/0909.4061). The corresponding routine called <tt>randomized_svd::sv</tt> approximates the smallest singular value of the solution matrix by using this technique.
- Solution of the Helmholtz transmission problem can be plotted for an arbitrary polygonal scatterer and for several types of incoming waves.

We will appreciate it if users of this library were to cite the paper [[1]](#1) in their work.

## Configuration and Dependencies
The library can be configured by running 
~~~
cmake CMakeLists.txt
~~~
in the base directory or
~~~
cmake ..
~~~
in a subdirectory for an out-of-source build.
This should automatically generate a target with which the Eigen library for matrix computations can be built by running
~~~
make Eigen
~~~
if it is not already available. This step is not required; just typing
~~~
make
~~~
will build the entire project together with Eigen. However, if you're using <tt>cmake</tt> with <tt>ninja</tt>, make sure you always do
~~~
cd build
ninja Eigen
~~~
from the project root directory after configuration. (In particular, you should do this in a terminal before building the project in Kdevelop for the first time.) Afterwards, the rest is compiled by running
~~~
ninja
~~~
from the <tt>build</tt> directory (or by issuing the build step in Kdevelop).

The [arpackpp library](https://github.com/m-reuter/arpackpp) which gives a <tt>c++</tt> interface to the [arpack library](https://github.com/opencollab/arpack-ng) is installed automatically. <tt>arpack</tt>, <tt>lapack</tt>, [<tt>gsl</tt>](https://www.gnu.org/software/gsl/) and <tt>tbb</tt> need to be installed separately and can usually be done so with your distributions packagemanager.

For <tt>arch</tt> based distros:
~~~
sudo pacman -S arpack
sudo pacman -S lapack
sudo pacman -S gsl
sudo pacman -S tbb2020
~~~
For <tt>debian</tt> based distros:
~~~
sudo apt install libboost-all-dev
sudo apt install libarpack2-dev 
sudo apt install liblapack3-dev
sudo apt install libgsl-dev
sudo apt install libtbb-dev
~~~

To generate the documentation <tt>latex</tt> and <tt>doxygen</tt> have to be installed as well.
For <tt>arch</tt> based distros:
~~~
sudo pacman -S texlive-most
sudo pacman -S doxygen
~~~
For <tt>debian</tt> based distros:
~~~
sudo apt install texlive-full
sudo apt install doxygen
~~~
Running CMake also configures some examples of how to use the library as <tt>make</tt> targets.
These can then be built by running 

~~~
make <target_name>
~~~

The compiled binary can be found in the <tt>bin</tt> directory.

### Compilation with MinGW in Windows

For Windows 10 and 11 users, the code can be compiled to native binaries as follows. If you already have <tt>MSYS2</tt> installed, skip the next section.

#### Installing <tt>MSYS2</tt>
Download the latest <tt>MSYS2</tt> installer from [here](https://www.msys2.org/) and run it. Use the defaults provided by the installer, but untick the "Run MSYS2" checkbox in the last page. Then run the <tt>MSYS2 MINGW64</tt> application from the Windows menu, which opens a terminal. Type
~~~
pacman -Syu
~~~
to update the system. You may be asked to relaunch the <tt>MINGW64</tt> terminal.

#### Installing necessary packages
To install dependencies, copy the following command to the <tt>MINGW64</tt> terminal:
~~~
pacman -S base-devel git mingw-w64-x86_64-gcc mingw-w64-x86_64-gcc-fortran mingw-w64-x86_64-lapack mingw-w64-x86_64-arpack mingw-w64-x86_64-boost mingw-w64-x86_64-gsl mingw-w64-x86_64-python mingw-w64-x86_64-tbb mingw-w64-x86_64-cmake
~~~
Next, add a symbolic link to the <tt>tbb</tt> shared library by issuing the command
~~~
ln -s /mingw64/lib/libtbb12.dll.a /mingw64/lib/libtbb.dll.a
~~~
After this is done, close the <tt>MSYS2</tt> terminal. You will not be needing it anymore.

In the Windows Start menu, start typing <tt>environment</tt>, which should offer you to edit the environment variables for your account. Select the entry and press Enter, which will open a Control Panel dialog with user variables. Select the <tt>Path</tt> variable in the upper list and click <tt>Edit</tt>. In the dialog that opens, click <tt>New</tt> and enter
~~~
C:\msys64\mingw64\bin
~~~
Press <tt>Enter</tt> and repeat the same process to add
~~~
C:\msys64\usr\bin
~~~
This will have the desired effect only if you have installed <tt>MSYS2</tt> in the <tt>C:\msys64</tt> directory (which is the default). Otherwise, modify the above paths accordingly.

#### Compilation in the Windows commandline
Launch the Windows PowerShell (or <tt>cmd</tt>, but the former is better). Navigate to the directory in which you wish to install the code. Alternatively, open that folder in File manager, right click to get the context menu and choose "Open in terminal". Then enter
~~~
git clone https://github.com/marohnicluka/HelmholtzBEM
~~~
This should download the project to the <tt>HelmholtzBEM</tt> folder. To configure the project, enter
~~~
cd HelmholtzBEM
~~~
and run
~~~
cmake .
~~~
After configuring is done, run
~~~
ninja Eigen
~~~
which builds the <tt>Eigen</tt> library. Then run
~~~
ninja
~~~
to compile the rest of the project. The target binaries are located in the <tt>bin</tt> subdirectory. Anytime you wish to update the code and recompile, simply enter
~~~
git pull; ninja
~~~
from the project root directory. If CMake files have been changed, you may have to configure the project again before recompiling:
~~~
git pull
rm CMakeCache.txt
cmake .
ninja
~~~

## Usage
We will show how the built targets are to be used.
We commonly refer to the wavenumber by $k$.

#### <tt>doxygen_HelmholtzTransmissionProblemBEM</tt>
This target generates a documentation of the library in the <tt>doxygen/generated_doc</tt> directory.
The documentation can be browsed using any common browser.

#### <tt>debugging_SVs</tt>
This target builds a script that computes the Eigenvalues of the BIO for Helmholtz Transmission Problem. The results are written to file. The script can be run as follows:
~~~
/path/to/debugging_SVs
~~~
The output file will contain a section for each set mesh resolution and each of those sections will contain one section each for every BIO where all Eigenvalues for different wavenumbers will be listed in columns. The Eigenvalues are computed using the facts stated in Lemma 3.22. [TODO: find reference]

#### <tt>direct_v_arnoldi</tt>
This target builds a script that computes the singular values of the Galerkin BEM approximated BIO for the second-kind direct BIEs of the Helmholtz transmission problem, once using the Arnoldi algorithm and once using s direct solver. The scatterer is set to be a circle. The results are written to file. The script can be run as follows: 
~~~
/path/to/direct_v_arnoldi <radius of circle>
    <#SVs to be computed> <accurracy of Arnoldi algorithm>
~~~
 The script will generate four files: <tt>file_vals_eig_<#SVs to be computed>\_\<accurracy of arnoldi algorithm\>.dat</tt>, <tt>file_vals_arpp\_<#SVs to be computed>\_\<accurracy of arnoldi algorithm\>.dat</tt>, <tt>file_timings_<#SVs to be computed>\_\<accurracy of arnoldi algorithm\>.dat</tt>, <tt>file_iter_<#SVs to be computed>_\<accurracy of arnoldi algorithm\>.dat</tt>. These will contain the SVs computed using the direct solver, the SVs computed using the Arnoldi algorithm, the time taken by the direct solver and the Arnoldi algorithm, and the number of iterations the Arnoldi algorithm took to converge respectively.

#### <tt>direct_v_arnoldi_1st_der</tt>
This target builds a script that computes the first derivative of the singular values of the Galerkin BEM approximated BIO for the second-kind direct BIEs of the Helmholtz transmission problem, once using the Arnoldi algorithm and once using s direct solver. The scatterer is set to be a circle. The results are written to file. The script can be run as follows: 
~~~
/path/to/direct_v_arnoldi_1st_der <radius of circle>
    <#SV derivatives to be computed> <accurracy of Arnoldi algorithm>
~~~
The script will generate four files:
<tt>file_vals_eig_<#SV derivatives to be computed>\_\<accurracy of arnoldi algorithm\>\_1stDer.dat</tt>,
<tt>file_vals_arpp_<#SV derivatives to be computed>\_\<accurracy of arnoldi algorithm\>\_1stDer.dat</tt>,
<tt>file_timings_<#SV derivatives to be computed>\_\<accurracy of arnoldi algorithm\>\_1stDer.dat</tt>,
<tt>file_iter_<#SV derivatives to be computed>_\<accurracy of arnoldi algorithm\>_1stDer.dat</tt>.
 These will contain the derivatives computed using the direct solver,
 the derivatives computed using the Arnoldi algorithm,
 the time taken by the direct solver and the Arnoldi algorithm,
 and the number of iterations the Arnoldi algorithm took to converge respectively.

#### <tt>direct_v_arnoldi_2nd_der</tt>
This target builds a script that computes the second derivative of the singular values of the Galerkin BEM approximated BIO for the second-kind direct BIEs of the Helmholtz transmission problem, once using the Arnoldi algorithm and once using s direct solver. The scatterer is set to be a circle. The results are written to file. The script can be run as follows:
~~~
/path/to/library/bin/direct_v_arnoldi_2nd_der <radius of circle>
    <#SV derivatives to be computed> <accurracy of arnoldi algorithm>
~~~
The script will generate four files: <tt>file_vals_eig_<#SV derivatives to be computed>\_\<accurracy of arnoldi algorithm\>\_2ndDer.dat</tt>, <tt>file_vals_arpp_<#SV derivatives to be computed>\_\<accurracy of arnoldi algorithm\>\_2ndDer.dat</tt>, <tt>file_timings_<#SV derivatives to be computed>\_\<accurracy of arnoldi algorithm\>\_2ndDer.dat</tt>, <tt>file_iter_<#SV derivatives to be computed>_\<accurracy of arnoldi algorithm\>_2ndDer.dat</tt>. These will contain the derivatives computed using the direct solver, the derivatives computed using the Arnoldi algorithm, the time taken by the direct solver and the Arnoldi algorithm, and the number of iterations the Arnoldi algorithm took to converge respectively.

#### <tt>dirichlet_example</tt>
This target builds a script that computes the solution to a Dirichlet problem using first kind direct BIEs. No command line parameters are necessary. Once built the script can be run as follows: 
~~~
/path/to/library/bin/dirichlet_example
~~~
The user will be updated over the residual error in the euclidean norm of the computed FEM-space interpolation coefficients to the known FEM-space interpolation coefficients for the current number of panels through the command line.
 
#### <tt>neumann_example</tt>
This target builds a script that computes the solution to a Neumann problem
using first kind direct BIEs. No command line parameters are necessary. Once built the script can be run as follows: 
~~~
/path/to/library/bin/neumann_example
~~~
The user will be updated over the residual error in the euclidean norm of the computed FEM-space interpolation coefficients to the known FEM-space interpolation coefficients for the current number of panels through the command line.

#### <tt>parabolic_approximation</tt>
This target builds a script that tries to find minimas in the smallest sinuglar values
of the Galerkin BEM approximated solutions operator for the second-kind direct BIEs of 
the Helmholtz Transmission problem.
The minimas are searched for using a parabolic approximation
based on evaluating the smallest singular values and their first
two derivatives.
The results are written to disk.
No command line arguments are necessary.
The script can be run as follows:
~~~
/path/to/library/bin/parabolic_approximation <outfile>
~~~
In the file the first column contains the initial point used for the parabolic approximation.
The next three columns contain the function value and the first two derivatives at the initial point that were used to compute the parabolic approximation.
The user also will get updates on the current best approximation for a minima and the value of the first derivatie at this point through the command line if <tt>-DCMDL</tt> is set.

### Finding minima of the smallest singular value of the BEM approximated BIO

#### <tt>roots_brent_circle</tt>
This target builds a script that computes minimas in the smallest singular value of the Galerkin BEM approximated solutions operator for the second-kind direct BIEs of the Helmholtz transmission problem using the Van Wijngaarden-Dekker-Brent method.
The scatterer is set to be a circle.
The results are written to disk.
The script can be run as follows:
~~~
/path/to/library/bin/roots_brent_circle <radius of circle> 
    <refraction inside> <refraction outside> <initial wavenumber> 
    <#grid points for root search> <#panels> 
    <order of quadrature rule> <outputfile>
~~~
The resulting file will contain the left boundary of the
interval used to compute the root in the first column.
Then in the next three columns will be the point, the
function value and the derivative at which the root was found.
The last column will contain the number of iterations used to find the root.
If no root was found the last four columns will be set to <tt>NAN</tt>.
The singular values and their derivatives are computed using the direct
Eigen algorithm.
The user will be updated through the command line about the
progress of the algorithm if <tt>-DCMDL</tt> is set.

#### <tt>roots_brent_circle_arnoldi</tt>
This target builds a script that computes minimas in the smallest singular value of the
Galerkin BEM approximated solutions operator for the second-kind direct BIEs of the Helmholtz
transmission problem using the Van Wijngaarden-Dekker-Brent method.
The scatterer is set to be a circle.
The results are written to disk.
The script can be run as follows:
~~~
/path/to/library/bin/roots_brent_circle <radius of circle> 
    <refraction inside> <refraction outside> <initial wavenumber> 
    <#grid points for root search> <#panels> 
    <order of quadrature rule> <outputfile>
~~~
The resulting file will contain the left boundary of the
interval used to compute the root in the first column.
Then in the next three columns will be the point, the
function value and the derivative at which the root was found.
The last column will contain the number of iterations used to find the root.
If no root was found the last four columns will be set to <tt>NAN</tt>.
The singular values and their derivatives are computed using the Arnoldi algorithm.
The user will be updated through the command line about the
progress of the algorithm if <tt>-DCMDL</tt> is set.

#### <tt>roots_brent_square</tt>
This target builds a script that computes minimas in the smallest singular value of the
Galerkin BEM approximated solutions operator for the sedond-kind direct BIEs of the Helmholtz
transmission problem using the Van Wijngaarden-Dekker-Brent method.
The scatterer is set to be a square.
The results are written to disk.
The script can be run as follows:
~~~
/path/to/library/bin/roots_brent_circle <half side length of square> 
    <refraction inside> <refraction outside> <initial wavenumber> 
    <#grid points for root search> <#panels> 
    <order of quadrature rule> <outputfile>
~~~
The resulting file will contain the left boundary of the
interval used to compute the root in the first column.
Then in the next three columns will be the point, the
function value and the derivative at which the root was found.
The last column will contain the number of iterations used to find the root.
If no root was found the last four columns will be set to <tt>NAN</tt>.
The singular values and their derivatives are computed using the direct
Eigen algorithm.
The user will be updated through the command line about the
progress of the algorithm if <tt>-DCMDL</tt> is set.

#### <tt>roots_brent_square_arnoldi</tt>
This target builds a script that computes minimas in the smallest singular value of the
Galerkin BEM approximated solutions operator for the sedond-kind direct BIEs of the Helmholtz
transmission problem using the Van Wijngaarden-Dekker-Brent method.
The scatterer is set to be a square.
The results are written to disk.
The script can be run as follows:
~~~
/path/to/library/bin/roots_brent_circle <half side length of square> 
    <refraction inside> <refraction outside> <initial wavenumber> 
    <#grid points for root search> <#panels> 
    <order of quadrature rule> <outputfile>
~~~
The resulting file will contain the left boundary of the
interval used to compute the root in the first column.
Then in the next three columns will be the point, the
function value and the derivative at which the root was found.
The last column will contain the number of iterations used to find the root.
If no root was found the last four columns will be set to <tt>NAN</tt>.
The singular values and their derivatives are computed using the Arnoldi algorithm.
The user will be updated through the command line about the
progress of the algorithm
if <tt>-DCMDL</tt> is set.

#### <tt>roots_brent_square_rsvd</tt>
This target builds a script that computes minimas in the smallest singular value of the Galerkin BEM approximated solutions operator for the sedond-kind direct BIEs of the Helmholtz transmission problem using the Van Wijngaarden-Dekker-Brent method. The scatterer is set to be a square. The results are written to the <tt>data</tt> directory. The script can be run as follows:
~~~
/path/to/library/bin/roots_brent_square_rsvd <half side length of square>
    <refraction inside> <refraction outside> <initial wavenumber>
    <#grid points for root search> <#panels> <order of quadrature rule>
    <accuracy> <#subspace iterations>
~~~
The resulting file will contain the boundaries of the interval used to compute the root in the first two columns, which are obtained by approximating the smallest singular value with randomized SVD. Then in the next two columns will be the point and the respective function value. The last column will contain the number of iterations used to find the root. The singular values are computed using the Arnoldi algorithm. The user will be updated through the command line about the progress of the algorithm if <tt>-DCMDL</tt> is set.

#### <tt>roots_brent_polygon_rsvd</tt>
This target builds a script that computes minimas in the smallest singular value of the Galerkin BEM approximated solutions operator for the sedond-kind direct BIEs of the Helmholtz transmission problem using Brent's method without derivatives. The scatterer is a polygon read from disk (see <tt>scatterer.hpp</tt> for a description of the input file syntax). The results are written to the <tt>data</tt> directory. The script can be run as follows:
~~~
/path/to/library/bin/roots_brent_square_rsvd <scatterer filename>
    <refraction inside> <refraction outside> <initial wavenumber>
    <#grid points for root search> <#panels> <quadrature order>
    <accuracy> <#subspace iterations>
~~~
The resulting file will contain the local minima in a single column. The singular values are computed using the randomized SVD algorithm with the specified number of subspace iterations. The user will be updated through the command line about the progress of the algorithm if <tt>-DCMDL</tt> is set.

#### <tt>roots_mixed_circle_arnoldi</tt>
This target builds a script that computes minimas in the smallest singular value of the
Galerkin BEM approximated solutions operator for the second-kind direct BIEs of the Helmholtz
transmission problem using a precursor to the Algorithm described in Listing 1 of
 https://www.sam.math.ethz.ch/sam_reports/reports_final/reports2022/2022-38.pdf.
The scatterer is set to be a circle.
The results are written to disk.
The script can be run as follows:
~~~
/path/to/library/bin/roots_brent_circle <radius of circle> 
    <refraction inside> <refraction outside> <initial wavenumber> 
    <#grid points for root search> <#panels> 
    <order of quadrature rule> <outputfile>
~~~
The resulting file will contain the left boundary of the
interval used to compute the root in the first column.
Then in the next three columns will be the point, the
function value and the derivative at which the root was found.
The last column will contain the number of iterations used to find the root.
If no root was found the last four columns will be set to <tt>NAN</tt>.
The singular values and their derivatives are computed using the Arnoldi algorithm.
The user will be updated through the command line about the
progress of the algorithm if <tt>-DCMDL</tt> is set.

#### <tt>roots_mixed_square_arnoldi</tt>
This target builds a script that computes minimas in the smallest singular value of the
Galerkin BEM approximated solutions operator for the sedond-kind direct BIEs of the Helmholtz
transmission problem using a precursor to the Algorithm described in Listing 1 of
https://www.sam.math.ethz.ch/sam_reports/reports_final/reports2022/2022-38.pdf.
The scatterer is set to be a square.
The results are written to disk.
The script can be run as follows:
~~~
/path/to/library/bin/roots_brent_circle <half side length of square> 
    <refraction inside> <refraction outside> <initial wavenumber> 
    <#grid points for root search> <#panels> 
    <order of quadrature rule> <outputfile>
~~~
The resulting file will contain the left boundary of the
interval used to compute the root in the first column.
Then in the next three columns will be the point, the
function value and the derivative at which the root was found.
The last column will contain the number of iterations used to find the root.
If no root was found the last four columns will be set to <tt>NAN</tt>.
The singular values and their derivatives are computed using the Arnoldi algorithm.
The user will be updated through the command line about the
progress of the algorithm if <tt>-DCMDL</tt> is set.

#### <tt>roots_newton_circle</tt>
This target builds a sript that computes minimas in the smallest singular value of the
Galerkin BEM approximated solutions operator for the second-kind direct BIEs of the Helmholtz
transmission problem using the Newton-Raphson method.
The scatterer is set to be a circle.
The results are written to disk.
The script can be run as follows:
~~~
/path/to/library/bin/roots_newton_circle <radius of circle> 
    <refraction inside> <refraction outside> <initial wavenumber> 
    <#grid points for root search> <#panels> 
    <order of quadrature rule> <outputfile>
~~~
The resulting file will contain the left boundary of the
interval used to compute the root in the first column.
Then in the next three columns will be the point,
the function value and the derivative at which the root was found.
The last column will contain the number of iterations used to find the root.
If no root was found the last four columns will be set to <tt>NAN</tt>.
The singular values and their derivatives are computed using the direct
Eigen algorithm.
The user will be updated through the command line about the
progress of the algorithm if <tt>-DCMDL</tt> is set.

#### <tt>roots_newton_circle_arnoldi</tt>
This target builds a sript that computes minimas in the smallest singular value of the
Galerkin BEM approximated solutions operator for the second-kind direct BIEs of the Helmholtz
transmission problem using the Newton-Raphson method.
The scatterer is set to be a circle.
The results are written to disk.
The script can be run as follows:
~~~
/path/to/library/bin/roots_newton_circle <radius of circle> 
    <refraction inside> <refraction outside> <initial wavenumber> 
    <#grid points for root search> <#panels> 
    <order of quadrature rule> <outputfile>
~~~
The resulting file will contain the left boundary of the
interval used to compute the root in the first column.
Then in the next three columns will be the point,
the function value and the derivative at which the root was found.
The last column will contain the number of iterations used to find the root.
If no root was found the last four columns will be set to <tt>NAN</tt>.
The singular values and their derivatives are computed using the Arnoldi algorithm.
The user will be updated through the command line about the
progress of the algorithm if <tt>-DCMDL</tt> is set.

#### <tt>roots_newton_square</tt>
This target builds a sript that computes minimas in the smallest singular value of the
Galerkin BEM approximated solutions operator for the second-kind direct BIEs of the Helmholtz
transmission problem using the Newton-Raphson method.
The scatterer is set to be a square.
The results are written to disk.
The script can be run as follows:
~~~
/path/to/library/bin/roots_newton_circle <side length of square> 
    <refraction inside> <refraction outside> <initial wavenumber> 
    <#grid points for root search> <#panels> 
    <order of quadrature rule> <outputfile>
~~~
The resulting file will contain the left boundary of the
interval used to compute the root in the first column.
Then in the next three columns will be the point,
the function value and the derivative at which the root was found.
The last column will contain the number of iterations used to find the root.
If no root was found the last four columns will be set to <tt>NAN</tt>.
The singular values and their derivatives are computed using the direct
Eigen algorithm.
The user will be updated through the command line about the
progress of the algorithm if <tt>-DCMDL</tt> is set.

#### <tt>roots_newton_square_arnoldi</tt>
This target builds a sript that computes minimas in the smallest singular value of the
Galerkin BEM approximated solutions operator for the second-kind direct BIEs of the Helmholtz
transmission problem using the Newton-Raphson method.
The scatterer is set to be a square.
The results are written to disk.
The script can be run as follows:
~~~
/path/to/library/bin/roots_newton_circle <side length of square> 
    <refraction inside> <refraction outside> <initial wavenumber> 
    <#grid points for root search> <#panels> 
    <order of quadrature rule> <outputfile>
~~~
The resulting file will contain the left boundary of the
interval used to compute the root in the first column.
Then in the next three columns will be the point,
the function value and the derivative at which the root was found.
The last column will contain the number of iterations used to find the root.
If no root was found the last four columns will be set to <tt>NAN</tt>.
The singular values and their derivatives are computed using the Arnoldi algorithm.
The user will be updated through the command line about the
progress of the algorithm if <tt>-DCMDL</tt> is set.

#### <tt>roots_newton_square_rsvd</tt>
This target builds a script that computes minimas in the smallest singular value of the Galerkin BEM approximated solutions operator for the sedond-kind direct BIEs of the Helmholtz transmission problem using the Newton-Raphson method. The scatterer is set to be a square. The results are written to the <tt>data</tt> directory. The script can be run as follows:
~~~
/path/to/library/bin/roots_newton_square_rsvd <half side length of square>
    <refraction inside> <refraction outside> <initial wavenumber>
    <#grid points for root search> <#panels> <quadrature order>
    <accuracy> <#subspace iterations>
~~~
The resulting file will contain the local minima in a single column. The singular values are computed using the randomized SVD algorithm with the specified number of subspace iterations. The user will be updated through the command line about the progress of the algorithm if <tt>-DCMDL</tt> is set.

#### <tt>roots_newton_polygon_rsvd</tt>
This target builds a script that computes minimas in the smallest singular value of the Galerkin BEM approximated solutions operator for the sedond-kind direct BIEs of the Helmholtz transmission problem using the Newton-Raphson method. The scatterer is a polygon read from disk (see <tt>scatterer.hpp</tt> for a description of the input file syntax). The results are written to the <tt>data</tt> directory. The script can be run as follows:
~~~
/path/to/library/bin/roots_newton_square_rsvd <scatterer filename>
    <refraction inside> <refraction outside> <initial wavenumber> 
    <#grid points for root search> <#panels> <quadrature order> 
    <accuracy> <#subspace iterations>
~~~
The resulting file will contain the local minima in a single column. The singular values are computed using the randomized SVD algorithm with the specified number of subspace iterations. The user will be updated through the command line about the progress of the algorithm if <tt>-DCMDL</tt> is set.

#### <tt>roots_seq_circle_arnoldi</tt>
This target builds a script that computes minimas in the smallest singular value of the
Galerkin BEM approximated solutions operator for the second-kind direct BIEs of the Helmholtz
transmission problem using the algorithm described in Listing 1 of
https://www.sam.math.ethz.ch/sam_reports/reports_final/reports2022/2022-38.pdf.
The scatterer is set to be a circle.
The results are written to disk.
The script can be run as follows:
~~~
/path/to/library/bin/roots_brent_circle <radius of circle> 
    <refraction inside> <refraction outside> <initial wavenumber> 
    <#grid points for root search> <#panels> 
    <order of quadrature rule> <outputfile>
~~~
The resulting file will contain the left boundary of the
interval used to compute the root in the first column.
Then in the next three columns will be the point, the
function value and the derivative at which the root was found.
The last column will contain the number of iterations used to find the root.
If no root was found the last four columns will be set to <tt>NAN</tt>.
The singular values and their derivatives are computed using the Arnoldi algorithm.
The user will be updated through the command line about the
progress of the algorithm if <tt>-DCMDL</tt> is set.

#### <tt>roots_seq_square_arnoldi</tt>
This target builds a script that computes minimas in the smallest singular value of the
Galerkin BEM approximated solutions operator for the sedond-kind direct BIEs of the Helmholtz
transmission problem using the algorithm described in Listing 1 of
https://www.sam.math.ethz.ch/sam_reports/reports_final/reports2022/2022-38.pdf.
The scatterer is set to be a square.
The results are written to disk.
The script can be run as follows:
~~~
/path/to/library/bin/roots_brent_circle <half side length of square> 
    <refraction inside> <refraction outside> <initial wavenumber> 
    <#grid points for root search> <#panels> 
    <order of quadrature rule> <outputfile>
~~~
The resulting file will contain the left boundary of the
interval used to compute the root in the first column.
Then in the next three columns will be the point, the
function value and the derivative at which the root was found.
The last column will contain the number of iterations used to find the root.
If no root was found the last four columns will be set to <tt>NAN</tt>.
The singular values and their derivatives are computed using the Arnoldi algorithm.
The user will be updated through the command line about the
progress of the algorithm if <tt>-DCMDL</tt> is set.

### Computing singular values of the BEM approximated BIO

#### <tt>sv_circle</tt>
This target builds a script that computes the singular values of the Galerkin BEM approximated BIO for the second-kind direct BIEs of the Helmholtz transmission problem. The direct algorithm from Eigen is used to compute the sinuglar values.
The scatterer is set to be a circle.
The results are written to file.
The script can be run as follows:

~~~
/path/to/library/bin/sv_circle <radius of circle> <refraction inside>
    <refraction outside> <initial wavenumber> <final wavenumber>
    <#panels> <order of quadrature rule> <outputfile>
~~~

The resulting file will contain the value of <tt>k</tt> in the first column.
The rest of the columns contain the singular values from
smallest to largest for this <tt>k</tt>.
The user will be updated through the command line about the
progress of the algorithm if <tt>-DCMDL</tt> is set.

#### <tt>sv_circle_arnoldi</tt>
This target builds a script that computes the singular values
of the Galerkin BEM approximated BIO for the
second-kind direct BIEs of the Helmholtz
transmission problem. The arnoldi algorithm from arpack is used to compute the
sinuglar values. The scatterer is set to be a circle.
The results are written to file.
The script can be run as follows:

~~~
/path/to/library/bin/sv_circle <radius of circle> <refraction inside>
    <refraction outside> <initial wavenumber> <final wavenumber>
    <#points to evaluate> <scan complex wavenumbers> <#panels>
    <order of quadrature rule> <accuracy of Arnoldi algorithm>
~~~

The resulting file will contain the value of <tt>k</tt> in the first column.
The rest of the columns contain the singular values from
smallest to largest for this <tt>k</tt>.
The user will be updated through the command line about the
progress of the algorithm if <tt>-DCMDL</tt> is set.

#### <tt>sv_derivative_full</tt>
This target builds a script that computes the singular values and
their first two derivatives of the Galerkin BEM
approximated BIO for the second-kind direct BIEs of the Helmholtz
transmission problem.
Minimas in the smallest singular value are determined as well
by use of the Newton-Raphson method.
The scatterer is set to be a circle.
The results are written to file.
The script can be run as follows:
~~~
/path/to/library/bin/sv_derivative_full <radius of circle> 
    <refraction inside> <refraction outside> <initial wavenumber>
    <#panels> <order of quadrature rule> <outputfile>
~~~
The resulting file will contain the value of k in the first column.
Then the singular values and their first two derivatives at k will be listed from smallest to largest in the columns.
The singular values and their derivatives occupy three neighboring columns.
The final three columns will contain the value of the root, the value of the first derivative at the root and the number of iterations taken to find the root in the interval between the current and the next evaluation point.
If no root was found these three columns will contain <tt>NAN</tt>.
The user will be updated through the command line about the progress of the algorithm if <tt>-DCMDL</tt> is set.

#### <tt>sv_derivative_verification_circle</tt>
This target builds a script that verifies the derivatives of the singular
values of the Galerkin BEM approximated BIO for the
second-kind direct BIEs of the Helmholtz transmsission problem
using extrapolation.
The scatterer is set to be a circle.
The results are written to file.
The script can be run as follows:
~~~
/path/to/library/bin/sv_derivative_verification_circle 
    <radius of circle> <refraction inside> <refraction outside> 
    <initial wavenumber> <#panels> <order of quadrature rule> <outputfile>
~~~
The resulting file will contain the value of k in the first column.
The second column will contain the value of the smallest singular value at this k.
Then the columns will contain the computed derivative, the extrapolated derivative, the computed second derivative and the extrapolated second derivative in this order.
The user will be updated through the command line about the progress of the algorithm if <tt>-DCMDL</tt> is set.

#### <tt>sv_derivative_verification_square</tt>
This target builds a script that verifies the derivatives of the singular
values and their derivatives of the Galerkin BEM BIO for the
second-kind direct BIEs of the Helmholtz transmsission problem
using extrapolation.
The scatterer is set to be a square.
The results are written to file.
The script can be run as follows:
~~~
/path/to/library/bin/sv_derivative_verification_circle 
    <half side length of square> <refraction inside> 
    <refraction outside> <initial wavenumber> <#panels> 
    <order of quadrature rule> <outputfile>
~~~
The resulting file will contain the value of k in the first column.
The second column will contain the value of the smallest singular value at this k.
Then the columns will contain the computed derivative, the extrapolated derivative, the computed second derivative and the extrapolated second derivative in this order.
The user will be updated through the command line about the progress of the algorithm if <tt>-DCMDL</tt> is set.

#### <tt>sv_square</tt>
This target builds a script that computes the singular values
of the Galerkin BEM approximated BIO for the
second-kind direct BIEs of the Helmholtz
transmission problem. The direct algorithm from Eigen is used to compute the
sinuglar values.
The scatterer is set to be a square.
The results are written to file.
The script can be run as follows:

~~~
/path/to/library/bin/sv_square <half of side length of square>
    <refraction inside> <refraction outside> <initial wavenumber>
    <#panels> <order of quadrature rule> <outputfile>
~~~

The resulting file will contain the value of <tt>k</tt> in the first column.
The rest of the columns contain the singular values from
smallest to largest for this <tt>k</tt>.
The user will be updated through the command line about the
progress of the algorithm if <tt>-DCMDL</tt> is set.

#### <tt>sv_square_arnoldi</tt>
This target builds a script that computes the singular values
of the Galerkin BEM approximated BIO for the
second-kind direct BIEs of the Helmholtz
transmission problem. The arnoldi algorithm from arpack is used to compute the
sinuglar values. The scatterer is set to be a square.
The results are written to file.
The script can be run as follows:

~~~
/path/to/library/bin/sv_circle <radius of circle> <refraction inside>
     <refraction outside> <initial wavenumber> <final wavenumber>
     <#points to evaluate> <scan complex wavenumbers> <#panels>
     <order of quadrature rule> <accuracy of Arnoldi algorithm>
~~~

The resulting file will contain the value of <tt>k</tt> in the first column.
The rest of the columns contain the singular values from
smallest to largest for this <tt>k</tt>.
The user will be updated through the command line about the
progress of the algorithm if <tt>-DCMDL</tt> is set.

### Solving the Helmholtz transmission problem

#### <tt>transmission_problem_verification</tt>
This target builds a script that computes solutions to
the analytically solvable case of the Helmholtz transmission
problem where the scatterer is a circle using second-kind direct
BIEs and Galerkin BEM.
The results are written to file.
The script can be run as follows:
~~~
/path/to/library/bin/transmission_problem_verification <radius of circle>
    <#coeffs for series expansion of solution> <refraction inside>
    <refraction outside> <initial wavenumber>
    <order of quadrature rule> <outputfile>
~~~
This output file will contain two columns.
The first will contain the current panel size.
The second will contain the residual error in the euclidean norm of the computed FEM-space interpolation coefficients to the known FEM-space interpolation coefficients for the current number of panels.
The user will be updated through the command line about the progress of the algorithm if <tt>-DCMDL</tt> is set.

#### <tt>verify_solution_analytic</tt>
This target builds a script that solves the Helmholtz transmission problem in a circle and compares the result with the analytic solution. The computed solution can be drawn by running the gnuplot script which is written to the <tt>data</tt> directory. The script can be run as follows:
~~~
/path/to/library/bin/verify_solution_analytic <circle radius>
    <#coeffs for series expansion of solution> 
    <refraction inside> <refraction outside> <wavenumber>
    <#panels> <quadrature order> <grid size>
~~~
The user will be updated through the command line about the
progress of the algorithm if <tt>CMDL</tt> is set.

#### <tt>plot_solution_polygon</tt>

This target builds a script that solves the Helmholtz transmission problem and outputs a gnuplot file which plots the solution. The scatterer and incoming wave are read from disk. The results are written to the <tt>data</tt> directory. The script can be run as follows:
~~~
/path/to/library/bin/plot_solution_polygon
    <scatterer file> <incoming wave file>
    <refraction inside> <refraction outside>
    <min wavenumber> <max wavenumber>
    <#panels> <quadrature order> <grid size>
    <lower left x> <lower left y> <upper right x> <upper right x> 
    <mode> <intensity>
~~~
* The first two input arguments are paths to text files. For scatterer/incoming wave file syntax refer to the corresponding header files <tt>scatterer.hpp</tt> and <tt>incoming.hpp</tt>. See the <tt>data</tt> directory for examples of input files.
* <tt>#panels</tt> is the desired number of panels or a fraction of the shortest side of the scatterer specifying the base panel length. Panels are generated automatically in a way that the panel length variance is minimal and the actual number of panels is close to the desired number.
* <tt>quadrature order</tt> refers to computing the Green indentity integrals when lifting the solution from traces.
* <tt>grid size</tt> is the number of points sampled at each side of the rectangular drawing area, which is specified by its lower left and upper right corners (the next four input arguments).
* <tt>mode</tt> is an integer from 0 to 5, where 0 (3), 1 (4) and 2 (5) specify the default drawing mode (still images), animation mode and amplitude plotting mode, respectively. The incoming wave is added to the scattered wave if <tt>mode</tt> is greater than 2.
* <tt>intensity</tt> is a positive real number which controls color intensity of the plot (1.0 is the default intensity).

If <tt>mode</tt> = 1 or 4, then the gnuplot script produces several images in the <tt>data/img</tt> directory. The following commandline produces an animation out of these frames (note that it should be run from the <tt>data</tt> directory):
~~~
convert -delay 4 -loop 0 img/file_plot_solution_square_XXXX_*.png
    output.gif
~~~
The user will be updated through the command line about the progress of the algorithm if <tt>-DCMDL</tt> is set.

## Acknowledgements
This software is a part of the project [Randomized low rank algorithms and applications to parameter dependent problems](https://www.croris.hr/projekti/projekt/4409) supported by the [Croatian Science Foundation](https://hrzz.hr/en/) (HRZZ). <img align='top' src='figures/HRZZ-eng.jpg' width='100'>

The development was supervised by Luka Grubišić.

## References
<a id="1">[1]</a> 
L. Grubišić, R. Hiptmair, and D. Renner,
"_Detecting Near Resonances in Acoustic Scattering_,"
Journal of Scientific Computing, vol. 96, no. 3, Sep. 2023,
doi: 10.1007/s10915-023-02284-5

