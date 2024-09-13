/**
 * \file cbessel.hpp
 *
 * \brief This is a C++ library for Bessel functions of complex argument and
 * real order, intended as a replacement for complex_bessel library which
 * uses the Fortran code by Donald E. Amos.
 *
 * The same theoretical basis is used, as explained in the two papers by
 * Amos (1983). This code includes optimizations for order 0 and 1, making it
 * slightly faster than the Fortran library. The routines below will throw
 * exceptions if CBESSEL_EXCEPT is defined (by default it is not defined,
 * the respective line in the CPP file is commented out).
 *
 * The following exceptions are thrown:
 *  - std::overflow_error in the case of overflow
 *  - std::underflow_error in the case of underflow (the result is 0)
 *  - std::invalid_argument in the case of undefined result (std::numeric_limits<Real>::quiet_NaN())
 *  - std::length_error in the case of convergence failure
 *  - std::domain_error in the case of invalid parameter value
 *
 * Throwing exceptions is more informative but slightly slower.
 * Convergence problems should not happen in practice.
 *
 * (c) 2023 Luka Marohnić
 */

#ifndef CBESSEL_HPP
#define CBESSEL_HPP

#include <complex>
#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <boost/math/interpolators/quintic_hermite.hpp>
#include <boost/circular_buffer.hpp>

using namespace std;

#define CBESSEL_MAXITER 1000
//#define CBESSEL_EXCEPT 1
//#define COMPUTE_BESSEL_STATS 1

namespace complex_bessel {

    /**
     * typedef for Real precision real type
     */
    typedef double Real;
    typedef Eigen::Array<Real,-1,-1> RealArray;
    typedef Eigen::Array<Real,-1,1> RealVector;

    /**
     * typedef for Real precision complex type
     */
    typedef complex<Real> Cplx;
    typedef Eigen::Array<Cplx,-1,-1> ComplexArray;

    /**
     * workspace for Olver expansions for large order
     */
    typedef struct olver_data {
        bool is_valid;
        Cplx S1,S2,xi,phi; // xi = 2/3*zeta^(3/2)
        olver_data();
    } OlverData;

    /**
     * This function computes the value of the modified
     * Bessel function of the first kind.
     *
     * @param v real order
     * @param z complex argument
     * @param scaled if true, the value is scaled by exp(-real(z))
     */
    Cplx BesselI (Real v,const Cplx &z,bool scaled=false);

    /**
     * This function computes the nth derivative of the modified
     * Bessel function of the first kind.
     *
     * @param v real order
     * @param z complex argument
     * @param n positive integer (the order of derivative)
     */
    Cplx BesselIp (Real v,const Cplx &z,int n=1);

    /**
     * This function computes the value of the
     * Bessel function of the first kind.
     *
     * @param v real order
     * @param z complex argument
     * @param scaled if true, the value is scaled by exp(-imag(z))
     */
    Cplx BesselJ (Real v,const Cplx &z,bool scaled=false);

    /**
     * This function computes the nth derivative of the
     * Bessel function of the first kind.
     *
     * @param v real order
     * @param z complex argument
     * @param n positive integer (the order of derivative)
     */
    Cplx BesselJp (Real v,const Cplx &z,int n=1);

    /**
     * This function computes the value of the modified
     * Bessel function of the second kind.
     *
     * @param v real order
     * @param z complex argument
     * @param scaled if true, the value is scaled by exp(z)
     */
    Cplx BesselK (Real v,const Cplx &z,bool scaled=false);

    /**
     * This function computes the nth derivative of the modified
     * Bessel function of the second kind.
     *
     * @param v real order
     * @param z complex argument
     * @param n positive integer (the order of derivative)
     */
    Cplx BesselKp (Real v,const Cplx &z,int n=1);

    /**
     * This function computes the value of the
     * Bessel function of the second kind.
     *
     * @param v real order
     * @param z complex argument
     * @param scaled if true, the value is scaled by exp(-imag(z))
     */
    Cplx BesselY (Real v,const Cplx &z,bool scaled=false);

    /**
     * This function computes the nth derivative of the
     * Bessel function of the second kind.
     *
     * @param v real order
     * @param z complex argument
     * @param n positive integer (the order of derivative)
     */
    Cplx BesselYp (Real v,const Cplx &z,int n=1);

    /**
     * This function computes the value of the
     * Hankel function of the first kind.
     *
     * @param v real order
     * @param z complex argument
     * @param scaled if true, the value is scaled by exp(-i*z)
     */
    Cplx HankelH1 (Real v,const Cplx &z,bool scaled=false);

    /**
     * This function computes the nth derivative of the
     * Hankel function of the first kind.
     *
     * @param v real order
     * @param z complex argument
     * @param n positive integer (the order of derivative)
     */
    Cplx HankelH1p (Real v,const Cplx &z,int n=1);

    /**
     * This function computes the value of the
     * Hankel function of the second kind.
     *
     * @param v real order
     * @param z complex argument
     * @param scaled if true, the value is scaled by exp(i*z)
     */
    Cplx HankelH2 (Real v,const Cplx &z,bool scaled=false);

    /**
     * This function computes the nth derivative of the
     * Hankel function of the second kind.
     *
     * @param v real order
     * @param z complex argument
     * @param n positive integer (the order of derivative)
     */
    Cplx HankelH2p (Real v,const Cplx &z,int n=1);

    /**
     * This function computes the value of the
     * Airy function of the first kind.
     *
     * @param z complex argument
     * @param scaled if true, the result is scaled by exp(2/3*z*sqrt(z))
     */
    Cplx AiryAi (const Cplx &z,bool scaled=false);

    /**
     * This function computes the first derivative of the
     * Airy function of the first kind.
     *
     * @param z complex argument
     * @param scaled if true, the result is scaled by exp(2/3*z*sqrt(z))
     */
    Cplx AiryAip (const Cplx &z,bool scaled=false);

    /**
     * This function computes the value of the
     * Airy function of the second kind.
     *
     * @param z complex argument
     * @param scaled if true, the result is scaled by exp(-abs(real(2/3*z*sqrt(z))))
     */
    Cplx AiryBi (const Cplx &z,bool scaled=false);

    /**
     * This function computes the first derivative of the
     * Airy function of the second kind.
     *
     * @param z complex argument
     * @param scaled if true, the result is scaled by exp(-abs(real(2/3*z*sqrt(z))))
     */
    Cplx AiryBip(const Cplx &z,bool scaled=false);

    /**
     * Vectorized computation of J0, J1, Y0, Y1, H0, H1
     */
    class Hankel1Real01 {
        size_t _m,_n,n_limit,n_small,n_normal,n_elem,n_zero;
        RealArray ws[13],m_0;
        RealArray x_limit,x_small,x_normal;
        RealArray j0_limit,j0_small,j0_normal,j1_limit,j1_small,j1_normal;
        RealArray y0_limit,y0_small,y0_normal,y1_limit,y1_small,y1_normal;
        std::vector<std::pair<size_t,size_t> > ind_limit,ind_small,ind_normal;
        Eigen::Array<bool,-1,-1> tmsk1,tmsk2;
        /**
         * Compute normalized sine function x->sin(pi*x).
         * Result is stored in res. Square of x must be
         * provided as xx.
         */
        void nsin(const RealArray &x,RealArray &w,RealArray &res);
        /**
         * Compute sin(x) and cos(x) and store then in s and c.
         * The workspace needs to be provided as xt and xx.
         */
        void sin_cos(const RealArray &x,RealArray &xt,RealArray &xx,RealArray &s,RealArray &c);
        void compute_limit(const RealArray &x,RealArray &j_0,RealArray &j_1,RealArray &y_0,RealArray &y_1);
        void compute_small(const RealArray &x,RealArray &j_0,RealArray &j_1,RealArray &y_0,RealArray &y_1);
        void compute_normal(const RealArray &x,RealArray &j_0,RealArray &j_1,RealArray &y_0,RealArray &y_1);
    public:
        Hankel1Real01() { }
        Hankel1Real01(size_t m,size_t n);
        void initialize(size_t m,size_t n);
        /**
        * Compute H^1_0(x) and H^1_1(x) for real x>0 (vectorized).
        *
        * @param x real argument (x>0)
        * @param h1_0 computed values (need to be initialized to the size of x)
        * @param h1_1 computed values (need to be initialized to the size of x)
        */
        void h1_01(const RealArray &x,ComplexArray &h1_0,ComplexArray &h1_1);
        /**
        * Compute i*H^1_0(x) and i*H^1_1(x) for real x>0 (vectorized).
        *
        * @param x real argument (x>0)
        * @param ih1_0 computed values (need to be initialized to the size of x)
        * @param ih1_1 computed values (need to be initialized to the size of x)
        */
        void ih1_01(const RealArray &x,ComplexArray &ih1_0,ComplexArray &ih1_1);
        /**
        * Compute j0(x), y0(x), j1(x) and y1(x) for real x>0 (vectorized).
        *
        * @param x real argument (x>0)
        */
        void jy_01(const RealArray &x,RealArray &j_0,RealArray &y_0,RealArray &j_1,RealArray &y_1);
    };

    void print_bessel_stats();

    class Hankel1Real01Interp {
        Real b;
        std::map<Real,std::tuple<Real,Real,Real,Real> > data;
        std::vector<Real> x_j0,x_y0,x_j1,x_y1,y_j0,y_y0,y_j1,y_y1,dy_j0,dy_y0,dy_j1,dy_y1,d2y_j0,d2y_j1,d2y_y0,d2y_y1;
        boost::math::interpolators::quintic_hermite<std::vector<Real> >
            *spline_j0=NULL,
            *spline_j1=NULL,
            *spline_y0=NULL,
            *spline_y1=NULL;
        void make_splines();
    public:
        Hankel1Real01Interp(Real x1,Real tol);
        ~Hankel1Real01Interp();
        void compute(Real x,Cplx &h0,Cplx &h1);
        void compute(const RealArray &x,ComplexArray &h0,ComplexArray &h1);
    };

    /**
     * Compute j0(x), y0(x), j1(x) and y1(x) at the same time (faster).
     *
     * @param x real argument (x>0)
     */
    void JY_01(Real x,Real &j_0,Real &y_0,Real &j_1,Real &y_1);

    /**
     * Compute H^1_0(x) and H^1_1(x) at the same time (faster).
     *
     * @param x real argument (x>0)
     */
    void H1_01(Real x, Cplx &h0, Cplx &h1);
    void iH1_01(Real x, Cplx &h0, Cplx &h1);

}

#endif // CBESSEL_HPP
