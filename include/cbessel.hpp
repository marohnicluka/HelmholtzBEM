/**
 * @file cbessel.hpp
 *
 * @brief This is a standalone C++ library for computing Bessel functions
 * of complex argument and real order, intended as a replacement for
 * @p complex_bessel library which uses the Fortran code by Donald E. Amos.
 *
 * The same theoretical basis is used, as explained in the two papers by
 * Amos (1983). This code includes optimizations for order 0 and 1, making it
 * slightly faster than the Fortran library. The routines below will throw
 * exceptions if @p CBESSEL_EXCEPT is defined (by default it is not defined,
 * the respective line in the source file is commented out).
 *
 * The following exceptions are thrown:
 *  - @p std::overflow_error in the case of overflow
 *  - @p std::underflow_error in the case of underflow (the result is 0)
 *  - @p std::invalid_argument in the case of undefined result (NaN)
 *  - @p std::length_error in the case of convergence failure
 *  - @p std::domain_error in the case of invalid parameter value
 *
 * Throwing exceptions is more informative but slightly slower.
 * Convergence problems should not happen in practice.
 *
 * (c) 2023-24 Luka Marohnić
 */

#ifndef CBESSEL_HPP
#define CBESSEL_HPP

#include <complex>
#include <Eigen/Dense>

using namespace std;

#define CBESSEL_MAXITER 1000
//#define CBESSEL_EXCEPT 1
//#define COMPUTE_BESSEL_STATS 1

namespace complex_bessel {

    /**
     * @brief Typedef for @p Real precision real type.
     */
    typedef double Real;
    typedef Eigen::Array<Real,-1,-1> RealArray;
    typedef Eigen::Array<Real,-1,1> RealVector;

    /**
     * @brief Typedef for @p Real precision complex type.
     */
    typedef complex<Real> Cplx;
    typedef Eigen::Array<Cplx,-1,-1> ComplexArray;

    /**
     * @brief Workspace for Olver expansions for large order.
     */
    typedef struct olver_data {
        bool is_valid;
        Cplx S1,S2,xi,phi; // xi = 2/3*zeta^(3/2)
        olver_data();
    } OlverData;

    /**
     * @brief Evaluate the modified Bessel function of
     * the first kind. If @p scaled is @p true, the result
     * will be scaled by @f$\exp(-\Re(z))@f$.
     *
     * @param v real order
     * @param z complex argument
     * @param scaled whether to scale the result (default: @p false)
     */
    Cplx BesselI (Real v,const Cplx &z,bool scaled=false);

    /**
     * @brief Compute the @p n th derivative of the modified
     * Bessel function of the first kind.
     *
     * @param v real order
     * @param z complex argument
     * @param n nonnegative integer (default: 1)
     */
    Cplx BesselIp (Real v,const Cplx &z,int n=1);

    /**
     * @brief Evaluate the Bessel function of the first kind.
     * If @p scaled is @p true, the result is scaled by
     * @f$\exp(-\Im(z))@f$.
     *
     * @param v real order
     * @param z complex argument
     * @param scaled whether to scale the result
     */
    Cplx BesselJ (Real v,const Cplx &z,bool scaled=false);

    /**
     * @brief Evaluate the @p n th derivative of the
     * Bessel function of the first kind.
     *
     * @param v real order
     * @param z complex argument
     * @param n nonnegative integer (default: 1)
     */
    Cplx BesselJp (Real v,const Cplx &z,int n=1);

    /**
     * @brief Evaluate the modified Bessel function of
     * the second kind. If @p scaled is @p true, the result
     * is scaled by @f$\exp(z)@f$.
     *
     * @param v real order
     * @param z complex argument
     * @param scaled whether to scale the result (default: @p false)
     */
    Cplx BesselK (Real v,const Cplx &z,bool scaled=false);

    /**
     * @brief Evaluate the @p n th derivative of
     * the modified Bessel function of the second kind.
     *
     * @param v real order
     * @param z complex argument
     * @param n nonnegative integer (default: 1)
     */
    Cplx BesselKp (Real v,const Cplx &z,int n=1);

    /**
     * @brief Evaluate the Bessel function of the second kind.
     * If @p scaled is @p true, the result is scaled by
     * @f$\exp(-\Im(z))@f$.
     *
     * @param v real order
     * @param z complex argument
     * @param scaled whether to scale the result (default: @p false)
     */
    Cplx BesselY (Real v,const Cplx &z,bool scaled=false);

    /**
     * @brief Evaluate the @p n th derivative of the
     * Bessel function of the second kind.
     *
     * @param v real order
     * @param z complex argument
     * @param n nonnegative integer (default: 1)
     */
    Cplx BesselYp (Real v,const Cplx &z,int n=1);

    /**
     * @brief Evaluate the Hankel function of the first kind.
     * If @p scaled is @p true, the value is scaled by
     * @f$\exp(-\mahrm{i} z)@f$.
     *
     * @param v real order
     * @param z complex argument
     * @param scaled whether to scale the result (default: @p false)
     */
    Cplx HankelH1 (Real v,const Cplx &z,bool scaled=false);

    /**
     * @brief Evaluate the @p n th derivative of the
     * Hankel function of the first kind.
     *
     * @param v real order
     * @param z complex argument
     * @param n nonnegative integer (default: 1)
     */
    Cplx HankelH1p (Real v,const Cplx &z,int n=1);

    /**
     * @brief Evaulate the Hankel function of the second kind.
     * If @p scaled is @p true, the value is scaled by
     * @f$\exp(\mathrm{i} z)@f$.
     *
     * @param v real order
     * @param z complex argument
     * @param scaled whether to scale the result (default: @p false)
     */
    Cplx HankelH2 (Real v,const Cplx &z,bool scaled=false);

    /**
     * @brief Evaluate the nth derivative of the
     * Hankel function of the second kind.
     *
     * @param v real order
     * @param z complex argument
     * @param n positive integer (the order of derivative)
     */
    Cplx HankelH2p (Real v,const Cplx &z,int n=1);

    /**
     * @brief Evaluate the Airy function of the first kind.
     * If @p scaled is @p true, the result is scaled by
     * @f$\exp(\frac23 z^{2/3})@f$.
     *
     * @param z complex argument
     * @param scaled whether to scale the result (default: @p false)
     */
    Cplx AiryAi (const Cplx &z,bool scaled=false);

    /**
     * @brief Evaluate the first derivative of the Airy function
     * of the first kind. If @p scaled is @p true, the result
     * is scaled by @f$\exp(\frac23 z^{2/3})@f$.
     *
     * @param z complex argument
     * @param scaled whether to scale the result (default: @p false)
     */
    Cplx AiryAip (const Cplx &z,bool scaled=false);

    /**
     * @brief Evaluate the Airy function of the second kind.
     * If @p scaled is @p true, the result is scaled by
     * @f$\exp(-\left|\Re\left(\frac23 z^{2/3}\right)|))@f$.
     *
     * @param z complex argument
     * @param scaled whether to scale the result (default: @p false)
     */
    Cplx AiryBi (const Cplx &z,bool scaled=false);

    /**
     * @brief Evaluate the first derivative of the
     * Airy function of the second kind.
     * If @p scaled is @p true, the result is scaled by
     * @f$\exp(-\left|\Re\left(\frac23 z^{2/3}\right)|))@f$.
     *
     * @param z complex argument
     * @param scaled whether to scale the result (default: @p false)
     */
    Cplx AiryBip(const Cplx &z,bool scaled=false);

    /**
     * @brief This class performs coupled vectorized computation
     * of Bessel and Hankel functions of order 0 and 1 when the
     * argument is a (array of) positive real number(s).
     * Asymptotic formulas for very small/large argument are used.
     */
    class Hankel1Real01 {
        size_t _m,_n,n_limit,n_small,n_normal,n_large,n_elem,n_zero;
        RealArray ws[13],m_0;
        RealArray x_limit,x_small,x_normal,x_large;
        RealArray j0_limit,j0_small,j0_normal,j0_large,j1_limit,j1_small,j1_normal,j1_large;
        RealArray y0_limit,y0_small,y0_normal,y0_large,y1_limit,y1_small,y1_normal,y1_large;
        std::vector<std::pair<size_t,size_t> > ind_limit,ind_small,ind_normal,ind_large;
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
        void compute_large(const RealArray &x,RealArray &j_0,RealArray &j_1,RealArray &y_0,RealArray &y_1);

    public:
        Hankel1Real01() { }
        Hankel1Real01(size_t m,size_t n);
        void initialize(size_t m,size_t n);

        /**
        * Compute @f$H^{(1)}_0(x)@f$ and @f$H^{(1)}_1(x)@f$
        * for real @p x > 0 (vectorized).
        *
        * @param x array of positive real numbers
        * @param h1_0 array of computed values of @f$H^{(1)}_0@f$.
        * @param h1_1 array of computed values of @f$H^{(1)}_1@f$.
        */
        void h1_01(const RealArray &x,ComplexArray &h1_0,ComplexArray &h1_1);

        /**
        * Compute @f$J_0(x)@f$, @f$Y_0(x)@f$, @f$J_1(x)@f$ and @f$Y_1(x)@f$
        * for real @p x > 0 (vectorized).
        *
        * @param x array of positive real numbers
        * @param j_0 array of computed values of @f$J_0@f$
        * @param y_0 array of computed values of @f$Y_0@f$
        * @param j_1 array of computed values of @f$J_1@f$
        * @param y_1 array of computed values of @f$Y_1@f$
        */
        void jy_01(const RealArray &x,RealArray &j_0,RealArray &y_0,RealArray &j_1,RealArray &y_1);
    };

    /**
     * @brief Evaluate @f$\operatorname{HankelH1}_0@f$ and
     * @f$\operatorname{HankelH1}_1@f$ at complex @p z.
     * The results are stored to @p h0 and @p h1.
     *
     * @param x complex number
     * @param h0 output value for order 0
     * @param h1 output value for order 1
     * @param realpos is the argument a positive real (default: @p false)
     */
    void H1_01(Cplx x,Cplx &h0,Cplx &h1,bool realpos=false);

    /**
     * @brief Print Bessel function computation profiling results.
     */
    void print_bessel_stats();

}

#endif // CBESSEL_HPP
