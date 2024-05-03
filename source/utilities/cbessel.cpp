/**
 * \file cbessel.cpp
 *
 * \brief This file contains the implementation of Bessel functions
 * of real order and complex argument.
 *
 * If CBESSEL_EXCEPT is defined, then routines will throw exceptions.
 *
 * (c) 2023 Luka Marohnić
 */

#include "cbessel.hpp"
#include <limits>
#include <vector>
#include <numeric>
#include <cmath>

//#define USE_STD_BESSEL
#define HI(x) *(1+(int*)&x)

namespace complex_bessel {

    /* utilities */
    Real sign1(Real x) {
        return x>=0.0?1.0:-1.0;
    }
    bool undef(const Cplx &z) {
        return isnan(real(z)) || isnan(imag(z));
    }
    bool zero(Real x) {
        int c=fpclassify(x);
        return c==FP_ZERO || c==FP_SUBNORMAL;
    }
    bool inf(const Cplx &z) {
        return isinf(real(z)) || isinf(imag(z));
    }
    bool zero(const Cplx &z) {
        return zero(real(z)) && zero(imag(z));
    }
    Real v_L(Real eps) {
        return -8.0-6.0*log10(eps);
    }

    /* Olver expansion routines for large order */
    olver_data::olver_data() {
        is_valid=false;
        S1=S2=0.0;
    }
    Cplx U(int k,const Cplx &t2) {
        size_t k0=(k*(k+1))/2;
        return accumulate(U_C+k0,U_C+k0+k+1,Cplx(0,0),[&t2](Cplx &a,Real c) { return a*t2+c; });
    }
    void olver(Real v,const Cplx &z,Real s,OlverData &data) {
        Cplx zv=-s*z*i/v,w2=1.0-zv*zv,w=sqrt(w2);
        Real iv2=1.0/(v*v),v13=pow(v,-f13);
        int j,k;
        if (abs(w2)>0.25) {
            data.xi=log((1.0+w)/zv)-w;
            data.phi=pow(12.0*data.xi,f16)/sqrt(w)*v13;
            Cplx t=1.0/w,t2=1.0/w2,tk,ix=f23/data.xi,A,B;
            for (k=6;k-->0;) {
                A=0.0,B=0.0,tk=w;
                for (j=2*(k+1);j-->0;) {
                    if (j<=2*k)
                        A=A*ix-mu_coef[j]*U(2*k-j,t2)*tk;
                    B=B*ix-lambda_coef[j]*U(2*k-j+1,t2)*(tk*=t);
                }
                data.S1=data.S1*iv2+A;
                data.S2=data.S2*iv2+B;
            }
            data.S2*=pow(ix,f13);
            Real u=sign1(imag(data.xi));
            if (s*u<0.0) {
                data.S2*=exp(u*M_PI*f23*i);
                data.phi*=exp(-u*M_PI*f13*i);
            }
        } else {
            Cplx a0(0.0);
            data.xi=w*accumulate(xi_coef,xi_coef+27,a0,[&w2](Cplx &a,Real c) { return a*w2+c; });
            data.phi=v13*accumulate(g_coef,g_coef+27,a0,[&w2](Cplx &a,Real c) { return a*w2+c; });
            const Real *ac=a_coef,*bc=b_coef;
            for (k=0;k<6;++k,ac+=27,bc+=27) {
                data.S1=data.S1*iv2+(k==5?1.0:accumulate(ac,ac+27,a0,[&w2](Cplx &a,Real c) { return a*w2+c; }));
                data.S2=data.S2*iv2+accumulate(bc,bc+27,a0,[&w2](Cplx &a,Real c) { return a*w2+c; });
            }
        }
        data.is_valid=true;
    }
    int airy_olver(Real v,const Cplx &xi,Real s,Cplx &ai,Cplx &aip,bool &corr,bool rot,bool scaled);

    /* workhorse routines */
    Cplx K_in(Real v,const Cplx &z,bool scaled,Cplx *K_val_1=nullptr,OlverData *data=nullptr);
    Cplx K0_in(const Cplx &z,bool scaled);
    Cplx K1_in(const Cplx &z,bool scaled);
    Cplx I_in(Real v,const Cplx &z,bool scaled,Cplx *K_val=nullptr,OlverData *data=nullptr) {
        if (undef(z) || (inf(z) && isinf(v)))
#ifdef CBESSEL_EXCEPT
            throw ia_err;
#else
            return NAN;
#endif
        if (isinf(v))
#ifdef CBESSEL_EXCEPT
            throw uf_err;
#else
            return 0.0;
#endif
        if (inf(z))
#ifdef CBESSEL_EXCEPT
            throw of_err;
#else
            return infty;
#endif
        if (zero(z))
            return zero(v)?1.0:0.0;
        if (v<0.0 && v==round(v))
            return I_in(-v,z,scaled);
        if (real(z)<0.0)
            return exp(v*M_PI*sign1(imag(z))*i)*I_in(v,-z,scaled);
        if (v<0.0) {
            OlverData data;
            Cplx Kv=NAN,Iv;
#ifdef CBESSEL_EXCEPT
            try {
#endif
                Iv=I_in(-v,z,scaled,&Kv,&data);
#ifdef CBESSEL_EXCEPT
            } catch (const underflow_error &e) { Iv=0.0; }
#endif
            if (scaled || undef(Kv))
#ifdef CBESSEL_EXCEPT
            try {
#endif
                Kv=K_in(-v,z,false,nullptr,&data)*(scaled?exp(-real(z)):1.0);
#ifdef CBESSEL_EXCEPT
            } catch (const underflow_error &e) { Kv=0.0; }
#endif
            return Iv-M_2_PI*sin(M_PI*v)*Kv;
        }
        if (zero(abs(2.0*arg(z)-M_PI)))
            return cyl_bessel_j(v,imag(z))*exp(i*v*M_PI_2);
        /* Basic case: Re(z)>=0, v>=0 */
        Cplx z2=2.0/z;
        Real R=abs(z),vL=v_L(eps);
        int maxiter=CBESSEL_MAXITER,j;
        if (R<=2.0*sqrt(v+1.0)) {
            Real tg=tgamma(1.0+v),scale=scaled?exp(-real(z)):1.0;
            Cplx A=1.0,z24=0.25*z*z,a=isinf(tg)?NAN:scale*pow(z2,-v)/tg,res=1.0;
            if (zero(a))
#ifdef CBESSEL_EXCEPT
                throw uf_err;
#else
                return 0.0;
#endif
            Real b=1.0,eps2=eps*0.5,R24=0.25*R*R,k=0.0;
            z24/=R24;
            while (b>k*eps2 && k++<maxiter)
                res+=(A*=z24)*(b*=R24/(k*(k+v)));
#ifdef CBESSEL_EXCEPT
            if (k>maxiter) throw cvg_err;
#endif
            if (undef(a)) {
                Real vfl=floor(v),vf=v-vfl;
                int vi=int(vfl);
                res*=scale*pow(z2,vf)/(vf*tgamma(vf));
                for (j=1;j<=vi;++j) res*=z2/(vf+j);
#ifdef CBESSEL_EXCEPT
                if (zero(res)) throw uf_err;
#endif
                return res;
            }
            return res*a;
        }
        if (v>vL) {
            Real theta=arg(z),s=sign1(theta);
            if (3.0*abs(theta)<M_PI) {
                Real iv=1.0/v;
                Cplx zv=z*iv,zv2=zv*zv,it2=1.0+zv2,it=sqrt(it2),zet=it-log((1.0+it)/zv);
                Cplx t=1.0/it,t2=1.0/it2,tk=pow(t2,6),res=0.0;
                for (int k=12;k-->1;)
                    res=res*iv+U(k,t2)*(tk*=it);
                Cplx ret=sqrt(t*M_1_PI*iv*0.5)*exp((scaled?-real(z):0.0)+v*zet)*(iv*res+1.0);
#ifdef CBESSEL_EXCEPT
                if (inf(ret)) throw of_err;
                if (zero(ret)) throw uf_err;
#endif
                return ret;
            }
            Cplx ai,aip,C,ret;
            OlverData od;
            if (data==nullptr) data=&od;
            olver(v,z,s,*data);
            bool corr;
            int e=airy_olver(v,data->xi,s,ai,aip,corr,false,scaled);
            C=exp(s*M_PI*v*i*0.5-(scaled?real(z)+(corr?v*data->xi:0.0):0.0))*data->phi;
#ifdef CBESSEL_EXCEPT
            if (e==1) { if (zero(C)) throw ia_err; throw of_err; }
            if (e==2) { if (inf(C)) throw ia_err; throw uf_err; }
            if (inf(C)) throw of_err;
            if (zero(C)) throw uf_err;
#endif
            ret=C*(ai*data->S1+aip*pow(v,-f43)*data->S2);
#ifdef CBESSEL_EXCEPT
            if (inf(ret)) throw of_err;
            if (zero(ret)) throw uf_err;
#endif
            return ret;
        }
        Real RL=1.2*(1.0-round(log10(eps)))+2.4,v2=v*v;
        if (R>RL && 2.0*R>v2) {
            Cplx f2=exp(sign1(imag(z))*(v+.5)*M_PI*i-z-(scaled?real(z):0.0));
            Cplx f1=exp(scaled?i*imag(z):z);
            if (inf(f1))
#ifdef CBESSEL_EXCEPT
                throw of_err;
#else
                return infty;
#endif
            Cplx sn=0.0,sp=1.0,a=1.0,z8=0.0625*z2;
            Real f=4.0*v2-1.0,z8a=0.125/R,b=1.0;
            int jmax=2+2*(int)floor(RL);
            for (j=1;j<jmax && b>eps;++j) {
                if (j>1) {
                    f=((f-8)*(j-1))/j;
                    b*=abs(f)*z8a;
                }
                a*=f*z8;
                if (j%2) sn+=a; else sp+=a;
            }
            Cplx ret=((sp-sn)*f1+(sp+sn)*f2)/sqrt(2.0*M_PI*z);
#ifdef CBESSEL_EXCEPT
            if (inf(ret)) throw of_err;
#endif
            return ret;
        }
        if (R>vL && 2.0*R<=v2) {
            v2=v,j=0;
            while (v2<=vL) ++j,++v2;
            Cplx i1=I_in(v2,z,scaled);
            if (inf(i1))
#ifdef CBESSEL_EXCEPT
                throw of_err;
#else
                return infty;
#endif
            Cplx i0=I_in(v2+1.0,z,scaled);
            for (;j-->0;v2--) {
                i0+=v2*z2*i1,swap(i0,i1);
#ifdef CBESSEL_EXCEPT
                if (inf(i1)) throw of_err;
#endif
            }
            return i1;
        }
        /* Miller algorithm -- Recurrence + Wronskian
         * (neither I nor K can overflow/underflow here) */
        int Ri=(int)floor(R),vi=(int)floor(v),nu=max(Ri,vi),k=nu+1;
        Cplx p0=1.0,p1=-(nu+1.0)*z2;
        Real T1=abs(p1)/eps,vf=v-floor(v);
        while (norm(p1)<=T1 && k++<maxiter)
            p0-=Real(k)*z2*p1,swap(p0,p1);
#ifdef CBESSEL_EXCEPT
        if (k>maxiter) throw cvg_err;
#endif
        Real k1R=(k+1.0)/R,betaN1=k1R+sqrt(k1R*k1R-1.0);
        Real rhoN1=min(betaN1,abs(p0/p1));
        T1*=rhoN1/(rhoN1*rhoN1-1.0);
        while (norm(p1)<T1 && k++<maxiter)
            p0-=Real(k)*z2*p1,swap(p0,p1);
#ifdef CBESSEL_EXCEPT
        if (k>maxiter) throw cvg_err;
#endif
        int M=k;
        bool with_y=R<=RL;
        if (with_y) {
            Real nu3R=(nu+3.0)/R,beta0=nu3R+sqrt(nu3R*nu3R-1.0),beta02=beta0*beta0;
            Real bnd=(2.0*beta02)/(eps*(beta02-1.0)*(beta0-1.0));
            while (abs(p1)<=(k+1)*(k+1)*bnd && k++<maxiter)
                p0-=Real(k)*z2*p1,swap(p0,p1);
#ifdef CBESSEL_EXCEPT
            if (k>maxiter) throw cvg_err;
#endif
            M=R>v?k:max(M+vi,k+Ri);
        }
        Cplx y0=0.0,y1=eps,yt;
        vector<Cplx> y;
        vector<Cplx>::iterator ybeg,it,jt;
        int jmax=int(M+(R>v?Ri-vi:0));
        if (with_y) {
            jmax+=vi;
            y.resize(jmax+1);
            ybeg=it=y.begin();
            *it=y1;
        }
        Real vfnuM=vf+nu+M,vf2=2.0*vf,scl;
        for (j=0;j<jmax;) {
            yt=y0+(vfnuM-j)*z2*y1;
            if (inf(yt)) {
                scl=1.0/max(abs(real(y1)),abs(imag(y1)));
                y1*=scl,y0*=scl;
                if (with_y) for (jt=ybeg;jt<=it;++jt) *jt*=scl;
                continue;
            }
            y0=yt; ++j;
            if (with_y) *(++it)=y0;
            swap(y0,y1);
        }
        if (with_y) {
            Real lambda=2.0; j=1;
            Cplx ip=accumulate(next(y.crbegin()),y.crend(),*it,[&](const Cplx &a,const Cplx &yk) {
                Cplx res=a+yk*lambda*(1.0+vf/j);
                lambda*=1.0+vf2/j++;
                return res;
            });
            return y[jmax-vi]*(scaled?exp(i*imag(z)):exp(z))*pow(z2,-vf)/(ip*tgamma(1.0+vf));
        }
        Cplx Kv_1=NAN,k1,k2,r=y0/y1,scale=scaled?exp(i*imag(z)):1.0;
        k1=K_in(v,z,scaled,&Kv_1);
        if (K_val!=nullptr) *K_val=k1;
        if (undef(Kv_1)) {
            k2=K_in(v+1.0,z,scaled);
            return scale/(z*(k2+r*k1));
        }
        return scale/(2.0*v*k1+z*(Kv_1+r*k1));
    }
    Cplx I0_in(const Cplx &z,bool scaled,Cplx *K_val=nullptr) {
        if (undef(z))
#ifdef CBESSEL_EXCEPT
            throw ia_err;
#else
            return NAN;
#endif
        if (inf(z))
#ifdef CBESSEL_EXCEPT
            throw of_err;
#else
            return infty;
#endif
        if (zero(z))
            return 1.0;
        if (real(z)<0.0)
            return I0_in(-z,scaled);
        if (zero(abs(2.0*arg(z)-M_PI)))
            return cyl_bessel_j(0.,imag(z));
        /* Basic case: Re(z)>=0, v>=0 */
        Cplx z2=2.0/z;
        Real R=abs(z);
        int maxiter=CBESSEL_MAXITER,j;
        if (R<=2.0) {
            Cplx A=1.0,z24=0.25*z*z,res=1.0;
            Real b=1.0,eps2=eps*0.5,R24=0.25*R*R,k=0.0;
            z24/=R24;
            while (b>k*eps2 && k++<maxiter)
                res+=(A*=z24)*(b*=R24/(k*k));
#ifdef CBESSEL_EXCEPT
            if (k>maxiter) throw cvg_err;
#endif
            return scaled?res*exp(-real(z)):res;
        }
        Real RL=1.2*(1.0-round(log10(eps)))+2.4;
        if (R>RL) {
            Cplx f2=exp(sign1(imag(z))*.5*M_PI*i-z-(scaled?real(z):0.0));
            Cplx f1=exp(scaled?i*imag(z):z);
            if (inf(f1))
#ifdef CBESSEL_EXCEPT
                throw of_err;
#else
                return infty;
#endif
            Cplx sn=0.0,sp=1.0,a=1.0,z8=0.0625*z2;
            Real f=-1.0,z8a=0.125/R,b=1.0;
            int jmax=2+2*(int)floor(RL);
            for (j=1;j<jmax && b>eps;++j) {
                if (j>1) {
                    f=((f-8)*(j-1))/j;
                    b*=abs(f)*z8a;
                }
                a*=f*z8;
                if (j%2) sn+=a; else sp+=a;
            }
            Cplx ret=((sp-sn)*f1+(sp+sn)*f2)/sqrt(2.0*M_PI*z);
#ifdef CBESSEL_EXCEPT
            if (inf(ret)) throw of_err;
#endif
            return ret;
        }
        /* Miller algorithm -- Recurrence + Wronskian
         * (neither I nor K can overflow/underflow here) */
        int Ri=(int)floor(R),nu=Ri,k=nu+1;
        Cplx p0=1.0,p1=-(nu+1.0)*z2;
        Real T1=abs(p1)/eps;
        while (norm(p1)<=T1 && k++<maxiter)
            p0-=Real(k)*z2*p1,swap(p0,p1);
#ifdef CBESSEL_EXCEPT
        if (k>maxiter) throw cvg_err;
#endif
        Real k1R=(k+1.0)/R,betaN1=k1R+sqrt(k1R*k1R-1.0);
        Real rhoN1=min(betaN1,abs(p0/p1));
        T1*=rhoN1/(rhoN1*rhoN1-1.0);
        while (norm(p1)<T1 && k++<maxiter)
            p0-=Real(k)*z2*p1,swap(p0,p1);
#ifdef CBESSEL_EXCEPT
        if (k>maxiter) throw cvg_err;
#endif
        int M=k;
        bool with_y=R<=RL;
        if (with_y) {
            Real nu3R=(nu+3.0)/R,beta0=nu3R+sqrt(nu3R*nu3R-1.0),beta02=beta0*beta0;
            Real bnd=(2.0*beta02)/(eps*(beta02-1.0)*(beta0-1.0));
            while (abs(p1)<=(k+1)*(k+1)*bnd && k++<maxiter)
                p0-=Real(k)*z2*p1,swap(p0,p1);
#ifdef CBESSEL_EXCEPT
            if (k>maxiter) throw cvg_err;
#endif
            M=k;
        }
        Cplx y0=0.0,y1=eps,yt,sum=0.0;
        int jmax=int(M+Ri);
        Real vfnuM=nu+M,scl;
        for (j=0;j<jmax;) {
            yt=y0+(vfnuM-j)*z2*y1;
            if (inf(yt)) {
                scl=1.0/max(abs(real(y1)),abs(imag(y1)));
                y1*=scl,y0*=scl;
                if (with_y) sum*=scl;
                continue;
            }
            y0=yt; ++j;
            if (j==jmax) sum*=2.0;
            if (with_y) sum+=y0;
            swap(y0,y1);
        }
        if (with_y)
            return y1*(scaled?exp(i*imag(z)):exp(z))/sum;
        Cplx k1,k2,r=y0/y1,scale=scaled?exp(i*imag(z)):1.0;
        k1=K0_in(z,scaled);
        k2=K1_in(z,scaled);
        if (K_val!=nullptr) *K_val=k1;
        return scale/(z*(k2+r*k1));
    }
    Cplx I1_in(const Cplx &z,bool scaled,Cplx *K_val=nullptr) {
        if (undef(z))
#ifdef CBESSEL_EXCEPT
            throw ia_err;
#else
            return NAN;
#endif
        if (inf(z))
#ifdef CBESSEL_EXCEPT
            throw of_err;
#else
            return infty;
#endif
        if (zero(z))
            return 0.0;
        if (real(z)<0.0)
            return -I1_in(-z,scaled);
        if (zero(abs(2.0*arg(z)-M_PI)))
            return cyl_bessel_j(1.0,imag(z))*i;
        /* Basic case: Re(z)>=0, v>=0 */
        Cplx z2=2.0/z;
        Real R=abs(z);
        int maxiter=CBESSEL_MAXITER,j;
        if (R<=2.0*M_SQRT2) {
            Cplx A=1.0,z24=0.25*z*z,res=1.0;
            Real b=1.0,eps2=eps*0.5,R24=0.25*R*R,k=0.0;
            z24/=R24;
            while (b>k*eps2 && k++<maxiter)
                res+=(A*=z24)*(b*=R24/(k*(k+1)));
#ifdef CBESSEL_EXCEPT
            if (k>maxiter) throw cvg_err;
#endif
            return res*(scaled?exp(-real(z)):1.0)*z*0.5;
        }
        Real RL=1.2*(1.0-round(log10(eps)))+2.4;
        if (R>RL) {
            Cplx f2=exp(sign1(imag(z))*1.5*M_PI*i-z-(scaled?real(z):0.0));
            Cplx f1=exp(scaled?i*imag(z):z);
            if (inf(f1))
#ifdef CBESSEL_EXCEPT
                throw of_err;
#else
                return infty;
#endif
            Cplx sn=0.0,sp=1.0,a=1.0,z8=0.0625*z2;
            Real f=3.0,z8a=0.125/R,b=1.0;
            int jmax=2+2*(int)floor(RL);
            for (j=1;j<jmax && b>eps;++j) {
                if (j>1) {
                    f=((f-8)*(j-1))/j;
                    b*=abs(f)*z8a;
                }
                a*=f*z8;
                if (j%2) sn+=a; else sp+=a;
            }
            Cplx ret=((sp-sn)*f1+(sp+sn)*f2)/sqrt(2.0*M_PI*z);
#ifdef CBESSEL_EXCEPT
            if (inf(ret)) throw of_err;
#endif
            return ret;
        }
        /* Miller algorithm -- Recurrence + Wronskian
         * (neither I nor K can overflow/underflow here) */
        int Ri=(int)floor(R),nu=max(Ri,1),k=nu+1;
        Cplx p0=1.0,p1=-(nu+1.0)*z2;
        Real T1=abs(p1)/eps;
        while (norm(p1)<=T1 && k++<maxiter)
            p0-=Real(k)*z2*p1,swap(p0,p1);
#ifdef CBESSEL_EXCEPT
        if (k>maxiter) throw cvg_err;
#endif
        Real k1R=(k+1.0)/R,betaN1=k1R+sqrt(k1R*k1R-1.0);
        Real rhoN1=min(betaN1,abs(p0/p1));
        T1*=rhoN1/(rhoN1*rhoN1-1.0);
        while (norm(p1)<T1 && k++<maxiter)
            p0-=Real(k)*z2*p1,swap(p0,p1);
#ifdef CBESSEL_EXCEPT
        if (k>maxiter) throw cvg_err;
#endif
        int M=k;
        bool with_y=R<=RL;
        if (with_y) {
            Real nu3R=(nu+3.0)/R,beta0=nu3R+sqrt(nu3R*nu3R-1.0),beta02=beta0*beta0;
            Real bnd=(2.0*beta02)/(eps*(beta02-1.0)*(beta0-1.0));
            while (k++<maxiter && abs(p1)<=k*k*bnd)
                p0-=Real(k)*z2*p1,swap(p0,p1);
#ifdef CBESSEL_EXCEPT
            if (k>maxiter) throw cvg_err;
#endif
            M=R>1.0?k:max(M+1,k+Ri);
        }
        Cplx y0=0.0,y1=eps,yt,sum=0.0;
        int jmax=int(M+(R>1.0?Ri-1:0))+(with_y?1:0);
        Real vfnuM=nu+M,scl;
        for (j=0;j<jmax;) {
            yt=y0+(vfnuM-j)*z2*y1;
            if (inf(yt)) {
                scl=1.0/max(abs(real(y1)),abs(imag(y1)));
                y1*=scl,y0*=scl;
                if (with_y) sum*=scl;
                continue;
            }
            y0=yt; ++j;
            if (with_y) {
                if (j==jmax) sum*=2.0;
                sum+=y0;
            }
            swap(y0,y1);
        }
        if (with_y)
            return y0*(scaled?exp(i*imag(z)):exp(z))/sum;
        Cplx k1,k2,r=y0/y1,scale=scaled?exp(i*imag(z)):1.0;
        k1=K1_in(z,scaled);
        k2=K_in(2.0,z,scaled);
        if (K_val!=nullptr) *K_val=k1;
        return scale/(z*(k2+r*k1));
    }
    Cplx K_in(Real v,const Cplx &z,bool scaled,Cplx *K_val_1,OlverData *data) {
        if (zero(z))
#ifdef CBESSEL_EXCEPT
            throw of_err;
#else
            return infty;
#endif
        if (inf(z) && isinf(v))
#ifdef CBESSEL_EXCEPT
            throw ia_err;
#else
            return NAN;
#endif
        if (inf(z))
#ifdef CBESSEL_EXCEPT
            throw uf_err;
#else
            return 0.0;
#endif
        if (isinf(v))
#ifdef CBESSEL_EXCEPT
            throw of_err;
#else
            return infty;
#endif
        if (v<0.0)
            return K_in(-v,z,scaled);
        if (real(z)<0.0) {
            Cplx Kv=NAN,Iv,si=sign1(imag(z))*M_PI*i;
            OlverData data;
#ifdef CBESSEL_EXCEPT
            try {
#endif
                Iv=I_in(v,-z,scaled,&Kv,&data)*(scaled?exp(i*imag(z)):1.0);
#ifdef CBESSEL_EXCEPT
            } catch (const underflow_error &e) { Iv=0.0; }
#endif
            if (scaled || undef(Kv))
#ifdef CBESSEL_EXCEPT
            try {
#endif
                Kv=K_in(v,-z,false,nullptr,&data)*(scaled?exp(z):1.0);
#ifdef CBESSEL_EXCEPT
            } catch (const underflow_error &e) { Kv=0.0; }
#endif
            Cplx ret=exp(-si*v)*Kv-si*Iv;
#ifdef CBESSEL_EXCEPT
            if (inf(ret)) throw of_err;
#endif
            return ret;
        }
        /* Basic case: Re(z)>=0, v>=0 */
        if (v>v_L(eps)) {
            Real theta=arg(z),s=sign1(theta);
            if (3.0*abs(theta)<M_PI) {
                Real iv=1.0/v;
                Cplx zv=z*iv,zv2=zv*zv,it2=1.0+zv2,it=sqrt(it2),zet=it-log((1.0+it)/zv);
                Cplx t=1.0/it,t2=1.0/it2,tk=pow(t2,6),res=0.0;
                for (int k=12;k-->1;)
                    res=res*iv+(k%2?-1.0:1.0)*U(k,t2)*(tk*=it);
                Cplx ret=sqrt(M_PI_2*t*iv)*exp((scaled?z:0.0)-v*zet)*(iv*res+1.0);
#ifdef CBESSEL_EXCEPT
                if (inf(ret)) throw of_err;
                if (zero(ret)) throw uf_err;
#endif
                return ret;
            }
            Cplx si=M_PI*s*i,ai,aip,C,ret;
            OlverData od;
            if (data==nullptr) data=&od;
            if (!data->is_valid)
                olver(v,z,s,*data);
            bool corr;
            int e=airy_olver(v,data->xi,s,ai,aip,corr,true,scaled);
            C=-si*exp(si*(f13-v*0.5)+(scaled?z+(corr?v*data->xi:0.0):0.0))*data->phi;
#ifdef CBESSEL_EXCEPT
            if (e==1) { if (zero(C)) throw ia_err; throw of_err; }
            if (e==2) { if (inf(C)) throw ia_err; throw uf_err; }
            if (inf(C)) throw of_err;
            if (zero(C)) throw uf_err;
#endif
            ret=C*(ai*data->S1+exp(-si*f23)*aip*pow(v,-f43)*data->S2);
#ifdef CBESSEL_EXCEPT
            if (inf(ret)) throw of_err;
            if (zero(ret)) throw uf_err;
#endif
            return ret;
        }
        Cplx k0,k1,z2=2.0/z;
        int jmax=0;
        while (v>0.5) jmax++,v--;
        Real R=abs(z),theta=arg(z);
        if (v==0.5) {
            k0=sqrt(M_PI_4*z2)*(scaled?1.0:exp(-z));
            k1=(z2*0.5+1.0)*k0;
        } else if (R<=2.0) {
            bool vz=zero(v);
            Real s=vz?1.0:M_PI*v/sin(M_PI*v),g2=1.0/tgamma(1.0+v),g1=1.0/(s*g2),gama1,gama2=(g1+g2)*0.5;
            if (vz)
                gama1=D[9];
            else if (abs(v)>0.1)
                gama1=(g1-g2)*0.5/v;
            else {
                Real v2=v*v;
                gama1=accumulate(D,D+10,0.0,[&v2](Real &a,Real d) { return a*v2+d; });
            }
            Cplx lz2=log(z2),mu=v*lz2,z2v=pow(z2,-v),z2d4=1.0/(z2*z2),scale=scaled?exp(z):1.0;
            Cplx p=0.5*scale/(g2*z2v),q=0.5*scale*z2v/g1,C=1.0;
            Cplx f=s*scale*(gama1*cosh(mu)+gama2*lz2*(vz?1.0:sinh(mu)/mu));
            Real A=0.5,a=0.25*R*R,k=1.0,kmv,kpv,ki,maxiter=CBESSEL_MAXITER;
            k0=f,k1=p;
            do {
                ki=1.0/k,kmv=1.0/(k-v),kpv=1.0/(k+v);
                C*=z2d4*ki,A*=a*ki*kmv;
                f=(k*f+p+q)*kmv*kpv;
                p*=kmv,q*=kpv;
                k0+=C*f,k1+=C*(p-k*f);
                if (inf(k0) || inf(k1))
#ifdef CBESSEL_EXCEPT
                    throw of_err;
#else
                    return infty;
#endif
            } while (A>=eps && k++<maxiter);
#ifdef CBESSEL_EXCEPT
            if (k>maxiter) throw cvg_err;
#endif
            k1*=z2;
            if (inf(k1))
#ifdef CBESSEL_EXCEPT
                throw of_err;
#else
                return infty;
#endif
        } else {
            Real v2=v*v,A=3.0/(1.0+R),B=14.7/(28.0+R);
            Real C=2.0*M_2_SQRTPI*(abs(abs(v)-.5)<eps?M_PI*eps:cos(M_PI*v))/(eps*pow(2.0*R,0.25));
            Real M=ceil(0.485/R*pow((log(C)+R*cos(A*theta)/(1.0+0.008*R))/(2.0*cos(B*theta)),2)+1.5);
            k0=0.0,k1=eps;
            Cplx S=0.0;
            for (Real n=M;n-->1.0;swap(k0,k1)) {
                S+=k1;
                k0=(n*(2.0*(z+n)*k1-(n+1.0)*k0))/((n-0.5)*(n-0.5)-v2);
            }
            Cplx r=z+v+.5-k0/k1;
            k0=(scaled?1.0:exp(-z))*sqrt(M_PI_4*z2)*k1/(S+k1);
            k1=0.5*k0*r*z2;
        }
        if (jmax==0)
            return k0;
        for (int j=1;j<jmax;++j,swap(k0,k1))
            if (inf(k0+=(++v)*z2*k1))
#ifdef CBESSEL_EXCEPT
                throw of_err;
#else
                return infty;
#endif
        if (K_val_1!=nullptr) *K_val_1=k0;
#ifdef CBESSEL_EXCEPT
        if (zero(k1)) throw uf_err;
#endif
        return k1;
    }
    Cplx K0_in(const Cplx &z,bool scaled) {
        if (zero(z))
#ifdef CBESSEL_EXCEPT
            throw of_err;
#else
            return infty;
#endif
        if (inf(z))
#ifdef CBESSEL_EXCEPT
            throw uf_err;
#else
            return 0.0;
#endif
        if (real(z)<0.0) {
            Cplx Kv=NAN,Iv,si=sign1(imag(z))*M_PI*i;
#ifdef CBESSEL_EXCEPT
            try {
#endif
                Iv=I0_in(-z,scaled,&Kv)*(scaled?exp(i*imag(z)):1.0);
#ifdef CBESSEL_EXCEPT
            } catch (const underflow_error &e) { Iv=0.0; }
#endif
            if (scaled || undef(Kv))
#ifdef CBESSEL_EXCEPT
            try {
#endif
                Kv=K0_in(-z,false)*(scaled?exp(z):1.0);
#ifdef CBESSEL_EXCEPT
            } catch (const underflow_error &e) { Kv=0.0; }
#endif
            Cplx ret=Kv-si*Iv;
#ifdef CBESSEL_EXCEPT
            if (inf(ret)) throw of_err;
#endif
            return ret;
        }
        /* Basic case: Re(z)>=0, v>=0 */
        Cplx k0,k1,z2=2.0/z;
        Real R=abs(z),theta=arg(z);
        if (R<=2.0) {
            Cplx lz2=log(z2),z2d4=z*z*0.25,scale=scaled?exp(z):1.0,p=scale,C=1.0,f=scale*(D[9]+lz2);
            Real A=0.5,a=0.25*R*R,k=1.0,ki,maxiter=CBESSEL_MAXITER;
            k0=f;
            do {
                ki=1.0/k;
                C*=z2d4*ki,A*=a*ki*ki;
                f=(f+(p*=ki))*ki;
                k0+=C*f;
#ifdef CBESSEL_EXCEPT
                if (inf(k0)) throw of_err;
#endif
            } while (A>=eps && k++<maxiter);
#ifdef CBESSEL_EXCEPT
            if (k>maxiter) throw cvg_err;
#endif
            return k0;
        }
        Real A=3.0/(1.0+R),B=14.7/(28.0+R),C=2.0*M_2_SQRTPI/(eps*pow(2.0*R,0.25));
        Real M=ceil(0.485/R*pow((log(C)+R*cos(A*theta)/(1.0+0.008*R))/(2.0*cos(B*theta)),2)+1.5);
        k0=0.0,k1=eps;
        Cplx S=0.0;
        for (Real n=M;n-->1.0;swap(k0,k1)) {
            S+=k1;
            k0=(n*(2.0*(z+n)*k1-(n+1.0)*k0))/((n-0.5)*(n-0.5));
        }
        return (scaled?1.0:exp(-z))*sqrt(M_PI_4*z2)*k1/(S+k1);
    }
    Cplx K1_in(const Cplx &z,bool scaled) {
        if (zero(z))
#ifdef CBESSEL_EXCEPT
            throw of_err;
#else
            return infty;
#endif
        if (inf(z))
#ifdef CBESSEL_EXCEPT
            throw uf_err;
#else
            return 0.0;
#endif
        if (real(z)<0.0) {
            Cplx Kv=NAN,Iv,si=sign1(imag(z))*M_PI*i;
#ifdef CBESSEL_EXCEPT
            try {
#endif
                Iv=I1_in(-z,scaled,&Kv)*(scaled?exp(i*imag(z)):1.0);
#ifdef CBESSEL_EXCEPT
            } catch (const underflow_error &e) { Iv=0.0; }
#endif
            if (scaled || undef(Kv))
#ifdef CBESSEL_EXCEPT
            try {
#endif
                Kv=K1_in(-z,false)*(scaled?exp(z):1.0);
#ifdef CBESSEL_EXCEPT
            } catch (const underflow_error &e) { Kv=0.0; }
#endif
            Cplx ret=-Kv-si*Iv;
#ifdef CBESSEL_EXCEPT
            if (inf(ret)) throw of_err;
#endif
            return ret;
        }
        /* Basic case: Re(z)>=0, v>=0 */
        Cplx k0,k1,z2=2.0/z;
        Real R=abs(z),theta=arg(z);
        if (R<=2.0) {
            Cplx lz2=log(z2),z2d4=z*z*0.25,scale=scaled?exp(z):1.0,p=0.5*scale,C=1.0,f=scale*(D[9]+lz2);
            Real A=0.5,a=0.25*R*R,k=1.0,ki,maxiter=CBESSEL_MAXITER;
            k1=p;
            do {
                ki=1.0/k;
                C*=z2d4*ki,A*=a*ki*ki;
                f=(f+2.0*(p*=ki))*ki;
                k1+=C*(p-k*f);
#ifdef CBESSEL_EXCEPT
                if (inf(k1)) throw of_err;
#endif
            } while (A>=eps && k++<maxiter);
#ifdef CBESSEL_EXCEPT
            if (k>maxiter) throw cvg_err;
#endif
#ifdef CBESSEL_EXCEPT
            if (inf(k1)) throw of_err;
#endif
            return k1*z2;
        }
        Real A=3.0/(1.0+R),B=14.7/(28.0+R),C=2.0*M_2_SQRTPI/(eps*pow(2.0*R,0.25));
        Real M=ceil(0.485/R*pow((log(C)+R*cos(A*theta)/(1.0+0.008*R))/(2.0*cos(B*theta)),2)+1.5);
        k0=0.0,k1=eps;
        Cplx S=0.0;
        for (Real n=M;n-->1.0;swap(k0,k1)) {
            S+=k1;
            k0=(n*(2.0*(z+n)*k1-(n+1.0)*k0))/((n-0.5)*(n-0.5));
        }
        Cplx r=z+.5-k0/k1;
        k0=(scaled?1.0:exp(-z))*sqrt(M_PI_4*z2)*k1/(S+k1);
        k1=0.5*k0*r*z2;
        return k1;
    }

    /* Bessel functions J_v(z) and Y_v(z) */
    Cplx J(Real v,const Cplx &z,bool scaled) {
        Cplx si=sign1(imag(z))*i;
#ifdef USE_STD_BESSEL
        if (!scaled && zero(z.imag()) && z.real()>0) {
            if (zero(std::round(v)-v))
                return jn((int)std::round(v),z.real());
            return cyl_bessel_j(v,z.real());
        }
#endif
        if (v>=0.0) {
            if (!scaled && abs(zero(arg(z))-M_PI))
                return exp(si*M_PI*v)*cyl_bessel_j(v,-real(z));
            if (v==0.0)
                return I0_in(-z*si,scaled);
            if (v==1.0)
                return si*I1_in(-z*si,scaled);
        }
        return exp(si*M_PI_2*v)*I_in(v,-z*si,scaled);
    }
    Cplx Y(Real v,const Cplx &z,bool scaled) {
#ifdef USE_STD_BESSEL
        if (!scaled && zero(imag(z)) && z.real()>0 && v>=0.0) {
            if (zero(std::round(v)-v))
                return yn((int)std::round(v),z.real());
            return cyl_neumann(v,z.real());
        }
#endif
        Real s=sign1(imag(z)),va=abs(v);
        Cplx si=i*s,a=exp(va*M_PI_2*si),iz=z*si,Kv=NAN,Iv;
#ifdef CBESSEL_EXCEPT
        try {
#endif
            if (va==0.0)
                Iv=I0_in(-iz,scaled,&Kv);
            else if (va==1.0)
                Iv=I1_in(-iz,scaled,&Kv);
            else Iv=I_in(va,-iz,scaled,&Kv);
#ifdef CBESSEL_EXCEPT
        } catch (const underflow_error &e) { Iv=0.0; }
#endif
        if (scaled || undef(Kv))
#ifdef CBESSEL_EXCEPT
        try
#endif
            {
            if (va==0.0)
                Kv=K0_in(-iz,false);
            else if (va==1.0)
                Kv=K1_in(-iz,false);
            else Kv=K_in(va,-iz,false);
            if (scaled)
                Kv*=exp(-abs(imag(z)));
            }
#ifdef CBESSEL_EXCEPT
        catch (const underflow_error &e) { Kv=0.0; }
#endif
        Cplx ret=si*a*Iv-M_2_PI*Kv/a;
        if (v>=0.0)
            return ret;
        int n;
        if (abs(2.0*v-(n=int(round(2.0*v))))<eps && n%2)
            return ((-n/2)%2?-1.0:1.0)*a*Iv;
        if (va==0.0)
            return ret;
        if (va==1.0)
            return -ret;
        return cos(M_PI*va)*ret+sin(M_PI*va)*a*Iv;
    }
    /* modified Bessel functions I_v(z) and K_v(z) */
    Cplx I(Real v,const Cplx &z,bool scaled) {
        if (!scaled && zero(z.imag()) && z.real()>=0)
            return cyl_bessel_i(v,z.real());
        if (v==0.0)
            return I0_in(z,scaled);
        if (v==1.0)
            return I1_in(z,scaled);
        return I_in(v,z,scaled);
    }
    Cplx K(Real v,const Cplx &z,bool scaled) {
        if (!scaled && zero(z.imag()) && z.real()>=0)
            return cyl_bessel_k(v,z.real());
        if (v==0.0)
            return K0_in(z,scaled);
        if (v==1.0)
            return K1_in(z,scaled);
        return K_in(v,z,scaled);
    }
    /* Hankel function H^(kind)_v(z) */
    inline Cplx H(Real v,const Cplx &z,int kind,bool scaled) {
#ifdef CBESSEL_EXCEPT
        if (kind<1 || kind>2)
            throw domain_error("Invalid kind");
#endif
        Real va=abs(v),s=kind==1?1.0:-1.0;
        Cplx si=i*s,a=exp(M_PI_2*va*si),iz=si*z,ret;
        if (s*arg(z)>=-M_PI_2) {
            ret=i*M_1_PI/a*K_in(va,-iz,scaled);
        } else {
            OlverData data;
            Cplx Kv=NAN,Iv;
#ifdef CBESSEL_EXCEPT
            try {
#endif
                Iv=I_in(va,iz,scaled,&Kv,&data);
#ifdef CBESSEL_EXCEPT
            } catch (const underflow_error &e) { Iv=0.0; }
#endif
            if (scaled || undef(Kv))
#ifdef CBESSEL_EXCEPT
            try {
#endif
                Kv=K_in(va,iz,false,nullptr,&data)*(scaled?exp(-iz):1.0);
#ifdef CBESSEL_EXCEPT
            } catch (const underflow_error &e) { Kv=0.0; }
#endif
            ret=i*M_1_PI*a*Kv-s*Iv/a*(scaled?exp(-si*real(z)):1.0);
        }
        return -2.0*s*(v>=0.0?ret:a*a*ret);
    }
    inline Cplx H_v0(const Cplx &z,int kind,bool scaled) {
#ifdef CBESSEL_EXCEPT
        if (kind<1 || kind>2)
            throw domain_error("Invalid kind");
#endif
        Real s=kind==1?1.0:-1.0;
        Cplx si=i*s,iz=si*z,ret;
        if (s*arg(z)>=-M_PI_2) {
            ret=i*M_1_PI*K0_in(-iz,scaled);
        } else {
            Cplx Kv=NAN,Iv;
#ifdef CBESSEL_EXCEPT
            try {
#endif
                Iv=I0_in(iz,scaled,&Kv);
#ifdef CBESSEL_EXCEPT
            } catch (const underflow_error &e) { Iv=0.0; }
#endif
            if (scaled || undef(Kv))
#ifdef CBESSEL_EXCEPT
            try {
#endif
                Kv=K0_in(iz,false)*(scaled?exp(-iz):1.0);
#ifdef CBESSEL_EXCEPT
            } catch (const underflow_error &e) { Kv=0.0; }
#endif
            ret=i*M_1_PI*Kv-s*Iv*(scaled?exp(-si*real(z)):1.0);
        }
        return -2.0*s*ret;
    }
    inline Cplx H_v1(const Cplx &z,int kind,bool scaled) {
#ifdef CBESSEL_EXCEPT
        if (kind<1 || kind>2)
            throw domain_error("Invalid kind");
#endif
        Real s=kind==1?1.0:-1.0;
        Cplx si=i*s,iz=si*z,ret;
        if (s*arg(z)>=-M_PI_2) {
            ret=s*M_1_PI*K1_in(-iz,scaled);
        } else {
            Cplx Kv=NAN,Iv;
#ifdef CBESSEL_EXCEPT
            try {
#endif
                Iv=I1_in(iz,scaled,&Kv);
#ifdef CBESSEL_EXCEPT
            } catch (const underflow_error &e) { Iv=0.0; }
#endif
            if (scaled || undef(Kv))
#ifdef CBESSEL_EXCEPT
            try {
#endif
                Kv=K1_in(iz,false)*(scaled?exp(-iz):1.0);
#ifdef CBESSEL_EXCEPT
            } catch (const underflow_error &e) { Kv=0.0; }
#endif
            ret=-s*M_1_PI*Kv+i*Iv*(scaled?exp(-si*real(z)):1.0);
        }
        return -2.0*s*ret;
    }
    /* Hankel functions */
    Cplx H1(Real v,const Cplx &z,bool scaled) {
        if (v==0.0) {
#ifdef USE_STD_BESSEL
            if (!scaled && zero(imag(z)) && real(z)>0)
                return Cplx(j0(real(z)),y0(real(z)));
#endif
            return H_v0(z,1,scaled);
        }
        if (v==1.0) {
#ifdef USE_STD_BESSEL
            if (!scaled && zero(imag(z)) && real(z)>0)
                return Cplx(j1(real(z)),y1(real(z)));
#endif
            return H_v1(z,1,scaled);
        }
        return H(v,z,1,scaled);
    }
    Cplx H2(Real v,const Cplx &z,bool scaled) {
        if (v==0.0) {
#ifdef USE_STD_BESSEL
            if (!scaled && zero(imag(z)) && real(z)>0)
                return Cplx(j0(real(z)),-y0(real(z)));
#endif
            return H_v0(z,2,scaled);
        }
        if (v==1.0) {
#ifdef USE_STD_BESSEL
            if (!scaled && zero(imag(z)) && real(z)>0)
                return Cplx(j1(real(z)),-y1(real(z)));
#endif
            return H_v1(z,2,scaled);
        }
        return H(v,z,2,scaled);
    }

    Real pzero(Real x) {
        const Real *p,*q;
        Real z,r,s;
        int ix=0x7fffffff&HI(x);
        if(ix>=0x40200000)     {p=pR8; q=pS8;}
        else if(ix>=0x40122E8B){p=pR5; q=pS5;}
        else if(ix>=0x4006DB6D){p=pR3; q=pS3;}
        else                   {p=pR2; q=pS2;}
        z=1./(x*x);
        r=p[0]+z*(p[1]+z*(p[2]+z*(p[3]+z*(p[4]+z*p[5]))));
        s=1.+z*(q[0]+z*(q[1]+z*(q[2]+z*(q[3]+z*q[4]))));
        return 1.+r/s;
    }
    Real qzero(Real x) {
        const Real *p,*q;
        Real s,r,z;
        int ix=0x7fffffff&HI(x);
        if(ix>=0x40200000)     {p=qR8; q=qS8;}
        else if(ix>=0x40122E8B){p=qR5; q=qS5;}
        else if(ix>=0x4006DB6D){p=qR3; q=qS3;}
        else                   {p=qR2; q=qS2;}
        z=1./(x*x);
        r=p[0]+z*(p[1]+z*(p[2]+z*(p[3]+z*(p[4]+z*p[5]))));
        s=1.+z*(q[0]+z*(q[1]+z*(q[2]+z*(q[3]+z*(q[4]+z*q[5])))));
        return (-.125+r/s)/x;
    }
    Real pone(Real x) {
        const Real *p,*q;
        Real z,r,s;
        int ix=0x7fffffff&HI(x);
        if(ix>=0x40200000)     {p=pr8; q=ps8;}
        else if(ix>=0x40122E8B){p=pr5; q=ps5;}
        else if(ix>=0x4006DB6D){p=pr3; q=ps3;}
        else                   {p=pr2; q=ps2;}
        z=1./(x*x);
        r=p[0]+z*(p[1]+z*(p[2]+z*(p[3]+z*(p[4]+z*p[5]))));
        s=1.+z*(q[0]+z*(q[1]+z*(q[2]+z*(q[3]+z*q[4]))));
        return 1.+r/s;
    }
    Real qone(Real x) {
        const Real *p,*q;
        Real  s,r,z;
        int ix=0x7fffffff&HI(x);
        if(ix>=0x40200000)     {p=qr8; q=qs8;}
        else if(ix>=0x40122E8B){p=qr5; q=qs5;}
        else if(ix>=0x4006DB6D){p=qr3; q=qs3;}
        else                   {p=qr2; q=qs2;}
        z=1./(x*x);
        r=p[0]+z*(p[1]+z*(p[2]+z*(p[3]+z*(p[4]+z*p[5]))));
        s=1.+z*(q[0]+z*(q[1]+z*(q[2]+z*(q[3]+z*(q[4]+z*q[5])))));
        return (.375+r/s)/x;
    }

    void JY_01(Real x,Real &J0,Real &Y0,Real &J1,Real &Y1) {
        assert(x>0);
        Real z,s,c,ss,cc,r,u,v,sqrtx,logx;
        int ix=HI(x)&0x7fffffff;
        if(ix>=0x7ff00000) {
            J0=1./(x*x);
            Y0=Y1=1./(x+x*x);
            J1=1./x;
        } else if(ix>=0x40000000) { /* |x| >= 2.0 */
            s=sin(x);
            c=cos(x);
            ss=s-c;
            cc=s+c;
            if(ix<0x7fe00000) { /* make sure x+x not overflow */
                z=-cos(x+x);
                if ((s*c)<0.) cc=z/ss;
                else ss=z/cc;
            }
            sqrtx=sqrt(x);
            if(ix>0x48000000) {
                J0=(invsqrtpi*cc)/sqrtx;
                Y0=J1=(invsqrtpi*ss)/sqrtx;
                Y1=-(invsqrtpi*cc)/sqrtx;
            } else {
                u=pzero(x); v=qzero(x);
                J0=invsqrtpi*(u*cc-v*ss)/sqrtx;
                Y0=invsqrtpi*(u*ss+v*cc)/sqrtx;
                u=pone(x); v=qone(x);
                J1=invsqrtpi*(u*ss+v*cc)/sqrtx;
                Y1=invsqrtpi*(v*ss-u*cc)/sqrtx;
            }
        } else {
            logx=log(x);
            z=x*x;
            r=z*(R0[2]+z*(R0[3]+z*(R0[4]+z*R0[5])));
            s=1.+z*(S0[1]+z*(S0[2]+z*(S0[3]+z*S0[4])));
            if(ix<0x3FF00000) { /* |x| < 1.00 */
                J0=1.+z*(-0.25+(r/s));
            } else {
                u=0.5*x;
                J0=((1.+u)*(1.-u)+z*(r/s));
            }
            u=u0[0]+z*(u0[1]+z*(u0[2]+z*(u0[3]+z*(u0[4]+z*(u0[5]+z*u0[6])))));
            v=1.+z*(v0[1]+z*(v0[2]+z*(v0[3]+z*v0[4])));
            Y0=u/v+tpi*J0*logx;
            r=z*(r0[0]+z*(r0[1]+z*(r0[2]+z*r0[3])));
            s=1.+z*(s0[1]+z*(s0[2]+z*(s0[3]+z*(s0[4]+z*s0[5]))));
            r*=x;
            J1=x*0.5+r/s;
            u=U0[0]+z*(U0[1]+z*(U0[2]+z*(U0[3]+z*U0[4])));
            v=1.+z*(V0[0]+z*(V0[1]+z*(V0[2]+z*(V0[3]+z*V0[4]))));
            Y1=x*(u/v)+tpi*(J1*logx-1./x);
        }
    }

    void H1_01(Real x, Cplx &h0, Cplx &h1) {
        Real J0, Y0, J1, Y1;
        JY_01(x, J0, Y0, J1, Y1);
        h0 = J0 + 1i * Y0;
        h1 = J1 + 1i * Y1;
    }

    /* n-th derivative of Bessel function f_v(z) */
    Cplx bessel_derive(Real v,const Cplx &z,int n,Cplx (*f)(Real,const Cplx&,bool),bool alt) {
        Cplx ret=0.0,t;
        int C=1,k=0;
        for (;k<=n;++k) {
            try { t=f(v-n+2*k,z,false); }
            catch (const underflow_error &e) { t=0.0; }
            ret+=(alt&&k%2?-1.0:1.0)*C*t;
            C=(C*(n-k))/(k+1);
        }
        if (zero(ret))
#ifdef CBESSEL_EXCEPT
            throw uf_err;
#else
            return 0.0;
#endif
        ret*=pow(2.0,-n);
        return ret;
    }
    Cplx Ip(Real v,const Cplx &z,int n) {
        return bessel_derive(v,z,n,&I,false);
    }
    Cplx Jp(Real v,const Cplx &z,int n) {
        return bessel_derive(v,z,n,&J,true);
    }
    Cplx Kp(Real v,const Cplx &z,int n) {
        return (n%2?-1.0:1.0)*bessel_derive(v,z,n,&K,false);
    }
    Cplx Yp(Real v,const Cplx &z,int n) {
        return bessel_derive(v,z,n,&Y,true);
    }
    Cplx H1p(Real v,const Cplx &z,int n) {
        return bessel_derive(v,z,n,&H1,true);
    }
    Cplx H2p(Real v,const Cplx& z,int n) {
        return bessel_derive(v,z,n,&H2,true);
    }

    /* Compute f and g (resp. f' and g' if p=true) for Airy power series.
     * MODE is 1 for f,g, 2 for f',g', and 3 for both f,g and f',g'. */
    void airy_series(const Cplx &z,int mode,Cplx &f,Cplx &g,Cplx &fp,Cplx &gp) {
        Cplx z3=pow(z,3),zp=1.0;
        Real z3a=1.0/abs(z3),tol=eps/(mode>1?1.07:1.05);
        f=g=fp=gp=1.0;
        const Real *fc=&fg_coef[0],*fpc=&fg_coef[14],*gc=&fg_coef[7],*gpc=&fg_coef[21];
        bool np=mode%2,p=mode>1;
        for (int j=0;j<7&&(*fc>=tol||*gc>=tol);++j,++fc,++gc,++fpc,++gpc) {
            tol*=z3a,zp*=z3;
            if (np) f+=*fc*zp,g+=*gc*zp;
            if (p) fp+=*fpc*zp,gp+=*gpc*zp;
        }
    }
    /* Airy functions and their first derivatives */
    Cplx Ai(const Cplx &z,bool scaled) {
        if (norm(z)<=1) {
            Cplx f,g,fp,gp;
            airy_series(z,1,f,g,fp,gp);
            return (scaled?exp(f23*pow(z,f32)):1.0)*(airy_C1*f-airy_C2*z*g);
        }
        Real a=arg(z);
        Cplx si=sign1(a)*i,sz=sqrt(z),w=f23*sz*z,Kv=NAN,Iv;
        if (abs(a)<=M_PI*f23)
            return M_1_PI*sqrt_1_3*sz*K_in(f13,w,scaled);
#ifdef CBESSEL_EXCEPT
        try {
#endif
            Iv=I_in(f13,-w,scaled,&Kv)*(scaled?exp(i*imag(w)):1.0);
#ifdef CBESSEL_EXCEPT
        } catch (const underflow_error &e) { Iv=0.0; }
#endif
        if (scaled || undef(Kv))
#ifdef CBESSEL_EXCEPT
        try {
#endif
            Kv=K_in(f13,-w,false);
#ifdef CBESSEL_EXCEPT
        } catch (const underflow_error &e) { Kv=0.0; }
#endif
        return sqrt_1_3*sz*(M_1_PI*Kv*exp((scaled?w:0.0)-si*M_PI*f13)-si*Iv);
    }
    Cplx Aip(const Cplx &z,bool scaled) {
        if (norm(z)<=1) {
            Cplx f,g,fp,gp;
            airy_series(z,2,f,g,fp,gp);
            return (scaled?exp(f23*pow(z,f32)):1.0)*(z*z*airy_C1*fp*0.5-airy_C2*gp);
        }
        Real a=arg(z),b=M_PI*f23;
        Cplx is=sign1(a)*i,w=f23*pow(z,1.5),Kv=NAN,Iv;
        if (abs(a)<=b)
            return -M_1_PI*z*sqrt_1_3*K_in(f23,w,scaled);
#ifdef CBESSEL_EXCEPT
        try {
#endif
            Iv=I_in(f23,-w,scaled,&Kv)*(scaled?exp(i*imag(w)):1.0);
#ifdef CBESSEL_EXCEPT
        } catch (const underflow_error &e) { Iv=0.0; }
#endif
        if (scaled || undef(Kv))
#ifdef CBESSEL_EXCEPT
        try {
#endif
            Kv=K_in(f23,-w,false);
#ifdef CBESSEL_EXCEPT
        } catch (const underflow_error &e) { Kv=0.0; }
#endif
        return z*sqrt_1_3*(is*Iv-M_1_PI*Kv*exp((scaled?w:0.0)-is*b));
    }
    Cplx Bi(const Cplx &z,bool scaled) {
        if (norm(z)<=1) {
            Cplx f,g,fp,gp;
            airy_series(z,1,f,g,fp,gp);
            return (scaled?exp(-f23*abs(real(pow(z,f32)))):1.0)
                *(airy_C1*f+airy_C2*z*g)/sqrt_1_3;
        }
        Real a=arg(z),s=sign1(a),b=M_PI*f23;
        Cplx is=s*i,sz=sqrt(z),w=f23*z*sz,Kv=NAN,Iv,h1,h2;
        if (abs(a)<=b) {
            int k=int(round((s+1.0)*0.5));
            Real t=sign1(real(w));
#ifdef CBESSEL_EXCEPT
            try {
#endif
                h1=H(f13,-is*w,2-k,scaled && t>0.0)
                    *exp((scaled?(t>0.0?i*imag(w):-abs(real(w))):0.0)+M_PI*is*f16);
#ifdef CBESSEL_EXCEPT
            } catch (const underflow_error &e) { h1=0.0; }
            try {
#endif
                h2=H(f13,-is*w,1+k,scaled && t<0.0)
                    *exp((scaled?(t<0.0?-i*imag(w):-abs(real(w))):0.0)-M_PI*is*f16);
#ifdef CBESSEL_EXCEPT
            } catch (const underflow_error &e) { h2=0.0; }
#endif
            return sqrt_1_3*sz*(h1+0.5*h2);
        }
#ifdef CBESSEL_EXCEPT
        try {
#endif
            Iv=I_in(f13,-w,scaled,&Kv);
#ifdef CBESSEL_EXCEPT
        } catch (const underflow_error &e) { Iv=0.0; }
#endif
        if (scaled || undef(Kv))
#ifdef CBESSEL_EXCEPT
        try {
#endif
            Kv=K_in(f13,-w,false);
#ifdef CBESSEL_EXCEPT
        } catch (const underflow_error &e) { Kv=0.0; }
#endif
        return sz*(Iv*sqrt_1_3+M_1_PI*exp((scaled?real(w):0.0)-is*M_PI*f13)*Kv);
    }
    Cplx Bip(const Cplx &z,bool scaled) {
        if (norm(z)<=1) {
            Cplx f,g,fp,gp;
            airy_series(z,2,f,g,fp,gp);
            return (scaled?exp(-f23*abs(real(pow(z,f32)))):1.0)
                *(z*z*airy_C1*fp*0.5+airy_C2*gp)/sqrt_1_3;
        }
        Real a=arg(z),s=sign1(a),b=M_PI*f23;
        Cplx is=s*i,w=f23*pow(z,1.5),Kv=NAN,Iv,h1,h2;
        if (abs(a)<=b) {
            int k=int(round((s+1.0)*0.5));
            Real t=sign1(real(w));
#ifdef CBESSEL_EXCEPT
            try {
#endif
                h1=H(f23,-is*w,2-k,scaled && t>0.0)
                    *exp((scaled?(t>0.0?i*imag(w):-abs(real(w))):0.0)+is*M_PI*f13);
#ifdef CBESSEL_EXCEPT
            } catch (const underflow_error &e) { h1=0.0; }
            try {
#endif
                h2=H(f23,-is*w,1+k,scaled && t<0.0)
                    *exp((scaled?(t<0.0?-i*imag(w):-abs(real(w))):0.0)-is*M_PI*f13);
#ifdef CBESSEL_EXCEPT
            } catch (const underflow_error &e) { h2=0.0; }
#endif
            return z*sqrt_1_3*(h1-0.5*h2);
        }
#ifdef CBESSEL_EXCEPT
        try {
#endif
            Iv=I_in(f23,-w,scaled,&Kv);
#ifdef CBESSEL_EXCEPT
        } catch (const underflow_error &e) { Iv=0.0; }
#endif
        if (scaled || undef(Kv))
#ifdef CBESSEL_EXCEPT
        try {
#endif
            Kv=K_in(f23,-w,false);
#ifdef CBESSEL_EXCEPT
        } catch (const underflow_error &e) { Kv=0.0; }
#endif
        return z*(M_1_PI*exp((scaled?real(w):0.0)-is*M_PI*f23)*Kv-Iv*sqrt_1_3);
    }
    /* Compute both Ai and Aip for Olver expansions.
     * Return 0 if AI and AIP are both normal, 1 if
     * one of them is infinite and 2 if both are zero. */
    int airy_olver(Real v,const Cplx &xi,Real s,Cplx &ai,Cplx &aip,bool &corr,bool rot,bool scaled) {
        Real b=M_PI*f23,t=sign1(imag(xi));
        Cplx alpha=s*t>=0.0?pow(v/f23*xi,f23):pow(v/f23*abs(xi),f23)*exp((arg(xi)-t*2.0*M_PI)*f23*i);
        Cplx beta=alpha*(rot?exp(-s*b*i):1.0);
        corr=false;
        if (abs(alpha)<=1) {
            Cplx f,g,fp,gp;
            airy_series(beta,3,f,g,fp,gp);
            ai=airy_C1*f-airy_C2*beta*g;
            aip=beta*beta*airy_C1*fp*0.5-airy_C2*gp;
        } else
#ifdef CBESSEL_EXCEPT
            try
#endif
        {
            corr=true;
            Real a=arg(beta),sa=sign1(a);
            Cplx is=sa*i,sz=sqrt(beta*f13),w=(rot?-v:v)*xi;
            if (abs(a)<=b) {
#ifdef CBESSEL_EXCEPT
                try {
#endif
                    ai=M_1_PI*sz*K(f13,w,scaled);
#ifdef CBESSEL_EXCEPT
                } catch (const underflow_error &e) { ai=0.0; }
                try {
#endif
                    aip=-M_1_PI*beta*sqrt_1_3*K(f23,w,scaled);
#ifdef CBESSEL_EXCEPT
                } catch (const underflow_error &e) { aip=0.0; }
#endif
            } else {
                Cplx sf=scaled?exp(i*imag(w)):1.0;
#ifdef CBESSEL_EXCEPT
                try {
#endif
                    ai=sz*(M_1_PI*K(f13,-w)*exp((scaled?w:0.0)-is*M_PI*f13)-is*I(f13,-w,scaled)*sf);
#ifdef CBESSEL_EXCEPT
                } catch (const underflow_error &e) { ai=0.0; }
                try {
#endif
                    aip=beta*sqrt_1_3*(is*I(f23,-w,scaled)*sf-M_1_PI*K(f23,-w)*exp((scaled?w:0.0)-is*b));
#ifdef CBESSEL_EXCEPT
                } catch (const underflow_error &e) { aip=0.0; }
#endif
            }
#ifdef CBESSEL_EXCEPT
        } catch (const overflow_error &e) {
            return 1;
#endif
        }
#ifdef CBESSEL_EXCEPT
        if (zero(ai) && zero(aip))
            return 2;
#endif
        return 0;
    }

}

