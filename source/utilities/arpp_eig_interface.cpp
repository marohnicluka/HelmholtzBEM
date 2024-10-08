#include <Eigen/Dense>
#include "arpp_eig_interface.hpp"

typedef std::complex<double> complex_t;

void arpp_to_eig(complex_t* in, Eigen::VectorXcd& out) {
    int size = out.size();
    for( int i = 0; i <size; i++ ) {
        out[i] = *(in + i);
    }
}

void eig_to_arpp(Eigen::VectorXcd& in, complex_t* out){
    int size = in.size();
    for( int i = 0; i <size; i++ ) {
        *(out + i) = in[i];
    }
}

void arpp_to_eig(ARrcCompStdEig<double>& in, Eigen::VectorXd& out_vals, Eigen::MatrixXcd& out_vectors){
    unsigned count = out_vals.size();
    out_vals[0] = 1/in.Eigenvalue(0).real();
    Eigen::VectorXcd temp_vec(out_vectors.rows());
    arpp_to_eig(in.RawEigenvector(0), temp_vec);
    out_vectors.col(0) = temp_vec;
    unsigned i_max;
    if (count%2==0){
        i_max = 2*(count-1);
        out_vals[(count-1)] = 1/in.Eigenvalue(2*(count-1)).real();
        arpp_to_eig(in.RawEigenvector(count-1), temp_vec);
        out_vectors.col(count-1) = temp_vec;
    } else {
        i_max = 2*(count);
    }

    for (unsigned i = 2; i < i_max; i+=4){
        out_vals[i/2] = 1/in.Eigenvalue(i).real();
        out_vals[i/2+1] = 1/in.Eigenvalue(i+1).real();
        arpp_to_eig(in.RawEigenvector(i), temp_vec);
        out_vectors.col(i/2) = temp_vec;
        arpp_to_eig(in.RawEigenvector(i+1), temp_vec);
        out_vectors.col(i/2+1) = temp_vec;
        if (abs(out_vals[i/2]) > abs(out_vals[i/2+1])){
            double temp = out_vals[i/2];
            out_vals[i/2] = out_vals[i/2+1];
            out_vals[i/2+1] = temp;
            temp_vec = out_vectors.col(i/2);
            out_vectors.col(i/2) = out_vectors.col(i/2+1);
            out_vectors.col(i/2+1) = temp_vec;
        }
    }
}
