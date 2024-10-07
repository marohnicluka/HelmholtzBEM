/**
 * \file hmatrix.hpp
 * \brief This file defines the hierarchical matrix class which
 * can approximate a dense matrix and implements basic arithmetic
 * operations with h-matrices.
 *
 * This File is a part of the HelmholtzTransmissionBEM
 *
 * (c) 2024 Luka Marohnić
 */

#ifndef HMATRIXHPP
#define HMATRIXHPP

#include <Eigen/Dense>
#include <vector>
#include <complex>
#include <limits>
#include <map>
#include "parametrized_mesh.hpp"

using namespace std;

namespace hierarchical {

    typedef std::function<void(std::vector<size_t>::const_iterator,std::vector<size_t>::const_iterator,Eigen::MatrixXcd&)> MatrixFunction;
    typedef std::pair<size_t,size_t> IPair;
    typedef std::pair<double,double> DPair;
    typedef Eigen::Vector2d Point;
    struct BoundingBox {
        DPair xspan, yspan;
        BoundingBox(std::vector<Point>::const_iterator it, std::vector<Point>::const_iterator itend);
        double diameter() const;
        double distance(const BoundingBox &other) const;
    };

    class PanelGeometry : public std::vector<Point> {
    public:
        PanelGeometry(const PanelVector &panels);
        const Point &vertex(size_t i) const { return this->operator[](i); }
        double cluster_distance_with_diameters(const IPair &span1, const IPair &span2, double &d1, double &d2) const;
    };

    class Block {
        IPair rowspan_, colspan_;
        std::vector<Block*> sons_;
        std::vector<Block*>::const_iterator sons_iterator_;
        const PanelGeometry &pg_;
        size_t divs_;
    public:
        // row/column span is a pair (pos,n), n is the span length
        Block(const IPair &rs, const IPair &cs, const PanelGeometry &pg, size_t divs) : rowspan_(rs), colspan_(cs), pg_(pg), divs_(divs) { }
        ~Block() { for (const auto &son : sons_) delete son; }
        Block *make_son(const IPair &rowspan, const IPair &colspan, const PanelGeometry &pg);
        size_t nsons() const { return sons_.size(); }
        bool is_leaf() const { return sons_.empty(); }
        bool is_admissible(double eta) const;
        bool is_divisible(size_t min_size) const { return rowspan_.second > min_size && colspan_.second > min_size; }
        const IPair &rowspan() const { return rowspan_; }
        const IPair &colspan() const { return colspan_; }
        const PanelGeometry &panel_geometry() const { return pg_; }
        void reset_sons_iterator() { sons_iterator_ = sons_.begin(); }
        Block *next_son() { if (sons_iterator_==sons_.end()) return NULL; return *(sons_iterator_++); }
        void delete_sons() { for (Block *son : sons_) delete son; sons_.clear(); }
        Block *get_son(size_t i, size_t j) const { return sons_[i + j * divs_]; }
    };

    class BlockTree {
        const PanelGeometry &pg_;
        double eta_;
        size_t min_size_;
        size_t divs_;
        Block *root_;
        size_t node_count_;
        size_t admissible_nodes_count_;
        void make_subtree(Block *node);
    public:
        BlockTree(const PanelGeometry &pg, double eta, size_t divs = 2);
        ~BlockTree() { delete root_; }
        bool is_low_rank(Block *node) const { return node->is_leaf() && node->is_divisible(min_size_); }
        Block *root() const { return root_; }
        size_t divisions() const { return divs_; }
    };

    class Matrix {
        const BlockTree &tree_;
        Block *root_;
        std::map<Block*,Eigen::MatrixXcd> dense_;
        std::map<Block*,std::pair<Eigen::MatrixXcd,Eigen::MatrixXcd> > lowrk_;
        double eps_;
        void create_blocks(MatrixFunction &mat, Block *block);
        void submat_dfs(Block *block, Matrix &other) const;
        void to_dense_dfs(Block *block, Eigen::MatrixXcd &mat, size_t i0, size_t j0) const;
        std::pair<Eigen::MatrixXcd,Eigen::MatrixXcd> aca(MatrixFunction &mat, const IPair &rowspan, const IPair &colspan, double eps);
    public:
        Matrix(const BlockTree &tree) : tree_(tree) { }
        Matrix(MatrixFunction &mat, const BlockTree &tree, double eps = 1e-8);
        Matrix(const Matrix &other, Block *root);
        Matrix(const Matrix &other) : tree_(other.tree_), root_(other.root_), dense_(other.dense_), lowrk_(other.lowrk_), eps_(other.eps_) { }
        ~Matrix() { }
        Matrix &operator=(const Matrix &h) {
            assert(&tree_==&h.tree_); root_ = h.root_; dense_ = h.dense_, lowrk_ = h.lowrk_; eps_ = h.eps_; return *this;
        }
        const BlockTree &tree() const { return tree_; }
        void set_tolerance(double eps) { eps_ = eps; }
        Eigen::MatrixXcd &dense_block(Block *block) { return dense_[block]; }
        std::pair<Eigen::MatrixXcd,Eigen::MatrixXcd> &low_rank_block(Block *block) { return lowrk_[block]; }
        Matrix submatrix(Block *root) const { return Matrix(*this, root); }
        /**
         * Truncates low-rank blocks to rank maxr or with respect to
         * the tolerance eps_ if maxr = 0.
         * Use the formula AB* = A(QR)* = (AR*)Q* = USV*Q* = US(QV)* = CD*,
         * where C = U and D = QVS with truncated S and V.
         *
         * @param maxr maximal rank for low-rank blocks
         */
        void truncate(size_t maxr = 0);
        Eigen::MatrixXcd to_dense_matrix() const;
        unsigned long param_count() const;
        IPair block_count() const { return {lowrk_.size(), dense_.size()}; }
        // H-matrix addition (with rank truncation)
        Matrix operator+(const Matrix &other) const;
        // H-matrix multiplication (exact)
        Matrix operator*(const Matrix &other) const;
        // H-matrix-vector multiplication (exact)
        Eigen::VectorXcd operator*(const Eigen::VectorXcd &v) const;
    };

    typedef enum block_type {
        BLOCK_TYPE_RECURSIVE = 0,
        BLOCK_TYPE_DENSE = 1,
        BLOCK_TYPE_LOW_RANK = 2,
        BLOCK_TYPE_NULL = 3
    } BlockType;

    template <typename T = std::complex<double> >
    class HMatrix {
        typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Mtrx;
        std::unique_ptr<Mtrx> coefficients_;
        std::unique_ptr<Eigen::PartialPivLU<Mtrx> > lu_, lu_adjoint_;
        std::vector<HMatrix> sons_;
        size_t divisions_, rows_, cols_;
        int rank_; // = -1 for dense blocks and 0 or larger for null/low-rank blocks
        double accuracy_;
        // SVD truncation for additional compression
        void truncate(size_t max_rank = 0);
        // split leaf block
        void split();
        // ACA algorithm implementation
        int aca(MatrixFunction &mat, size_t row, size_t col, size_t rows, size_t cols);
    public:
        /**
         * Approximate the implicitly given matrix using an
         * existing hierarchical tree and specified accuracy.
         *
         * @param mat function that computes elements of the input matrix
         * @param tree block tree
         * @param acc accuracy
         */
        HMatrix(MatrixFunction &mat, const BlockTree &tree, double acc);
        void approximate(MatrixFunction &mat, const BlockTree &tree, double acc);
        /**
         * Return the sum of this matrix and other matrix.
         * The accuracy of the lesser good approximation will be used.
         */
        HMatrix operator+(const HMatrix &other) const;
        /**
         * Return the product of this matrix and other matrix.
         * This is exact operation.
         */
        HMatrix operator*(const HMatrix &other) const;
        /**
         * Compute matrix-vector multiplication A*v.
         *
         * @param other left operand v in dense form (matrix/vector with elements of type T)
         */
        template <typename Derived>
        Eigen::MatrixBase<Derived> operator*(const Eigen::MatrixBase<Derived> &other) const;
        /**
         * Compute the hierarchical LU of this.
         * This works only if the matrix is square.
         */
        void compute_lu();
        /**
         * Solve Ax = b, where A is this matrix.
         * If adjoint argument is true, then A*x=b
         * is solved (the adjoint system).
         * LU decomposition is required and must be
         * computed in advance (no check is performed).
         *
         * @param b left-hand side (matrix/vector with elements of type T)
         * @param adjoint whether to solve the adjoint system (by default false)
         */
        template <typename Derived>
        Eigen::MatrixBase<Derived> solve(const Eigen::MatrixBase<Derived> &b, bool adjoint = false);
        /**
         * Return the block type (recursive, dense,
         * low-rank or null).
         */
        BlockType type() const;
        /**
         * Return the number of rows of this matrix.
         */
        size_t rows() const { return rows_; }
        /**
         * Return the number of columns of this matrix.
         */
        size_t cols() const { return cols_; }
        /**
         * Return the number of divisions for this matrix.
         * This is the number of equal parts in which row
         * and column spans will be subdivided.
         */
        size_t divisions() const { return divisions_; }
        /**
         * Return the accuracy of this h-matrix approximation.
         */
        double accuracy() const { return accuracy_; }
        /**
         * Return the matrix corresponding to this (leaf) block.
         */
        const Eigen::MatrixXcd &matrix() const { return coefficients_; }
        /**
         * Return modifiable/const reference to the son
         * at position (i, j).
         */
        HMatrix &son(size_t i, size_t j) { return sons_[i + j * divisions_]; }
        const HMatrix &son(size_t i, size_t j) const { return sons_[i + j * divisions_]; }
        /**
         * Return true iff this is actually a dense matrix with
         * no low-rank blocks.
         */
        bool is_dense() const;
        /**
         * Return the dense representation of this.
         */
        Mtrx to_dense_matrix() const;
        /**
         * Return true iff this is leaf block.
         */
        bool is_leaf() const { return sons_.empty(); }
        /**
         * Return true iff this matrix is compatible
         * with the other one, i.e. iff the size and
         * divisions are the same (and hence arithmetic
         * operations can be performed).
         */
        bool is_compatible(const HMatrix &other) const;
    };

}
#endif // HMATRIXHPP
