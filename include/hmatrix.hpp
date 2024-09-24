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

    typedef std::function<std::complex<double>(size_t,size_t)> MatrixFunction;
    typedef std::pair<size_t,size_t> IPair;
    typedef std::pair<double,double> DPair;
    struct Point {
        double x, y;
        Point operator-(const Point &other) const { return {x - other.x, y - other.y}; }
    };
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
        double cluster_diameter(const IPair &span) const;
        double cluster_distance(const IPair &span1, const IPair &span2) const;
    };

    class Block {
        IPair rowspan_, colspan_;
        std::vector<Block*> sons_;
        std::vector<Block*>::const_iterator sons_iterator_;
        const PanelGeometry &pg_;
    public:
        // row/column span is a pair (pos,n), n is the span length
        Block(const IPair &rs, const IPair &cs, const PanelGeometry &pg) : rowspan_(rs), colspan_(cs), pg_(pg) { }
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
    };

    class Matrix {
        const BlockTree &tree_;
        Block *root_;
        std::map<Block*,Eigen::MatrixXcd> dense_;
        std::map<Block*,std::pair<Eigen::MatrixXcd,Eigen::MatrixXcd> > lowrk_;
        double eps_;
        std::vector<IPair> aca_sample_indices_;
        Eigen::MatrixXcd U, V;
        Eigen::VectorXcd u_, v_;
        std::vector<size_t> i_indices, j_indices;
        size_t admissible_dense_block_count_;
        void create_blocks(const MatrixFunction &f, Block *block);
        void submat_dfs(Block *block, Matrix &other) const;
        void to_dense_dfs(Block *block, Eigen::MatrixXcd &mat, size_t i0, size_t j0) const;
        int aca(const MatrixFunction &f, const IPair &rowspan, const IPair &colspan, double eps);
    public:
        Matrix(const BlockTree &tree) : tree_(tree) { }
        Matrix(const MatrixFunction &f, const BlockTree &tree, double eps = 1e-8);
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
         * Use the formula AB = A(B*)* = (AR*)Q* = USV*Q* = US(QV)* = A'B'.
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

}
#endif // HMATRIXHPP
