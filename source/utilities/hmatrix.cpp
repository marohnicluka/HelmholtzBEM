/**
 * \file hmatrix.cpp
 *
 * \brief This file contains the implementation of the
 * hierarchical matrix data type class.
 *
 * (c) 2024 Luka Marohnić
 */

#include "hmatrix.hpp"
#include <iostream>
#include <chrono>
#include <set>
#include <numeric>

#define PARALLELIZE 1
#ifdef PARALLELIZE
#include <execution>
#define EXEC_POLICY std::execution::par,
#else
#define EXEC_POLICY
#endif

using namespace std::chrono;

namespace hierarchical {

    PanelGeometry::PanelGeometry(const PanelVector& panels) {
        reserve(panels.size() + 1);
        bool first = true;
        for (const auto &panel : panels) {
            const auto &p1 = (*panel)[0], &p2 = (*panel)[1];
            if (first) {
                Point p0 = {p1(0), p1(1)};
                push_back(p0);
                first = false;
            }
            Point p = {p2(0), p2(1)};
            push_back(p);
        }
    }

    bool do_spans_intersect(const DPair &span1, const DPair &span2) {
        double a = span1.first, b = span1.second, c = span2.first, d = span2.second;
        return (a <= c && c <= b) || (c <= a && a <= d);
    }

    double distance_between_intervals(const DPair &span1, const DPair &span2) {
        if (do_spans_intersect(span1, span2))
            return 0.;
        return std::min(std::abs(span1.first - span2.second), std::abs(span1.second - span2.first));
    }

    BoundingBox::BoundingBox(std::vector<Point>::const_iterator it, std::vector<Point>::const_iterator itend) {
        double L = 1e12;
        xspan = yspan = {L, -L};
        auto iter = it;
        for (; iter != itend; ++iter) {
            xspan.first = std::min(xspan.first, iter->x);
            xspan.second = std::max(xspan.second, iter->x);
            yspan.first = std::min(yspan.first, iter->y);
            yspan.second = std::max(yspan.second, iter->y);
        }
    }

    double BoundingBox::diameter() const {
        double dx = xspan.second - xspan.first, dy = yspan.second - yspan.first;
        return std::sqrt(dx * dx + dy * dy);
    }

    double BoundingBox::distance(const hierarchical::BoundingBox& other) const {
        double dx = distance_between_intervals(xspan, other.xspan);
        double dy = distance_between_intervals(yspan, other.yspan);
        return std::sqrt(dx * dx + dy * dy);
    }


    double PanelGeometry::cluster_distance(const hierarchical::IPair& span1, const hierarchical::IPair& span2) const {
        auto bb1 = BoundingBox(begin() + span1.first, begin() + span1.first + span1.second + 1);
        auto bb2 = BoundingBox(begin() + span2.first, begin() + span2.first + span2.second + 1);
        return bb1.distance(bb2);
    }

    double PanelGeometry::cluster_diameter(const hierarchical::IPair& span) const {
        auto bb = BoundingBox(begin() + span.first, begin() + span.first + span.second + 1);
        return bb.diameter();
    }

    Block * Block::make_son(const IPair& rowspan, const IPair& colspan, const PanelGeometry& pg) {
        sons_.push_back(new Block(rowspan, colspan, pg));
        return sons_.back();
    }

    bool Block::is_admissible(double eta) const {
        assert(eta > 0.);
        double dist = pg_.cluster_distance(rowspan_, colspan_);
        if (dist <= 0.)
            return false;
        return std::min(pg_.cluster_diameter(rowspan_), pg_.cluster_diameter(colspan_)) <= eta * dist;
    }

    BlockTree::BlockTree(const PanelGeometry& pg, double eta, size_t divs)
        : pg_(pg), eta_(eta), min_size_(1 + (size_t)std::round(std::log2((double)pg.size()))), divs_(divs),
          node_count_(0), admissible_nodes_count_(0)
    {
        IPair span = {0, pg.size() - 1};
        root_ = new Block(span, span, pg);
        make_subtree(root_);
#ifdef CMDL
        std::cout << "Hierarchical block tree has " << node_count_ << " nodes, "
                  << admissible_nodes_count_ << " admissible" << std::endl;
#endif
    }

    // split a span into the list of N equally sized spans
    std::vector<IPair> span_split(const IPair &span, size_t divs) {
        std::vector<IPair> res;
        res.reserve(divs);
        size_t pos = span.first, n = span.second;
        assert((n % divs) == 0);
        size_t step = n / divs, p = pos;
        for (size_t i = 0; i < divs; ++i, p+=step)
            res.push_back({p, step});
        return res;
    }

    void BlockTree::make_subtree(Block* node) {
        if (!node->is_divisible(min_size_))
            return;
        if (node->is_admissible(eta_)) {
            admissible_nodes_count_++;
            return;
        }
        auto rowspans = span_split(node->rowspan(), divs_);
        auto colspans = span_split(node->colspan(), divs_);
        for (const auto &rowspan : rowspans) {
            for (const auto &colspan : colspans) {
                node_count_++;
                make_subtree(node->make_son(rowspan, colspan, pg_));
            }
        }
    }

    size_t random_index(size_t n) {
        return (size_t)(rand() % (int)n);
    }

    size_t max_coeff_index(const Eigen::VectorXcd &v) {
        size_t n = v.size(), i_max;
        auto a = v.cwiseAbs();
        double max_cf = 0;
        for (size_t i = 0; i < n; ++i) {
            if (a(i) > max_cf) {
                max_cf = a(i);
                i_max = i;
            }
        }
        return i_max;
    }

    Eigen::MatrixXcd extract_dense_submatrix(const MatrixFunction &f, const IPair &rowspan, const IPair &colspan) {
        Eigen::MatrixXcd m(rowspan.second, colspan.second);
        for (size_t j = 0; j < colspan.second; ++j)
            for (size_t i = 0; i < rowspan.second; ++i)
                m(i, j) = f(i + rowspan.first, j + colspan.first);
        return m;
    }

    size_t argmax(const Eigen::VectorXd &v, std::vector<size_t> *indices) {
        double maxval = -std::numeric_limits<double>::infinity(), val;
        size_t ret;
        if (indices == NULL) {
            v.maxCoeff(&ret);
        } else {
            for (auto index : *indices) {
                val = v(index);
                if (val > maxval) {
                    maxval = val;
                    ret = index;
                }
            }
            indices->erase(std::find(indices->begin(), indices->end(), ret));
        }
        return ret;
    }

    int Matrix::aca(const MatrixFunction &f, const IPair &rowspan, const IPair &colspan, double eps) {
        size_t m = rowspan.second, n = colspan.second, k;
        size_t i0 = rowspan.first, j0 = colspan.first;
        size_t max_rank = std::min(m, n) / 2;
        bool failed = false, done = false;
        // Initialize empty matrices U and V
        auto u = u_.head(m), v = v_.head(n);
        size_t M = m + n, best;
        size_t rank = 0, i_pivot, j_pivot;
        size_t l;
        double max_coeff = 0.0, cf;
        for (k = 0; k < M; ++k) {
            IPair index = aca_sample_indices_[k];
            index.first %= m;
            index.second %= n;
            cf = std::abs(f(i0 + index.first, j0 + index.second));
            if (cf >= max_coeff) {
                best = index.first;
                max_coeff = cf;
            }
        }
        if (std::fpclassify(max_coeff) == FP_ZERO) {
            // the block is probably nul-matrix
            return 0;
        }
        i_indices.resize(m);
        j_indices.resize(n);
        std::iota(i_indices.begin(), i_indices.end(), 0);
        std::iota(j_indices.begin(), j_indices.end(), 0);
        i_pivot = best;
        double mu = 0.0, u2v2;
        size_t restart_count = 0, max_restarts = 2; //(size_t)std::round(std::log2(std::min(m, n)));
        for (k = 1; k <= max_rank; ++k) {
            for (l = 0; l < n; ++l)
                v(l) = f(i0 + i_pivot, j0 + l);
            if (k > 1)
                v -= U.row(i_pivot).head(rank) * V.block(0, 0, rank, n);
            if (v.array().abs().isZero()) {
                // ACA failure
                if (restart_count++ < max_restarts) {
                    k--;
                    i_pivot = i_indices[random_index(i_indices.size())];
                    continue;
                }
                failed = true;
                break;
            }
            j_pivot = argmax(v.array().abs(), &j_indices);
            v /= v(j_pivot);
            for (l = 0; l < m; ++l)
                u(l) = f(i0 + l, j0 + j_pivot);
            if (k > 1)
                u -= U.block(0, 0, m, rank) * V.col(j_pivot).head(rank);
            u2v2 = u.squaredNorm() * v.squaredNorm();
            mu += u2v2;
            if (k > 1) {
                mu += 2. * ((u.transpose() * U.block(0, 0, m, rank)).array() * (V.block(0, 0, rank, n) * v).transpose().array()).abs().sum();
                if (std::sqrt(u2v2) <= eps * std::sqrt(mu))
                    done = true;
            }
            rank++;
            if (U.cols() < rank)
                U.conservativeResize(Eigen::NoChange, rank);
            if (V.rows() < rank)
                V.conservativeResize(rank, Eigen::NoChange);
            U.col(rank - 1).head(m) = u;
            V.row(rank - 1).head(n) = v;
            if (done)
                break;
            i_pivot = argmax(u.array().abs(), &i_indices);
        }
        //std::cout << "Rank: " << rank << ", log(n) = " << (int)std::log2(std::min(m, n)) << std::endl;
        if (failed) {
            // error handling
            //auto mat = extract_dense_submatrix(f, rowspan, colspan);
            //std::cout << "ACA failed, exact error: " << (mat - approx_U * approx_V).norm() / mat.norm() << ", rank: " << rank << std::endl;
            return -(int)rank;
        }
        return rank;
    }

    Matrix::Matrix(const hierarchical::MatrixFunction& f, const hierarchical::BlockTree& tree, double eps)
        : tree_(tree), root_(tree.root()), eps_(eps)
    {
        size_t m = root_->rowspan().second, n = root_->colspan().second, M = (m + n) / 2;
        aca_sample_indices_.resize(M);
        for (size_t i = 0; i < M; ++i)
            aca_sample_indices_[i] = {rand(), rand()};
        u_.resize(m / 2);
        v_.resize(n / 2);
        i_indices.reserve(m / 2);
        j_indices.reserve(n / 2);
        size_t logn = (size_t)std::round(std::log2(std::min(m, n)));
        U.resize(m / 2, logn);
        V.resize(logn, n / 2);
        admissible_dense_block_count_ = 0;
        create_blocks(f, root_);
#ifdef CMDL
        if (admissible_dense_block_count_ > 0)
            std::cerr << "Warning: found " << admissible_dense_block_count_ << " dense admissible blocks" << std::endl;
#endif
    }

    void Matrix::create_blocks(const MatrixFunction &f, Block *block) {
        if (tree_.is_low_rank(block)) {
            int res = aca(f, block->rowspan(), block->colspan(), eps_);
            size_t m = block->rowspan().second, n = block->colspan().second, rk = std::abs(res);
            if (rk * (m + n) >= m * n) {
                admissible_dense_block_count_++;
                dense_.insert({block, extract_dense_submatrix(f, block->rowspan(), block->colspan())});
            } else if (rk == 0)
                lowrk_.insert({block, {Eigen::MatrixXcd(), Eigen::MatrixXcd()}});
            else
                lowrk_.insert({block, {U.block(0, 0, m, rk), V.block(0, 0, rk, n)}});
        } else if (block->is_leaf())
            dense_.insert({block, extract_dense_submatrix(f, block->rowspan(), block->colspan())});
        else {
            block->reset_sons_iterator();
            Block *son;
            while ((son = block->next_son()) != NULL)
                create_blocks(f, son);
        }
    }

    void Matrix::submat_dfs(hierarchical::Block* block, hierarchical::Matrix& other) const {
        if (tree_.is_low_rank(block))
            other.lowrk_.insert(*lowrk_.find(block));
        else if (block->is_leaf())
            other.dense_.insert(*dense_.find(block));
        else {
            block->reset_sons_iterator();
            Block *son;
            while ((son = block->next_son()) != NULL)
                submat_dfs(son, other);
        }
    }

    Matrix::Matrix(const hierarchical::Matrix& other, hierarchical::Block* root)
        : tree_(other.tree_), root_(root), eps_(other.eps_)
    {
        other.submat_dfs(root, *this);
    }

    unsigned long Matrix::param_count() const {
        unsigned long ret = 0;
        for (const auto &lowrk : lowrk_) {
            const auto &U = lowrk.second.first, &V = lowrk.second.second;
            ret += U.rows() * U.cols() + V.rows() * V.cols();
        }
        for (const auto &dense : dense_) {
            const auto &M = dense.second;
            ret += M.rows() * M.cols();
        }
        return ret;
    }

    void Matrix::to_dense_dfs(Block *block, Eigen::MatrixXcd &mat, size_t i0, size_t j0) const {
        size_t i = block->rowspan().first - i0, m = block->rowspan().second;
        size_t j = block->colspan().first - j0, n = block->colspan().second;
        if (block->is_leaf()) {
            auto it = lowrk_.find(block);
            if (it != lowrk_.end()) {
                if (it->second.first.cols() == 0)
                    mat.block(i, j, m, n).setZero();
                else {
                    assert(it->second.first.rows() == m && it->second.second.cols() == n);
                    mat.block(i, j, m, n) = it->second.first * it->second.second;
                }
            } else {
                auto jt = dense_.find(block);
                assert(jt != dense_.end());
                mat.block(i, j, m, n) = jt->second;
            }
        } else {
            block->reset_sons_iterator();
            Block *son;
            while ((son = block->next_son()) != NULL)
                to_dense_dfs(son, mat, i0, j0);
        }
    }

    Eigen::MatrixXcd hierarchical::Matrix::to_dense_matrix() const {
        size_t i0 = root_->rowspan().first, m = root_->rowspan().second;
        size_t j0 = root_->colspan().first, n = root_->colspan().second;
        Eigen::MatrixXcd mat(m, n);
        to_dense_dfs(root_, mat, i0, j0);
        return mat;
    }

    void Matrix::truncate(size_t maxr) {
        std::for_each (EXEC_POLICY lowrk_.begin(), lowrk_.end(), [&](auto &lowrk) {
            size_t n = lowrk.second.second.cols(), r = lowrk.second.first.cols();
            if (r < 2 || (maxr > 0 && r <= maxr))
                return;
            Eigen::MatrixXcd A = lowrk.second.first, B = lowrk.second.second;
            Eigen::HouseholderQR<Eigen::MatrixXcd> qr(B.adjoint());
            Eigen::MatrixXcd Q = qr.householderQ() * Eigen::MatrixXd::Identity(n, r);
            Eigen::MatrixXcd R = qr.matrixQR().topRows(r).triangularView<Eigen::Upper>();
            Eigen::JacobiSVD<Eigen::MatrixXcd> svd(A * R.adjoint(), Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::VectorXd sv = svd.singularValues();
            Eigen::MatrixXd sigma = sv.asDiagonal();
            if (maxr > 0) {
                lowrk.second = {svd.matrixU().leftCols(maxr), sigma.topRows(maxr) * (Q * svd.matrixV()).adjoint()};
            } else { // max_rank = 0
                // discard singular values which are small enough
                Eigen::VectorXd sv2 = sv.array().square();
                double err = sv2.sum(), denom = std::sqrt(err);
                for (size_t k = 1; k < r; ++k) {
                    err -= sv2(k - 1);
                    if (std::sqrt(err) < eps_ * denom) { // truncate to rank = k
                        lowrk.second = {svd.matrixU().leftCols(k), sigma.topRows(k) * (Q * svd.matrixV()).adjoint()};
                        //std::cout << "Error: " << (A * B - lowrk.second.first * lowrk.second.second).norm() / (A * B).norm() << std::endl;
                        break;
                    }
                }
            }
        });
    }


}


