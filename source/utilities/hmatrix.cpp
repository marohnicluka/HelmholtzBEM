/**
 * \file hmatrix.cpp
 *
 * \brief This file contains the implementation of the
 * hierarchical matrix data type class.
 *
 * (c) 2024 Luka Marohnić
 */

//#define USE_H2LIB_ACA 1
//#define ACA_CONVERGENCE_INFO 1
#define USE_BOUNDING_BOX_DISTANCE 1
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/adapted/boost_tuple.hpp>
BOOST_GEOMETRY_REGISTER_BOOST_TUPLE_CS(cs::cartesian)
#define PARALLELIZE 1
#ifdef PARALLELIZE
#include <execution>
#define EXEC_POLICY std::execution::par,
#else
#define EXEC_POLICY
#endif
#include "hmatrix.hpp"
#include "h2lib_interface.hpp"
#include <iostream>
#include <chrono>
#include <set>
#include <numeric>

using namespace std::chrono;

namespace hierarchical {

    double polygon_diameter(const boost::geometry::model::polygon<boost::tuple<double, double> > &poly) {
#if 0
        double a = boost::geometry::area(poly);
        double p = boost::geometry::perimeter(poly);
        return 4.0 * a / p;
#else
        std::vector<Point> p;
        p.reserve(poly.outer().size());
        for (const auto &bpt : poly.outer())
            p.push_back({bpt.get<0>(), bpt.get<1>()});
        double max_dst = 0.0;
        size_t i, j = 1, n = p.size();
        if (n < 2)
            return 0.0;
        for (i = 0; i < n; ++i) {
            while (true) {
                double cur_dst = (p[i] - p[j]).squaredNorm();
                double nxt_dst = (p[i] - p[(j + 1) % n]).squaredNorm();
                if (nxt_dst > cur_dst)
                    j = (j + 1) % n;
                else break;
            }
            max_dst = std::max(max_dst, (p[i] - p[j]).squaredNorm());
        }
        return std::sqrt(max_dst);
#endif
    }

    // Compute the distance between polygonal lines p1 and p2
    double distance_between_polygons_with_diameters(const std::vector<Point> &p1, const std::vector<Point> &p2, double &d1, double &d2) {
        boost::geometry::model::polygon<boost::tuple<double, double> > bp1, bp2, ch1, ch2;
        for (const auto &p : p1)
            boost::geometry::append(bp1, boost::make_tuple(p(0), p(1)));
        for (const auto &p : p2)
            boost::geometry::append(bp2, boost::make_tuple(p(0), p(1)));
        boost::geometry::convex_hull(bp1, ch1);
        boost::geometry::convex_hull(bp2, ch2);
        d1 = polygon_diameter(ch1);
        d2 = polygon_diameter(ch2);
        return boost::geometry::distance(ch1, ch2);
    }

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
            xspan.first = std::min(xspan.first, (*iter)(0));
            xspan.second = std::max(xspan.second, (*iter)(0));
            yspan.first = std::min(yspan.first, (*iter)(1));
            yspan.second = std::max(yspan.second, (*iter)(1));
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


    double PanelGeometry::cluster_distance_with_diameters(const hierarchical::IPair& span1, const hierarchical::IPair& span2, double &d1, double &d2) const {
#ifdef USE_BOUNDING_BOX_DISTANCE
        auto bb1 = BoundingBox(begin() + span1.first, begin() + span1.first + span1.second + 1);
        auto bb2 = BoundingBox(begin() + span2.first, begin() + span2.first + span2.second + 1);
        d1 = bb1.diameter();
        d2 = bb2.diameter();
        return bb1.distance(bb2);
#else
        std::vector<Point> p(begin() + span1.first, begin() + span1.first + span1.second + 1);
        std::vector<Point> q(begin() + span2.first, begin() + span2.first + span2.second + 1);
        return distance_between_polygons_with_diameters(p, q, d1, d2);
#endif
    }

    Block * Block::make_son(const IPair& rowspan, const IPair& colspan, const PanelGeometry& pg) {
        sons_.push_back(new Block(rowspan, colspan, pg, divs_));
        return sons_.back();
    }

    bool Block::is_admissible(double eta) const {
        assert(eta > 0.);
        double d1, d2;
        double dist = pg_.cluster_distance_with_diameters(rowspan_, colspan_, d1, d2);
        if (dist <= 0.)
            return false;
        return std::min(d1, d2) <= eta * dist;
    }

    BlockTree::BlockTree(const PanelGeometry& pg, double eta, size_t divs)
        : pg_(pg), eta_(eta), min_size_(1 + (size_t)std::round(std::log2((double)pg.size()))), divs_(divs),
          node_count_(0), admissible_nodes_count_(0)
    {
        IPair span = {0, pg.size() - 1};
        root_ = new Block(span, span, pg, divs_);
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
        for (const auto &colspan : colspans) {
            for (const auto &rowspan : rowspans) {
                node_count_++;
                make_subtree(node->make_son(rowspan, colspan, pg_));
            }
        }
    }

    size_t random_index(size_t n) {
        return (size_t)(rand() % (int)n);
    }

    template<typename T>
    T random_element(const std::vector<T> &v) {
        return v[random_index(v.size())];
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
        Eigen::MatrixXcd mat(rowspan.second, colspan.second);
        std::vector<size_t> ind(rowspan.second + colspan.second);
        std::iota(ind.begin(), ind.begin() + rowspan.second, rowspan.first);
        std::iota(ind.begin() + rowspan.second, ind.end(), colspan.first);
        f(ind.begin(), ind.begin() + rowspan.second, mat);
        return mat;
    }

    size_t argmax(const Eigen::ArrayXd &v, std::vector<size_t> *indices, double &maxelem, bool rm = false) {
        assert(indices == NULL || (!indices->empty() && indices->back() < v.size()));
        size_t ret;
        if (indices == NULL) {
            maxelem = v.maxCoeff(&ret);
        } else {
            maxelem = -std::numeric_limits<double>::infinity();
            auto it = indices->begin(), it_max = indices->end();
            for (; it != indices->end(); ++it) {
                double val = v(*it);
                if (val > maxelem) {
                    maxelem = val;
                    it_max = it;
                }
            }
            assert (it_max != indices->end()); // v must contain a finite element
            ret = *it_max;
            if (rm)
                indices->erase(it_max);
        }
        return ret;
    }

    void remove_element(std::vector<size_t> &indices, size_t i) {
        auto it = std::lower_bound(indices.begin(), indices.end(), i);
        assert(it != indices.end() && *it == i);
        indices.erase(it);
    }

    Eigen::VectorXcd row_entries(MatrixFunction &f, size_t row, const IPair &colspan) {
        Eigen::MatrixXcd mat(1, colspan.second);
        std::vector<size_t> ind(colspan.second + 1);
        ind.front() = row;
        std::iota(ind.begin() + 1, ind.end(), colspan.first);
        f(ind.begin(), ind.begin() + 1, mat);
        return mat.row(0);
    }

    Eigen::VectorXcd column_entries(MatrixFunction &f, const IPair &rowspan, size_t col) {
        Eigen::MatrixXcd mat(rowspan.second, 1);
        std::vector<size_t> ind(rowspan.second + 1);
        ind.front() = col;
        std::iota(ind.begin() + 1, ind.end(), rowspan.first);
        f(ind.begin() + 1, ind.begin(), mat);
        return mat.col(0);
    }

    bool is_practically_zero(double x) {
#if 1
        auto res = std::fpclassify(x);
        return res == FP_ZERO || res == FP_SUBNORMAL;
#else
        return std::abs(x) < std::numeric_limits<double>::epsilon();
#endif
    }

    std::pair<Eigen::MatrixXcd,Eigen::MatrixXcd> Matrix::aca(MatrixFunction &mat, const IPair &rowspan, const IPair &colspan, double eps) {
        // number of rows/columns of the submatrix A_{rowspan,colspan}
        size_t m = rowspan.second, n = colspan.second, k;
        // offset in the input matrix, (i0,j0) corresponds to the top-left corner of the submatrix
        size_t i0 = rowspan.first, j0 = colspan.first;
        // maximal allowed rank for low-rank matrices, estimated rank
        size_t max_rank = std::min(m, n) / 2, est_rank = std::round(-std::log(eps));
        // storage for matrices U, V
        Eigen::MatrixXcd U(m, est_rank), V(est_rank, n);
        // pivot/reference row/column indices, initial rank
        size_t rank = 0, row_piv, col_piv;
        size_t row_ref, col_ref, new_row_ref, new_col_ref;
        // randomly choose reference row and column
        row_ref = random_index(m);
        col_ref = random_index(n);
        // get reference row and column
        auto reference_row = row_entries(mat, i0 + row_ref, colspan);
        auto reference_column = column_entries(mat, rowspan, j0 + col_ref);
        // pivoting row/column indices, which will be dropped one by one
        std::vector<size_t> available_row_indices(m), available_column_indices(n);
        std::iota(available_row_indices.begin(), available_row_indices.end(), 0);
        std::iota(available_column_indices.begin(), available_column_indices.end(), 0);
        // tracking norms and absolute values of pivoting elements
        double mu = 0.0, u2v2, r, s;
        bool converged = false;
        auto get_pivot_row = [&](Eigen::VectorXcd &v) {
            if (row_piv == row_ref) {
                // pivot row coincides with the reference row
                v = reference_row;
                // choose another reference row
                do new_row_ref = random_element(available_row_indices);
                while (new_row_ref == row_ref);
                row_ref = new_row_ref;
                reference_row = row_entries(mat, i0 + row_ref, colspan);
                if (rank)
                    reference_row -= U.row(row_ref).head(rank) * V.topRows(rank);
            } else {
                v = row_entries(mat, i0 + row_piv, colspan);
                if (rank)
                    v -= U.row(row_piv).head(rank) * V.topRows(rank);
            }
        };
        auto get_pivot_column = [&](Eigen::VectorXcd &u) {
            if (col_piv == col_ref) {
                // pivot column coincides with the reference column
                u = reference_column;
                // choose another reference column
                do new_col_ref = random_element(available_column_indices);
                while (new_col_ref == col_ref);
                col_ref = new_col_ref;
                reference_column = column_entries(mat, rowspan, j0 + col_ref);
                if (rank)
                    reference_column -= U.leftCols(rank) * V.col(col_ref).head(rank);
            } else {
                u = column_entries(mat, rowspan, j0 + col_piv);
                if (rank)
                    u -= U.leftCols(rank) * V.col(col_piv).head(rank);
            }
        };
        for (k = 1; k <= max_rank; ++k) {
            assert(!available_row_indices.empty() && !available_column_indices.empty());
            Eigen::VectorXcd u, v;
            // find the largest available element in the reference row/column
            row_piv = argmax(reference_column.array().abs(), &available_row_indices, r);
            col_piv = argmax(reference_row.array().abs(), &available_column_indices, s);
            if (is_practically_zero(r) && is_practically_zero(s))
                break; // there are no more pivots to select
            // compute pivot row and column, stop if one of them is zero vector
            if (r >= s) {
                get_pivot_row(v);
                col_piv = argmax(v.array().abs(), &available_column_indices, s, true);
                if (is_practically_zero(s)) break;
                get_pivot_column(u);
                if (u.isZero()) break;
                v /= v(col_piv);
            } else {
                get_pivot_column(u);
                row_piv = argmax(u.array().abs(), &available_row_indices, r, true);
                if (is_practically_zero(r)) break;
                get_pivot_row(v);
                if (v.isZero()) break;
                u /= u(row_piv);
            }
            // estimate residual norm and input matrix norm
            u2v2 = u.squaredNorm() * v.squaredNorm();
            mu += u2v2;
            if (rank)
                mu += 2. * ((U.leftCols(rank).transpose() * u).array() * (V.topRows(rank) * v).array()).abs().sum();
            // update the residual
            reference_row -= u(row_ref) * v;
            reference_column -= v(col_ref) * u;
            // append new column to U and new row to V
            rank++;
            if (rank > est_rank) {
                U.conservativeResize(Eigen::NoChange, rank);
                V.conservativeResize(rank, Eigen::NoChange);
            }
            U.col(rank - 1) = u;
            V.row(rank - 1) = v;
            // if the relative residual error is small enough, stop
            if (std::sqrt(u2v2) < eps * std::sqrt(mu)) {
                converged = true;
                break;
            }
        }
        if (rank) {
            if (rank < est_rank) {
                U.conservativeResize(Eigen::NoChange, rank);
                V.conservativeResize(rank, Eigen::NoChange);
            }
#ifdef ACA_CONVERGENCE_INFO
            if (!converged) {
                auto C = extract_dense_submatrix(mat, rowspan, colspan);
                double err = (C - U * V).norm() / C.norm();
                std::cout << "ACA did not converge, size = " << m << ", rank = " << rank << ", error = " << err << std::endl;
            }
#endif
            return {U, V};
        }
        // rank = 0, return empty U and V
        return {Eigen::MatrixXcd(), Eigen::MatrixXcd()};
    }

    Matrix::Matrix(hierarchical::MatrixFunction& mat, const hierarchical::BlockTree& tree, double eps)
        : tree_(tree), root_(tree.root()), eps_(eps)
    {
        create_blocks(mat, root_);
    }

    void Matrix::create_blocks(MatrixFunction &mat, Block *block) {
        if (tree_.is_low_rank(block)) {
            size_t m = block->rowspan().second, n = block->colspan().second, rk;
            std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd> UV;
#ifdef USE_H2LIB_ACA
            Eigen::MatrixXcd U, V;
            rk = h2lib::aca_with_partial_pivoting(mat, block->rowspan(), block->colspan(), eps_, U, V);
            UV = {U, V.adjoint()};
#else
            UV = aca(mat, block->rowspan(), block->colspan(), eps_);
            rk = UV.first.cols();
#endif
            if (rk * (m + n) >= m * n) {
                std::cout << "Storing admissible block as dense" << std::endl;
                dense_.insert({block, extract_dense_submatrix(mat, block->rowspan(), block->colspan())});
            } else if (rk == 0) {
                lowrk_.insert({block, {Eigen::MatrixXcd(), Eigen::MatrixXcd()}});
            } else {
                lowrk_.insert({block, {UV.first, UV.second.adjoint()}});
            }
        } else if (block->is_leaf())
            dense_.insert({block, extract_dense_submatrix(mat, block->rowspan(), block->colspan())});
        else {
            block->reset_sons_iterator();
            Block *son;
            while ((son = block->next_son()) != NULL)
                create_blocks(mat, son);
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
                    assert(it->second.first.rows() == m && it->second.second.rows() == n);
                    mat.block(i, j, m, n) = it->second.first * it->second.second.adjoint();
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
            size_t r = lowrk.second.first.cols();
            if (r < 2 || (maxr > 0 && r <= maxr))
                return;
            assert(lowrk.second.second.cols() == r);
            Eigen::HouseholderQR<Eigen::MatrixXcd> qr(lowrk.second.second);
            Eigen::MatrixXcd Q = qr.householderQ() * Eigen::MatrixXd::Identity(lowrk.second.second.rows(), r);
            Eigen::MatrixXcd R = qr.matrixQR().topRows(r).triangularView<Eigen::Upper>();
            Eigen::JacobiSVD<Eigen::MatrixXcd> svd(lowrk.second.first * R.adjoint(), Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::VectorXd sv = svd.singularValues();
            Eigen::MatrixXd sigma = sv.asDiagonal();
            if (maxr > 0) {
                lowrk.second = {svd.matrixU().leftCols(maxr),  Q * svd.matrixV().leftCols(maxr) * sigma.topLeftCorner(maxr, maxr)};
            } else { // max_rank = 0
                // discard singular values which are small enough
                Eigen::VectorXd sv2 = sv.array().square();
                double err = sv2.sum(), denom = std::sqrt(err);
                for (size_t k = 1; k < r; ++k) {
                    err -= sv2(k - 1);
                    if (std::sqrt(err) < eps_ * denom) { // truncate to rank = k
                        lowrk.second = {svd.matrixU().leftCols(k), Q * svd.matrixV().leftCols(k) * sigma.topLeftCorner(k, k)};
                        break;
                    }
                }
            }
        });
    }

}


