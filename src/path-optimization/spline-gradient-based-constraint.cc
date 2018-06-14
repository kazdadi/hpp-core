// Copyright (c) 2017, Joseph Mirabel
// Authors: Joseph Mirabel (joseph.mirabel@laas.fr)
//
// This file is part of hpp-core.
// hpp-core is free software: you can redistribute it
// and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version.
//
// hpp-core is distributed in the hope that it will be
// useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Lesser Public License for more details.  You should have
// received a copy of the GNU Lesser General Public License along with
// hpp-core. If not, see <http://www.gnu.org/licenses/>.

#define HPP_DEBUG 1

#include <hpp/core/path-optimization/spline-gradient-based-constraint.hh>

#include <hpp/util/exception-factory.hh>
#include <hpp/util/timer.hh>

#include <hpp/pinocchio/device.hh>

#include <hpp/constraints/svd.hh>

#include <hpp/core/config-projector.hh>
#include <hpp/core/problem.hh>
#include <hpp/core/collision-path-validation-report.hh>

#include <path-optimization/spline-gradient-based/cost.hh>
#include <path-optimization/spline-gradient-based/collision-constraint.hh>
#include <path-optimization/spline-gradient-based/eiquadprog_2011.hpp>

namespace hpp {
  namespace core {
    using pinocchio::Device;

    namespace pathOptimization {
      typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorMatrix_t;
      typedef Eigen::Map<const vector_t> ConstVectorMap_t;
      typedef Eigen::Map<      vector_t>      VectorMap_t;

      typedef Eigen::BlockIndex BlockIndex;

      HPP_DEFINE_TIMECOUNTER(SGB_constraintDecomposition);
      HPP_DEFINE_TIMECOUNTER(SGB_qpDecomposition);
      // HPP_DEFINE_TIMECOUNTER(SGB_findNewConstraint);
      HPP_DEFINE_TIMECOUNTER(SGB_qpSolve);

      template <int NbRows>
        VectorMap_t reshape (Eigen::Matrix<value_type, NbRows, Eigen::Dynamic, Eigen::RowMajor>& parameters)
        {
          return VectorMap_t (parameters.data(), parameters.size());
        }

      template <int NbRows>
        ConstVectorMap_t reshape (const Eigen::Matrix<value_type, NbRows, Eigen::Dynamic, Eigen::RowMajor>& parameters)
        {
          return ConstVectorMap_t (parameters.data(), parameters.size());
        }

      template <int _PB, int _SO>
        SplineGradientBasedConstraint<_PB, _SO>::SplineGradientBasedConstraint (const Problem& problem)
        : Base (problem)
          , checkOptimum_ (false)
      {}

      // ----------- Convenience class -------------------------------------- //

      /** \f{eqnarray*}{
       *  min & 0.5 * x^T H x + b^T x \\
       *      & lc.J * x = lc.b
       *  \f}
       **/
      template <int _PB, int _SO>
        struct SplineGradientBasedConstraint<_PB, _SO>::QuadraticProblem
        {
          typedef Eigen::JacobiSVD < matrix_t > Decomposition_t;
          typedef Eigen::LLT <matrix_t, Eigen::Lower> LLT_t;

          QuadraticProblem (size_type inputSize) :
            H (inputSize, inputSize), b (inputSize),
            dec (inputSize, inputSize, Eigen::ComputeThinU | Eigen::ComputeThinV),
            xStar (inputSize)
          {
            H.setZero();
            b.setZero();
            bIsZero = true;
          }

          QuadraticProblem (const QuadraticProblem& QP, const LinearConstraint& lc) :
            H (lc.PK.cols(), lc.PK.cols()), b (lc.PK.cols()), bIsZero (false),
            dec (lc.PK.cols(), lc.PK.cols(), Eigen::ComputeThinU | Eigen::ComputeThinV),
            xStar (lc.PK.cols())
          {
            QP.reduced (lc, *this);
          }

          QuadraticProblem (const QuadraticProblem& QP) :
            H (QP.H), b (QP.b), bIsZero (QP.bIsZero),
            dec (QP.dec), xStar (QP.xStar)
          {}

          void addRows (const std::size_t& nbRows)
          {
            H.conservativeResize(H.rows() + nbRows, H.cols());
            b.conservativeResize(b.rows() + nbRows, b.cols());

            H.bottomRows(nbRows).setZero();
          }

          /*/ Compute the problem
           *  \f{eqnarray*}{
           *  min & 0.5 * x^T H x + b^T x \\
           *      & lc.J * x = lc.b
           *  \f}
           **/
          void reduced (const LinearConstraint& lc, QuadraticProblem& QPr) const
          {
            matrix_t H_PK (H * lc.PK);
            QPr.H.noalias() = lc.PK.transpose() * H_PK;
            QPr.b.noalias() = H_PK.transpose() * lc.xStar;
            if (!bIsZero) {
              QPr.b.noalias() += lc.PK.transpose() * b;
            }
            QPr.bIsZero = false;
          }

          void decompose ()
          {
            HPP_SCOPE_TIMECOUNTER(SGB_qpDecomposition);
            dec.compute(H);
            assert(dec.rank() == H.rows());
          }

          void solve ()
          {
            xStar.noalias() = - dec.solve(b);
          }

          void computeLLT()
          {
            HPP_SCOPE_TIMECOUNTER(SGB_qpDecomposition);
            trace = H.trace();
            llt.compute(H);
          }

          double solve(const LinearConstraint& ce, const LinearConstraint& ci)
          {
            HPP_SCOPE_TIMECOUNTER(SGB_qpSolve);
            // min   0.5 * x G x + g0 x
            // s.t.  CE^T x + ce0 = 0
            //       CI^T x + ci0 >= 0
            return solve_quadprog2 (llt, trace, b,
                ce.J.transpose(), - ce.b,
                ci.J.transpose(), - ci.b,
                xStar, activeConstraint, activeSetSize);
          }

          // model
          matrix_t H;
          vector_t b;
          bool bIsZero;

          // Data
          LLT_t llt;
          value_type trace;
          Eigen::VectorXi activeConstraint;
          int activeSetSize;

          // Data
          Decomposition_t dec;
          vector_t xStar;
        };

      template <int _PB, int _SO>
        struct SplineGradientBasedConstraint<_PB, _SO>::CollisionFunctions
        {
          void addConstraint (const CollisionFunctionPtr_t& f,
              const std::size_t& idx,
              const size_type& row,
              const value_type& r)
          {
            assert (f->outputSize() == 1);
            functions.push_back(f);
            splineIds.push_back(idx);
            rows.push_back(row);
            ratios.push_back(r);
          }

          void removeLastConstraint (const std::size_t& n, LinearConstraint& lc)
          {
            assert (functions.size() >= n && std::size_t(lc.J.rows()) >= n);

            const std::size_t nSize = functions.size() - n;
            functions.resize(nSize);
            splineIds.resize(nSize);
            rows.resize(nSize);
            ratios.resize(nSize);

            lc.J.conservativeResize(lc.J.rows() - n, lc.J.cols());
            lc.b.conservativeResize(lc.b.rows() - n, lc.b.cols());
          }

          // Compute linearization
          // b = f(S(t))
          // J = Jf(S(p, t)) * dS/dp
          // f(S(t)) = b -> J * P = b
          void linearize (const SplinePtr_t& spline, const SplineOptimizationData& sod,
              const std::size_t& fIdx, LinearConstraint& lc)
          {
            const CollisionFunctionPtr_t& f = functions[fIdx];

            const size_type row = rows[fIdx],
            nbRows = 1,
            rDof = f->inputDerivativeSize();
            const value_type t = spline->length() * ratios[fIdx];

            q.resize(f->inputSize());
            (*spline) (q, t);

            // Evaluate explicit functions
            if (sod.es) sod.es->solve(q);

            LiegroupElement v (f->outputSpace());
            f->value(v, q);

            J.resize(f->outputSize(), f->inputDerivativeSize());
            f->jacobian(J, q);

            // Apply chain rule if necessary
            if (sod.es) {
              Js.resize(sod.es->derSize(), sod.es->derSize());
              sod.es->jacobian(Js, q);

              sod.es->inDers().lview(J) =
                sod.es->inDers().lview(J).eval() +
                sod.es->outDers().transpose().rview(J).eval()
                * sod.es->viewJacobian(Js).eval();
              sod.es->outDers().transpose().lview(J).setZero();
            }

            spline->parameterDerivativeCoefficients(paramDerivativeCoeff, t);

            const size_type col = splineIds[fIdx] * Spline::NbCoeffs * rDof;
            for (size_type i = 0; i < Spline::NbCoeffs; ++i)
              lc.J.block (row, col + i * rDof, nbRows, rDof).noalias()
                = paramDerivativeCoeff(i) * J;

            lc.b.segment(row, nbRows) =
              lc.J.block (row, col, nbRows, Spline::NbCoeffs * rDof)
              * spline->rowParameters();
          }

          void linearize (const Splines_t& splines, const SplineOptimizationDatas_t& ss, LinearConstraint& lc)
          {
            for (std::size_t i = 0; i < functions.size(); ++i)
              linearize(splines[splineIds[i]], ss[i], i, lc);
          }

          std::vector<CollisionFunctionPtr_t> functions;
          std::vector<std::size_t> splineIds;
          std::vector<size_type> rows;
          std::vector<value_type> ratios;

          mutable Configuration_t q;
          mutable matrix_t J, Js;
          mutable typename Spline::BasisFunctionVector_t paramDerivativeCoeff;
        };

      // ----------- Resolution steps --------------------------------------- //

      template <int _PB, int _SO>
        typename SplineGradientBasedConstraint<_PB, _SO>::Ptr_t SplineGradientBasedConstraint<_PB, _SO>::create
        (const Problem& problem)
        {
          SplineGradientBasedConstraint* ptr = new SplineGradientBasedConstraint (problem);
          Ptr_t shPtr (ptr);
          return shPtr;
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::addProblemConstraints
        (const PathVectorPtr_t& init, const Splines_t& splines, LinearConstraint& lc, SplineOptimizationDatas_t& ss) const
        {
          assert (init->numberPaths() == splines.size() && ss.size() == splines.size());
          for (std::size_t i = 0; i < splines.size(); ++i) {
            addProblemConstraintOnPath (init->pathAtRank(i), i, splines[i], lc, ss[i]);
          }
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::addProblemConstraintOnPath
        (const PathPtr_t& path, const size_type& idxSpline, const SplinePtr_t& spline, LinearConstraint& lc, SplineOptimizationData& sod) const
        {
          ConstraintSetPtr_t cs = path->constraints();
          if (cs) {
            ConfigProjectorPtr_t cp = cs->configProjector();
            if (cp) {
              const HybridSolver& hs = cp->solver();
              const constraints::ExplicitSolver& es = hs.explicitSolver();

              // Get the active parameter row selection.
              value_type guessThreshold = problem().getParameter ("SplineGradientBasedConstraint/guessThreshold", value_type(-1));
              Eigen::RowBlockIndices select = computeActiveParameters (path, hs, guessThreshold);

              const size_type rDof = robot_->numberDof(),
                    col  = idxSpline * Spline::NbCoeffs * rDof,
                    row = lc.J.rows(),
                    nOutVar = select.nbIndices();

              sod.set = cs;
              sod.es.reset(new ExplicitSolver(es));
              sod.activeParameters = RowBlockIndices (BlockIndex::difference
                  (BlockIndex::segment_t(0, rDof),
                   select.indices()));
              hppDout (info, "Path " << idxSpline << ": do not change this dof " << select);
              hppDout (info, "Path " << idxSpline << ": active dofs " << sod.activeParameters);

              // Add nOutVar constraint per coefficient.
              lc.addRows(Spline::NbCoeffs * nOutVar);
              matrix_t I = select.rview(matrix_t::Identity(rDof, rDof));
              for (size_type k = 0; k < Spline::NbCoeffs; ++k) {
                lc.J.block  (row + k * nOutVar, col + k * rDof, nOutVar, rDof) = I;
                lc.b.segment(row + k * nOutVar, nOutVar) = I * spline->parameters().row(k).transpose();
              }

              assert ((lc.J.block(row, col, Spline::NbCoeffs * nOutVar, rDof * Spline::NbCoeffs) * spline->rowParameters())
                  .isApprox(lc.b.segment(row, Spline::NbCoeffs * nOutVar)));
            }
          }
        }

      template <int _PB, int _SO>
        Eigen::RowBlockIndices SplineGradientBasedConstraint<_PB, _SO>::computeActiveParameters
        (const PathPtr_t& path, const HybridSolver& hs, const value_type& guessThr, const bool& useExplicitInput) const
        {
          const constraints::ExplicitSolver& es = hs.explicitSolver();

          BlockIndex::segments_t implicitBI, explicitBI;

          // Handle implicit part
          if (hs.reducedDimension() > 0) {
            implicitBI = hs.implicitDof();

            hppDout (info, "Solver " << hs
                << '\n' << Eigen::RowBlockIndices(implicitBI));

            // in the case of PR2 passing a box from right to left hand,
            // the double grasp is a loop closure so the DoF of the base are
            // not active (one can see this in the Jacobian).
            // They should be left unconstrained.
            // TODO I do not see any good way of guessing this since it is
            // the DoF of the base are not active only on the submanifold
            // satisfying the constraint. It has to be dealt with in
            // hpp-manipulation.

            // If requested, check if the jacobian has columns of zeros.
            BlockIndex::segments_t passive;
            if (guessThr >= 0) {
              matrix_t J (hs.dimension(), es.inDers().nbIndices());
              hs.computeValue<true>(path->initial());
              hs.updateJacobian(path->initial());
              hs.getReducedJacobian(J);
              size_type j = 0, k = 0;
              for (size_type r = 0; r < J.cols(); ++r) {
                if (J.col(r).isZero(guessThr)) {
                  size_type idof = es.inDers().indices()[j].first + k;
                  passive.push_back(BlockIndex::segment_t (idof, 1));
                  hppDout (info, "Deactivated dof (thr=" << guessThr
                      << ") " << idof << ". J = " << J.col(r).transpose());
                }
                k++;
                if (k >= es.inDers().indices()[j].second) {
                  j++;
                  k = 0;
                }
              }
              BlockIndex::sort(passive);
              BlockIndex::shrink(passive);
              hppDout (info, "Deactivated dof (thr=" << guessThr
                  << ") " << Eigen::ColBlockIndices(passive)
                  << "J = " << J);
              implicitBI = BlockIndex::difference (implicitBI, passive);
            }
          } else if (useExplicitInput) {
            Eigen::ColBlockIndices esadp = es.activeDerivativeParameters();
            implicitBI = esadp.indices();
          }

          // Handle explicit part
          explicitBI = es.outDers().indices();

          // Add both
          implicitBI.insert (implicitBI.end(),
              explicitBI.begin(), explicitBI.end());
          Eigen::RowBlockIndices rbi (implicitBI);
          rbi.updateIndices<true, true, true>();
          return rbi;
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::addCollisionConstraint
        (const std::size_t idxSpline,
         const SplinePtr_t& spline, const SplinePtr_t& nextSpline,
         const SplineOptimizationData& sod,
         const CollisionPathValidationReportPtr_t& report,
         LinearConstraint& collision,
         CollisionFunctions& functions) const
        {
          hppDout (info, "Collision on spline " << idxSpline << " at ratio (in [0,1]) = " << report->parameter / nextSpline->length());
          CollisionFunctionPtr_t cc =
            CollisionFunction::create (robot_, spline, nextSpline, report);

          collision.addRows(cc->outputSize());
          functions.addConstraint (cc, idxSpline,
              collision.J.rows() - 1,
              report->parameter / nextSpline->length());

          functions.linearize(spline, sod, functions.functions.size() - 1, collision);
        }

      template <int _PB, int _SO>
        bool SplineGradientBasedConstraint<_PB, _SO>::findNewConstraint
        (LinearConstraint& constraint,
         LinearConstraint& collision,
         LinearConstraint& collisionReduced,
         CollisionFunctions& functions,
         const std::size_t iF,
         const SplinePtr_t& spline,
         const SplineOptimizationData& sod) const
        {
          // HPP_SCOPE_TIMECOUNTER(SGB_findNewConstraint);
          bool solved = false;
          Configuration_t q (robot_->configSize());
          CollisionFunctionPtr_t function = functions.functions[iF];

          solved = constraint.reduceConstraint(collision, collisionReduced);

          size_type i = 5;
          while (not solved) {
            if (i == 0) {
              functions.removeLastConstraint (1, collision);
              hppDout (warning, "Could not find a suitable collision constraint. Removing it.");
              return false;
            }
            hppDout (info, "Looking for collision which does not make the constraint rank deficient.");
            // interpolate at alpha
            pinocchio::interpolate<hpp::pinocchio::LieGroupTpl>
              (robot_, function->qFree_, function->qColl_, 0.5, q);
            hppDout (info, "New q: " << q.transpose());
            // update the constraint
            function->updateConstraint (q);
            functions.linearize(spline, sod, iF, collision);
            // check the rank
            solved = constraint.reduceConstraint(collision, collisionReduced);
            --i;
          }
          return true;
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::interrupt()
        {
          interrupt_ = true;
        }

      template <int _PB, int _SO>
        value_type SplineGradientBasedConstraint<_PB, _SO>::backtrackingLineSearch
        (SquaredLength<Spline, 1> cost, Splines_t& splines, LinearConstraint constraint,
         const vector_t x, vector_t direction, value_type derivative,
         value_type initialStep, value_type factor, value_type threshold) const
        {
          direction = direction/direction.norm();
          value_type step = initialStep/factor;
          value_type f_x;
          value_type f_step;
          getFullSplines(x, splines, constraint);
          cost.value(f_x, splines);
          do {
            step *= factor;
            vector_t x_new = x + step*direction;
            getFullSplines(x_new, splines, constraint);
            cost.value(f_step, splines);
          }
          while (f_x - f_step < threshold * step * derivative);
          getFullSplines(x, splines, constraint);
          return step;
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::getFullSplines
        (const vector_t reducedParams, Splines_t& fullSplines, LinearConstraint constraint) const
        {
          vector_t param = constraint.xStar + (constraint.PK*reducedParams);
          updateSplines(fullSplines, param);
        }

      template <int _PB, int _SO>
        size_type SplineGradientBasedConstraint<_PB, _SO>::getNbConstraints
        (const Splines_t splines) const
        {
          size_type nbConstraints = 0;
          for (std::size_t i = 1; i < splines.size(); ++i)
          {
            ConfigProjectorPtr_t configProj = splines[i]->constraints()->configProjector();
            if (configProj) {
              nbConstraints += configProj->dimension();
            }
          }
          return nbConstraints;
        }

      template <int _PB, int _SO>
        value_type SplineGradientBasedConstraint<_PB, _SO>::solveSubSubProblem
        (vector_t c, vector_t k, value_type r, value_type error) const
        {
          // Upper bound : ||c||/r
          // Lower bound : sum (c_i^2)^.5 / r; i s.t k_i > 0
          value_type u = 0;
          for (std::size_t i = 0; i < k.size(); ++i)
          {
            if (k[i] > 0) u += std::pow(c[i], 2);
          }
          u = std::sqrt(u)/r;
          if (u == 0) return 0;
          value_type diff;
          value_type derivative;
          do
          {
            derivative = 0;
            diff = -std::pow(r, 2);
            for (std::size_t i = 0; i < k.size(); ++i)
            {
              diff += std::pow(c[i]/(k[i] + u), 2);
              derivative += -2 * std::pow(c[i], 2)/std::pow(k[i] + u, 3);
            }
            if (diff < 0) u /= 2;
            else u = u - diff/derivative;
          }
          while (std::abs(diff/derivative) > error);
          return u;
        }

      template <int _PB, int _SO>
        vector_t SplineGradientBasedConstraint<_PB, _SO>::solveQP
        (matrix_t A, vector_t b, value_type r) const
        {
          vector_t x(b.size());
          Eigen::SelfAdjointEigenSolver<matrix_t> eigenSolver(A);
          matrix_t V = eigenSolver.eigenvectors();
          vector_t d = eigenSolver.eigenvalues();
          if (d[0] > 0)
          {
            x = A.llt().solve(b);
            if (x.norm() <= r)
            {
              return x;
            }
          }
          vector_t c(b.size());
          c = V.transpose() * b;

          vector_t k(b.size());
          vector_t ones(b.size());
          ones.setOnes();
          k = d - d[0] * ones;

          // Solve sum_1^n [c_i/(u + k_i)]^2 = r^2, u > 0
          value_type u = solveSubSubProblem(c, k, r);

          // x = V * ((d + u*ones).cwiseInverse().asDiagonal() * (V.transpose() * b));
          
          return (A + u * matrix_t::Identity(b.size(), b.size())).llt().solve(b);;
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::getHessianFiniteDiff
        (Splines_t fullSplines, const vector_t reducedParams,
         std::vector<matrix_t>& hessianStack, value_type stepSize,
         LinearConstraint constraint) const
        {
          const std::size_t nParameters = fullSplines.size() * Spline::NbCoeffs;
          const std::size_t rDof = robot_->numberDof();
          const std::size_t dim = nParameters * rDof;

          vector_t x(dim);
          x = constraint.xStar + constraint.PK*reducedParams;
          vector_t step(rDof);
          step.setZero();
          Splines_t splines_plus;
          Splines_t splines_minus;
          Base::copy(fullSplines, splines_plus);
          Base::copy(fullSplines, splines_minus);
          for (std::size_t j = 0; j < rDof; ++j)
          {
            std::size_t constraintIndex = 0;
            step[j] = stepSize;

            for (std::size_t i = 1; i < fullSplines.size(); ++i)
            {
              ConfigProjectorPtr_t configProj = fullSplines[i]->constraints()->configProjector();
              if (configProj)
              {
                std::size_t numberOfConstraints = configProj->dimension();
                // TODO: update only current spline's first rDof parameters
                vector_t paramsPlus = x.segment (i*rDof*Spline::NbCoeffs,
                    rDof*Spline::NbCoeffs) + step;
                vector_t paramsMinus = x.segment (i*rDof*Spline::NbCoeffs,
                    rDof*Spline::NbCoeffs) - step;

                splines_plus[i]->rowParameters(paramsPlus);
                splines_minus[i]->rowParameters(paramsMinus);

                Configuration_t initial_plus = splines_plus[i]->initial();
                Configuration_t initial_minus = splines_minus[i]->initial();

                vector_t tmp_value(numberOfConstraints);

                matrix_t jac_plus(numberOfConstraints, rDof);
                matrix_t jac_minus(numberOfConstraints, rDof);
                jac_plus.setZero();
                jac_minus.setZero();

                configProj->computeValueAndJacobian(initial_plus, tmp_value, jac_plus);
                configProj->computeValueAndJacobian(initial_minus, tmp_value, jac_minus);

                matrix_t jac_diff(numberOfConstraints, rDof);
                jac_diff = (jac_plus - jac_minus)/(2*stepSize);

                for (std::size_t k = 0; k < numberOfConstraints; ++k)
                {
                  hessianStack[constraintIndex+k].row(j).
                    segment(rDof*Spline::NbCoeffs*i, rDof) = jac_diff.row(k);
                }
                constraintIndex += numberOfConstraints;
              }
            }
            step[j] = 0;
          }
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::getValueJacobianReduced
        (const Splines_t fullSplines, const vector_t reducedParams,
         vector_t& value, matrix_t& jacobian, LinearConstraint constraint) const
        {
          const size_type nParameters = fullSplines.size() * Spline::NbCoeffs;
          const size_type rDof = robot_->numberDof();
          value.resize(0);
          jacobian.resize(0, reducedParams.size());
          for (std::size_t i = 1; i < fullSplines.size(); ++i)
          {
            Configuration_t initial = fullSplines[i]->initial();
            ConfigProjectorPtr_t configProj = fullSplines[i]->constraints()->configProjector();
            if (configProj)
            {
              size_type numberOfConstraints = configProj->dimension();
              jacobian.conservativeResize(jacobian.rows()+numberOfConstraints, Eigen::NoChange);
              value.conservativeResize(value.size()+numberOfConstraints);
              value.bottomRows(numberOfConstraints).setZero();
              matrix_t fullJacobian(numberOfConstraints, nParameters*rDof);
              fullJacobian.setZero();
              configProj->computeValueAndJacobian(initial, value.bottomRows(numberOfConstraints),
                  fullJacobian.block(0, rDof*Spline::NbCoeffs*i, numberOfConstraints, rDof));
              jacobian.bottomRows(numberOfConstraints) = fullJacobian * constraint.PK;
            }
          }
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::addCollisionConstraintsValueJacobian
        (const Splines_t fullSplines, const vector_t reducedParams,
         vector_t& value, matrix_t& jacobian, LinearConstraint constraint,
         std::vector<DifferentiableFunctionPtr_t> collFunctions,
         std::vector<value_type> collValues,
         std::vector<std::size_t> indices,
         std::vector<value_type> ratios) const
        {
          std::size_t row = value.size();
          value.conservativeResize(row + collFunctions.size());
          jacobian.conservativeResize(row + collFunctions.size(), Eigen::NoChange);
          for (std::size_t i = 0; i < collFunctions.size(); ++i)
          {
            size_type rDof = robot_->numberDof();

            vector_t BernsteinCoeffs(int (Spline::NbCoeffs));
            matrix_t splinesJacobian(rDof, fullSplines.size()*Spline::NbCoeffs*rDof);
            fullSplines[indices[i]]->parameterDerivativeCoefficients(BernsteinCoeffs, ratios[i]);
            splinesJacobian.setZero();
            for (std::size_t j = 0; j < BernsteinCoeffs.size(); ++j)
            {
              splinesJacobian.block(0, Spline::NbCoeffs*rDof*indices[i] + rDof*j,
                  rDof, rDof) = BernsteinCoeffs[j] * matrix_t::Identity(rDof, rDof);
            }

            vector_t currentConfig(collFunctions[i]->inputSize());
            matrix_t collJacobian(1, collFunctions[i]->inputDerivativeSize());
            (*(fullSplines[indices[i]])) (currentConfig, ratios[i]);
            value[row] = (((*(collFunctions[i])) (currentConfig)).vector()[0] - collValues[i]);
            collFunctions[i]->jacobian(collJacobian, currentConfig);
            jacobian.row(row) = collJacobian*splinesJacobian*constraint.PK;
            ++row;
          }
        }

      template <int _PB, int _SO>
        bool SplineGradientBasedConstraint<_PB, _SO>::validateConstraints
        (const Splines_t fullSplines, const vector_t value,
         LinearConstraint constraint, value_type factor) const
        {
          std::size_t row = 0;
          for (std::size_t i = 1; i < fullSplines.size(); ++i)
          {
            ConfigProjectorPtr_t configProj = fullSplines[i]->constraints()->configProjector();
            if (configProj)
            {
              std::size_t numberOfConstraints = configProj->dimension();
              if (value.segment(row, numberOfConstraints).norm() > factor * configProj->errorThreshold())
                return false;
              row += numberOfConstraints;
            }
          }
          return true;
        }
      // ----------- Optimize ----------------------------------------------- //

      template <int _PB, int _SO>
        PathVectorPtr_t SplineGradientBasedConstraint<_PB, _SO>::optimize (const PathVectorPtr_t& path)
        {
          size_type maxIterations = problem().getParameter(
              "SplineGradientBasedConstraint/maxIterations", size_type(200));
          value_type stepThreshold = problem().getParameter(
              "SplineGradientBasedConstraint/stepThreshold", value_type(.01));
          value_type lineSearchTrustRadius = problem().getParameter(
              "SplineGradientBasedConstraint/lineSearchTrustRadius", value_type(1));
          value_type trustRadius = problem().getParameter(
              "SplineGradientBasedConstraint/trustRadius", value_type(.5));
          value_type stepSize = problem().getParameter(
              "SplineGradientBasedConstraint/stepSize", value_type(.00001));
          bool useHessian = problem().getParameter(
              "SplineGradientBasedConstraint/useHessian", true);
          bool checkJointBound = problem().getParameter(
              "SplineGradientBasedConstraint/checkJointBound", true);
          bool checkCollisions = problem().getParameter(
              "SplineGradientBasedConstraint/checkCollisions", true);

          PathVectorPtr_t tmp = PathVector::create (robot_->configSize(), robot_->numberDof());
          path->flatten(tmp);
          // Remove zero length path
          PathVectorPtr_t input = PathVector::create (robot_->configSize(), robot_->numberDof());
          for (std::size_t i = 0; i < tmp->numberPaths(); ++i) {
            PathPtr_t p = tmp->pathAtRank (i);
            if (p->length() > 0) input->appendPath (p);
          }
          robot_->controlComputation ((Device::Computation_t)(robot_->computationFlag() | Device::JACOBIAN));
          const size_type rDof = robot_->numberDof();

          // 1
          Splines_t splines;
          this->appendEquivalentSpline (input, splines);
          this->initializePathValidation(splines);
          const size_type nParameters = splines.size() * Spline::NbCoeffs;

          // 2
          enum { MaxContinuityOrder = int( (SplineOrder - 1) / 2) };
          const size_type orderContinuity = MaxContinuityOrder;

          LinearConstraint constraint (nParameters * rDof, 0);
          SplineOptimizationDatas_t solvers (splines.size(), SplineOptimizationData(rDof));
          this->addContinuityConstraints (splines, orderContinuity, solvers, constraint);

          // 3
          // TODO add weights
          SquaredLength<Spline, 1> cost (splines, rDof, rDof);

          // 4
          // splines = xStar + PK*reducedParams
          constraint.decompose();
          size_type reducedDim = nParameters*rDof-constraint.rank;
          vector_t reducedParams(reducedDim);
          vector_t fullParams(nParameters*rDof);
          getParameters(splines, fullParams);
          reducedParams = constraint.PK.transpose() * (fullParams-constraint.xStar);

          // Assuming the constraints are independent
          size_type nbConstraints = getNbConstraints(splines);
          hppDout(info, nbConstraints);
          matrix_t zeroMatrix(nParameters*rDof, nParameters*rDof);
          zeroMatrix.setZero();
          std::vector<matrix_t> hessianStack(nbConstraints, zeroMatrix);
          matrix_t fullObjectiveHessian(nParameters*rDof, nParameters*rDof);

          LinearConstraint boundConstraint (nParameters * rDof, 0);
          if (checkJointBound) {
            this->jointBoundConstraint(splines, boundConstraint);
            if (!this->validateBounds(splines, boundConstraint).empty())
              throw std::invalid_argument("Input path does not satisfy joint bounds");
          }
          if (checkCollisions) {
            if (!(this->validatePath(splines, true)).empty())
              throw std::invalid_argument("Input path contains a collision");
          }
          LinearConstraint boundConstraintReduced (constraint.PK.rows(), 0);
          constraint.reduceConstraint(boundConstraint, boundConstraintReduced, false);
          std::vector<std::size_t> activeBoundIndices;

          size_type numberOfIterations = 0;

          Splines_t newSplines;
          Base::copy(splines, newSplines);
          Splines_t collisionFreeSplines;
          Base::copy(splines, collisionFreeSplines);
          vector_t collisionFreeParams = reducedParams;
          value_type lengthSinceLastCheck = 0;
          std::size_t nCollisionChecks = 0;

          std::vector<DifferentiableFunctionPtr_t> collFunctions;
          std::vector<value_type> collValues;
          std::vector<std::size_t> indices;
          std::vector<value_type> ratios;
          // 5
          interrupt_ = false;
          while (!interrupt_)
          {
            if (++numberOfIterations > maxIterations) break;
            hppDout(info, numberOfIterations);
            getFullSplines(reducedParams, splines, constraint);
            vector_t value;
            matrix_t jacobian;

            getValueJacobianReduced(splines, reducedParams, value, jacobian, constraint);
            if (checkCollisions) addCollisionConstraintsValueJacobian(splines, reducedParams,
                value, jacobian, constraint, collFunctions, collValues, indices, ratios);
            hppDout(info, "Constraints error: " << value.norm());
            if (useHessian)
            {
              getHessianFiniteDiff(splines, reducedParams, hessianStack, stepSize, constraint);
              cost.hessian(fullObjectiveHessian, splines);
            }

            vector_t fullGradient(nParameters*rDof);
            cost.jacobian(fullGradient, splines);

            LinearConstraint jacobianConstraint(reducedDim, value.size());
            jacobianConstraint.J = jacobian;
            jacobianConstraint.b = -value;
            bool feasible = jacobianConstraint.decompose(true);
            if (!feasible) 
            {
              hppDout(info, "Not enough degrees of freedom to satisfy equality constraints");
              break;
            }

            // 5.1
            // Solve first order approximation of h(p + correction) = 0
            vector_t correction(reducedDim);
            correction = jacobianConstraint.xStar;

            if (useHessian)
            {
              vector_t fullCorrection = constraint.PK * correction;
              fullGradient += fullObjectiveHessian * fullCorrection;
              for (std::size_t i = 0; i < nbConstraints; ++i)
              {
                jacobian.row(i) += (constraint.PK.transpose() * (hessianStack[i] * (constraint.PK * correction))).transpose();
              }
            }

            // TODO: Calculate determinant
            matrix_t inverseGram(nbConstraints, nbConstraints);
            inverseGram = (jacobian * jacobian.transpose()).inverse();
            matrix_t fullHessian(nParameters*rDof, nParameters*rDof);
            matrix_t PK;

            matrix_t hessian(reducedDim, reducedDim);
            if (useHessian)
            {
              fullHessian = fullObjectiveHessian;
              matrix_t tmp = fullObjectiveHessian;
              for (std::size_t i = 0; i < nbConstraints; ++i)
              {
                value_type coeff = 0;
                for (std::size_t j = 0; j < nbConstraints; ++j) {
                  coeff += fullGradient.dot(jacobian.row(i)) * inverseGram.coeff(i, j);
                }
                fullHessian -= coeff * hessianStack[i];
              }
              PK = constraint.PK * jacobianConstraint.PK;
              hessian = PK.transpose() * fullHessian * PK;
            }

            bool optimumReached = false;
            vector_t step(reducedDim);
            step = correction;
            if (validateConstraints(splines, value, constraint, 500))
            {
              vector_t gradient(reducedDim);
              if (!useHessian)
              {
                gradient = constraint.PK.transpose() * fullGradient;

                value_type alpha = backtrackingLineSearch(cost, splines, constraint,
                    reducedParams, -gradient, gradient.norm(), lineSearchTrustRadius);
                hppDout(info, alpha);
                gradient = - alpha * gradient/gradient.norm();

                gradient = jacobianConstraint.PK.transpose()*gradient;

                LinearConstraint projectedBounds(jacobianConstraint.PK.cols(), 0);
                jacobianConstraint.reduceConstraint(boundConstraintReduced, projectedBounds);
                if (checkJointBound)
                {
                  // 5.2
                  // Remove gradient components that break the inequality constraints
                  activeBoundIndices.clear();
                  for (std::size_t i = 0; i < projectedBounds.J.rows(); ++i)
                  {
                    if (projectedBounds.J.row(i)*gradient <= projectedBounds.b[i] - boundConstraintReduced.J.row(i)*reducedParams)
                    {
                      activeBoundIndices.push_back(i);
                      hppDout(info, "Bound " << i << " is active: "
                          << projectedBounds.J.row(i)*gradient - projectedBounds.b[i]
                          + boundConstraintReduced.J.row(i)*reducedParams);
                    }
                  }
                  if (activeBoundIndices.size() > 0)
                  {
                    LinearConstraint activeBoundsProjected(jacobianConstraint.PK.cols(), activeBoundIndices.size());
                    for (std::size_t i = 0; i < activeBoundIndices.size(); ++i)
                    {
                      activeBoundsProjected.J.row(i) = projectedBounds.J.row(activeBoundIndices[i]);
                      activeBoundsProjected.b[i] = projectedBounds.b[activeBoundIndices[i]] - boundConstraintReduced.J.row(activeBoundIndices[i])*reducedParams;
                    }
                    activeBoundsProjected.decompose();
                    gradient = activeBoundsProjected.xStar + activeBoundsProjected.PK*(activeBoundsProjected.PK.transpose()*gradient);
                  }
                }

                // Project gradient back to parameter space
                gradient = jacobianConstraint.PK*gradient;
                for (std::size_t i = 0; i < activeBoundIndices.size(); ++i)
                {
                  hppDout(info, (boundConstraintReduced.J*(reducedParams + correction + gradient) - boundConstraintReduced.b)[activeBoundIndices[i]]);
                }
                step = correction + gradient;
              }

              if (useHessian) {
                // minimize 1/2 xT H x - bT x
                vector_t s = jacobianConstraint.PK*solveQP(hessian,
                    -(jacobianConstraint.PK.transpose()*(constraint.PK.transpose()*fullGradient)), trustRadius);
                hppDout(info, "Current gradient " << (jacobianConstraint.PK.transpose()*(constraint.PK.transpose()*fullGradient)).norm());
                hppDout(info, "Solution norm " << s.norm());
                // TODO: External function
                vector_t secondOrderCorrection(reducedDim);
                secondOrderCorrection.setZero();
                for (std::size_t j = 0; j < nbConstraints; ++j)
                {
                  vector_t v_j(reducedDim);
                  v_j.setZero();
                  for (std::size_t i = 0; i < nbConstraints; ++i)
                  {
                    v_j += inverseGram.coeff(i, j) * jacobian.row(i);
                  }
                  secondOrderCorrection += (s.transpose() * (constraint.PK.transpose() * (hessianStack[j] * (constraint.PK * s)))).coeff(0, 0) * v_j;
                }
                step = correction + s + secondOrderCorrection;

                vector_t sol = jacobianConstraint.PK.transpose() * s;
                value_type predictedDecrease = (1/2 * sol.transpose() * hessian * sol - (jacobianConstraint.PK.transpose()*(constraint.PK.transpose()*fullGradient)).transpose()*sol).coeff(0,0);
                vector_t tmp = reducedParams + step;
                Splines_t tmpSpl;
                Base::copy(splines, tmpSpl);
                getFullSplines(tmp, tmpSpl, constraint);
                value_type newCost;
                value_type oldCost;
                cost.value(newCost, tmpSpl);
                cost.value(oldCost, splines);

                vector_t tmpValue;
                matrix_t tmpJacobian;
                getValueJacobianReduced(tmpSpl, reducedParams+step, tmpValue, tmpJacobian, constraint);

                hppDout(info, "Actual/Predicted " << (oldCost - newCost)/predictedDecrease);
                bool validNewState = validateConstraints(tmpSpl, tmpValue, constraint, 500);
                if ((oldCost - newCost)/predictedDecrease <= 0. || !validNewState) {
                  hppDout(info, "New trust radius " << trustRadius);
                  trustRadius *= .5;
                  continue;
                }
                if ((oldCost - newCost)/predictedDecrease < .3) trustRadius *= 0.25;
                if ((oldCost - newCost)/predictedDecrease > .8
                    && sol.norm() >= .5 * trustRadius && validNewState) trustRadius *= 2;
                hppDout(info, "New trust radius " << trustRadius);
              }

              // 5.3
              // Check if optimum is reached
              optimumReached = (step.norm() < stepThreshold && validateConstraints(splines, value, constraint))
                || (numberOfIterations == maxIterations);
            }

            value_type costValue;
            cost.value(costValue, splines);
            hppDout(info, "Cost: " << costValue);

            bool noCollision = true;
            // 5.4
            // If a collision is detected, add collision constraints
            // and reset path to valid state
            if (checkCollisions && (lengthSinceLastCheck >= trustRadius || optimumReached))
            {
              lengthSinceLastCheck = 0;
              hppDout(info, "Checking for collision at iteration " << numberOfIterations);
              hppDout(info, ++nCollisionChecks << " collision checks so far");
              Reports_t reports = this->validatePath (splines, true);
              noCollision = reports.empty();
              if (noCollision)
              {
                collisionFreeParams = reducedParams;
                hppDout(info, "Collision free path at iteration " << numberOfIterations);
              }
              else
              {
                hppDout(info, "Collision detected");
                getFullSplines(collisionFreeParams, collisionFreeSplines, constraint);
                DifferentiableFunctionPtr_t cc =
                  CollisionFunction::create(robot_,
                      collisionFreeSplines[reports[0].second],
                      splines[reports[0].second],
                      reports[0].first);
                vector_t freeConfig(cc->inputSize());
                (*(collisionFreeSplines[reports[0].second])) (freeConfig, reports[0].first->parameter);
                collFunctions.push_back(cc);
                collValues.push_back((((*cc) (freeConfig)).vector())[0]);
                indices.push_back(reports[0].second);
                ratios.push_back(reports[0].first->parameter);
              }
            }
            if (noCollision && optimumReached) break;
            if (noCollision) {
              reducedParams = reducedParams + step;
              lengthSinceLastCheck += step.norm();
            }
            else reducedParams = collisionFreeParams;
          }
          // 6
          // getFullSplines(reducedParams, splines, constraint);
          return this->buildPathVector(splines);
        }

      // ----------- Convenience functions ---------------------------------- //

      template <int _PB, int _SO>
        template <typename Cost_t>
        bool SplineGradientBasedConstraint<_PB, _SO>::checkHessian
        (const Cost_t& cost, const matrix_t& H, const Splines_t& splines) const
        {
          value_type expected;
          cost.value(expected, splines);

          vector_t P (H.rows());

          const size_type size = robot_->numberDof() * Spline::NbCoeffs;
          for (std::size_t i = 0; i < splines.size(); ++i)
            P.segment (i * size, size) = splines[i]->rowParameters();
          value_type result = 0.5 * P.transpose() * H * P;

          bool ret = std::fabs(expected - result) < Eigen::NumTraits<value_type>::dummy_precision();
          if (!ret) {
            hppDout (error, "Hessian of the cost is not correct: " << expected << " - " << result << " = " << expected - result);
          }
          return ret;
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::copy
        (const Splines_t& in, Splines_t& out) const
        {
          out.resize(in.size());
          for (std::size_t i = 0; i < in.size(); ++i)
            out[i] = HPP_STATIC_PTR_CAST(Spline, in[i]->copy());
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::updateSplines
        (Splines_t& splines, const vector_t& param) const
        {
          size_type row = 0, size = robot_->numberDof() * Spline::NbCoeffs;
          for (std::size_t i = 0; i < splines.size(); ++i) {
            splines[i]->rowParameters(param.segment(row, size));
            row += size;
          }
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::getParameters
        (const Splines_t& splines, vector_t& param) const
        {
          size_type row = 0, size = robot_->numberDof() * Spline::NbCoeffs;
          for (std::size_t i = 0; i < splines.size(); ++i) {
            param.segment(row, size) = splines[i]->rowParameters();
            row += size;
          }
        }

      // ----------- Instanciate -------------------------------------------- //

      // template class SplineGradientBasedConstraint<path::CanonicalPolynomeBasis, 1>; // equivalent to StraightPath
      // template class SplineGradientBasedConstraint<path::CanonicalPolynomeBasis, 2>;
      // template class SplineGradientBasedConstraint<path::CanonicalPolynomeBasis, 3>;
      template class SplineGradientBasedConstraint<path::BernsteinBasis, 1>; // equivalent to StraightPath
      // template class SplineGradientBasedConstraint<path::BernsteinBasis, 2>;
      template class SplineGradientBasedConstraint<path::BernsteinBasis, 3>;
    } // namespace pathOptimization
  }  // namespace core

} // namespace hpp
