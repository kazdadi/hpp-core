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
      HPP_DEFINE_TIMECOUNTER(SGB_findNewConstraint);
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
          HPP_SCOPE_TIMECOUNTER(SGB_findNewConstraint);
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

      // ----------- Optimize ----------------------------------------------- //

      template <int _PB, int _SO>
        PathVectorPtr_t SplineGradientBasedConstraint<_PB, _SO>::optimize (const PathVectorPtr_t& path)
        {
          size_type maxIterations = problem().getParameter("SplineGradientBasedConstraint/maxIterations", size_type(200));
          value_type gradStepSize = problem().getParameter("SplineGradientBasedConstraint/gradStepSize", value_type(.1));
          value_type gradientEpsilon = problem().getParameter("SplineGradientBasedConstraint/gradientEpsilon", value_type(.01));
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
          const size_type nParameters = splines.size() * Spline::NbCoeffs;

          this->initializePathValidation(splines);

          // 2
          enum { MaxContinuityOrder = int( (SplineOrder - 1) / 2) };
          const size_type orderContinuity = MaxContinuityOrder;

          LinearConstraint constraint (nParameters * rDof, 0);
          SplineOptimizationDatas_t solvers (splines.size(), SplineOptimizationData(rDof));
          this->addContinuityConstraints (splines, orderContinuity, solvers, constraint);
          size_type numberOfContinuityConstraints = constraint.J.rows();

          // 4
          // TODO add weights
          SquaredLength<Spline, 1> cost (splines, rDof, rDof);
          matrix_t H(Spline::NbCoeffs*rDof*splines.size(), Spline::NbCoeffs*rDof*splines.size());
          cost.hessian(H, splines);
          Eigen::SelfAdjointEigenSolver<matrix_t> es(H);
          hppDout(info, "Cost hessian eigenvectors: \n" << es.eigenvectors());
          hppDout(info, "Cost hessian eigenvalues: \n" << H*es.eigenvectors());

          size_type numberOfIterations = 0;
          while (true)
          {
            if (numberOfIterations > 200) {
              hppDout (info, "Exceeded maximum number of iterations");
              break;
            }
            // Collision test
            // If collision, add constraint and go back
            Reports_t reports = this->validatePath(splines, false);
            bool noCollision = reports.empty();
            if (!noCollision)
            {
            }

            // If no collision, optimize
            value_type costValue;
            cost.value(costValue, splines);
            vector_t gradient(Spline::NbCoeffs*rDof*splines.size());
            cost.jacobian(gradient, splines);

            constraint.J.conservativeResize(numberOfContinuityConstraints, Eigen::NoChange);
            constraint.b.conservativeResize(numberOfContinuityConstraints, Eigen::NoChange);
            size_type constraintIndex = 0;
            for (std::size_t i = 1; i < splines.size(); ++i)
            {
              ConstraintSetPtr_t cs = splines[i]->constraints();
              ConfigProjectorPtr_t configProj;
              configProj = cs->configProjector();

              if (configProj)
              {
                size_type numberOfConstraints = configProj->numericalConstraints().size();
                constraint.J.conservativeResize(constraint.J.rows() + 2*numberOfConstraints, Eigen::NoChange);
                constraint.J.bottomRows(2*numberOfConstraints).setZero();
                constraint.b.conservativeResize(constraint.b.size() + 2*numberOfConstraints);
                constraint.b.bottomRows(2*numberOfConstraints).setZero();
                vector_t constraintValue;
                constraintValue.conservativeResize (2*numberOfConstraints);
                configProj->computeValueAndJacobian(splines[i]->initial(), constraintValue,
                    constraint.J.block(constraintIndex, rDof*Spline::NbCoeffs*i,
                      numberOfConstraints, rDof));
                constraintIndex += numberOfConstraints;
                configProj->computeValueAndJacobian(splines[i]->end(), constraintValue,
                    constraint.J.block(constraintIndex, rDof*Spline::NbCoeffs*(i+1)-rDof,
                      numberOfConstraints, rDof));
                constraintIndex += numberOfConstraints;
              }
            }
            bool feasible = constraint.decompose (true);
            if (!feasible) {
              hppDout (info, "Constraints not feasible");
              break;
            }
            gradient = constraint.PK*constraint.PK.transpose()*gradient;
            hppDout (info, std::setw(4) << ++numberOfIterations
                << " Cost " << std::setw(10) << costValue
                << " -- |grad| " << std::setw(10) << gradient.norm()
                << " -- |J*grad| " << std::setw(12) << (constraint.J*gradient).norm());
            if (gradient.norm() < gradientEpsilon) break;
            Splines_t oldSplines;
            Base::copy(splines, oldSplines);
            step(splines, -gradStepSize*gradient, (-gradStepSize*gradient).norm(), splines);
          }
          hppDout (info, "Finished after " << numberOfIterations << " iterations");
          return this->buildPathVector (splines);
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
        void SplineGradientBasedConstraint<_PB, _SO>::projectOnConstraints
        (const Splines_t& a, const vector_t direction,
         Splines_t res, vector_t resultDirection,
         bool calculatedResultDirection) const
        {
          assert (a.size() == res.size() && direction.size() == resultDirection.size());
          Base::copy(a, res);
          resultDirection = direction;
          size_type rDof = robot_->numberDof();
          const size_type orderContinuity = int( (SplineOrder - 1) / 2);
          hppDout(info, "projectOnConstraints");
          for (std::size_t i = 0; i < res.size(); ++i)
          {
            ConfigProjectorPtr_t initialConstraints;
            ConfigProjectorPtr_t endConstraints;
            if (i > 0)
            {
              initialConstraints = ConfigProjector::createUnion
                (res[i-1]->constraints()->configProjector(),
                 res[i]->constraints()->configProjector());
            }
            if (i < res.size()-1)
            {
              endConstraints = ConfigProjector::createUnion
                (res[i]->constraints()->configProjector(),
                 res[i+1]->constraints()->configProjector());
            }
            matrix_t parameters = res[i]->parameters();

            if (initialConstraints) {
              // hppDout(info, "Applying initial constraints");
              // Configuration_t initial = res[i]->base();
              // integrate(problem().robot(),
              //     res[i]->base(),
              //     res[i]->parameters().row(0),
              //     initial);
              Configuration_t initial = res[i]->initial();
              Configuration_t oldInitial = a[i]->initial();
              initialConstraints->apply(initial);
              // hppDout(info, "Difference after initial projection:\n" << initial - oldInitial);
              vector_t initialParameter(rDof);
              difference(problem().robot(), initial, res[i]->base(), initialParameter);
              vector_t oldInitialParameter = parameters.row(0);
              parameters.row(0) = initialParameter;
              // hppDout(info, "Difference after initial projection - parameter:\n" << initialParameter - oldInitialParameter);
              if (orderContinuity > 0)
              {
                hppDout(info, "Applying initial velocity constraints...");
                Configuration_t afterInitial = res[i]->base();
                integrate(problem().robot(),
                    res[i]->base(),
                    res[i]->parameters().row(1),
                    afterInitial);
                vector_t initialTangentVector(rDof);
                difference(problem().robot(), afterInitial, initial, initialTangentVector);
                hppDout(info, "v_i " << initialTangentVector);
                initialConstraints->projectVectorOnKernel(initial,
                    initialTangentVector,
                    initialTangentVector);
                hppDout(info, "new v_i " << initialTangentVector);
                integrate(problem().robot(), initial, initialTangentVector, afterInitial);
                vector_t afterInitialParameter(rDof);
                difference(problem().robot(), afterInitial, res[i]->base(), afterInitialParameter);
                parameters.row(1) = afterInitialParameter;
              }

              if (calculatedResultDirection)
              {
                integrate(problem().robot(),
                    res[i]->base(),
                    res[i]->parameters().row(0),
                    initial);
                initialConstraints->projectVectorOnKernel(initial,
                    direction.segment(res.size()*Spline::NbCoeffs*rDof*i, rDof),
                    resultDirection.segment(res.size()*Spline::NbCoeffs*rDof*i, rDof));
              }
            }

            if (endConstraints) {
              // Configuration_t end = res[i]->base();
              // integrate(problem().robot(),
              //     res[i]->base(),
              //     res[i]->parameters().row(Spline::NbCoeffs-1),
              //     end);
              Configuration_t end = res[i]->end();
              Configuration_t oldEnd = a[i]->end();
              endConstraints->apply(end);
              // hppDout(info, "Difference after end projection:\n" << end - oldEnd);
              vector_t endParameter(rDof);
              difference(problem().robot(), end, res[i]->base(), endParameter);
              vector_t oldEndParameter = parameters.row(Spline::NbCoeffs-1);
              parameters.row(Spline::NbCoeffs-1) = endParameter;
              // hppDout(info, "Difference after end projection - parameter:\n" << endParameter - oldEndParameter);
              if (orderContinuity > 0)
              {
                hppDout(info, "Applying velocity contraints...");
                Configuration_t beforeEnd = res[i]->base();
                integrate(problem().robot(),
                    res[i]->base(),
                    res[i]->parameters().row(Spline::NbCoeffs-2),
                    beforeEnd);
                vector_t endTangentVector(rDof);
                difference(problem().robot(), beforeEnd, end, endTangentVector);
                hppDout(info, "v " << endTangentVector);
                endConstraints->projectVectorOnKernel(end,
                    endTangentVector,
                    endTangentVector);
                hppDout(info, "new v " << endTangentVector);
                integrate(problem().robot(), end, endTangentVector, beforeEnd);
                vector_t beforeEndParameter(rDof);
                difference(problem().robot(), beforeEnd, res[i]->base(), beforeEndParameter);
                parameters.row(Spline::NbCoeffs-2) = beforeEndParameter;
              }
              if (calculatedResultDirection)
              {
                integrate(problem().robot(),
                    res[i]->base(),
                    res[i]->parameters().row(Spline::NbCoeffs-1),
                    end);
                endConstraints->projectVectorOnKernel(end,
                    direction.segment(res.size()*Spline::NbCoeffs*rDof*(i+1) - rDof, rDof),
                    resultDirection.segment(res.size()*Spline::NbCoeffs*rDof*(i+1) - rDof, rDof));
              }
            }
            if (initialConstraints || endConstraints) res[i]->parameters(parameters);
          }
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::manifoldStep
        (const Splines_t& a, const vector_t direction,
         Splines_t res, vector_t transportedDirection,
         bool calculateTransportedDirection) const
        {
          assert (a.size() == res.size() && direction.size() == transportedDirection.size());
          // res = a + direction
          size_type rDof = a[0]->parameterSize();
          for (std::size_t i = 0; i < a.size(); ++i) {
            res[i]->rowParameters(a[i]->rowParameters());
            res[i]->parameterIntegrate(direction.segment(Spline::NbCoeffs*rDof*i,Spline::NbCoeffs*rDof));
          }
          projectOnConstraints(res, direction, res, transportedDirection, calculateTransportedDirection);
          if (calculateTransportedDirection){
          transportedDirection = (direction.norm()/transportedDirection.norm())*transportedDirection;
          }
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::step
        (const Splines_t& a, vector_t gradient, value_type stepSize,Splines_t& res) const
        {
          assert (a.size() == res.size());
          res = a;
          while (gradient.norm() > stepSize)
          {
            manifoldStep(res, stepSize/gradient.norm()*gradient, res, gradient, true);
            gradient = (1 - stepSize/gradient.norm())*gradient;
          }
          manifoldStep(res, gradient, res, gradient, false);
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
