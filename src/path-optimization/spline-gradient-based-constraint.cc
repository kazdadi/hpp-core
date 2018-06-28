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

#include <hpp/core/explicit-numerical-constraint.hh>
#include <../src/steering-method/cross-state-optimization/function.cc>
#include <hpp/core/config-projector.hh>
#include <hpp/core/locked-joint.hh>
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
        (const Splines_t splines, HybridSolver& hybridSolver, LinearConstraint& constraint,
         std::vector<size_type>& dofPerSpline, std::vector<size_type>& argPerSpline,
         size_type& nbConstraints) const
        {
          // TODO: Explicit constraints derivatives
          // TODO: Add continuity constraints
          size_type nArgs = splines.size() * Spline::NbCoeffs * robot_->configSize();
          size_type nDers = splines.size() * Spline::NbCoeffs * robot_->numberDof();
          size_type order = int ((SplineOrder-1)/2);
          for (std::size_t i = 0; i < splines.size(); ++i)
          {
            Configuration_t endPrev;
            Configuration_t initial = splines[i]->initial();
            Configuration_t end = splines[i]->end();
            Configuration_t initialNext;

            ConfigProjectorPtr_t configProjectorPrev;
            ConfigProjectorPtr_t configProjector = splines[i]->constraints()->configProjector();
            ConfigProjectorPtr_t configProjectorNext;

            if (i >= 1) {
              endPrev = splines[i-1]->end();
              configProjectorPrev = splines[i-1]->constraints()->configProjector();
            }
            if (i <= splines.size() - 2) {
              initialNext = splines[i+1]->end();
              configProjectorNext = splines[i+1]->constraints()->configProjector();
            }

            if (configProjector)
            {
              NumericalConstraints_t numericalConstraints = configProjector->numericalConstraints();
              for (std::size_t j = 0; j < numericalConstraints.size(); ++j)
              {
                // TODO: Mixed comparison type constraints
                bool equalToZero = numericalConstraints[j]->comparisonType()[0] == constraints::Equality;

                ExplicitNumericalConstraintPtr_t explicitConstraint =
                  HPP_DYNAMIC_PTR_CAST (ExplicitNumericalConstraint, numericalConstraints[j]);
                if (explicitConstraint)
                {
                  // Explicit constraints
                  if (equalToZero)
                  {
                    // Adding initial constraints
                    segments_t outArg = explicitConstraint->outputConf();
                    segments_t outDer = explicitConstraint->outputVelocity();
                    segments_t inArg = explicitConstraint->inputConf();
                    segments_t inDer = explicitConstraint->inputVelocity();
                    for (std::size_t k = 0; k < inArg.size(); ++k)
                    {
                      inArg[k].first += robot_->configSize() * Spline::NbCoeffs * i;
                      inDer[k].first += robot_->numberDof() * Spline::NbCoeffs * i;
                    }
                    for (std::size_t k = 0; k < outArg.size(); ++k)
                    {
                      outArg[k].first += robot_->configSize() * Spline::NbCoeffs * i;
                      outDer[k].first += robot_->numberDof() * Spline::NbCoeffs * i;
                    }
                    hybridSolver.explicitSolver().add(explicitConstraint->explicitFunction(), inArg, outArg, inDer, outDer);
                    dofPerSpline[i] -= explicitConstraint->functionPtr()->outputDerivativeSize();
                    argPerSpline[i] -= explicitConstraint->functionPtr()->outputSize();

                    // Adding derivative constraints to remove them from the parameters vector
                    // The actual nature of the constraint is irrelevant
                    for (std::size_t o = 0; o < 2*order; ++o)
                    {
                      for (std::size_t k = 0; k < outArg.size(); ++k)
                      {
                        outArg[k].first += robot_->configSize();
                        outDer[k].first += robot_->numberDof();
                      }
                      hybridSolver.explicitSolver().add(explicitConstraint->explicitFunction(), inArg, outArg, inDer, outDer);
                    }

                    // Adding end constraints
                    for (std::size_t k = 0; k < inArg.size(); ++k)
                    {
                      inArg[k].first += robot_->configSize() * (Spline::NbCoeffs - 1);
                      inDer[k].first += robot_->numberDof() * (Spline::NbCoeffs - 1);
                    }
                    for (std::size_t k = 0; k < outArg.size(); ++k)
                    {
                      outArg[k].first += robot_->configSize();
                      outDer[k].first += robot_->numberDof();
                    }
                    hybridSolver.explicitSolver().add(explicitConstraint->explicitFunction(), inArg, outArg, inDer, outDer);
                  }
                  else // Equality type constraints
                  {
                    // Add as linear implicit constraint
                    segments_t outDer = explicitConstraint->outputVelocity();
                    for (std::size_t k = 0; k < outDer.size(); ++k)
                      outDer[k].first += robot_->numberDof() * Spline::NbCoeffs * i;

                    for (std::size_t k = 0; k < outDer.size(); ++k)
                    {
                      constraint.addRows(outDer[k].second);
                      constraint.J.block(constraint.J.rows() - outDer[k].second, outDer[k].first,
                          outDer[k].second, outDer[k].second) = matrix_t::Identity(outDer[k].second, outDer[k].second);
                      constraint.J.block(constraint.J.rows() - outDer[k].second, outDer[k].first + (Spline::NbCoeffs-1)*robot_->numberDof(),
                          outDer[k].second, outDer[k].second) = -matrix_t::Identity(outDer[k].second, outDer[k].second);
                      constraint.b.bottomRows(outDer[k].second).setZero();
                    }
                  }
                }
                else
                {
                  NumericalConstraintPtr_t numericalConstraint = numericalConstraints[j];
                  segment_t inArgs;
                  segment_t inDers;
                  inArgs.second = robot_->configSize();
                  inDers.second = robot_->numberDof();
                  if (i >= 1 and (!configProjectorPrev or !(configProjectorPrev->contains(numericalConstraint))))
                  {
                    inArgs.first = robot_->configSize()*Spline::NbCoeffs*i;
                    inDers.first = robot_->numberDof() *Spline::NbCoeffs*i;
                    StateFunction::Ptr_t initialNumericalconstraint (new StateFunction(numericalConstraint->functionPtr(),
                          nArgs, nDers, inArgs, inDers));
                    // TODO Priority?
                    hybridSolver.add(initialNumericalconstraint, 0, numericalConstraint->comparisonType());
                    nbConstraints += numericalConstraint->functionPtr()->outputDerivativeSize();
                  }
                  if (i <= splines.size() - 2)
                  {
                    inArgs.first = robot_->configSize() * (Spline::NbCoeffs*(i+1) - 1);
                    inDers.first = robot_->numberDof() * (Spline::NbCoeffs*(i+1) - 1);
                    StateFunction::Ptr_t endNumericalconstraint (new StateFunction(numericalConstraint->functionPtr(),
                          nArgs, nDers, inArgs, inDers));
                    hybridSolver.add(endNumericalconstraint, 0, numericalConstraint->comparisonType());
                    nbConstraints += numericalConstraint->functionPtr()->outputDerivativeSize();
                  }
                }
              }
              // Explicit constraints
              LockedJoints_t lockedJointsList = configProjector->lockedJoints();
              for (LockedJoints_t::iterator it = lockedJointsList.begin(); it != lockedJointsList.end(); ++it)
              {
                bool equalToZero = (*it)->comparisonType()[0] == constraints::Equality;
                if (equalToZero)
                {
                  // Adding initial constraints
                  segments_t outArg = (*it)->outputConf();
                  segments_t outDer = (*it)->outputVelocity();
                  segments_t inArg = (*it)->inputConf();
                  segments_t inDer = (*it)->inputVelocity();
                  for (std::size_t k = 0; k < inArg.size(); ++k)
                  {
                    inArg[k].first += robot_->configSize() * Spline::NbCoeffs * i;
                    inDer[k].first += robot_->numberDof() * Spline::NbCoeffs * i;
                  }
                  for (std::size_t k = 0; k < outArg.size(); ++k)
                  {
                    outArg[k].first += robot_->configSize() * Spline::NbCoeffs * i;
                    outDer[k].first += robot_->numberDof() * Spline::NbCoeffs * i;
                  }
                  hybridSolver.explicitSolver().add((*it)->explicitFunction(), inArg, outArg, inDer, outDer);
                  dofPerSpline[i] -= (*it)->functionPtr()->outputDerivativeSize();
                  argPerSpline[i] -= (*it)->functionPtr()->outputSize();

                  // Adding derivative constraints to remove them from the parameters vector
                  // The actual nature of the constraint is irrelevant
                  for (std::size_t o = 0; o < 2*order; ++o)
                  {
                    for (std::size_t k = 0; k < outArg.size(); ++k)
                    {
                      outArg[k].first += robot_->configSize();
                      outDer[k].first += robot_->numberDof();
                    }
                    hybridSolver.explicitSolver().add((*it)->explicitFunction(), inArg, outArg, inDer, outDer);
                  }

                  // Adding end constraints
                  for (std::size_t k = 0; k < inArg.size(); ++k)
                  {
                    inArg[k].first += robot_->configSize() * (Spline::NbCoeffs - 1);
                    inDer[k].first += robot_->numberDof() * (Spline::NbCoeffs - 1);
                  }
                  for (std::size_t k = 0; k < outArg.size(); ++k)
                  {
                    outArg[k].first += robot_->configSize();
                    outDer[k].first += robot_->numberDof();
                  }
                  hybridSolver.explicitSolver().add((*it)->explicitFunction(), inArg, outArg, inDer, outDer);
                }
                else // Equality type constraints
                {
                  // Add as linear implicit constraint
                  segments_t outDer = (*it)->outputVelocity();
                  for (std::size_t k = 0; k < outDer.size(); ++k)
                    outDer[k].first += robot_->numberDof() * Spline::NbCoeffs * i;

                  for (std::size_t k = 0; k < outDer.size(); ++k)
                  {
                    constraint.addRows(outDer[k].second);
                    constraint.J.block(constraint.J.rows() - outDer[k].second, outDer[k].first,
                        outDer[k].second, outDer[k].second) = matrix_t::Identity(outDer[k].second, outDer[k].second);
                    constraint.J.block(constraint.J.rows() - outDer[k].second, outDer[k].first + (Spline::NbCoeffs-1)*robot_->numberDof(),
                        outDer[k].second, outDer[k].second) = -matrix_t::Identity(outDer[k].second, outDer[k].second);
                    constraint.b.bottomRows(outDer[k].second).setZero();
                  }
                }
              }
            }
          }
          hybridSolver.explicitSolverHasChanged();
        }

      bool contains (const size_type index, const segments_t segments)
      {
        for (std::size_t i = 0; i < segments.size(); ++i)
        {
          if (index - segments[i].first >= 0 and index - segments[i].first <= segments[i].second)
            return true;
        }
        return false;
      }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::processContinuityConstraints
        (const Splines_t splines, HybridSolver hybridSolver,
         LinearConstraint& continuityConstraints, LinearConstraint& linearConstraints) const
        {
          LinearConstraint nonlinearContinuityConstraints (continuityConstraints.J.cols(), 0);
          segments_t freeDofs = hybridSolver.explicitSolver().freeDers().indices();
          std::size_t order = int ((SplineOrder-1)/2);
          for (std::size_t i = 0; i < splines.size() + 1; ++i)
          {
            for (std::size_t j = 0; j < robot_->numberDof(); ++j)
            {
              bool leftFree = true;
              bool rightFree = true;
              if (i > 0)
                leftFree = contains(robot_->numberDof() * (Spline::NbCoeffs * i - 1) + j, freeDofs);
              if (i < splines.size())
                rightFree = contains(robot_->numberDof() * Spline::NbCoeffs * i + j, freeDofs);

              if (leftFree and rightFree)
              {
                for (std::size_t o = 0; o < order + 1; ++o) {
                  linearConstraints.addRows(1);
                  linearConstraints.J.bottomRows(1) = continuityConstraints.J.row(
                      o * (splines.size() + 1) * robot_->numberDof() + robot_->numberDof() * i + j);
                  linearConstraints.b.bottomRows(1) = continuityConstraints.b.row(
                    o * (splines.size() + 1) * robot_->numberDof() + robot_->numberDof() * i + j);
                }
              }
              else if (leftFree and i > 0)
              {
                for (std::size_t o = 0; o < order + 1; ++o) {
                  nonlinearContinuityConstraints.addRows(1);
                  nonlinearContinuityConstraints.J.bottomRows(1) = continuityConstraints.J.row(
                      o * (splines.size() + 1) * robot_->numberDof() + robot_->numberDof() * i + j);
                  nonlinearContinuityConstraints.b.bottomRows(1) = continuityConstraints.b.row(
                      o * (splines.size() + 1) * robot_->numberDof() + robot_->numberDof() * i + j);
                }
              }
              else if (rightFree and i < splines.size())
              {
                for (std::size_t o = 0; o < order + 1; ++o) {
                  nonlinearContinuityConstraints.addRows(1);
                  nonlinearContinuityConstraints.J.bottomRows(1) = continuityConstraints.J.row(
                      o * (splines.size() + 1) * robot_->numberDof() + robot_->numberDof() * i + j);
                  nonlinearContinuityConstraints.b.bottomRows(1) = continuityConstraints.b.row(
                      o * (splines.size() + 1) * robot_->numberDof() + robot_->numberDof() * i + j);
                }
              }
              else if (!leftFree and !rightFree)
              {
                // Do nothing
              }
            }
          }
          continuityConstraints.J = nonlinearContinuityConstraints.J;
          linearConstraints.J = hybridSolver.explicitSolver().freeDers().rview(linearConstraints.J).eval();
        }

      template <int _PB, int _SO>
        matrix_t SplineGradientBasedConstraint<_PB, _SO>::costHessian(const Splines_t splines,
            const LinearConstraint linearConstraints, std::vector<size_type> dofPerSpline) const
        {
          matrix_t hessian(linearConstraints.J.cols(), linearConstraints.J.cols());
          hessian.setZero();
          size_type splineIndex = 0;
          for (std::size_t i = 0; i < dofPerSpline.size(); ++i)
          {
            size_type nbCoeffs = Spline::NbCoeffs;
            matrix_t integrals(nbCoeffs, nbCoeffs);
            splines[i]->squaredNormBasisFunctionIntegral(1, integrals);
            size_type nDof = dofPerSpline[i];
            for (std::size_t j = 0; j < nbCoeffs; ++j)
            {
              for (std::size_t k = 0; k < nbCoeffs; ++k)
              {
                hessian.block(splineIndex + nDof*j, splineIndex + nDof*k, nDof, nDof) =
                  integrals.coeff(j, k) * matrix_t::Identity(nDof, nDof);
              }
            }
            splineIndex += nDof*nbCoeffs;
          }

          matrix_t reducedHessian = linearConstraints.PK.transpose() * hessian * linearConstraints.PK;
          return reducedHessian;
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
        void SplineGradientBasedConstraint<_PB, _SO>::getFullSplines
        (const vector_t reducedParams, Splines_t& splines, LinearConstraint linearConstraints, HybridSolver hybridSolver) const
        {
          vector_t freeParameters = linearConstraints.xStar + (linearConstraints.PK*reducedParams);

          vector_t fullParameters(hybridSolver.explicitSolver().derSize());
          fullParameters.setZero();
          hybridSolver.explicitSolver().freeDers().transpose().lview(fullParameters) = freeParameters;
          updateSplines(splines, fullParameters);
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
            hppDout(info, "Estimated distance to optimum is: " << x.norm());
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
          return (A + (u - d[0]) * matrix_t::Identity(b.size(), b.size())).llt().solve(b);
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

          vector_t x = constraint.xStar + constraint.PK*reducedParams;
          vector_t step(rDof*Spline::NbCoeffs);
          step.setZero();
          Splines_t splinesPlus;
          Splines_t splinesMinus;
          Base::copy(fullSplines, splinesPlus);
          Base::copy(fullSplines, splinesMinus);
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

                vector_t paramsPlus = x.segment (i*rDof*Spline::NbCoeffs,
                    rDof*Spline::NbCoeffs) + step;
                vector_t paramsMinus = x.segment (i*rDof*Spline::NbCoeffs,
                    rDof*Spline::NbCoeffs) - step;

                splinesPlus[i]->rowParameters(paramsPlus);
                splinesMinus[i]->rowParameters(paramsMinus);

                Configuration_t initial_plus = splinesPlus[i]->initial();
                Configuration_t initial_minus = splinesMinus[i]->initial();

                vector_t tmp_value(numberOfConstraints);

                matrix_t J_plus(numberOfConstraints, configProj->numberNonLockedDof());
                matrix_t J_minus(numberOfConstraints, configProj->numberNonLockedDof());
                matrix_t jacPlus(numberOfConstraints, rDof);
                matrix_t jacMinus(numberOfConstraints, rDof);
                jacPlus.setZero();
                jacMinus.setZero();

                configProj->computeValueAndJacobian(initial_plus, tmp_value, J_plus);
                configProj->computeValueAndJacobian(initial_minus, tmp_value, J_minus);

                configProj->solver().explicitSolver().freeDers().lview(jacPlus) = J_plus;
                configProj->solver().explicitSolver().freeDers().lview(jacMinus) = J_minus;

                matrix_t jacDiff(numberOfConstraints, rDof);
                jacDiff = (jacPlus - jacMinus)/(2*stepSize);

                for (std::size_t k = 0; k < numberOfConstraints; ++k)
                {
                  hessianStack[constraintIndex+k].row(j).
                    segment(rDof*Spline::NbCoeffs*i, rDof) = jacDiff.row(k);
                }
                constraintIndex += numberOfConstraints;
              }
            }
            step[j] = 0;
          }
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::getConstraintsValue
        (const vector_t x, Splines_t& splines, vector_t& value,
         const LinearConstraint linearConstraints, HybridSolver& hybridSolver) const
        {
          getFullSplines(x, splines, linearConstraints, hybridSolver);

          vector_t stateConfiguration(splines.size() * Spline::NbCoeffs * robot_->configSize());
          stateConfiguration.setZero();
          for (std::size_t i = 0; i < splines.size(); ++i) {
            splines[i]->at (splines[i]->timeRange().first,
            stateConfiguration.segment(
                i*Spline::NbCoeffs*robot_->configSize(), robot_->configSize()));
            splines[i]->at (splines[i]->timeRange().second,
            stateConfiguration.segment(
                ((i+1)*Spline::NbCoeffs-1)*robot_->configSize(), robot_->configSize()));
          }
          hybridSolver.explicitSolver().solve(stateConfiguration);

          hybridSolver.computeValue<false>(stateConfiguration);
          hybridSolver.getValue(value);
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::getConstraintsValueJacobian
        (const vector_t x, Splines_t& splines, vector_t& value, matrix_t& jacobian,
         const LinearConstraint linearConstraints, HybridSolver& hybridSolver) const
        {
          getFullSplines(x, splines, linearConstraints, hybridSolver);
          vector_t stateConfiguration(splines.size() * Spline::NbCoeffs * robot_->configSize());
          stateConfiguration.setZero();
          for (std::size_t i = 0; i < splines.size(); ++i) {
            splines[i]->at (splines[i]->timeRange().first,
            stateConfiguration.segment(
                i*Spline::NbCoeffs*robot_->configSize(), robot_->configSize()));
            splines[i]->at (splines[i]->timeRange().second,
            stateConfiguration.segment(
                ((i+1)*Spline::NbCoeffs-1)*robot_->configSize(), robot_->configSize()));
          }
          hybridSolver.explicitSolver().solve(stateConfiguration);

          hybridSolver.computeValue<true>(stateConfiguration);
          hybridSolver.updateJacobian(stateConfiguration);
          hybridSolver.getValue(value);
          hybridSolver.getReducedJacobian(jacobian);
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::addCollisionConstraintsValueJacobian
        (const Splines_t fullSplines, const vector_t reducedParams,
         vector_t& value, matrix_t& jacobian, LinearConstraint constraint,
         std::vector<DifferentiableFunctionPtr_t> collFunctions,
         std::vector<value_type> collValues,
         std::vector<std::size_t> indices,
         std::vector<value_type> times) const
        {
          std::size_t row = value.size();
          value.conservativeResize(row + collFunctions.size());
          jacobian.conservativeResize(row + collFunctions.size(), Eigen::NoChange);
          for (std::size_t i = 0; i < collFunctions.size(); ++i)
          {
            size_type rDof = robot_->numberDof();

            vector_t BernsteinCoeffs(int (Spline::NbCoeffs));
            matrix_t splinesJacobian(rDof, fullSplines.size()*Spline::NbCoeffs*rDof);
            fullSplines[indices[i]]->parameterDerivativeCoefficients(BernsteinCoeffs, times[i]);
            splinesJacobian.setZero();
            for (std::size_t j = 0; j < BernsteinCoeffs.size(); ++j)
            {
              splinesJacobian.block(0, Spline::NbCoeffs*rDof*indices[i] + rDof*j,
                  rDof, rDof) = BernsteinCoeffs[j] * matrix_t::Identity(rDof, rDof);
            }

            vector_t currentConfig(collFunctions[i]->inputSize());
            matrix_t collJacobian(1, collFunctions[i]->inputDerivativeSize());
            (*(fullSplines[indices[i]])) (currentConfig, times[i]);
            value[row] = (((*(collFunctions[i])) (currentConfig)).vector()[0] - collValues[i]);
            collFunctions[i]->jacobian(collJacobian, currentConfig);
            jacobian.row(row) = (collJacobian*splinesJacobian)*constraint.PK;
            ++row;
          }
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::addCollisionHessianFiniteDiff
        (const Splines_t fullSplines, const vector_t reducedParams,
         std::vector<matrix_t>& hessianStack, LinearConstraint constraint,
         std::vector<DifferentiableFunctionPtr_t> collFunctions,
         std::vector<value_type> collValues, std::vector<std::size_t> indices,
         std::vector<value_type> times, value_type stepSize, std::size_t nbConstraints) const
        {
          const std::size_t rDof = robot_->numberDof();
          std::size_t row = nbConstraints;

          Splines_t splinesPlus;
          Splines_t splinesMinus;
          Base::copy(fullSplines, splinesPlus);
          Base::copy(fullSplines, splinesMinus);

          for (std::size_t i = 0; i < collFunctions.size(); ++i)
          {
            vector_t BernsteinCoeffs(int (Spline::NbCoeffs));
            matrix_t splinesJacobian(rDof, fullSplines.size()*Spline::NbCoeffs*rDof);
            fullSplines[indices[i]]->parameterDerivativeCoefficients(BernsteinCoeffs, times[i]);
            splinesJacobian.setZero();

            for (std::size_t j = 0; j < BernsteinCoeffs.size(); ++j)
            {
              splinesJacobian.block(0, Spline::NbCoeffs*rDof*indices[i] + rDof*j,
                  rDof, rDof) = BernsteinCoeffs[j] * matrix_t::Identity(rDof, rDof);
            }

            matrix_t collHessian(collFunctions[i]->inputDerivativeSize(),
                collFunctions[i]->inputDerivativeSize());

            vector_t step(Spline::NbCoeffs * rDof);
            vector_t x = fullSplines[indices[i]]->rowParameters();
            for (std::size_t j = 0; j < rDof; ++j)
            {
              step.setZero();
              for (std::size_t k = 0; k < Spline::NbCoeffs; ++k) {
                step[j + k*rDof] = stepSize;
              }
              splinesPlus[indices[i]]->rowParameters(x + step);
              splinesMinus[indices[i]]->rowParameters(x - step);

              vector_t configPlus(collFunctions[i]->inputSize());
              vector_t configMinus(collFunctions[i]->inputSize());

              (*(splinesPlus[indices[i]])) (configPlus, times[i]);
              (*(splinesMinus[indices[i]])) (configMinus, times[i]);

              matrix_t collJacobianPlus(1, collFunctions[i]->inputDerivativeSize());
              matrix_t collJacobianMinus(1, collFunctions[i]->inputDerivativeSize());

              collFunctions[i]->jacobian(collJacobianPlus, configPlus);
              collFunctions[i]->jacobian(collJacobianMinus, configMinus);

              collHessian.row(j) = (collJacobianPlus - collJacobianMinus)/(2*stepSize);
            }

            hessianStack[row] = splinesJacobian.transpose() * collHessian * splinesJacobian;
            ++row;
          }
        }

      template <int _PB, int _SO>
        bool SplineGradientBasedConstraint<_PB, _SO>::validateConstraints
        (const Splines_t fullSplines, const vector_t value,
         const std::vector<value_type> collValues,
         LinearConstraint constraint, value_type factor) const
        {
          // TODO: Return max ratio of error over threshold
          std::size_t row = 0;
          for (std::size_t i = 1; i < fullSplines.size(); ++i)
          {
            ConfigProjectorPtr_t configProj = fullSplines[i]->constraints()->configProjector();
            if (configProj)
            {
              std::size_t numberOfConstraints = configProj->dimension();
              if (value.segment(row, numberOfConstraints).norm() > factor * configProj->errorThreshold())
              {
                return false;
              }
              row += numberOfConstraints;
            }
          }
          for (std::size_t i = 0; i < collValues.size(); ++i)
          {
            if (std::abs(value[row]) > factor * collValues[i]*0.01) return false;
            ++row;
          }
          return true;
        }

      template <int _PB, int _SO>
        vector_t SplineGradientBasedConstraint<_PB, _SO>::getSecondOrderCorrection
        (const vector_t step, const matrix_t jacobian, const std::vector<matrix_t> hessianStack,
         const matrix_t inverseGram, const matrix_t PK) const
        {
          vector_t secondOrderCorrection(step.size());
          secondOrderCorrection.setZero();
          for (std::size_t j = 0; j < inverseGram.rows(); ++j)
          {
            vector_t v_j(step.size());
            v_j.setZero();
            for (std::size_t i = 0; i < inverseGram.rows(); ++i)
            {
              v_j += inverseGram.coeff(i, j) * jacobian.row(i);
            }
            secondOrderCorrection += 0.5*
              (step.transpose() * (PK.transpose() * (hessianStack[j] * (PK * step))))
              .coeff(0, 0) * v_j;
          }
          return secondOrderCorrection;
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::analyzeHessians
        (const std::vector<matrix_t> hessianStack, const matrix_t PK) const
        {
          matrix_t restrictedHessian(PK.cols(), PK.cols());
          for (std::size_t i = 0; i < hessianStack.size(); ++i)
          {
            restrictedHessian = PK.transpose() * hessianStack[i] * PK;
            Eigen::SelfAdjointEigenSolver<matrix_t> solver(restrictedHessian);
            hppDout(info, solver.eigenvalues().transpose());
          }
        }
      // ----------- Optimize ----------------------------------------------- //

      template <int _PB, int _SO>
        PathVectorPtr_t SplineGradientBasedConstraint<_PB, _SO>::optimize (const PathVectorPtr_t& path)
        {
          size_type maxIterations = problem().getParameter(
              "SplineGradientBasedConstraint/maxIterations", size_type(200));
          bool checkJointBound = problem().getParameter(
              "SplineGradientBasedConstraint/checkJointBound", true);
          bool checkCollisions = problem().getParameter(
              "SplineGradientBasedConstraint/checkCollisions", true);

          PathVectorPtr_t tmp = PathVector::create (robot_->configSize(), robot_->numberDof());
          path->flatten(tmp);
          // Remove zero length paths
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
          size_type nArgs = nParameters * robot_->configSize();
          size_type nDers = nParameters * rDof;

          // 2
          enum { MaxContinuityOrder = int( (SplineOrder - 1) / 2) };
          const size_type orderContinuity = MaxContinuityOrder;

          LinearConstraint continuityConstraints (nDers, 0);
          SplineOptimizationDatas_t solvers (splines.size(), SplineOptimizationData(rDof));
          this->addContinuityConstraints (splines, orderContinuity, solvers, continuityConstraints);

          matrix_t tmpA(continuityConstraints.b.size(), continuityConstraints.J.cols()+1);
          tmpA.leftCols(continuityConstraints.J.cols()) = continuityConstraints.J;
          tmpA.rightCols(1) = continuityConstraints.b;
          hppDout(info, continuityConstraints.b.transpose());
          hppDout(info, "\n" << tmpA);

          HybridSolver hybridSolver(nArgs, nDers);

          std::vector<size_type> dofPerSpline(splines.size(), rDof);
          std::vector<size_type> argPerSpline(splines.size(), robot_->configSize());
          LinearConstraint linearConstraints(nDers, 0);
          size_type nbConstraints = 0;
          addProblemConstraints(splines, hybridSolver, linearConstraints, dofPerSpline, argPerSpline, nbConstraints);

          processContinuityConstraints(splines, hybridSolver, continuityConstraints, linearConstraints);

          vector_t fullParameters(nParameters*rDof);
          getParameters(splines, fullParameters);

          vector_t freeParameters = hybridSolver.explicitSolver().freeDers().transpose().rview(fullParameters);

          linearConstraints.decompose();
          hppDout(info, linearConstraints.b.size());
          hppDout(info, linearConstraints.rank);
          vector_t reducedParameters = linearConstraints.PK.transpose() * (freeParameters - linearConstraints.xStar);

          // 3
          // Cost value is: 1/2 x^T * cost * x, where x = reducedParameters
          // Cost gradient is: cost * x, where x = reducedParameters
          matrix_t cost = costHessian(splines, linearConstraints, dofPerSpline);

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

          vector_t value(nbConstraints);
          matrix_t jacobian(nbConstraints, freeParameters.size());
          while (!this->interrupt_)
          {
            getConstraintsValueJacobian(reducedParameters, splines, value, jacobian, linearConstraints, hybridSolver);
            matrix_t reducedJacobian = jacobian * linearConstraints.PK;

            matrix_t finiteDiffJacobian(nbConstraints, reducedParameters.size());
            finiteDiffJacobian.setZero();
            vector_t dx(reducedParameters.size());
            dx.setZero();
            vector_t tmpValue(nbConstraints);
            for (std::size_t i = 0; i < dx.size(); ++i)
            {
              value_type step = .00001;
              dx[i] = step;
              getConstraintsValue(reducedParameters+dx, splines, tmpValue, linearConstraints, hybridSolver);
              getConstraintsValue(reducedParameters-dx, splines, value, linearConstraints, hybridSolver);

              finiteDiffJacobian.col(i) = (tmpValue-value)/(2*step);
              dx[i] = 0;
            }
            hppDout(info, "jacobian\n" << reducedJacobian);
            hppDout(info, "finitediff\n" << finiteDiffJacobian);
            matrix_t M = reducedJacobian - finiteDiffJacobian;
            M = (M.array().abs() < 1e-6).select(0, M);
            hppDout(info, "diff\n" << M);

            hppDout(info, (reducedJacobian - finiteDiffJacobian).norm());
            break;
            // Get constraints jacobian
            // Compare result with finite differences
            // Correct constraints error
            // Get constraints hessians
            // Add jacobian correction term?
            // Get cost gradient/hessian
            // Calculate tangent space and choose subset of constraints to work with if full set is rank deficient
            // Calculate effective hessian
            // Solve QP subproblem
            // Calculate correction term
            // Check result validity and adjust trust region accordingly
            // Collision testing (if optimum reached, or sufficient distance crossed since last collision check)
            // If in collision, add constraint and go back to collision free state
            // Else, go to new state
          }
          getFullSplines(reducedParameters, splines, linearConstraints, hybridSolver);
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
