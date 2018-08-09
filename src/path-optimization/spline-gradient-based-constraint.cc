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
#include <hpp/core/path-optimization/quadratic-program.hh>

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
      HPP_DEFINE_TIMECOUNTER(SGB_hessianJacobians);
      HPP_DEFINE_TIMECOUNTER(SGB_hessianAssignment);
      HPP_DEFINE_TIMECOUNTER(SGB_hessian);
      HPP_DEFINE_TIMECOUNTER(SGB_jacobian1);
      HPP_DEFINE_TIMECOUNTER(SGB_jacobian21);
      HPP_DEFINE_TIMECOUNTER(SGB_jacobian22);
      HPP_DEFINE_TIMECOUNTER(SGB_jacobian23);
      HPP_DEFINE_TIMECOUNTER(SGB_jacobian24);
      HPP_DEFINE_TIMECOUNTER(SGB_jacobian3);
      HPP_DEFINE_TIMECOUNTER(SGB_jacobian4);

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
        {}

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
         std::vector<size_type>& constraintSplineIndex,
         std::vector<size_type>& constraintOutputSize,
         std::vector<value_type>& errorThreshold)
        {
          this->nbConstraints = 0;
          constraintSplineIndex.empty();
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
                bool equalToZero = numericalConstraints[j]->comparisonType()[0] == constraints::EqualToZero;

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
                  if (equalToZero)
                  {
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
                      hybridSolver.add(initialNumericalconstraint, 0, numericalConstraint->comparisonType());
                      this->nbConstraints += numericalConstraint->functionPtr()->outputDerivativeSize();
                      for (std::size_t k = 0; k < numericalConstraint->functionPtr()->outputDerivativeSize(); ++k)
                      {
                        constraintSplineIndex.push_back(i);
                        constraintOutputSize.push_back(numericalConstraint->functionPtr()->outputDerivativeSize());
                        errorThreshold.push_back(configProjector->errorThreshold());
                      }
                    }
                    if (i <= splines.size() - 2)
                    {
                      // Add end constraints to next spline's initial point
                      inArgs.first = robot_->configSize() * Spline::NbCoeffs*(i+1);
                      inDers.first = robot_->numberDof() * Spline::NbCoeffs*(i+1);
                      StateFunction::Ptr_t endNumericalconstraint (new StateFunction(numericalConstraint->functionPtr(),
                            nArgs, nDers, inArgs, inDers));
                      hybridSolver.add(endNumericalconstraint, 0, numericalConstraint->comparisonType());
                      this->nbConstraints += numericalConstraint->functionPtr()->outputDerivativeSize();
                      for (std::size_t k = 0; k < numericalConstraint->functionPtr()->outputDerivativeSize(); ++k)
                      {
                        constraintSplineIndex.push_back(i+1);
                        constraintOutputSize.push_back(numericalConstraint->functionPtr()->outputDerivativeSize());
                        errorThreshold.push_back(configProjector->errorThreshold());
                      }
                    }
                  }
                  else
                  {
                    // TODO: Equality (mixed?) type constraints
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
          if (index - segments[i].first >= 0 and index - segments[i].first < segments[i].second)
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
        void SplineGradientBasedConstraint<_PB, _SO>::processInequalityConstraints
        (const LinearConstraint& boundConstraint, LinearConstraint& boundConstraintReduced,
         const Splines_t& splines, const HybridSolver& hybridSolver,
         const LinearConstraint& linearConstraints, const std::vector<size_type>& dofPerSpline) const
        {
          int MaxContinuityOrder = int( (SplineOrder - 1) / 2);
          matrix_t boundConstraintFree = hybridSolver.explicitSolver().freeDers().rview(boundConstraint.J);

          Eigen::RowBlockIndices nonZeroRows;
          for (std::size_t i = 0; i < boundConstraintFree.rows(); ++i) {
            if (not boundConstraintFree.row(i).isZero(1e-8))
              nonZeroRows.addRow(i, 1);
          }
          boundConstraintFree = nonZeroRows.rview(boundConstraintFree).eval();
          vector_t boundConstraintFreeRhs = nonZeroRows.rview(boundConstraint.b).eval();

          nonZeroRows.clearRows();
          if (MaxContinuityOrder == 0)
          {
            size_type i = 0;
            size_type index = 0;
            size_type row = 0;
            while (i < splines.size())
            {
              if (boundConstraintFree.row(row).middleCols(index, 2*dofPerSpline[i]).isZero(1e-8)) {
                index += 2*dofPerSpline[i];
                ++i;
              }
              else
              {
                if (i >= 1 and boundConstraintFree.row(row).middleCols
                    (index + dofPerSpline[i], dofPerSpline[i]).isZero(1e-8))
                  nonZeroRows.addRow(row, 1);
                ++row;
              }
            }
            nonZeroRows.updateRows<true, true, true>();
          }
          else if (MaxContinuityOrder == 1)
          {
          }

          boundConstraintFree = nonZeroRows.rview(boundConstraintFree).eval();
          boundConstraintFreeRhs = nonZeroRows.rview(boundConstraintFreeRhs).eval();
          boundConstraintReduced.J.resize(boundConstraintFree.rows(), Eigen::NoChange);
          boundConstraintReduced.b.resize(boundConstraintFree.rows(), Eigen::NoChange);
          boundConstraintReduced.J = boundConstraintFree * linearConstraints.PK;
          boundConstraintReduced.b = boundConstraintFreeRhs - boundConstraintFree * linearConstraints.xStar;
        }

      template <int _PB, int _SO>
        LiegroupSpacePtr_t SplineGradientBasedConstraint<_PB, _SO>::createProblemLieGroup
        (std::vector<LiegroupSpacePtr_t>& splineSpaces, const Splines_t splines, const HybridSolver hybridSolver) const
        {
          LiegroupSpacePtr_t stateSpace = LiegroupSpace::empty();
          segments_t freeDofs = hybridSolver.explicitSolver().freeDers().indices();
          std::size_t j = 0;
          std::size_t splineIndex;
          for (std::size_t i = 0; i < splines.size(); ++i)
          {
            LiegroupSpacePtr_t splineSpace = LiegroupSpace::empty();

            splineIndex = j;
            while (j - splineIndex < robot_->numberDof())
            {
              JointPtr_t joint = robot_->getJointAtVelocityRank(j-splineIndex);
              LiegroupSpacePtr_t jointSpace = joint->configurationSpace();
              if (contains(j, freeDofs)) {
                splineSpace = splineSpace * jointSpace;
              }
              j += jointSpace->nv();
            }
            j += (Spline::NbCoeffs - 1)*(j - splineIndex);
            // State space for initial and final spline coefficients only
            stateSpace = stateSpace * (splineSpace * splineSpace);

            splineSpace->mergeVectorSpaces();
            splineSpaces[i] = splineSpace;
          }
          stateSpace->mergeVectorSpaces();
          return stateSpace;
        }


      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::costFunction(const Splines_t splines,
            const LinearConstraint& linearConstraints, std::vector<size_type> dofPerSpline,
            matrix_t& hessian, vector_t& gradientAtZero, value_type& valueAtZero) const
        {
          matrix_t fullHessian(linearConstraints.J.cols(), linearConstraints.J.cols());
          fullHessian.setZero();
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
                fullHessian.block(splineIndex + nDof*j, splineIndex + nDof*k, nDof, nDof) =
                  integrals.coeff(j, k) * matrix_t::Identity(nDof, nDof);
              }
            }
            splineIndex += nDof*nbCoeffs;
          }

          hessian = linearConstraints.PK.transpose() * fullHessian * linearConstraints.PK;
          gradientAtZero = linearConstraints.xStar.transpose() * fullHessian * linearConstraints.PK;
          valueAtZero = linearConstraints.xStar.transpose() * fullHessian * linearConstraints.xStar;
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::getFullSplines
        (const vector_t freeParameters, Splines_t& splines, HybridSolver hybridSolver) const
        {
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
          value_type u = 1;
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
        (matrix_t& A, vector_t b, value_type r) const
        {
          Eigen::SelfAdjointEigenSolver<matrix_t> eigenSolver(A);
          matrix_t V = eigenSolver.eigenvectors();
          vector_t d = eigenSolver.eigenvalues();
          if (d[0] > 0)
          {
            vector_t x(b.size());
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

          return (A + (u - d[0]) * matrix_t::Identity(b.size(), b.size())).llt().solve(b);
        }


      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::addCollisionConstraint (
            Splines_t collisionFreeSplines, Splines_t collisionSplines,
            std::pair<CollisionPathValidationReportPtr_t, std::size_t> collisionReport)
        {
          DifferentiableFunctionPtr_t cc =
            CollisionFunction::create(robot_,
                collisionFreeSplines[collisionReport.second],
                collisionSplines[collisionReport.second],
                collisionReport.first);
          vector_t collisionFreeConfig(cc->inputSize());
          vector_t configOnSpline(cc->inputSize());
          collisionFreeSplines[collisionReport.second]->at (collisionReport.first->parameter, configOnSpline);
          (*(collisionFreeSplines[collisionReport.second])) (collisionFreeConfig, collisionReport.first->parameter);
          this->collFunctions.push_back(cc);
          this->collValues.push_back((((*cc) (configOnSpline)).vector())[0]);
          this->collIndices.push_back(collisionReport.second);
          this->collTimes.push_back(collisionReport.first->parameter);
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::getConstraintsValue
        (const vector_t x, Splines_t& splines, vectorOut_t value,
         HybridSolver& hybridSolver) const
        {
          getFullSplines(x, splines, hybridSolver);

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
        void SplineGradientBasedConstraint<_PB, _SO>::getCollisionConstraintsValue
        (const vector_t x, Splines_t& splines, vectorOut_t value, HybridSolver& hybridSolver) const
        {
          getFullSplines(x, splines, hybridSolver);
          vector_t stateConfiguration(splines.size() * Spline::NbCoeffs * robot_->configSize());
          stateConfiguration.setZero();
          for (std::size_t i = 0; i < this->collFunctions.size(); ++i)
          {
            splines[this->collIndices[i]]->at (this->collTimes[i],
                stateConfiguration.segment(this->collIndices[i]*Spline::NbCoeffs*robot_->configSize(), robot_->configSize()));
            hybridSolver.explicitSolver().solve(stateConfiguration);
            value.row(this->nbConstraints+i) = ((*(this->collFunctions[i])) (stateConfiguration.segment
                 (this->collIndices[i]*Spline::NbCoeffs*robot_->configSize(), robot_->configSize()))).vector();
            value[this->nbConstraints+i] -= this->collValues[i];
          }
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::getConstraintsValueJacobian
        (const vector_t x, Splines_t& splines, vectorOut_t value, matrixOut_t jacobian,
         const std::vector<size_type>& dofPerSpline,
         HybridSolver& hybridSolver, const LiegroupSpacePtr_t stateSpace) const
        {
          HPP_START_TIMECOUNTER(SGB_jacobian1);
          getFullSplines(x, splines, hybridSolver);
          vector_t stateConfiguration(splines.size() * Spline::NbCoeffs * robot_->configSize());
          vector_t baseConfiguration(splines.size() * 2 * robot_->configSize());
          stateConfiguration.setZero();
          baseConfiguration.setZero();
          for (std::size_t i = 0; i < splines.size(); ++i) {
            splines[i]->at (splines[i]->timeRange().first,
                stateConfiguration.segment(
                  i*Spline::NbCoeffs*robot_->configSize(), robot_->configSize()));
            splines[i]->at (splines[i]->timeRange().second,
                stateConfiguration.segment(
                  ((i+1)*Spline::NbCoeffs-1)*robot_->configSize(), robot_->configSize()));

            baseConfiguration.segment(i*2*robot_->configSize(),
                robot_->configSize()) = splines[i]->base();
            baseConfiguration.segment(i*2*robot_->configSize() + robot_->configSize(),
                robot_->configSize()) = splines[i]->base();
          }

          hybridSolver.explicitSolver().solve(stateConfiguration);
          HPP_STOP_TIMECOUNTER(SGB_jacobian1);
          HPP_START_TIMECOUNTER(SGB_jacobian21);
          hybridSolver.computeValue<true>(stateConfiguration);
          HPP_STOP_TIMECOUNTER(SGB_jacobian21);
          HPP_START_TIMECOUNTER(SGB_jacobian22);
          hybridSolver.updateJacobian(stateConfiguration);
          HPP_STOP_TIMECOUNTER(SGB_jacobian22);
          HPP_START_TIMECOUNTER(SGB_jacobian23);
          hybridSolver.getValue(value);
          HPP_STOP_TIMECOUNTER(SGB_jacobian23);
          HPP_START_TIMECOUNTER(SGB_jacobian24);
          hybridSolver.getReducedJacobian(jacobian);
          HPP_STOP_TIMECOUNTER(SGB_jacobian24);

          HPP_START_TIMECOUNTER(SGB_jacobian3);

          Eigen::ColBlockIndices initialEndIndices;
          size_type index = 0;
          for (std::size_t i = 0; i < splines.size(); ++i)
          {
            initialEndIndices.addCol(index, dofPerSpline[i]);
            initialEndIndices.addCol(index + (Spline::NbCoeffs-1)*dofPerSpline[i], dofPerSpline[i]);
            index += Spline::NbCoeffs * dofPerSpline[i];
          }
          vector_t initialEndParameters =
            initialEndIndices.transpose().rview(x);

          LiegroupElement baseElement(hybridSolver.explicitSolver().freeArgs().rview(baseConfiguration), stateSpace);
          matrix_t initialEndJacobian = initialEndIndices.rview(jacobian);
          HPP_STOP_TIMECOUNTER(SGB_jacobian3);
          HPP_START_TIMECOUNTER(SGB_jacobian4);
          stateSpace->dIntegrate_dv <false> (baseElement, initialEndParameters, initialEndJacobian); 
          initialEndIndices.lview(jacobian) = initialEndJacobian;
          HPP_STOP_TIMECOUNTER(SGB_jacobian4);
          matrix_t J (hybridSolver.explicitSolver().derSize(), hybridSolver.explicitSolver().derSize());
          hybridSolver.explicitSolver().jacobian(J, stateConfiguration);
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::getCollisionConstraintsValueJacobian
        (const vector_t x, Splines_t& splines, vectorOut_t value, matrixOut_t jacobian, HybridSolver& hybridSolver,
         const std::vector<size_type>& dofPerSpline, const std::vector<LiegroupSpacePtr_t>& splineSpaces) const
        {
          std::vector<size_type> splineIndex;
          size_type index = 0;
          for (std::size_t i = 0; i < splines.size(); ++i)
          {
            splineIndex.push_back(index);
            index += Spline::NbCoeffs * dofPerSpline[i];
          }
          vector_t stateConfiguration(splines.size() * Spline::NbCoeffs * robot_->configSize());
          stateConfiguration.setZero();
          matrix_t collisionJacobian (this->collFunctions.size(), splines.size()*Spline::NbCoeffs*robot_->numberDof());
          getFullSplines(x, splines, hybridSolver);
          for (std::size_t i = 0; i < this->collFunctions.size(); ++i)
          {
            index = splineIndex[this->collIndices[i]];
            size_type dof = dofPerSpline[this->collIndices[i]];

            splines[this->collIndices[i]]->at (this->collTimes[i],
                stateConfiguration.segment(this->collIndices[i]*
                  Spline::NbCoeffs*robot_->configSize(), robot_->configSize()));
            hybridSolver.explicitSolver().solve(stateConfiguration);

            value.row(this->nbConstraints+i) = ((*(this->collFunctions[i])) (stateConfiguration.segment
                 (this->collIndices[i]*Spline::NbCoeffs*robot_->configSize(), robot_->configSize()))).vector();
            value[this->nbConstraints+i] -= this->collValues[i];

            collisionJacobian.setZero();
            this->collFunctions[i]->jacobian(collisionJacobian.middleRows(i, 1)
                .middleCols(this->collIndices[i]*Spline::NbCoeffs*robot_->numberDof(), robot_->numberDof()),
                stateConfiguration.segment
                 (this->collIndices[i]*Spline::NbCoeffs*robot_->configSize(), robot_->configSize()));

            hybridSolver.reduceJacobian(stateConfiguration,
                collisionJacobian.middleRows(i, 1), jacobian.middleRows(this->nbConstraints+i, 1));

            vector_t BernsteinCoeffs(int (Spline::NbCoeffs));
            matrix_t splinesJacobian(dof, x.size());
            splinesJacobian.setZero();
            splines[this->collIndices[i]]->parameterDerivativeCoefficients(BernsteinCoeffs, this->collTimes[i]);

            vector_t v (dof);
            v.setZero();
            for (std::size_t j = 0; j < BernsteinCoeffs.size(); ++j)
            {
              splinesJacobian.block(0, index + dof*j, dof, dof) =
                BernsteinCoeffs[j] * matrix_t::Identity(dof, dof);
              v += BernsteinCoeffs[j] * x.segment(index + dof*j, dof);
            }

            std::size_t tmp = this->collIndices[i];
            splineSpaces[tmp]->dIntegrate_dv <false> (splineSpaces[this->collIndices[i]]->neutral(),
                v, jacobian.middleRows(this->nbConstraints+i, 1).middleCols(index, dof));

            jacobian.middleRows(this->nbConstraints+i, 1) =
              (jacobian.middleRows(this->nbConstraints+i, 1).
               middleCols(index, dof) * splinesJacobian).eval();
          }
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::getConstraintsHessian
        (const vector_t x, Splines_t& splines, std::vector<matrix_t>& hessianStack,
         const std::vector<size_type>& dofPerSpline,
         HybridSolver& hybridSolver, const LiegroupSpacePtr_t stateSpace,
         const std::vector<size_type>& constraintSplineIndex) const
        {
          HPP_SCOPE_TIMECOUNTER(SGB_hessian);
          // TODO: Equality type constraints
          size_type maxDof = *std::max_element(dofPerSpline.begin(), dofPerSpline.end());
          std::vector<size_type> splineIndex;
          size_type index = 0;
          for (std::size_t i = 0; i < splines.size(); ++i)
          {
            splineIndex.push_back(index);
            index += Spline::NbCoeffs * dofPerSpline[i];
          }

          matrix_t jacobianPlus (hessianStack.size(), x.size());
          matrix_t jacobianMinus (hessianStack.size(), x.size());
          matrix_t jacobianPlusFinite (hessianStack.size(), x.size());
          matrix_t jacobianMinusFinite (hessianStack.size(), x.size());
          jacobianPlus.setZero();
          jacobianMinus.setZero();
          matrix_t jacobianDiff (hessianStack.size(), x.size());
          matrix_t jacobianDiffFinite (hessianStack.size(), x.size());
          vector_t tmpValue (this->nbConstraints);
          vector_t step(x.size());
          step.setZero();

          size_type variableIndex;
          for (std::size_t i = 0; i < maxDof; ++i)
          {
            for (std::size_t j = 0; j < splines.size(); ++j)
            {
              if (i < dofPerSpline[j]) step[splineIndex[j] + i] = this->stepSize;
            }

            HPP_START_TIMECOUNTER(SGB_hessianJacobians);
            getConstraintsValueJacobian(x + step, splines, tmpValue.topRows(this->nbConstraints),
                jacobianPlus.topRows(this->nbConstraints), dofPerSpline, hybridSolver, stateSpace);
            getConstraintsValueJacobian(x - step, splines, tmpValue.topRows(this->nbConstraints),
                jacobianMinus.topRows(this->nbConstraints), dofPerSpline, hybridSolver, stateSpace);
            HPP_STOP_TIMECOUNTER(SGB_hessianJacobians);

            jacobianDiff = (jacobianPlus - jacobianMinus)/(2*this->stepSize);

            HPP_START_TIMECOUNTER(SGB_hessianAssignment);
            for (std::size_t k = 0; k < this->nbConstraints; ++k)
            {
              if (i < dofPerSpline[constraintSplineIndex[k]]) {
                variableIndex = splineIndex[constraintSplineIndex[k]] + i;
                hessianStack[k].row(variableIndex) = jacobianDiff.row(k);
              }
            }
            HPP_STOP_TIMECOUNTER(SGB_hessianAssignment);

            for (std::size_t j = 0; j < splines.size(); ++j)
            {
              if (i < dofPerSpline[j]) step[splineIndex[j] + i] = 0;
            }
          }
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::getCollisionConstraintsHessians
        (const vector_t x, Splines_t& splines, std::vector<matrix_t>& hessianStack,
         const std::vector<size_type>& dofPerSpline,
         HybridSolver& hybridSolver, const std::vector<LiegroupSpacePtr_t>& splineSpaces) const
        {
          std::vector<size_type> splineIndex;
          size_type index = 0;
          for (std::size_t i = 0; i < splines.size(); ++i)
          {
            splineIndex.push_back(index);
            index += Spline::NbCoeffs * dofPerSpline[i];
          }

          vector_t stateConfiguration(splines.size() * Spline::NbCoeffs * robot_->configSize());
          stateConfiguration.setZero();
          matrix_t collisionJacobianPlus (1, splines.size()*Spline::NbCoeffs*robot_->numberDof());
          matrix_t collisionJacobianMinus (1, splines.size()*Spline::NbCoeffs*robot_->numberDof());
          matrix_t reducedCollisionJacobianPlus (1, x.size());
          matrix_t reducedCollisionJacobianMinus (1, x.size());

          getFullSplines(x, splines, hybridSolver);
          vector_t v (x.size());
          vector_t full_v (hybridSolver.explicitSolver().derSize());
          vector_t step (x.size());
          v.setZero();
          full_v.setZero();
          step.setZero();
          for (std::size_t k = 0; k < this->collFunctions.size(); ++k)
          {
            index = splineIndex[this->collIndices[k]];
            size_type dof = dofPerSpline[this->collIndices[k]];
            matrix_t collisionHessian (dof, dof);
            collisionHessian.setZero();

            vector_t BernsteinCoeffs(int (Spline::NbCoeffs));
            matrix_t splinesJacobian(dof, x.size());
            splinesJacobian.setZero();
            splines[this->collIndices[k]]->parameterDerivativeCoefficients(BernsteinCoeffs, this->collTimes[k]);

            for (std::size_t j = 0; j < BernsteinCoeffs.size(); ++j) {
              splinesJacobian.block(0, index + dof*j, dof, dof) =
                BernsteinCoeffs[j] * matrix_t::Identity(dof, dof);
              v.segment(index, dof) += BernsteinCoeffs[j] * x.segment(index + dof*j, dof);
            }

            for (std::size_t i = 0; i < dof; ++i)
            {
              step[index + i] = this->stepSize;
              // Jacobian at v + step
              hybridSolver.explicitSolver().freeDers().transpose().lview(full_v) = v + step;

              splines[this->collIndices[k]]->rowParameters(full_v.segment
                  (this->collIndices[k]*Spline::NbCoeffs*robot_->numberDof(), Spline::NbCoeffs*robot_->numberDof()));
              splines[this->collIndices[k]]->at (splines[this->collIndices[k]]->timeRange().first,
                  stateConfiguration.segment(this->collIndices[k]*Spline::NbCoeffs*robot_->configSize(), robot_->configSize()));

              hybridSolver.explicitSolver().solve(stateConfiguration);

              this->collFunctions[k]->jacobian(collisionJacobianPlus.middleCols(this->collIndices[k]
                    *Spline::NbCoeffs*robot_->numberDof(), robot_->numberDof()),
                    stateConfiguration.segment
                    (this->collIndices[k]*Spline::NbCoeffs*robot_->configSize(), robot_->configSize()));

              hybridSolver.reduceJacobian(stateConfiguration,
                  collisionJacobianPlus, reducedCollisionJacobianPlus);

              std::size_t tmp = this->collIndices[k];
              splineSpaces[tmp]->dIntegrate_dv <false> (splineSpaces[this->collIndices[k]]->neutral(),
                  (v+step).segment(index, dof), reducedCollisionJacobianPlus.middleCols(index, dof));

              // Jacobian at v - step
              hybridSolver.explicitSolver().freeDers().transpose().lview(full_v) = v - step;

              splines[this->collIndices[k]]->rowParameters(full_v.segment
                  (this->collIndices[k]*Spline::NbCoeffs*robot_->numberDof(), Spline::NbCoeffs*robot_->numberDof()));
              splines[this->collIndices[k]]->at (splines[this->collIndices[k]]->timeRange().first,
                  stateConfiguration.segment(this->collIndices[k]*Spline::NbCoeffs*robot_->configSize(), robot_->configSize()));

              hybridSolver.explicitSolver().solve(stateConfiguration);

              this->collFunctions[k]->jacobian(collisionJacobianMinus.middleCols(this->collIndices[k]
                    *Spline::NbCoeffs*robot_->numberDof(), robot_->numberDof()),
                    stateConfiguration.segment
                    (this->collIndices[k]*Spline::NbCoeffs*robot_->configSize(), robot_->configSize()));

              hybridSolver.reduceJacobian(stateConfiguration,
                  collisionJacobianMinus, reducedCollisionJacobianMinus);

              splineSpaces[tmp]->dIntegrate_dv <false> (splineSpaces[this->collIndices[k]]->neutral(),
                  (v-step).segment(index, dof), reducedCollisionJacobianMinus.middleCols(index, dof));

              collisionHessian.row(i) = (reducedCollisionJacobianPlus.middleCols(index, dof)
                  - reducedCollisionJacobianMinus.middleCols(index,dof))/(2*this->stepSize);

              step[index + i] = 0;
            }
            hessianStack[this->nbConstraints + k] = splinesJacobian.transpose() * collisionHessian * splinesJacobian;
          }
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::getJacobianFiniteDiff
        (const vector_t x, Splines_t& splines, matrixOut_t jacobian, HybridSolver& hybridSolver) const
        {
          vector_t value(jacobian.rows());
          vector_t valuePlus(jacobian.rows());
          vector_t valueMinus(jacobian.rows());

          vector_t step(x.size());
          step.setZero();

          getConstraintsValue(x, splines, value, hybridSolver);
          for (std::size_t i = 0; i < step.size(); ++i)
          {
            step[i] = this->stepSize;
            getConstraintsValue(x+step, splines, valuePlus.topRows(this->nbConstraints), hybridSolver);
            getCollisionConstraintsValue(x+step, splines, valuePlus,
                hybridSolver);

            getConstraintsValue(x-step, splines, valueMinus.topRows(this->nbConstraints), hybridSolver);
            getCollisionConstraintsValue(x-step, splines, valueMinus,
                hybridSolver);
            jacobian.col(i) = (valuePlus - valueMinus)/(2*this->stepSize);
            step[i] = 0;
          }
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::getHessianFiniteDiff
        (const vector_t x, Splines_t& splines, std::vector<matrix_t>& hessianStack,
         HybridSolver& hybridSolver, const std::vector<size_type>& dofPerSpline,
         const LiegroupSpacePtr_t stateSpace) const
        {
          vector_t step(x.size());
          step.setZero();
          matrix_t jacobian(hessianStack.size(), step.size());
          matrix_t jacobianStep(hessianStack.size(), step.size());
          matrix_t jacobianDiff(hessianStack.size(), step.size());
          vector_t tmpValue (hessianStack.size());
          getConstraintsValueJacobian(x, splines, tmpValue.topRows(this->nbConstraints),
              jacobian, dofPerSpline, hybridSolver, stateSpace);
          for (std::size_t i = 0; i < step.size(); ++i)
          {
            step[i] = this->stepSize;
            getConstraintsValueJacobian(x + step, splines, tmpValue.topRows(this->nbConstraints),
                jacobianStep, dofPerSpline, hybridSolver, stateSpace);
            step[i] = 0;
            jacobianDiff = (jacobianStep - jacobian)/(this->stepSize);
            for (std::size_t k = 0; k < hessianStack.size(); ++k)
            {
              hessianStack[k].row(i) = jacobianDiff.row(k);
            }
          }
        }

      template <int _PB, int _SO>
        void SplineGradientBasedConstraint<_PB, _SO>::getHessianDoubleFiniteDiff
        (const vector_t x, Splines_t& splines, std::vector<matrix_t>& hessianStack,
         HybridSolver hybridSolver) const
        {
          vector_t valuePlusPlus(hessianStack.size());
          vector_t valuePlusMinus(hessianStack.size());
          vector_t valueMinusPlus(hessianStack.size());
          vector_t valueMinusMinus(hessianStack.size());
          vector_t step(x.size());
          vector_t step2(x.size());
          step.setZero();
          step2.setZero();
          vector_t d2f(hessianStack.size());
          for (std::size_t i = 0; i < step.size(); ++i)
          {
            hppDout(info, i << "|" << step.size());
            step[i] = this->stepSize;
            for (std::size_t j = 0; j < step.size(); ++j)
            {
              step2[j] = this->stepSize;

              getConstraintsValue(x+step+step2, splines, valuePlusPlus.topRows(this->nbConstraints), hybridSolver);
              getCollisionConstraintsValue(x+step+step2, splines,
                  valuePlusPlus, hybridSolver);

              getConstraintsValue(x+step-step2, splines, valuePlusMinus.topRows(this->nbConstraints), hybridSolver);
              getCollisionConstraintsValue(x+step-step2, splines,
                  valuePlusMinus, hybridSolver);

              getConstraintsValue(x-step+step2, splines, valueMinusPlus.topRows(this->nbConstraints), hybridSolver);
              getCollisionConstraintsValue(x-step+step2, splines,
                  valuePlusMinus, hybridSolver);

              getConstraintsValue(x-step-step2, splines, valueMinusMinus.topRows(this->nbConstraints), hybridSolver);
              getCollisionConstraintsValue(x-step-step2, splines,
                  valuePlusMinus, hybridSolver);

              d2f = (valuePlusPlus - valuePlusMinus - valueMinusPlus + valueMinusMinus)/(4*std::pow(this->stepSize, 2));
              for (std::size_t k = this->nbConstraints; k < hessianStack.size(); ++k)
                hessianStack[k](i,j) = d2f[k];
              step2[j] = 0;
            }
            step[i] = 0;
          }
        }

      template <int _PB, int _SO>
        vector_t SplineGradientBasedConstraint<_PB, _SO>::getSecondOrderCorrection
        (const vector_t step, const matrix_t& jacobian, const std::vector<matrix_t>& hessianStack,
         const matrix_t& inverseGram, const matrix_t& PK) const
        {
          vector_t secondOrderCorrection(step.size());
          secondOrderCorrection.setZero();
          for (std::size_t j = 0; j < inverseGram.rows(); ++j)
          {
            vector_t v_j(step.size());
            v_j.setZero();
            for (std::size_t i = 0; i < inverseGram.rows(); ++i)
            {
              v_j -= inverseGram.coeff(i, j) * jacobian.row(i);
            }
            secondOrderCorrection += 0.5*
              (step.transpose() * (PK.transpose() * (hessianStack[j] * (PK * step))))
              .coeff(0, 0) * v_j;
          }
          return secondOrderCorrection;
        }

      template <int _PB, int _SO>
        value_type SplineGradientBasedConstraint<_PB, _SO>::errorRelativeToThreshold
        (const vector_t& value, const std::vector<size_type>& constraintOutputSize,
         const std::vector<value_type>& errorThreshold) const
        {
          std::size_t i = 0;
          value_type maxError = 0.;
          while (i < value.size())
          {
            maxError = (value.segment(i, constraintOutputSize[i]).norm() / errorThreshold[i] > maxError) ?
              value.segment(i, constraintOutputSize[i]).norm()/errorThreshold[i] : maxError;
            i += constraintOutputSize[i];
          }
          return maxError;
        }

      // ----------- Optimize ----------------------------------------------- //

      template <int _PB, int _SO>
        PathVectorPtr_t SplineGradientBasedConstraint<_PB, _SO>::optimize (const PathVectorPtr_t& path)
        {
          size_type maxIterations = problem().getParameter(
              "SplineGradientBasedConstraint/maxIterations", size_type(200));
          bool returnEquivalentSpline = problem().getParameter(
              "SplineGradientBasedConstraint/returnEquivalentSpline", false);
          bool checkJointBound = problem().getParameter(
              "SplineGradientBasedConstraint/checkJointBound", true);
          bool checkCollisions = problem().getParameter(
              "SplineGradientBasedConstraint/checkCollisions", true);
          this->stepSize = problem().getParameter(
              "SplineGradientBasedConstraint/stepSize", 1e-6);
          value_type trustRadius = problem().getParameter(
              "SplineGradientBasedConstraint/trustRadius", 1);
          value_type stepThreshold = problem().getParameter(
              "SplineGradientBasedConstraint/stepThreshold", 1e-3);
          value_type stepMin = problem().getParameter(
              "SplineGradientBasedConstraint/stepMin", .5);
          value_type collisionCheckThreshold = problem().getParameter(
              "SplineGradientBasedConstraint/collisionCheckThreshold", 1.);

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
          if (returnEquivalentSpline) {
            return this->buildPathVector(splines);
          }
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

          HybridSolver hybridSolver(nArgs, nDers);

          std::vector<size_type> dofPerSpline(splines.size(), rDof);
          std::vector<size_type> argPerSpline(splines.size(), robot_->configSize());
          LinearConstraint linearConstraints(nDers, 0);
          std::vector<size_type> constraintSplineIndex;
          std::vector<size_type> constraintOutputSize;
          std::vector<value_type> errorThreshold;
          addProblemConstraints(splines, hybridSolver, linearConstraints,
              dofPerSpline, argPerSpline, constraintSplineIndex,
              constraintOutputSize, errorThreshold);

          std::vector<LiegroupSpacePtr_t> splineSpaces (splines.size());
          LiegroupSpacePtr_t stateSpace = createProblemLieGroup(splineSpaces, splines, hybridSolver);

          processContinuityConstraints(splines, hybridSolver, continuityConstraints, linearConstraints);

          vector_t fullParameters(nParameters*rDof);
          getParameters(splines, fullParameters);

          vector_t freeParameters = hybridSolver.explicitSolver().freeDers().transpose().rview(fullParameters);

          linearConstraints.decompose();
          vector_t reducedParameters = linearConstraints.PK.transpose() * (freeParameters - linearConstraints.xStar);

          // 3
          // Cost value is: 1/2 x^T Q x + L^T x + C
          // Cost gradient is: Qx + L
          // where x = reducedParameters
          matrix_t costQuadratic(reducedParameters.size(), reducedParameters.size());
          vector_t costLinear(reducedParameters.size());
          value_type costConstant(reducedParameters.size());
          costFunction(splines, linearConstraints, dofPerSpline, costQuadratic, costLinear, costConstant);

          Eigen::RowBlockIndices activeInequalities;
          LinearConstraint boundConstraint (nDers, 0);
          LinearConstraint boundConstraintReduced (reducedParameters.size(), 0);
          if (checkJointBound)
          {
            this->jointBoundConstraint(splines, boundConstraint);
            if (!this->validateBounds(splines, boundConstraint).empty())
              throw std::invalid_argument("Input path does not satisfy joint bounds");
            // Remove redundant inequalities
            processInequalityConstraints(boundConstraint, boundConstraintReduced,
                splines, hybridSolver, linearConstraints, dofPerSpline);
          }
          // Compute initial active set
          activeInequalities.clearRows();
          for (std::size_t k = 0; k < boundConstraintReduced.J.rows(); ++k)
          {
            if (std::abs(boundConstraintReduced.J.row(k) * reducedParameters
                  - boundConstraintReduced.b[k]) < 1e-8) activeInequalities.addRow(k, 1);
          }

          if (checkCollisions) {
            if (!(this->validatePath(splines, true)).empty())
              throw std::invalid_argument("Input path contains a collision");
            this->collFunctions.clear();
            this->collValues.clear();
            this->collTimes.clear();
            this->collIndices.clear();
          }


          vector_t value(this->nbConstraints);
          matrix_t jacobian(this->nbConstraints, freeParameters.size());
          matrix_t reducedJacobian(this->nbConstraints, reducedParameters.size());
          jacobian.setZero();

          matrix_t zeroMatrix(freeParameters.size(), freeParameters.size());
          zeroMatrix.setZero();
          std::vector<matrix_t> hessianStack(this->nbConstraints, zeroMatrix);
          matrix_t hessian(reducedParameters.size(), reducedParameters.size());
          matrix_t hessianCorrection(freeParameters.size(), freeParameters.size());

          Splines_t collisionSplines;
          Splines_t newSplines;
          Base::copy (splines, collisionSplines);
          Base::copy (splines, newSplines);
          vector_t collisionFreeParameters = reducedParameters;
          vector_t newParameters (reducedParameters.size());
          value_type lengthSinceLastCheck = 0;

          size_type nbIterations = 0;
          while (!this->interrupt_)
          {
            if (++nbIterations > maxIterations) break;
            hppDout(info, "Iteration: " << nbIterations);
            // Get constraints jacobian
            freeParameters = linearConstraints.xStar + linearConstraints.PK * reducedParameters;
            getConstraintsValueJacobian(freeParameters, splines, value.topRows(this->nbConstraints),
                jacobian.topRows(this->nbConstraints), dofPerSpline, hybridSolver, stateSpace);
            getCollisionConstraintsValueJacobian(freeParameters, splines, value,
                jacobian, hybridSolver, dofPerSpline, splineSpaces);

            reducedJacobian = jacobian * linearConstraints.PK;

            hppDout(info, "Constraint error: " << value.norm());
            hppDout(info, "Constraint error infty norm: " << errorRelativeToThreshold(value, constraintOutputSize, errorThreshold));
            hppDout(info, value.transpose());
            hppDout(info, "Cost: " << (.5*reducedParameters.transpose() * (costQuadratic * reducedParameters)
                  + costLinear.transpose() * reducedParameters)(0, 0) + costConstant);
            // Correct constraints error

            vector_t correction (reducedParameters.size());
            correction.setZero();
            if (checkJointBound)
            {
              Eigen::VectorXi activeSet (reducedJacobian.rows() + boundConstraintReduced.J.rows());
              int activeSetSize = 0;
              activeSet.setZero();
              // TODO: Define once
              matrix_t identity = matrix_t::Identity (reducedParameters.size(), reducedParameters.size());
              vector_t zero = vector_t::Zero (reducedParameters.size());
              vector_t rhs = boundConstraintReduced.b - boundConstraintReduced.J * reducedParameters;
              // TODO: Active set is ambiguous
              // [-1 -2 0 0 0]: Is inequality constraint 0 active?
              // TODO: Set active inequalities as equalities?
              solve_quadprog (identity, zero, reducedJacobian.transpose(), value,
                  boundConstraintReduced.J.transpose(), -rhs,
                  correction, activeSet, activeSetSize);
              activeInequalities.clearRows();
              for (std::size_t k = 0; k < boundConstraintReduced.J.rows(); ++k)
              {
                if (activeSet[reducedJacobian.rows() + k] == 0
                    and activeSet[reducedJacobian.rows() + k+1] == 0)
                  break;
                else
                  activeInequalities.addRow(activeSet[reducedJacobian.rows() + k], 1);
              }
              hppDout(info, activeInequalities);
            }
            else
            {
              correction = reducedJacobian.colPivHouseholderQr().solve(-value);
            }

            reducedParameters += correction;
            hppDout(info, "Newton correction norm: " << correction.norm());

            bool optimumReached = false;
            if (errorRelativeToThreshold(value, constraintOutputSize, errorThreshold) < .1)
            {
              vector_t gradient = costQuadratic * reducedParameters + costLinear;

              // Compute new active set
              if (checkJointBound and activeInequalities.nbRows() > 0)
              {
                hppDout(info, activeInequalities);
                vector_t optimalDirection = vector_t::Zero(gradient.size());
                Eigen::VectorXi activeSet (reducedJacobian.rows() + activeInequalities.nbRows());
                activeSet.setZero();
                int activeSetSize = 0;
                matrix_t identity = matrix_t::Identity (reducedParameters.size(), reducedParameters.size());
                vector_t equalityRhs = vector_t::Zero(reducedJacobian.rows());
                matrix_t activeInequalityMatrix = activeInequalities.rview(boundConstraintReduced.J).eval();
                vector_t inequalityRhs = vector_t::Zero(activeInequalityMatrix.rows());
                solve_quadprog(identity, gradient, reducedJacobian.transpose(), equalityRhs,
                    activeInequalityMatrix.transpose(), inequalityRhs,
                    optimalDirection, activeSet, activeSetSize);
                activeInequalities.clearRows();
                for (std::size_t k = 0; k < activeInequalityMatrix.rows(); ++k)
                {
                  if (activeSet[reducedJacobian.rows() + k] == 0
                      and (k+1 == activeInequalityMatrix.rows() or
                        activeSet[reducedJacobian.rows() + k+1] == 0))
                    break;
                  else
                    activeInequalities.addRow(activeSet[reducedJacobian.rows() + k], 1);
                }
              }

              LinearConstraint jacobianConstraint
                (reducedJacobian.cols() + activeInequalities.nbRows(), reducedJacobian.rows());
              jacobianConstraint.J.topRows(reducedJacobian.rows()) = reducedJacobian;
              jacobianConstraint.J.bottomRows(activeInequalities.nbRows()) =

              activeInequalities.rview(boundConstraintReduced.J).eval();

              jacobianConstraint.decompose(false);

              freeParameters = linearConstraints.xStar + linearConstraints.PK * reducedParameters;
              getConstraintsHessian(freeParameters, splines, hessianStack, dofPerSpline,
                  hybridSolver, stateSpace, constraintSplineIndex);
              getCollisionConstraintsHessians(freeParameters, splines, hessianStack, dofPerSpline,
                  hybridSolver, splineSpaces);

              HPP_DISPLAY_TIMECOUNTER(SGB_hessianJacobians);
              HPP_DISPLAY_TIMECOUNTER(SGB_hessianAssignment);
              HPP_DISPLAY_TIMECOUNTER(SGB_hessian);

              hppDout(info, "Testing hessian approximation");
              vector_t freeParametersStep = vector_t::Random(freeParameters.size());
              hppDout(info, freeParametersStep.transpose());
              freeParametersStep *= 1e-2 / freeParametersStep.norm();

              vector_t tmpValue (value.size());
              getConstraintsValue(linearConstraints.xStar + linearConstraints.PK * reducedParameters,
                  splines, value.topRows(this->nbConstraints), hybridSolver);
              getCollisionConstraintsValue (linearConstraints.xStar + linearConstraints.PK *
                  reducedParameters, splines, value, hybridSolver);
              getConstraintsValue(linearConstraints.xStar + linearConstraints.PK * reducedParameters + freeParametersStep,
                  splines, tmpValue.topRows(this->nbConstraints), hybridSolver);
              getCollisionConstraintsValue (linearConstraints.xStar + linearConstraints.PK *
                  (reducedParameters + freeParametersStep), splines, tmpValue, hybridSolver);
              vector_t ddv(value.size());
              for (std::size_t k = 0; k < hessianStack.size(); ++k)
              {
                ddv.row(k) = .5 * freeParametersStep.transpose() * hessianStack[k] * freeParametersStep;
              }
              hppDout(info, (tmpValue - value).norm());
              hppDout(info, (tmpValue - (value + jacobian * freeParametersStep)).norm());
              hppDout(info, (tmpValue - (value + jacobian * freeParametersStep + ddv)).norm());
              hppDout(info, (tmpValue - value).transpose());
              hppDout(info, (tmpValue - (value + jacobian * freeParametersStep)).transpose());
              hppDout(info, (tmpValue - (value + jacobian * freeParametersStep + ddv)).transpose());

              hppDout(info, "\n" << (tmpValue - (value + jacobian * freeParametersStep)).
                  cwiseQuotient(tmpValue - value).transpose());
              hppDout(info, "\n" << (tmpValue - (value + jacobian * freeParametersStep + ddv)).
                  cwiseQuotient(tmpValue - (value + jacobian * freeParametersStep)).transpose());
              hppDout(info, "Testing done");

              Eigen::JacobiSVD<matrix_t> svd(reducedJacobian.transpose(), Eigen::ComputeThinU | Eigen::ComputeFullV);
              hppDout(info, svd.singularValues().transpose().reverse());
              vector_t pseudoInverse = svd.singularValues().cwiseInverse().cwiseAbs2();

              for (std::size_t i = 0; i < svd.singularValues().size(); ++i)
              {
                if (svd.singularValues()[i] < std::pow(10, -8))
                {
                  pseudoInverse[i] = 0;
                  vector_t linearRelation = svd.matrixV().col(i);
                  hppDout(info, "Linear dependence " << linearRelation.transpose());
                }
              }

              matrix_t inverseGram = svd.matrixV() * pseudoInverse.asDiagonal() * svd.matrixV().transpose();

              hessian = costQuadratic;
              hessianCorrection.setZero();
              for (std::size_t i = 0; i < hessianStack.size(); ++i)
              {
                value_type coeff = 0;
                for (std::size_t j = 0; j < hessianStack.size(); ++j) {
                  coeff += gradient.dot(reducedJacobian.row(i)) * inverseGram.coeff(i, j);
                }
                hessianCorrection -= coeff * hessianStack[i];
              }
              matrix_t projectedHessian = jacobianConstraint.PK.transpose()
                * (hessian + linearConstraints.PK.transpose() * hessianCorrection * linearConstraints.PK)
                * jacobianConstraint.PK;

              vector_t step (reducedParameters.size());
              bool noCollision = true;
              bool radiusLimitReached = false;
              bool collisionFound = false;
              Reports_t collisionReports;

              while (true)
              {
                hppDout(info, "Distance since last check: " << lengthSinceLastCheck);
                hppDout(info, "Trust radius: " << trustRadius);
                // TODO: reuse SVD
                vector_t solution = solveQP (projectedHessian,
                    -jacobianConstraint.PK.transpose() * gradient, trustRadius);

                step = jacobianConstraint.PK * solution;
                hppDout(info, "Step: " << solution.norm());
                hppDout(info, "Correction term: " << getSecondOrderCorrection(step,
                    reducedJacobian, hessianStack, inverseGram, linearConstraints.PK).norm());
                step += getSecondOrderCorrection(step,
                    reducedJacobian, hessianStack, inverseGram, linearConstraints.PK);
                vector_t tmpValue = value;

                getConstraintsValue(linearConstraints.xStar + linearConstraints.PK * (reducedParameters + step),
                    splines, value.topRows(this->nbConstraints), hybridSolver);
                getCollisionConstraintsValue (linearConstraints.xStar + linearConstraints.PK *
                    (reducedParameters + step), splines, value, hybridSolver);

                getConstraintsValue(linearConstraints.xStar +
                    linearConstraints.PK * (reducedParameters + jacobianConstraint.PK * solution),
                    splines, tmpValue.topRows(this->nbConstraints), hybridSolver);
                getCollisionConstraintsValue(linearConstraints.xStar + linearConstraints.PK *
                    (reducedParameters + jacobianConstraint.PK * solution), splines,
                    tmpValue, hybridSolver);

                hppDout(info, "First order error: " << tmpValue.norm());
                hppDout(info, "Second order error: " << value.norm());

                hppDout(info, "First order error infty norm: "
                    << errorRelativeToThreshold(tmpValue, constraintOutputSize, errorThreshold));
                hppDout(info, "Second order error infty norm: "
                    << errorRelativeToThreshold(value, constraintOutputSize, errorThreshold));

                value_type costDifference = ((reducedParameters + .5*step).transpose() * (costQuadratic * step)
                    + costLinear.transpose() * step)(0, 0);
                hppDout(info, "Cost difference: " << costDifference);
                if (step.norm() < trustRadius*.9) radiusLimitReached = true;

                if (errorRelativeToThreshold(value, constraintOutputSize, errorThreshold) > 100
                    or costDifference > 0) {
                  trustRadius *= .5;
                  radiusLimitReached = true;
                  continue;
                }
                else if (not radiusLimitReached) {
                  if (step.norm() > trustRadius*.9)
                    trustRadius *= 2;
                  else
                    radiusLimitReached = true;
                  continue;
                }
                optimumReached = (step.norm() < stepThreshold and
                    errorRelativeToThreshold(value, constraintOutputSize, errorThreshold) < .1)
                  or nbIterations == maxIterations;
                if (optimumReached) step.setZero();

                if (checkCollisions and (lengthSinceLastCheck+step.norm() >= collisionCheckThreshold or optimumReached))
                {
                  hppDout(info, "Collision check...");
                  getFullSplines(linearConstraints.xStar + linearConstraints.PK*(reducedParameters + step),
                      newSplines, hybridSolver);
                  Reports_t reports = this->validatePath(newSplines, true);
                  noCollision = reports.empty();
                  if (noCollision)
                  {
                    hppDout(info, "No collision found");
                    collisionFreeParameters = reducedParameters + step;
                    newParameters = collisionFreeParameters;
                    if (collisionFound) {
                      getFullSplines(linearConstraints.xStar + linearConstraints.PK*collisionFreeParameters,
                          newSplines, hybridSolver);
                      addCollisionConstraint(newSplines, collisionSplines, collisionReports[0]);
                      constraintOutputSize.push_back(1);
                      errorThreshold.push_back(this->collValues[this->collValues.size()-1]*1e-3);
                      hppDout(info, this->collValues[this->collValues.size()-1]*1e-3);

                      hessianStack.push_back(zeroMatrix);
                      jacobian.conservativeResize(jacobian.rows() + 1, Eigen::NoChange);
                      reducedJacobian.conservativeResize(reducedJacobian.rows() + 1, Eigen::NoChange);
                      value.conservativeResize(value.size() + 1, Eigen::NoChange);
                      jacobian.row(jacobian.rows()-1).setZero();
                      reducedJacobian.row(jacobian.rows()-1).setZero();
                      value[value.size()-1] = 0;
                    }
                    lengthSinceLastCheck = 0;
                    break;
                  }
                  else
                  {
                    hppDout(info, "Collision found");
                    collisionFound = true;
                    if (solution.norm() <= stepMin)
                    {
                      getFullSplines(linearConstraints.xStar + linearConstraints.PK*collisionFreeParameters,
                          newSplines, hybridSolver);
                      getFullSplines(linearConstraints.xStar + linearConstraints.PK*(reducedParameters + step),
                          collisionSplines, hybridSolver);
                      addCollisionConstraint(newSplines, collisionSplines, reports[0]);
                      constraintOutputSize.push_back(1);
                      errorThreshold.push_back(this->collValues[this->collValues.size()-1]*1e-3);
                      hppDout(info, this->collValues[this->collValues.size()-1]*1e-3);

                      hessianStack.push_back(zeroMatrix);
                      jacobian.conservativeResize(jacobian.rows() + 1, Eigen::NoChange);
                      reducedJacobian.conservativeResize(reducedJacobian.rows() + 1, Eigen::NoChange);
                      value.conservativeResize(value.size() + 1, Eigen::NoChange);
                      newParameters = collisionFreeParameters;
                      lengthSinceLastCheck = 0;
                      break;
                    }
                    else
                    {
                      getFullSplines(linearConstraints.xStar + linearConstraints.PK*(reducedParameters + step),
                          collisionSplines, hybridSolver);
                      collisionReports = reports;
                      trustRadius = solution.norm()/2;
                      continue;
                    }
                  }
                }
                else
                {
                  lengthSinceLastCheck += step.norm();
                  newParameters = reducedParameters + step;
                  break;
                }
              }

              if (noCollision) {
                value_type oldCost = (.5*reducedParameters.transpose() * (costQuadratic * reducedParameters)
                    + costLinear.transpose() * reducedParameters)(0, 0) + costConstant;
                reducedParameters = newParameters;
                value_type newCost = (.5*reducedParameters.transpose() * (costQuadratic * reducedParameters)
                    + costLinear.transpose() * reducedParameters)(0, 0) + costConstant;
                hppDout(info, "Actual decrease " << newCost - oldCost);
              }

              if (noCollision and optimumReached) {
                Eigen::SelfAdjointEigenSolver<matrix_t> eigenSolver(projectedHessian);
                vector_t d = eigenSolver.eigenvalues();
                hppDout(info, "Condition number at optimum: " << d[d.size()-1]/d[0]);
                break;
              }

              reducedParameters = newParameters;
            }
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
          HPP_DISPLAY_TIMECOUNTER(SGB_jacobian1);
          HPP_DISPLAY_TIMECOUNTER(SGB_jacobian21);
          HPP_DISPLAY_TIMECOUNTER(SGB_jacobian22);
          HPP_DISPLAY_TIMECOUNTER(SGB_jacobian23);
          HPP_DISPLAY_TIMECOUNTER(SGB_jacobian24);
          HPP_DISPLAY_TIMECOUNTER(SGB_jacobian3);
          HPP_DISPLAY_TIMECOUNTER(SGB_jacobian4);
          getFullSplines(linearConstraints.xStar + linearConstraints.PK * reducedParameters, splines, hybridSolver);
          return this->buildPathVector(splines);
        }

      // ----------- Convenience functions ---------------------------------- //

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
