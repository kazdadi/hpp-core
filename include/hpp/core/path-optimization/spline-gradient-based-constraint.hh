// Copyright (c) 2017 CNRS
// Authors: Joseph Mirabel
//
// This file is part of hpp-core
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
// hpp-core  If not, see
// <http://www.gnu.org/licenses/>.

#ifndef HPP_CORE_PATH_OPTIMIZATION_SPLINE_GRADIENT_BASED_CONSTRAINT_HH
# define HPP_CORE_PATH_OPTIMIZATION_SPLINE_GRADIENT_BASED_CONSTRAINT_HH

#include <hpp/constraints/explicit-solver.hh>

#include <hpp/constraints/hybrid-solver.hh>
#include <hpp/core/config-projector.hh>

#include <hpp/core/path-optimization/spline-gradient-based-abstract.hh>
#include <hpp/core/path-vector.hh>
#include <hpp/core/path/spline.hh>
#include <path-optimization/spline-gradient-based/cost.hh>

#include <hpp/core/steering-method/spline.hh>

namespace hpp {
  namespace core {
    /// \addtogroup path_optimization
    /// \{
    namespace pathOptimization {
      template <int _PolynomeBasis, int _SplineOrder>
        class HPP_CORE_DLLAPI SplineGradientBasedConstraint : public SplineGradientBasedAbstract<_PolynomeBasis, _SplineOrder>
      {
        public:
          typedef SplineGradientBasedAbstract<_PolynomeBasis, _SplineOrder> Base;
          enum {
            PolynomeBasis = _PolynomeBasis,
            SplineOrder = _SplineOrder
          };
          typedef boost::shared_ptr<SplineGradientBasedConstraint> Ptr_t;

          using typename Base::Spline;
          using typename Base::SplinePtr_t;
          using typename Base::Splines_t;

          /// Return shared pointer to new object.
          /// Default cost is path length.
          static Ptr_t create (const Problem& problem);

          /// Optimize path
          /// 1 - Transform straight paths into splines
          /// 2 - Add continuity constraints
          /// 3 - Make cost function
          /// 4 - Compute explicit representation of linear constraints.
          /// 5 :
          ///    1 - Solve first order approximation of equality and active bound constraints
          ///    2 - Project gradient on problem constraints
          ///    3 - Check if optimum is reached
          ///    4 - Check for collisions and add constraints if any are detected
          ///    5 - Check inequality constraints and add them to the active sets if they are violated
          /// 6 - Build result path.
          virtual PathVectorPtr_t optimize (const PathVectorPtr_t& path);

        protected:
          using typename Base::ExplicitSolver;
          using typename Base::RowBlockIndices;
          using typename Base::SplineOptimizationData;
          using typename Base::SplineOptimizationDatas_t;
          using Base::robot_;
          using Base::problem;

          SplineGradientBasedConstraint (const Problem& problem);

          // Constraint creation
          virtual void addProblemConstraints (const Splines_t splines,
              HybridSolver& hybridSolver, LinearConstraint& constraint,
              std::vector<size_type>& dofPerSpline, std::vector<size_type>& argPerSpline,
              size_type& nbConstraints, std::vector<size_type>& constraintSplineIndex) const;

          void processContinuityConstraints (const Splines_t splines, HybridSolver hybridSolver,
              LinearConstraint& continuityConstraints, LinearConstraint& linearConstraints) const;

          LiegroupSpacePtr_t createProblemLieGroup(std::vector<LiegroupSpacePtr_t>& splineSpaces,
              const Splines_t splines, const HybridSolver hybridSolver) const;

          void costFunction (const Splines_t splines, const LinearConstraint& linearConstraints,
              std::vector<size_type> dofPerSpline, matrix_t& hessian, vector_t& gradientAtZero, value_type& valueAtZero) const;

        private:
          typedef std::vector <std::pair <CollisionPathValidationReportPtr_t,
                  std::size_t> > Reports_t;
          struct CollisionFunctions;

          template <typename Cost_t> bool checkHessian (const Cost_t& cost, const matrix_t& H, const Splines_t& splines) const;

          size_type getNbConstraints(const Splines_t splines) const;

          /// Solve subproblem for constrained QP
          value_type solveSubSubProblem(vector_t c, vector_t k, value_type r, value_type error = std::pow(10, -8)) const;

          /// Solve min 1/2 xT A x - bT x s.t ||x|| < r
          vector_t solveQP(matrix_t A, vector_t b, value_type r) const;

          void getHessianFiniteDiff (const vector_t x, Splines_t& splines,
              std::vector<matrix_t>& hessianStack, const value_type stepSize, HybridSolver& hybridSolver,
              const std::vector<size_type>& dofPerSpline, const size_type nbConstraints,
              const LiegroupSpacePtr_t stateSpace) const;

          void getFullSplines (const vector_t reducedParams, Splines_t& fullSplines,
              HybridSolver hybridSolver) const;

          void addCollisionConstraint(Splines_t collisionFreeSplines, Splines_t collisionSplines,
              std::pair<CollisionPathValidationReportPtr_t, std::size_t> collisionReport,
              std::vector<DifferentiableFunctionPtr_t> collFunctions, std::vector<value_type> collValues,
              std::vector<size_type> collIndices, std::vector<value_type> collTimes) const;

          void getConstraintsValue(const vector_t x, Splines_t& splines, vectorOut_t value,
              HybridSolver& hybridSolver) const;

          void getCollisionConstraintsValue (
              const vector_t x, Splines_t& splines, vectorOut_t value, HybridSolver& hybridSolver,
              const std::vector<DifferentiableFunctionPtr_t> collFunctions,
              const std::vector<value_type> collValues, const std::vector<size_type> collIndices,
              const std::vector<value_type> collTimes) const;

          void getConstraintsValueJacobian(const vector_t x, Splines_t& splines, vectorOut_t value,
              matrixOut_t jacobian, const std::vector<size_type>& dofPerSpline,
              HybridSolver& hybridSolver, const LiegroupSpacePtr_t stateSpace) const;

          void getCollisionConstraintsValueJacobian (const vector_t x, Splines_t& splines, vectorOut_t value,
              matrixOut_t jacobian, HybridSolver& hybridSolver, const std::vector<size_type>& dofPerSpline,
              const std::vector<LiegroupSpacePtr_t>& splineSpaces,
              const std::vector<DifferentiableFunctionPtr_t>& collFunctions,
              const std::vector<value_type>& collValues, const std::vector<size_type>& collIndices,
              const std::vector<value_type>& collTimes) const;

          void getConstraintsHessian (const vector_t x, Splines_t& splines, std::vector<matrix_t>& hessianStack,
              const std::vector<size_type>& dofPerSpline, const value_type stepSize,
              HybridSolver& hybridSolver, const LiegroupSpacePtr_t stateSpace, size_type nbConstraints,
              const std::vector<size_type>& constraintSplineIndex) const;

          void getJacobianFiniteDiff(const vector_t x, Splines_t& splines, matrixOut_t jacobian,
              const value_type stepSize, HybridSolver& hybridSolver) const;

          void getHessianDoubleFiniteDiff
            (const vector_t x, Splines_t& splines, std::vector<matrix_t>& hessianStack, const value_type stepSize,
             HybridSolver hybridSolver) const;

          vector_t getSecondOrderCorrection (const vector_t step, const matrix_t& jacobian,
              const std::vector<matrix_t>& hessianStack, const matrix_t& inverseGram, const matrix_t& PK) const;

          /// \todo static
          void copy (const Splines_t& in, Splines_t& out) const;

          void updateSplines (Splines_t& spline, const vector_t& param) const;

          void getParameters (const Splines_t& spline, vector_t& param) const;

          // Continuity constraints
          // matrix_t Jcontinuity_;
          // vector_t rhsContinuity_;
      }; // GradientBasedConstraint
    } // namespace pathOptimization
  }  // namespace core
} // namespace hpp

#endif // HPP_CORE_PATH_OPTIMIZATION_GRADIENT_BASED_HH
