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

#include <hpp/core/path-optimization/spline-gradient-based-abstract.hh>
#include <hpp/core/path-vector.hh>
#include <hpp/core/path/spline.hh>

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
          /// 1* - Transform straight paths into splines
          /// 2* - Add continuity constraints
          /// 3  - Add problem constraints
          /// 4* - Make cost function
          /// 5* - Compute explicit representation of linear constraints.
          /// 6* :
          ///    1 - Compute cost hessian
          ///    2 - Compute optimum
          ///    3 - Check path for collision. If no collision, return solution
          ///    4 - add collision constraint
          ///    5 - re-compute explicit representation of linear constraints.
          /// 7* - Build result path.
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

          virtual void addProblemConstraints (const PathVectorPtr_t& init, const Splines_t& splines, LinearConstraint& lc, SplineOptimizationDatas_t& sods) const;

          void addProblemConstraintOnPath (const PathPtr_t& path, const size_type& idxSpline, const SplinePtr_t& spline, LinearConstraint& lc, SplineOptimizationData& sod) const;

          /// \param guessThr Threshold used to check whether the Jacobian
          ///                 contains rows of zeros, in which case the
          ///                 corresponding DoF is considered passive.
          Eigen::RowBlockIndices computeActiveParameters (const PathPtr_t& path,
              const constraints::HybridSolver& hs,
              const value_type& guessThr = -1,
              const bool& useExplicitInput = false) const;

          bool checkOptimum_;

        private:
          struct QuadraticProblem;
          typedef std::vector <std::pair <CollisionPathValidationReportPtr_t,
                  std::size_t> > Reports_t;
          struct CollisionFunctions;

          void addCollisionConstraint (const std::size_t idxSpline,
              const SplinePtr_t& spline, const SplinePtr_t& nextSpline,
              const SplineOptimizationData& sod,
              const CollisionPathValidationReportPtr_t& report,
              LinearConstraint& collision, CollisionFunctions& functions) const;

          bool findNewConstraint (LinearConstraint& constraint,
              LinearConstraint& collision, LinearConstraint& collisionReduced,
              CollisionFunctions& functions, const std::size_t iF,
              const SplinePtr_t& spline, const SplineOptimizationData& sod) const;

          template <typename Cost_t> bool checkHessian (const Cost_t& cost, const matrix_t& H, const Splines_t& splines) const;

          /// \todo static
          void copy (const Splines_t& in, Splines_t& out) const;

          /// Returns res = (1 - alpha) * a + alpha * b
          void updateSplines (Splines_t& spline, const vector_t& param) const;

          /// Returns res = (1 - alpha) * a + alpha * b
          /// \todo static
          void interpolate (const Splines_t& a, const Splines_t& b,
              const value_type& alpha, Splines_t& res) const;

          void step(const Splines_t& a, vector_t gradient, value_type stepSize,Splines_t& res) const;

          void manifoldStep(const Splines_t& a, const vector_t direction, Splines_t res, vector_t transportedDirection, bool calculateTransportedDirection) const;

          void projectOnConstraints (const Splines_t& a, const vector_t direction, Splines_t res, vector_t projectedDirection, bool calculatedProjectedDirection) const;

          void copyParam (const Splines_t& in, Splines_t& out) const;

          // Continuity constraints
          // matrix_t Jcontinuity_;
          // vector_t rhsContinuity_;
      }; // GradientBasedConstraint
    } // namespace pathOptimization
  }  // namespace core
} // namespace hpp

#endif // HPP_CORE_PATH_OPTIMIZATION_GRADIENT_BASED_HH
