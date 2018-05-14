
//
// Copyright (c) 2014 CNRS
// Authors: Florent Lamiraux
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

#include <hpp/core/problem-solver.hh>

#include <boost/bind.hpp>

#include <hpp/fcl/collision_utility.h>

#include <pinocchio/multibody/fcl.hpp>
#include <pinocchio/multibody/geometry.hpp>

#include <hpp/util/debug.hh>
#include <hpp/util/exception-factory.hh>

#include <hpp/pinocchio/collision-object.hh>

#include <hpp/constraints/differentiable-function.hh>

#include <hpp/core/basic-configuration-shooter.hh>
#include <hpp/core/bi-rrt-planner.hh>
#include <hpp/core/config-projector.hh>
#include <hpp/core/constraint-set.hh>
#include <hpp/core/continuous-collision-checking/dichotomy.hh>
#include <hpp/core/continuous-collision-checking/progressive.hh>
#include <hpp/core/diffusing-planner.hh>
#include <hpp/core/distance/reeds-shepp.hh>
#include <hpp/core/distance-between-objects.hh>
#include <hpp/core/discretized-collision-checking.hh>
#include <hpp/core/locked-joint.hh>
#include <hpp/core/numerical-constraint.hh>
#include <hpp/core/path-projector/global.hh>
#include <hpp/core/path-projector/dichotomy.hh>
#include <hpp/core/path-projector/progressive.hh>
#include <hpp/core/path-projector/recursive-hermite.hh>
#include <hpp/core/path-optimization/gradient-based.hh>
#include <hpp/core/path-optimization/spline-gradient-based.hh>
#include <hpp/core/path-optimization/spline-gradient-based-constraint.hh>
#include <hpp/core/path-optimization/partial-shortcut.hh>
#include <hpp/core/path-optimization/config-optimization.hh>
#include <hpp/core/path-optimization/simple-time-parameterization.hh>
#include <hpp/core/path-validation-report.hh>
// #include <hpp/core/problem-target/task-target.hh>
#include <hpp/core/problem-target/goal-configurations.hh>
#include <hpp/core/random-shortcut.hh>
#include <hpp/core/roadmap.hh>
#include <hpp/core/steering-method/dubins.hh>
#include <hpp/core/steering-method/hermite.hh>
#include <hpp/core/steering-method/reeds-shepp.hh>
#include <hpp/core/steering-method/snibud.hh>
#include <hpp/core/steering-method/straight.hh>
#include <hpp/core/visibility-prm-planner.hh>
#include <hpp/core/weighed-distance.hh>
#include <hpp/core/collision-validation.hh>
#include <hpp/core/joint-bound-validation.hh>

namespace hpp {
  namespace core {
    using boost::bind;
    using pinocchio::GeomIndex;

    using pinocchio::Model;
    using pinocchio::GeomModel;
    using pinocchio::GeomModelPtr_t;
    using pinocchio::GeomData;
    using pinocchio::GeomDataPtr_t;
    using pinocchio::CollisionObject;

    namespace {
      Model initObstacleModel () {
        Model model;
        model.addFrame(se3::Frame("obstacle_frame", 0, 0, Transform3f::Identity(), se3::BODY));
        return model;
      }

      template<typename Container> void remove(Container& vector, std::size_t pos)
      {
        assert (pos < vector.size());
        typename Container::iterator it = vector.begin();
        std::advance(it, pos);
        vector.erase(it);
      }

      struct FindCollisionObject {
        FindCollisionObject (const GeomIndex& i) : geomIdx_ (i) {}
        bool operator() (const CollisionObjectPtr_t co) const {
          return co->indexInModel() == geomIdx_;
        }
        GeomIndex geomIdx_;
      };

      void remove(ObjectStdVector_t& vector, const GeomIndex& i)
      {
        ObjectStdVector_t::iterator it =
          std::find_if(vector.begin(), vector.end(), FindCollisionObject(i));
        if (it != vector.end()) vector.erase(it);
      }

      const Model obsModel = initObstacleModel();
    }

    // Struct that constructs an empty shared pointer to PathOptimizer.
    struct NoneOptimizer
    {
      static PathOptimizerPtr_t create (const Problem&)
      {
	return PathOptimizerPtr_t ();
      }
    }; // struct NoneOptimizer

    // Struct that constructs an empty shared pointer to PathProjector.
    struct NonePathProjector
    {
      static PathProjectorPtr_t create (const Problem&,
					const value_type&)
      {
	return PathProjectorPtr_t ();
      }
    }; // struct NonePathProjector

    template <typename Derived> struct Factory {
      static boost::shared_ptr<Derived> create (const Problem& problem) { return Derived::create (problem); }
    };
    template <typename Derived> struct FactoryPP {
      static boost::shared_ptr<Derived> create (const Problem& problem, const value_type& value) { return Derived::create (problem, value); }
    };

    ProblemSolverPtr_t ProblemSolver::latest_ = 0x0;
    ProblemSolverPtr_t ProblemSolver::create ()
    {
      latest_ = new ProblemSolver ();
      return latest_;
    }

    ProblemSolverPtr_t ProblemSolver::latest ()
    {
      return latest_;
    }

    ProblemSolver::ProblemSolver () :
      constraints_ (), robot_ (), problem_ (NULL), pathPlanner_ (),
      roadmap_ (), paths_ (),
      pathProjectorType_ ("None"), pathProjectorTolerance_ (0.2),
      pathPlannerType_ ("DiffusingPlanner"),
      target_ (problemTarget::GoalConfigurations::create(NULL)),
      initConf_ (), goalConfigurations_ (),
      robotType_ ("hpp::pinocchio::Device"),
      configurationShooterType_ ("BasicConfigurationShooter"),
      distanceType_("WeighedDistance"),
      steeringMethodType_ ("Straight"),
      pathOptimizerTypes_ (), pathOptimizers_ (),
      pathValidationType_ ("Discretized"), pathValidationTolerance_ (0.05),
      configValidationTypes_ (),
      collisionObstacles_ (), distanceObstacles_ (),
      obstacleModel_ (new GeomModel()), obstacleData_
      (new GeomData(*obstacleModel_)),
      errorThreshold_ (1e-4), maxIterProjection_ (20),
      maxIterPathPlanning_ (std::numeric_limits
			    <unsigned long int>::max ()),
      passiveDofsMap_ (), comcMap_ (),
      distanceBetweenObjects_ ()
    {
      robots.add (robotType_, Device_t::create);
      pathPlanners.add ("DiffusingPlanner",     DiffusingPlanner::createWithRoadmap);
      pathPlanners.add ("VisibilityPrmPlanner", VisibilityPrmPlanner::createWithRoadmap);
      pathPlanners.add ("BiRRTPlanner", BiRRTPlanner::createWithRoadmap);

      configurationShooters.add ("BasicConfigurationShooter", BasicConfigurationShooter::create);

      // TODO "WeighedDistance" is kept for backward compatibility
      distances.add ("WeighedDistance", WeighedDistance::createFromProblem);
      distances.add ("Weighed",         WeighedDistance::createFromProblem);
      distances.add ("ReedsShepp",      bind (distance::ReedsShepp::create, _1));

      // TODO "SteeringMethodStraight" is kept for backward compatibility
      steeringMethods.add ("SteeringMethodStraight",
          Factory<steeringMethod::Straight>::create);
      steeringMethods.add ("Straight",
          Factory<steeringMethod::Straight>::create);
      steeringMethods.add ("ReedsShepp", steeringMethod::ReedsShepp::createWithGuess);
      steeringMethods.add ("Dubins",     steeringMethod::Dubins::createWithGuess);
      steeringMethods.add ("Snibud",     steeringMethod::Snibud::createWithGuess);
      steeringMethods.add ("Hermite",    steeringMethod::Hermite::create);

      // Store path optimization methods in map.
      pathOptimizers.add ("RandomShortcut",     RandomShortcut::create);
      pathOptimizers.add ("GradientBased",      pathOptimization::GradientBased::create);
      pathOptimizers.add ("PartialShortcut",    pathOptimization::PartialShortcut::create);
      pathOptimizers.add ("ConfigOptimization", pathOptimization::ConfigOptimization::create);
      pathOptimizers.add ("SimpleTimeParameterization", pathOptimization::SimpleTimeParameterization::create);
      pathOptimizers.add ("None",               NoneOptimizer::create); // TODO: Delete me

      // pathOptimizers.add ("SplineGradientBased_cannonical1",pathOptimization::SplineGradientBased<path::CanonicalPolynomeBasis, 1>::create);
      // pathOptimizers.add ("SplineGradientBased_cannonical2",pathOptimization::SplineGradientBased<path::CanonicalPolynomeBasis, 2>::create);
      // pathOptimizers.add ("SplineGradientBased_cannonical3",pathOptimization::SplineGradientBased<path::CanonicalPolynomeBasis, 3>::create);
      pathOptimizers.add ("SplineGradientBased_bezier1",pathOptimization::SplineGradientBased<path::BernsteinBasis, 1>::create);
      // pathOptimizers.add ("SplineGradientBased_bezier2",pathOptimization::SplineGradientBased<path::BernsteinBasis, 2>::create);
      pathOptimizers.add ("SplineGradientBased_bezier3",pathOptimization::SplineGradientBased<path::BernsteinBasis, 3>::create);
      pathOptimizers.add ("SplineGradientBasedConstraint",pathOptimization::SplineGradientBasedConstraint<path::BernsteinBasis, 1>::create);
      pathOptimizers.add ("SplineGradientBasedConstraint3",pathOptimization::SplineGradientBasedConstraint<path::BernsteinBasis, 3>::create);
      // Store path validation methods in map.
      pathValidations.add ("Discretized", DiscretizedCollisionChecking::create);
      pathValidations.add ("Progressive", continuousCollisionChecking::Progressive::create);
      pathValidations.add ("Dichotomy",   continuousCollisionChecking::Dichotomy::create);

      // Store config validation methods in map.
      configValidations.add ("CollisionValidation", CollisionValidation::create);
      configValidations.add ("JointBoundValidation", JointBoundValidation::create);

      // Set default config validation methods.
      configValidationTypes_.push_back("CollisionValidation");
      configValidationTypes_.push_back("JointBoundValidation");

      // Store path projector methods in map.
      pathProjectors.add ("None",             NonePathProjector::create);
      pathProjectors.add ("Progressive",      FactoryPP<pathProjector::Progressive>::create);
      pathProjectors.add ("Dichotomy",        FactoryPP<pathProjector::Dichotomy>::create);
      pathProjectors.add ("Global",           FactoryPP<pathProjector::Global>::create);
      pathProjectors.add ("RecursiveHermite", FactoryPP<pathProjector::RecursiveHermite>::create);
    }

    ProblemSolver::~ProblemSolver ()
    {
      if (problem_) delete problem_;
    }

    void ProblemSolver::distanceType (const std::string& type)
    {
      if (!distances.has (type)) {
    throw std::runtime_error (std::string ("No distance method with name ") +
                  type);
      }
      distanceType_ = type;
    }

    void ProblemSolver::steeringMethodType (const std::string& type)
    {
      if (!steeringMethods.has (type)) {
	throw std::runtime_error (std::string ("No steering method with name ") +
				  type);
      }
      steeringMethodType_ = type;
      if (problem_) initSteeringMethod();
    }

    void ProblemSolver::pathPlannerType (const std::string& type)
    {
      if (!pathPlanners.has (type)) {
	throw std::runtime_error (std::string ("No path planner with name ") +
				  type);
      }
      pathPlannerType_ = type;
    }

    void ProblemSolver::configurationShooterType (const std::string& type)
    {
      if (!configurationShooters.has (type)) {
    throw std::runtime_error (std::string ("No configuration shooter with name ") +
                  type);
      }
      configurationShooterType_ = type;
    }

    void ProblemSolver::addPathOptimizer (const std::string& type)
    {
      if (!pathOptimizers.has (type)) {
	throw std::runtime_error (std::string ("No path optimizer with name ") +
				  type);
      }
      pathOptimizerTypes_.push_back (type);
    }

    void ProblemSolver::clearPathOptimizers ()
    {
      pathOptimizerTypes_.clear ();
      pathOptimizers_.clear ();
    }

    void ProblemSolver::optimizePath (PathVectorPtr_t path)
    {
      createPathOptimizers ();
      for (PathOptimizers_t::const_iterator it = pathOptimizers_.begin ();
	   it != pathOptimizers_.end (); ++it) {
	path = (*it)->optimize (path);
	paths_.push_back (path);
      }
    }

    void ProblemSolver::pathValidationType (const std::string& type,
					    const value_type& tolerance)
    {
      if (!pathValidations.has (type)) {
	throw std::runtime_error (std::string ("No path validation method with "
					       "name ") + type);
      }
      pathValidationType_ = type;
      pathValidationTolerance_ = tolerance;
      // If a robot is present, set path validation method
      if (robot_ && problem_) {
        initPathValidation();
      }
    }

    void ProblemSolver::initPathValidation ()
    {
      if (!problem_) throw std::runtime_error ("The problem is not defined.");
      PathValidationPtr_t pathValidation =
        pathValidations.get (pathValidationType_)
        (robot_, pathValidationTolerance_);
      problem_->pathValidation (pathValidation);
    }

    void ProblemSolver::pathProjectorType (const std::string& type,
					    const value_type& tolerance)
    {
      if (!pathProjectors.has (type)) {
	throw std::runtime_error (std::string ("No path projector method with "
					       "name ") + type);
      }
      pathProjectorType_ = type;
      pathProjectorTolerance_ = tolerance;
      // If a robot is present, set path projector method
      if (robot_ && problem_) {
	initPathProjector ();
      }
    }

    void ProblemSolver::addConfigValidationBuilder
    (const std::string& type, const ConfigValidationBuilder_t& builder)
    {
      configValidations.add (type, builder);
    }

    void ProblemSolver::addConfigValidation (const std::string& type)
    {
      if (!configValidations.has (type)) {
	throw std::runtime_error (std::string ("No config validation method with "
					       "name ") + type);
      }
      configValidationTypes_.push_back (type);
      if (!problem_) throw std::runtime_error ("The problem is not defined.");
      // If a robot is present, set config validation methods
      if (robot_) {
        ConfigValidationPtr_t configValidation =
	      configValidations.get (type) (robot_);
        problem_->addConfigValidation (configValidation);
      }
    }

    void ProblemSolver::clearConfigValidations ()
    {
      configValidationTypes_.clear ();
      problem_->clearConfigValidations ();
    }

    void ProblemSolver::robotType (const std::string& type)
    {
      robotType_ = type;
    }

    const std::string& ProblemSolver::robotType () const
    {
      return robotType_;
    }

    DevicePtr_t ProblemSolver::createRobot (const std::string& name)
    {
      RobotBuilder_t factory (robots.get (robotType_));
      assert (factory);
      return factory (name);
    }

    void ProblemSolver::robot (const DevicePtr_t& robot)
    {
      robot_ = robot;
      constraints_ = ConstraintSet::create (robot_, "Default constraint set");
      // Reset obstacles
      obstacleModel_ = pinocchio::GeomModelPtr_t (new GeomModel());
      obstacleData_ = pinocchio::GeomDataPtr_t (new GeomData(*obstacleModel_));
      resetProblem ();
    }

    const DevicePtr_t& ProblemSolver::robot () const
    {
      return robot_;
    }

    void ProblemSolver::initConfig (const ConfigurationPtr_t& config)
    {
      initConf_ = config;
    }

    const Configurations_t& ProblemSolver::goalConfigs () const
    {
      return goalConfigurations_;
    }

    void ProblemSolver::addGoalConfig (const ConfigurationPtr_t& config)
    {
      target_ = problemTarget::GoalConfigurations::create(NULL);
      goalConfigurations_.push_back (config);
    }

    void ProblemSolver::resetGoalConfigs ()
    {
      goalConfigurations_.clear ();
    }

    /* Setting goal constraint is disabled for now.
     * To re-enable it :
     * - add a function called setGoalConstraints that:
     *   - takes all the NumericalConstraintPtr_t and LockedJointPtr_t
     *   - creates a TaskTarget and fills it.
     * - remove all the addGoalConstraint
    void ProblemSolver::addGoalConstraint (const ConstraintPtr_t& constraint)
    {
      if (!goalConstraints_) {
        if (!robot_) throw std::logic_error ("You must provide a robot first");
        goalConstraints_ = ConstraintSet::create (robot_, "Goal constraint set");
      }
      goalConstraints_->addConstraint (constraint);
    }

    void ProblemSolver::addGoalConstraint (const LockedJointPtr_t& lj)
    {
      if (!goalConstraints_) {
        if (!robot_) throw std::logic_error ("You must provide a robot first");
        goalConstraints_ = ConstraintSet::create (robot_, "Goal constraint set");
      }
      ConfigProjectorPtr_t  configProjector = goalConstraints_->configProjector ();
      if (!configProjector) {
	configProjector = ConfigProjector::create
	  (robot_, "Goal ConfigProjector", errorThreshold_, maxIterProjection_);
	goalConstraints_->addConstraint (configProjector);
      }
      configProjector->add (lj);
    }

    void ProblemSolver::addGoalConstraint (const std::string& constraintName,
        const std::string& functionName, const std::size_t priority)
    {
      if (!goalConstraints_) {
        if (!robot_) throw std::logic_error ("You must provide a robot first");
        goalConstraints_ = ConstraintSet::create (robot_, "Goal constraint set");
      }
      ConfigProjectorPtr_t  configProjector = goalConstraints_->configProjector ();
      if (!configProjector) {
	configProjector = ConfigProjector::create
	  (robot_, constraintName, errorThreshold_, maxIterProjection_);
	goalConstraints_->addConstraint (configProjector);
      }
      configProjector->add (numericalConstraint (functionName),
			    segments_t (0), priority);
    }

    void ProblemSolver::resetGoalConstraint ()
    {
      goalConstraints_.reset ();
    }
    */

    void ProblemSolver::addConstraint (const ConstraintPtr_t& constraint)
    {
      if (robot_)
	constraints_->addConstraint (constraint);
      else
	hppDout (error, "Cannot add constraint while robot is not set");
    }

    void ProblemSolver::addLockedJoint (const LockedJointPtr_t& lj)
    {
      if (!robot_) {
	hppDout (error, "Cannot add constraint while robot is not set");
      }
      ConfigProjectorPtr_t  configProjector = constraints_->configProjector ();
      if (!configProjector) {
	configProjector = ConfigProjector::create
	  (robot_, "ConfigProjector", errorThreshold_, maxIterProjection_);
	constraints_->addConstraint (configProjector);
      }
      configProjector->add (lj);
    }

    void ProblemSolver::resetConstraints ()
    {
      if (robot_) {
	constraints_ = ConstraintSet::create (robot_, "Default constraint set");
        if (problem_) {
          problem_->constraints (constraints_);
        }
      }
    }

    void ProblemSolver::addNumericalConstraintToConfigProjector
    (const std::string& configProjName, const std::string& constraintName,
     const std::size_t priority)
    {
      if (!robot_) {
	hppDout (error, "Cannot add constraint while robot is not set");
      }
      ConfigProjectorPtr_t  configProjector = constraints_->configProjector ();
      if (!configProjector) {
	configProjector = ConfigProjector::create
	  (robot_, configProjName, errorThreshold_, maxIterProjection_);
	constraints_->addConstraint (configProjector);
      }
      if (!numericalConstraints.has (constraintName)) {
        std::stringstream ss; ss << "Function " << constraintName <<
                                " does not exists";
        throw std::invalid_argument (ss.str());
      }
      configProjector->add (numericalConstraints.get(constraintName),
			    segments_t (0), priority);
    }

    void ProblemSolver::addLockedJointToConfigProjector
    (const std::string& configProjName, const std::string& lockedJointName)
    {
      if (!robot_) {
	hppDout (error, "Cannot add constraint while robot is not set");
      }
      ConfigProjectorPtr_t  configProjector = constraints_->configProjector ();
      if (!configProjector) {
	configProjector = ConfigProjector::create
	  (robot_, configProjName, errorThreshold_, maxIterProjection_);
	constraints_->addConstraint (configProjector);
      }
      if (!lockedJoints.has (lockedJointName)) {
        std::stringstream ss; ss << "Function " << lockedJointName <<
                                " does not exists";
        throw std::invalid_argument (ss.str());
      }
      configProjector->add (lockedJoints.get(lockedJointName));
    }

    void ProblemSolver::comparisonType (const std::string& name,
        const ComparisonTypes_t types)
    {
      NumericalConstraintPtr_t nc;
      if (numericalConstraints.has (name))
        nc = numericalConstraints.get(name);
      else if (lockedJoints.has (name))
        nc = lockedJoints.get(name);
      else
        throw std::runtime_error (name + std::string (" is neither a numerical "
              "constraint nor a locked joint"));
      nc->comparisonType (types);
    }

    void ProblemSolver::comparisonType (const std::string& name,
        const ComparisonType &type)
    {
      NumericalConstraintPtr_t nc;
      if (numericalConstraints.has (name))
        nc = numericalConstraints.get(name);
      else if (lockedJoints.has (name))
        nc = lockedJoints.get(name);
      else
        throw std::runtime_error (name + std::string (" is neither a numerical "
              "constraint nor a locked joint"));
      ComparisonTypes_t eqtypes (nc->function().outputDerivativeSize(), type);
      nc->comparisonType (eqtypes);
    }

    ComparisonTypes_t ProblemSolver::comparisonType (const std::string& name) const
    {
      NumericalConstraintPtr_t nc;
      if (numericalConstraints.has (name))
        nc = numericalConstraints.get(name);
      else if (lockedJoints.has (name))
        nc = lockedJoints.get(name);
      else
        throw std::runtime_error (name + std::string (" is neither a numerical "
              "constraint nor a locked joint"));
      return nc->comparisonType ();
    }

    void ProblemSolver::computeValueAndJacobian
    (const Configuration_t& configuration, vector_t& value, matrix_t& jacobian)
      const
    {
      if (!robot ()) throw std::runtime_error ("No robot loaded");
      ConfigProjectorPtr_t configProjector
	(constraints ()->configProjector ());
      if (!configProjector) {
	throw std::runtime_error ("No constraints have assigned.");
      }
      // resize value and Jacobian
      value.resize (configProjector->solver().dimension());
      size_type rows = configProjector->solver().reducedDimension();
      jacobian.resize (rows, configProjector->numberNonLockedDof ());
      configProjector->computeValueAndJacobian (configuration, value, jacobian);
    }

    void ProblemSolver::maxIterProjection (size_type iterations)
    {
      maxIterProjection_ = iterations;
      if (constraints_ && constraints_->configProjector ()) {
        constraints_->configProjector ()->maxIterations (iterations);
      }
    }

    void ProblemSolver::maxIterPathPlanning (size_type iterations)
    {
      maxIterPathPlanning_ = iterations;
      if (constraints_ && constraints_->configProjector ()) {
        constraints_->configProjector ()->maxIterations (iterations);
      }
    }

    void ProblemSolver::errorThreshold (const value_type& threshold)
    {
      errorThreshold_ = threshold;
      if (constraints_ && constraints_->configProjector ()) {
        constraints_->configProjector ()->errorThreshold (threshold);
      }
    }

    void ProblemSolver::resetProblem ()
    {
      if (problem_)
	delete problem_;
      initializeProblem (new Problem (robot_));
    }

    void ProblemSolver::initializeProblem (ProblemPtr_t problem)
    {
      problem_ = problem;
      resetRoadmap ();
      // Set constraints
      problem_->constraints (constraints_);
      // Set path validation method
      PathValidationPtr_t pathValidation =
	pathValidations.get (pathValidationType_)
        (robot_, pathValidationTolerance_);
      problem_->pathValidation (pathValidation);
      // Set config validation methods
      for (ConfigValidationTypes_t::const_iterator it =
          configValidationTypes_.begin (); it != configValidationTypes_.end ();
          ++it)
      {
        ConfigValidationPtr_t configValidation =
	      configValidations.get (*it) (robot_);
        problem_->addConfigValidation (configValidation);
      }
      // Set obstacles
      problem_->collisionObstacles(collisionObstacles_);
      // Distance to obstacles
      distanceBetweenObjects_ = DistanceBetweenObjectsPtr_t
	(new DistanceBetweenObjects (robot_));
      distanceBetweenObjects_->obstacles(distanceObstacles_);
    }

    void ProblemSolver::problem (ProblemPtr_t problem)
    {
      if (problem_)
        delete problem_;
      problem_ = problem;
    }

    void ProblemSolver::resetRoadmap ()
    {
      if (!problem_)
        throw std::runtime_error ("The problem is not defined.");
      roadmap_ = Roadmap::create (problem_->distance (), problem_->robot());
    }

    void ProblemSolver::createPathOptimizers ()
    {
      if (!problem_) throw std::runtime_error ("The problem is not defined.");
      pathOptimizers_.clear();
      for (PathOptimizerTypes_t::const_iterator it =
          pathOptimizerTypes_.begin (); it != pathOptimizerTypes_.end ();
          ++it) {
        PathOptimizerBuilder_t createOptimizer = pathOptimizers.get (*it);
        pathOptimizers_.push_back (createOptimizer (*problem_));
      }
    }

    void ProblemSolver::initDistance ()
    {
      if (!problem_) throw std::runtime_error ("The problem is not defined.");
      DistancePtr_t dist (distances.get (distanceType_) (*problem_));
      problem_->distance (dist);
    }

    void ProblemSolver::initSteeringMethod ()
    {
      if (!problem_) throw std::runtime_error ("The problem is not defined.");
      SteeringMethodPtr_t sm (
          steeringMethods.get (steeringMethodType_) (*problem_)
          );
      problem_->steeringMethod (sm);
    }

    void ProblemSolver::initPathProjector ()
    {
      if (!problem_) throw std::runtime_error ("The problem is not defined.");
      PathProjectorBuilder_t createProjector =
        pathProjectors.get (pathProjectorType_);
      // The PathProjector will store a copy of the current steering method.
      // This means:
      // - when constraints are relevant, they should have been added before.
      //   TODO The path projector should update the constraint according to the path they project.
      // - the steering method type must match the path projector type.
      PathProjectorPtr_t pathProjector_ =
        createProjector (*problem_, pathProjectorTolerance_);
      problem_->pathProjector (pathProjector_);
    }

    void ProblemSolver::initProblemTarget ()
    {
      if (!problem_) throw std::runtime_error ("The problem is not defined.");
      target_->problem(problem_);
      problem_->target (target_);
    }

    void ProblemSolver::initProblem ()
    {
      if (!problem_) throw std::runtime_error ("The problem is not defined.");

      // Set shooter
      problem_->configurationShooter
        (configurationShooters.get (configurationShooterType_) (robot_));
      // Set steeringMethod
      initSteeringMethod ();
      PathPlannerBuilder_t createPlanner = pathPlanners.get (pathPlannerType_);
      pathPlanner_ = createPlanner (*problem_, roadmap_);
      pathPlanner_->maxIterations (maxIterPathPlanning_);
      roadmap_ = pathPlanner_->roadmap();
      /// create Path projector
      initPathProjector ();
      /// create Path optimizer
      // Reset init and goal configurations
      problem_->initConfig (initConf_);
      problem_->resetGoalConfigs ();
      for (Configurations_t::const_iterator itConfig =
	     goalConfigurations_.begin ();
	   itConfig != goalConfigurations_.end (); ++itConfig) {
	problem_->addGoalConfig (*itConfig);
      }
      initProblemTarget();
    }

    bool ProblemSolver::prepareSolveStepByStep ()
    {
      initProblem ();

      pathPlanner_->startSolve ();
      pathPlanner_->tryDirectPath ();
      return roadmap_->pathExists ();
    }

    bool ProblemSolver::executeOneStep ()
    {
      pathPlanner_->oneStep ();
      return roadmap_->pathExists ();
    }

    void ProblemSolver::finishSolveStepByStep ()
    {
      if (!roadmap_->pathExists ())
        throw std::logic_error ("No path exists.");
      PathVectorPtr_t planned =  pathPlanner_->computePath ();
      paths_.push_back (pathPlanner_->finishSolve (planned));
    }

    void ProblemSolver::solve ()
    {
      initProblem ();

      PathVectorPtr_t path = pathPlanner_->solve ();
      paths_.push_back (path);
      optimizePath (path);
    }

    bool ProblemSolver::directPath
    (ConfigurationIn_t start, ConfigurationIn_t end, bool validate,
     std::size_t& pathId, std::string& report)
    {
      report = "";
      if (!problem_) throw std::runtime_error ("The problem is not defined.");

      // Create steering method using factory
      SteeringMethodPtr_t sm (steeringMethods.get (steeringMethodType_)
          (*problem_));
      problem_->steeringMethod (sm);
      PathPtr_t dp = (*sm) (start, end);
      if (!dp) {
	report = "Steering method failed to build a path.";
	pathId = -1;
	return false;
      }
      PathPtr_t dp1, dp2;
      PathValidationReportPtr_t r;
      bool projValid = true, projected = true, pathValid = true;
      if (validate) {
        PathProjectorPtr_t proj (problem ()->pathProjector ());
        if (proj) {
          projected = proj->apply (dp, dp1);
        } else {
          dp1 = dp;
        }
	projValid = problem()->pathValidation ()->validate (dp1, false, dp2, r);
        pathValid = projValid && projected;
        if (!projValid) {
          hppDout (info, *r);
          std::ostringstream oss;
          oss << *r; report = oss.str ();
        }
      } else {
        dp2 = dp;
      }
      // Add Path in problem
      PathVectorPtr_t path (core::PathVector::create (dp2->outputSize (),
                            dp2->outputDerivativeSize ()));
      path->appendPath (dp2);
      pathId = addPath (path);
      return pathValid;
    }

    void ProblemSolver::addConfigToRoadmap (const ConfigurationPtr_t& config)
    {
      roadmap_->addNode(config);
    }

    void ProblemSolver::addEdgeToRoadmap (const ConfigurationPtr_t& config1,
					  const ConfigurationPtr_t& config2,
					  const PathPtr_t& path)
    {
      NodePtr_t node1, node2;
      value_type accuracy = 10e-6;
      value_type distance1, distance2;
      node1 = roadmap_->nearestNode(config1, distance1);
      node2 = roadmap_->nearestNode(config2, distance2);
      if (distance1 >= accuracy) {
	throw std::runtime_error ("No node of the roadmap contains config1");
      }
      if (distance2 >= accuracy) {
	throw std::runtime_error ("No node of the roadmap contains config2");
      }
      roadmap_->addEdge(node1, node2, path);
    }

    void ProblemSolver::interrupt ()
    {
      if (pathPlanner ()) pathPlanner ()->interrupt ();
      for (PathOptimizers_t::iterator it = pathOptimizers_.begin ();
	   it != pathOptimizers_.end (); ++it) {
	(*it)->interrupt ();
      }
    }

    void ProblemSolver::addObstacle (const CollisionObjectPtr_t& object,
				     bool collision, bool distance)
    {
      // FIXME propagate object->mesh_path ?
      addObstacle(object->name(), *object->fcl(), collision, distance);
    }

    void ProblemSolver::addObstacle (const std::string& name,
                                     /*const*/ FclCollisionObject &inObject,
				     bool collision, bool distance)
    {
      if (obstacleModel_->existGeometryName(name)) {
        HPP_THROW(std::runtime_error, "object with name " << name
            << " already added! Choose another name (prefix).");
      }

      se3::GeomIndex id = obstacleModel_->addGeometryObject(se3::GeometryObject(
            name, obsModel.getFrameId("obstacle_frame", se3::BODY), 0,
            inObject.collisionGeometry(),
            se3::toPinocchioSE3(inObject.getTransform()),
            "",
            vector3_t::Ones()),
          obsModel);
      // Update obstacleData_
      // FIXME This should be done in Pinocchio
      {
        se3::GeometryModel& model = *obstacleModel_;
        se3::GeometryData& data = *obstacleData_;
        data.oMg.resize(model.ngeoms);
        //data.activeCollisionPairs.resize(model.collisionPairs.size(), true)
        //data.distance_results(model.collisionPairs.size())
        //data.collision_results(model.collisionPairs.size())
        //data.radius()
        data.collisionObjects.push_back (fcl::CollisionObject(
              model.geometryObjects[id].fcl));
        data.oMg[id] =  model.geometryObjects[id].placement;
        data.collisionObjects[id].setTransform( se3::toFclTransform3f(data.oMg[id]) );
      }
      CollisionObjectPtr_t object (
          new CollisionObject(obstacleModel_,obstacleData_,id));

      if (collision){
        collisionObstacles_.push_back (object);
        resetRoadmap ();
      }
      if (distance)
        distanceObstacles_.push_back (object);
      if (problem ())
        problem ()->addObstacle (object);
      if (distanceBetweenObjects_) {
	distanceBetweenObjects_->addObstacle (object);
      }
    }

    void ProblemSolver::removeObstacle (const std::string& name)
    {
      if (!obstacleModel_->existGeometryName(name)) {
        HPP_THROW(std::invalid_argument, "No obstacle with name " << name);
      }
      se3::GeomIndex id = obstacleModel_->getGeometryId(name);

      // Update obstacle model
      remove(obstacleModel_->geometryObjects, id);
      obstacleModel_->ngeoms--;
      remove(obstacleData_->oMg, id);
      remove(obstacleData_->collisionObjects, id);

      remove(collisionObstacles_, id);
      remove(distanceObstacles_, id);
      resetProblem(); // resets problem_ and distanceBetweenObjects_
      resetRoadmap();
    }

    void ProblemSolver::cutObstacle (const std::string& name,
                                     const fcl::AABB& aabb)
    {
      if (!obstacleModel_->existGeometryName(name)) {
        HPP_THROW(std::invalid_argument, "No obstacle with name " << name);
      }
      se3::GeomIndex id = obstacleModel_->getGeometryId(name);

      fcl::CollisionObject& fclobj = obstacleData_->collisionObjects[id];
      fclobj.computeAABB();
      if (!fclobj.getAABB().overlap(aabb)) {
        // No intersection. Geom should be removed.
        removeObstacle(name);
        return;
      }
      fcl::CollisionGeometryPtr_t fclgeom = obstacleModel_->geometryObjects[id].fcl;
      fcl::CollisionGeometryPtr_t newgeom (extract(fclgeom.get(), fclobj.getTransform(), aabb));
      if (!newgeom) {
        // No intersection. Geom should be removed.
        removeObstacle(name);
      } else {
        obstacleModel_->geometryObjects[id].fcl = newgeom;
        obstacleData_->collisionObjects[id] =
          fcl::CollisionObject(newgeom, se3::toFclTransform3f(obstacleData_->oMg[id]));
      }
    }

    void ProblemSolver::removeObstacleFromJoint
    (const std::string& obstacleName, const std::string& jointName)
    {
      if (!robot_) {
	throw std::runtime_error ("No robot defined.");
      }
      JointPtr_t joint = robot_->getJointByName (jointName);
      if (!joint) {
	std::ostringstream oss;
	oss << "Robot has no joint with name " << jointName << ".";
	throw std::runtime_error (oss.str ().c_str ());
      }
      const CollisionObjectPtr_t& object = obstacle (obstacleName);
      if (!object)  {
	std::ostringstream oss;
	oss << "No obstacle with with name " << obstacleName << ".";
	throw std::runtime_error (oss.str ().c_str ());
      }
      problem ()->removeObstacleFromJoint (joint, object);
    }

    void ProblemSolver::filterCollisionPairs ()
    {
      problem()->filterCollisionPairs ();
    }

    CollisionObjectPtr_t ProblemSolver::obstacle (const std::string& name) const
    {
      if (obstacleModel_->existGeometryName(name)) {
        se3::GeomIndex id = obstacleModel_->getGeometryId(name);
        return CollisionObjectPtr_t (
            new CollisionObject(obstacleModel_,obstacleData_,id));
      }
      HPP_THROW(std::invalid_argument, "No obstacle with name " << name);
    }

    std::list <std::string> ProblemSolver::obstacleNames
    (bool collision, bool distance) const
    {
      std::list <std::string> res;
      if (collision) {
    for (ObjectStdVector_t::const_iterator it = collisionObstacles_.begin ();
	     it != collisionObstacles_.end (); ++it) {
	  res.push_back ((*it)->name ());
	}
      }
      if (distance) {
    for (ObjectStdVector_t::const_iterator it = distanceObstacles_.begin ();
	     it != distanceObstacles_.end (); ++it) {
	  res.push_back ((*it)->name ());
	}
      }
      return res;
    }

    const ObjectStdVector_t& ProblemSolver::collisionObstacles () const
    {
      return collisionObstacles_;
    }

    const ObjectStdVector_t& ProblemSolver::distanceObstacles () const
    {
      return distanceObstacles_;
    }

  } //   namespace core
} // namespace hpp
