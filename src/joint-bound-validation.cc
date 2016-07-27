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

#include <sstream>
#include <hpp/pinocchio/device.hh>
#include <hpp/pinocchio/joint.hh>
#include <hpp/core/joint-bound-validation.hh>

namespace hpp {
  namespace core {
    typedef pinocchio::JointConfiguration* JointConfigurationPtr_t;
    JointBoundValidationPtr_t JointBoundValidation::create
    (const DevicePtr_t& robot)
    {
      JointBoundValidation* ptr = new JointBoundValidation (robot);
      return JointBoundValidationPtr_t (ptr);
    }

    bool JointBoundValidation::validate
    (const Configuration_t& config, ValidationReportPtr_t& validationReport)
    {
      const JointVector_t jv = robot_->getJointVector ();
      for (JointVector_t::const_iterator itJoint = jv.begin ();
	   itJoint != jv.end (); ++itJoint) {
	size_type index = (*itJoint)->rankInConfiguration ();
	for (size_type i=0; i < (*itJoint)->configSize (); ++i) {
      if ((*itJoint)->isBounded (i)) {
        value_type lower = (*itJoint)->lowerBound (i);
        value_type upper = (*itJoint)->upperBound (i);
	    value_type value = config [index + i];
	    if (value < lower || upper < value) {
          JointBoundValidationReportPtr_t report(new JointBoundValidationReport (*itJoint, i, lower, upper, value));
	      validationReport = report;
	      return false;
	    }
	  }
	}
      }
      const pinocchio::ExtraConfigSpace& ecs = robot_->extraConfigSpace();
      // Check the extra config space
      // FIXME This was introduced at the same time as the integration of Pinocchio
      size_type index = robot_->model().nq;
      for (size_type i=0; i < ecs.dimension(); ++i) {
        value_type lower = ecs.lower (i);
        value_type upper = ecs.upper (i);
        value_type value = config [index + i];
        if (value < lower || upper < value) {
          JointBoundValidationReportPtr_t report(new JointBoundValidationReport (JointPtr_t(), i, lower, upper, value));
          validationReport = report;
          return false;
        }
      }
      return true;
    }

    JointBoundValidation::JointBoundValidation (const DevicePtr_t& robot) :
      robot_ (robot)
    {
    }
  } // namespace core
} // namespace hpp
