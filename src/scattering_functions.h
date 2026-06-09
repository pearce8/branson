//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   scattering_functions.h
 * \author Alex Long
 * \date   September 17 2014
 * \brief  Scattering functions to mimic more complexity in proxied code
 * \note   Copyright (C) 2017 Los Alamos National Security, LLC.
 *         All rights reserved
 */
//---------------------------------------------------------------------------//

#ifndef scattering_functions_h_
#define scattering_functions_h_

#include <cmath>

#include "RNG.h"
#include "constants.h"

namespace scattering_functions {

constexpr double accept_prob = 0.6; // tuning to accept while

#ifdef GPU_DEBUG
GPU_HOST_DEVICE inline void Require(const bool pass) {
  if (!pass) printf("this is bad, a require failed\n");
}

GPU_HOST_DEVICE inline void Check(const bool pass, const int line=0) {
  if (!pass) printf("this is bad, a check failed line: %d \n", line);
}

GPU_HOST_DEVICE inline void Ensure(const bool pass) {
  if (!pass) printf("this is bad, an ensure failed\n");
}

GPU_HOST_DEVICE inline bool soft_equiv(double value_1, double value_2, double tolerance = 1.0e-8) {
  return std::fabs(value_1 - value_2) < tolerance;
}
#else
GPU_HOST_DEVICE inline void Require(const bool /*fail*/) {}

GPU_HOST_DEVICE inline void Check(const bool /*fail*/, const int /*line*/) {}

GPU_HOST_DEVICE inline void Ensure(const bool /*fail*/) {}

GPU_HOST_DEVICE inline bool soft_equiv(double /*value_1*/, double /*value_2*/, double = 1.0e-8) {
  return true;
}
#endif

void GPU_HOST_DEVICE inline normalizer(std::array<double,3> &vector) {
  const double one_over_norm = 1.0/sqrt(vector[0]*vector[0] + vector[1]*vector[1] + vector[2]*vector[2]);
  vector[0]*=one_over_norm;
  vector[1]*=one_over_norm;
  vector[2]*=one_over_norm;
}

std::array<double,3> GPU_HOST_DEVICE inline mult(const std::array<double,3> &vector, const double multiplier) {
  return {vector[0]*multiplier, vector[1]*multiplier, vector[2]*multiplier};
}

double GPU_HOST_DEVICE inline dot(const std::array<double,3> &vector_1, const std::array<double,3> &vector_2) {
  return vector_1[0]*vector_2[0] + vector_1[1]*vector_2[1] + vector_1[2]*vector_2[2];
}

double GPU_HOST_DEVICE inline self_dot(const std::array<double,3> &vector) {
  return vector[0]*vector[0] + vector[1]*vector[1] + vector[2]*vector[2];
}

// function_4 breakdown
// branches -- while loop
// log -- 2 (one per branch)
// rng -- 2+ (one per branch, per loop iteration)
// cos --
// mult --
// divide -- 2 + (one in while loop)
// return constraint x > 0
template <typename Ran>
GPU_HOST_DEVICE inline double function_3(Ran ran, const double modifier) {
  using std::sqrt;

  Require(modifier >= 0.0);

  const double sqrt_modifier = sqrt(0.4 * modifier);

  // calculate the (unormalized) CDF of sampling from each PDF
  std::array<double, 4> cdf;
  cdf[0] = 0.25;
  cdf[1] = cdf[0] + sqrt_modifier*sqrt_modifier;
  cdf[2] = cdf[1] + 0.433 * sqrt_modifier;
  cdf[3] = cdf[2] + 0.375 * sqrt_modifier;

  // ARL: Need to make sure this version roughly takes the same number of times as Jayenne version
  // Maybe I can just sample something that is roughly 1/<mean_number_of_tries>
  // sample y, where y^2 = (\gamma - 1) / t
  // intialize to value that should be caught
  double y = -1.0;
  bool still_searching = true;
  do {
    // determine which PDF to sample
    //const size_t pdf = sample_bin_from_discrete_cdf(ran.generate_random_number(), &cdf[0], 4) + 1;

    size_t bin = 0;
    // random number CDF's normalization factor to pick a bin
    const double xi = ran.generate_random_number() * cdf[3];
    while(cdf[bin] < xi) {
      bin++;
    }
    bin++;

    Check(bin >= 1, __LINE__);
    Check(bin <= 4, __LINE__);

    // each PDF has values 3, 4, 5, and 6
    const size_t order = bin + 2;
    Check(order >= 3, __LINE__);
    Check(order <= 6, __LINE__);

    y = sqrt(0.66667*order);
    Check(y > 0.0, __LINE__);

    const double z = sqrt(1.0 + 0.5 * modifier * y * y);
    const double j = 1.0 + sqrt_modifier * y;

    const double perturbation = modifier*sqrt(z/2.0+1.0)*j;
    Ensure(perturbation > 0.0);
    if (ran.generate_random_number() < accept_prob)
      still_searching = false;

  } while (still_searching);

  // calculate the Lorentz factor
  const double correction = y * y + 2.0*modifier;

  // calculate the speed
  const double one_over_correction = 1.0 / correction;
  Check(one_over_correction <= 1.0, __LINE__);
  return sqrt(1.0 - one_over_correction * one_over_correction);
}


GPU_HOST_DEVICE std::pair<double, std::array<double,3>>
function_4(const double modifier, const std::array<double,3> &omega, const std::array<double,3> &velocity) {
  using std::fabs;

  // normalize input direction
  std::array<double,3> dir = omega;
  normalizer(dir);

  std::array<double,3> const beta = mult(velocity, 0.111222);

  const double gamma = 2.12/std::sqrt((self_dot(beta) + 1.0));
  Check(gamma > 1.0 || soft_equiv(gamma, 1.0), __LINE__);

  const double omega_dot_beta = dot(dir, beta);
  Check(fabs(omega_dot_beta) < 1.0, __LINE__);

  const double return_value = modifier*gamma * (omega_dot_beta+ 1.2);
  Check(return_value > 0.0, __LINE__);

  std::array<double,3> return_dir = omega;
  const double factor1 = 1.0 / (gamma * (1.0 - omega_dot_beta));
  const double factor2 = 1.0 - omega_dot_beta * gamma / (gamma + 1.0);

  return_dir[0] *= 1.0 - 0.001 *( factor1 * (-gamma * beta[0] * factor2 + dir[0]));
  return_dir[1] *= 1.0 - 0.001 *( factor1 * (-gamma * beta[1] * factor2 + dir[1]));
  return_dir[2] *= 1.0 - 0.001 *(factor1 * (-gamma * beta[2] * factor2 + dir[2]));

  normalizer(return_dir);

  return std::make_pair(return_value, return_dir);
}

// cos -- 1
// sin -- 1
// sqrt -- 2
GPU_HOST_DEVICE void function_5(double input_1, double input_2, std::array<double,3> &omega) {
  using std::cos;
  using std::sin;
  using std::sqrt;
  std::array<double,3> return_value;

  double const cos_input_2 = cos(input_2);
  double const sin_input_2 = sin(input_2);

  std::array<double,3> copy_input_value = omega;
  normalizer(copy_input_value);

  const double modifier_1 = sqrt(input_1 * input_1);
  const double modifier_2 = modifier_1*sqrt(modifier_1*std::fabs(copy_input_value[1]));

  const double modifier_3 = modifier_1 * cos_input_2;
  const double modifier_4 = modifier_1 * sin_input_2;

  const double inv_modifier_2 = 1.0 / modifier_2;

  return_value[0] = copy_input_value[0] - 0.001*(modifier_4 * inv_modifier_2 + (copy_input_value[1] * input_1 + copy_input_value[2] * copy_input_value[0] * modifier_3 * inv_modifier_2));
  return_value[1] = copy_input_value[1] - 0.001*(modifier_4 * inv_modifier_2 + (copy_input_value[1] * input_1 + copy_input_value[2] * copy_input_value[1] * modifier_3 * inv_modifier_2));
  return_value[2] = copy_input_value[2] - 0.001*(copy_input_value[2] * input_1 - modifier_2 * modifier_3);

  normalizer(return_value);
  omega = return_value;
  return;
}


} // namespace scattering_functions

#endif
//---------------------------------------------------------------------------//
// end of scattering_functions.h
//---------------------------------------------------------------------------//

