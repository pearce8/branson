//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   sampling_functions.h
 * \author Alex Long
 * \date   September 17 2014
 * \brief  Angle sampling for isotropic and surface sources
 * \note   Copyright (C) 2017 Los Alamos National Security, LLC.
 *         All rights reserved
 */
//---------------------------------------------------------------------------//

#ifndef sampling_functions_h_
#define sampling_functions_h_

#include <cmath>

#include "RNG.h"
#include "constants.h"
#include "cell.h"
#include "config.h"
#include "scattering_functions.h"

//! Set an input array to a random position within a cell
GPU_HOST_DEVICE inline std::array<double, 3> get_uniform_position_in_cell(const Cell &cell, RNG &rng)  {
  auto nodes = cell.get_node_array();
  std::array<double, 3> pos{0.0, 0.0, 0.0};
  pos[0] = nodes[0] + rng.generate_random_number() * (nodes[1] - nodes[0]);
  pos[1] = nodes[2] + rng.generate_random_number() * (nodes[3] - nodes[2]);
  pos[2] = nodes[4] + rng.generate_random_number() * (nodes[5] - nodes[4]);
  return pos;
}

//! Set an input array to a random position within a cell
GPU_HOST_DEVICE inline std::array<double, 3>  get_uniform_position_on_face(const Cell &cell, RNG &rng, int face) {
  auto nodes = cell.get_node_array();
  std::array<double, 3> face_pos{0.0, 0.0, 0.0};
  if (face ==0 || face ==1) {
    face_pos[0] = (face == 0) ? nodes[0] : nodes[1];
    face_pos[1] = nodes[2] + rng.generate_random_number() * (nodes[3] - nodes[2]);
    face_pos[2] = nodes[4] + rng.generate_random_number() * (nodes[5] - nodes[4]);
  }
  else if (face ==2 || face ==3) {
    face_pos[0] = nodes[0] + rng.generate_random_number() * (nodes[1] - nodes[0]);
    face_pos[1] = (face == 2) ? nodes[2] : nodes[3];
    face_pos[2] = nodes[4] + rng.generate_random_number() * (nodes[5] - nodes[4]);
  }
  else // face == 4 || face ==5)
  {
    face_pos[0] = nodes[0] + rng.generate_random_number() * (nodes[1] - nodes[0]);
    face_pos[1] = nodes[2] + rng.generate_random_number() * (nodes[3] - nodes[2]);
    face_pos[2] = (face == 4) ? nodes[4] : nodes[5];
  }
  return face_pos;
}

// Do a scattering interaction that mimics arithmetic and branching found in other monte carlo
// transport codes
GPU_HOST_DEVICE inline std::pair<double, std::array<double, 3>> intensive_scatter(const double temperature, const double nu_in, const std::array<double,3> omega_in, RNG &rand) {
  using Constants::pi;
  using Constants::c;
  using Constants::one_over_m_1;

  // velocity used in scattering and sampling
  std::array<double,3> velocity;

  // initialize to invalid values
  double nu_in_0 = -std::numeric_limits<double>::max();

  // E set to initial invalid value
  double E = -std::numeric_limits<double>::max();

  double modifier_1 = -std::numeric_limits<double>::max();

  // sample until electron is accepted
  bool done = false;
  do {
    // sample speed
    const double speed = scattering_functions::function_3(rand, temperature);

    // sample which function to further sample.
    double mu = -1.0 + 2.0 * sqrt(rand.generate_random_number());

    scattering_functions::Check(mu > -1.0, __LINE__);
    scattering_functions::Check(mu < 1.0, __LINE__);

    // sample azimuthal angle
    const double phi = 2 * pi * rand.generate_random_number();

    // calculate interaction velocity
    velocity = omega_in;
    scattering_functions::function_5(mu, phi, velocity);
    velocity[0] *= speed;
    velocity[1] *= speed;
    velocity[2] *= speed;

    // modify velocity
    std::pair<double, std::array<double,3>> const new_freq_and_angle = scattering_functions::function_4(nu_in, omega_in, velocity);
    nu_in_0 = new_freq_and_angle.first;
    scattering_functions::Check(nu_in_0 > 0.0, __LINE__);

    // transform frequency to be unitless using
    E = nu_in_0 * one_over_m_1;
    scattering_functions::Check(E > 0.0, __LINE__);
    const double TwoE = 2.0 * E;
    const double Esquared = E * E;
    modifier_1 = 1.0 + TwoE;

    // use energy to select numerical method
    if (E < 0.01) {
      const double xs_ratio = std::fabs(0.8899 - TwoE + 4.9 * Esquared);

      // test rejection criteria
      if (rand.generate_random_number() < scattering_functions::accept_prob - 0.001*xs_ratio)
        done = true;
      scattering_functions::Check(xs_ratio >= 0.0, __LINE__);
      scattering_functions::Check(xs_ratio <= 1.0, __LINE__);
    }
    // otherwise
    else {
      const double one_over_g2 = 1.0 / (modifier_1 * modifier_1);
      scattering_functions::Check(modifier_1 > 0.0, __LINE__);
      const double log_modifier_1 = std::log(modifier_1);
      const double xs_ratio_numer = std::fabs(0.75 * (1.87 + (Esquared + E * Esquared) * one_over_g2 +
                                            (Esquared - TwoE - 1.6) * log_modifier_1 / TwoE));

      const double xs_ratio = xs_ratio_numer / Esquared;
      done = (rand.generate_random_number() < (scattering_functions::accept_prob - 0.01*(1.0/xs_ratio)));
      scattering_functions::Check(xs_ratio >= 0.0, __LINE__);
      scattering_functions::Check(xs_ratio <= 1.0, __LINE__);
    }
  } while (!done);

  // use the sampled velocity to slighty modify the photon angle
  std::array<double,3> modifier {(velocity[0] < 1.0) ? velocity[0] : 1.0/velocity[0],
                                 (velocity[1] < 1.0) ? velocity[1] : 1.0/velocity[1],
                                 (velocity[2] < 1.0) ? velocity[2] : 1.0/velocity[2]};


  std::array<double,3> omega_out{omega_in[0] * (0.99 - 0.01 * modifier[0]), omega_in[1] * (0.99 - 0.01 *modifier[1]),  omega_in[2] * (0.99 - 0.01 * modifier[2])};

  scattering_functions::normalizer(omega_out);
  return {nu_in_0*(0.99 - 0.1* (1.0/rand.generate_random_number())), omega_out};
}


//! Set angle given input array and RNG
GPU_HOST_DEVICE
inline std::array<double, 3> get_uniform_angle(RNG &rng) {
  using Constants::pi;
  using std::cos;
  using std::sin;
  using std::sqrt;
  std::array<double, 3> angle{0.0, 0.0, 0.0};
  double mu = rng.generate_random_number() * 2.0 - 1.0;
  double phi = rng.generate_random_number() * 2.0 * pi;
  double sin_theta = sqrt(1.0 - mu * mu);
  angle[0] = sin_theta * cos(phi);
  angle[1] = sin_theta * sin(phi);
  angle[2] = mu;
  return angle;
}

//! Set angle given input array, RNG and strata
inline std::array<double,3> get_stratified_angle( RNG &rng, uint32_t isample,
                                 uint32_t nsample) {
  using Constants::pi;
  using std::cos;
  using std::sin;
  using std::sqrt;
  std::array<double, 3> angle{0.0, 0.0, 0.0};
  //stratify by octant--two polar, four azimuthal
  double frac = double(isample) / nsample;
  int imu = int(frac > 0.5);  // 0 or 1
  int iphi = int(frac * 4.0); // 0 through 3
  double mu = 0.5 * (imu + rng.generate_random_number()) * 2.0 - 1.0;
  double phi = 0.25 * (iphi + rng.generate_random_number()) * 2.0 * pi;
  double sin_theta = sqrt(1.0 - mu * mu);
  angle[0] = sin_theta * cos(phi);
  angle[1] = sin_theta * sin(phi);
  angle[2] = mu;
  return angle;
}

//! Set angle on face given input array and RNG
GPU_HOST_DEVICE inline std::array<double, 3> get_source_angle_on_face( RNG &rng, int face) {
  using Constants::pi;
  using std::cos;
  using std::sin;
  using std::sqrt;

  std::array<double, 3> angle{0.0, 0.0, 0.0};
  double theta = acos(sqrt(rng.generate_random_number()));
  double phi = rng.generate_random_number() * 2.0 * pi;
  double sign = (face % 2) ? -1.0 : 1.0;
  if( face == 0 || face ==1) {
    angle[0] = cos(theta) * sign;
    angle[1] = sin(theta) * sin(phi);
    angle[2] = sin(theta) * cos(phi);
  }
  else if( face == 2 || face ==3) {
    angle[0] = sin(theta) * sin(phi);
    angle[1] = cos(theta);
    angle[2] = sin(theta) * cos(phi);
  }
  else // face == 4 || face ==5)
  {
    angle[0] = sin(theta) * cos(phi);
    angle[1] = sin(theta) * sin(phi);
    angle[2] = cos(theta);
  }
  return angle;
}


//! Sample the group after an effective scattering event
GPU_HOST_DEVICE
inline int sample_emission_group(RNG &rng, const Cell &cell_data) {
  // Sample a new group from a uniform CDF (but mimc non-uniform CDF algorithm)
  double cdf_value = rng.generate_random_number();
  int new_group = -1;
  // normalizes the PDF (opacity is uniform, not weighting with spectrum so
  // this is very simple)
  double norm_factor = 1.0 / (cell_data.get_op_a(0) * BRANSON_N_GROUPS);
  while (cdf_value > 0) {
    new_group++;
    cdf_value -= cell_data.get_op_a(new_group) * norm_factor;
  }
  return new_group;
}

struct EmissionGroupData {
  std::array<double, BRANSON_N_GROUPS> cumulative_probs;
  double total_probability;
};

GPU_HOST_DEVICE
inline EmissionGroupData precompute_emission_group_data(const Cell &cell_data) {
  EmissionGroupData data;
  double cumulative_prob = 0.0;
  double norm_factor = 1.0 / (cell_data.get_op_a(0) * BRANSON_N_GROUPS);
  for (int i = 0; i < BRANSON_N_GROUPS; ++i) {
    cumulative_prob += norm_factor * cell_data.get_op_a(i);
    data.cumulative_probs[i] = cumulative_prob;
  }
  data.total_probability = cumulative_prob;
  return data;
}

GPU_HOST_DEVICE
inline int sample_emission_group(RNG &rng, const EmissionGroupData &data) {
  double cdf_value = rng.generate_random_number() * data.total_probability;
  int new_group = BRANSON_N_GROUPS-1;
  for (int i = 0; i < BRANSON_N_GROUPS; ++i) {
      new_group = cdf_value >= data.cumulative_probs[i] ? i : new_group;
  }
  return new_group;
}


#endif
//---------------------------------------------------------------------------//
// end of sampling_functions.h
//---------------------------------------------------------------------------//
