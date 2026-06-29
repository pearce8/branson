//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   particle_pass_driver.h
 * \author Alex Long
 * \date   March 3 2017
 * \brief  Functions to run IMC with particle passing
 * \note   Copyright (C) 2017 Los Alamos National Security, LLC.
 *         All rights reserved
 */
//---------------------------------------------------------------------------//

#ifndef particle_pass_driver_h_
#define particle_pass_driver_h_

#include <functional>
#include <iostream>
#include <mpi.h>
#include <vector>

#include "config.h"
#include "census_functions.h"
#include "imc_parameters.h"
#include "imc_state.h"
#include "info.h"
#include "mesh.h"
#include "message_counter.h"
#include "mpi_types.h"
#include "particle_pass_transport.h"
#include "source.h"
#include "timer.h"
#include "write_silo.h"


template <typename Census_T>
void imc_particle_pass_driver(Mesh &mesh, IMC_State &imc_state,
                              const IMC_Parameters &imc_parameters,
                              const MPI_Types &mpi_types,
                              const Info &mpi_info) {
  using std::vector;
#ifdef caliper_FOUND
    CALI_MARK_BEGIN("vec_imc_particle_pass_driver");
#endif
  vector<double> abs_E(mesh.get_n_local_cells(), 0.0);
  vector<double> track_E(mesh.get_n_local_cells(), 0.0);
#ifdef caliper_FOUND
    CALI_MARK_END("vec_imc_particle_pass_driver");
#endif
  auto n_user_photons = imc_parameters.get_n_user_photons();
  Message_Counter mctr;
  const int rank = mpi_info.get_rank();
  const int n_ranks = mpi_info.get_n_rank();
  const uint32_t n_adjacent = mesh.get_proc_adjacency_list().size();
  const uint32_t max_buffer_size = imc_parameters.get_particle_message_size();
  const uint64_t particle_pass_capacity =
      n_user_photons + mesh.get_n_local_cells() +
      static_cast<uint64_t>(n_adjacent) * max_buffer_size;

  const uint32_t seed = imc_parameters.get_rng_seed();
  Source_Scratch source_scratch(mesh.get_n_local_cells());
  ParticlePassScratch<Census_T> particle_pass_scratch(
      mesh.get_n_local_cells(), n_adjacent, max_buffer_size,
      imc_parameters.get_dd_batch_size(), particle_pass_capacity);

  while (!imc_state.finished()) {
    if (rank == 0) {
#ifdef caliper_FOUND
    CALI_MARK_BEGIN("ppa_print_timestep_header");
#endif
      imc_state.print_timestep_header();
#ifdef caliper_FOUND
    CALI_MARK_END("ppa_print_timestep_header");
#endif
    }
#ifdef caliper_FOUND
    CALI_MARK_BEGIN("ppa_reset_ctrs");
#endif
    mctr.reset_counters();
#ifdef caliper_FOUND
    CALI_MARK_END("ppa_reset_ctrs");
#endif

    //set opacity, Fleck factor, all energy to source
#ifdef caliper_FOUND
    CALI_MARK_BEGIN("ppa_calculate_photon_energy");
#endif
    mesh.calculate_photon_energy(imc_state, n_user_photons);
#ifdef caliper_FOUND
    CALI_MARK_END("ppa_calculate_photon_energy");
#endif

    // all reduce to get total source energy to make correct number of
    // particles on each rank
    double global_source_energy = mesh.get_total_photon_E();
    MPI_Allreduce(MPI_IN_PLACE, &global_source_energy, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

#ifdef caliper_FOUND
    CALI_MARK_BEGIN("ppa_gpu_setup");
#endif
    // make gpu setup object, may want to source on GPU later so make it before sourcing here
    GPU_Setup<Census_T> gpu_setup(rank, n_ranks, imc_parameters.get_use_gpu_transporter_flag(), mesh.get_cells(), n_user_photons);
#ifdef caliper_FOUND
    CALI_MARK_END("ppa_gpu_setup");
#endif

    // setup source
    Timer t_source;
    t_source.start_timer("source");

    make_photons<Census_T>(imc_state.get_dt(), mesh, rank, imc_state.get_step(), seed,
                           n_user_photons, global_source_energy, source_scratch, gpu_setup);
    auto &all_photons = gpu_setup.get_census_photons();
    imc_state.set_pre_census_E(get_photon_list_census_E(all_photons));

    MPI_Barrier(MPI_COMM_WORLD);
    t_source.stop_timer("source");
    if (rank ==0)
      std::cout<<"source time: "<<t_source.get_time("source")<<std::endl;

#ifdef caliper_FOUND
    CALI_MARK_BEGIN("ppa_set_transported_particles");
#endif
    imc_state.set_transported_particles(all_photons.size());
#ifdef caliper_FOUND
    CALI_MARK_END("ppa_set_transported_particles");
#endif

#ifdef caliper_FOUND
    CALI_MARK_BEGIN("ppa_print_memory_estimate");
#endif
    imc_state.print_memory_estimate(rank, n_ranks, mesh.get_n_local_cells(), all_photons.size());
#ifdef caliper_FOUND
    CALI_MARK_END("ppa_print_memory_estimate");
#endif

    // add barrier here to make sure the transport timer starts at roughly the same time
    MPI_Barrier(MPI_COMM_WORLD);
#ifdef caliper_FOUND
    CALI_MARK_BEGIN("ppa_particle_pass_transport");
#endif
    particle_pass_transport(mesh, gpu_setup, imc_parameters, mpi_info, mpi_types,
                            imc_state, mctr, abs_E, track_E, all_photons,
                            particle_pass_scratch);
#ifdef caliper_FOUND
    CALI_MARK_END("ppa_particle_pass_transport");
#endif

#ifdef caliper_FOUND
    CALI_MARK_BEGIN("ppa_update_temperature");
#endif
    mesh.update_temperature(abs_E, track_E, imc_state);
#ifdef caliper_FOUND
    CALI_MARK_END("ppa_update_temperature");
#endif

    // update time for next step
#ifdef caliper_FOUND
    CALI_MARK_BEGIN("ppa_print_conservation");
#endif
    imc_state.print_conservation(imc_parameters.get_dd_mode());
#ifdef caliper_FOUND
    CALI_MARK_END("ppa_print_conservation");
#endif

    // write SILO file if it's enabled and it's the right cycle
#ifdef caliper_FOUND
    CALI_MARK_BEGIN("ppa_write_silo");
#endif
    if (imc_parameters.get_write_silo_flag() &&
        !(imc_state.get_step() % imc_parameters.get_output_frequency())) {
      // write SILO file
      constexpr bool replicated_flag = false;
      double fake_mpi_runtime = 0.0;
      write_silo(mesh, imc_state.get_time(), imc_state.get_step(),
                 imc_state.get_rank_transport_runtime(), fake_mpi_runtime, rank,
                 n_ranks, replicated_flag);
    }
#ifdef caliper_FOUND
    CALI_MARK_END("ppa_write_silo");
#endif

#ifdef caliper_FOUND
    CALI_MARK_BEGIN("ppa_next_time_step");
#endif
    imc_state.next_time_step();
#ifdef caliper_FOUND
    CALI_MARK_END("ppa_next_time_step");
#endif
  }
}

#endif // particle_pass_driver_h_

//---------------------------------------------------------------------------//
// end of particle_pass_driver.h
//---------------------------------------------------------------------------//
