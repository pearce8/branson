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
  vector<double> abs_E(mesh.get_n_local_cells(), 0.0);
  vector<double> track_E(mesh.get_n_local_cells(), 0.0);
  auto n_user_photons = imc_parameters.get_n_user_photons();
  Message_Counter mctr;
  const int rank = mpi_info.get_rank();
  const int n_ranks = mpi_info.get_n_rank();

  const uint32_t seed = imc_parameters.get_rng_seed();

  #ifdef caliper_FOUND
    CALI_MARK_BEGIN("Time Step Loop");
  #else
    Timer driver;
    driver.start_timer("Time Step Loop");
  #endif

  while (!imc_state.finished()) {
    if (rank == 0)
      imc_state.print_timestep_header();

    mctr.reset_counters();

    //set opacity, Fleck factor, all energy to source
    // #ifdef caliper_FOUND
    //   CALI_MARK_BEGIN("Calc Consts for Source");
    // #else 
    //   driver.start_timer("Calc Consts for Source");
    // #endif

    mesh.calculate_photon_energy(imc_state, n_user_photons);

    // #ifdef caliper_FOUND
    //   CALI_MARK_END("Calc Consts for Source");
    // #else 
    //   driver.stop_timer("Calc Consts for Source");
    // #endif

    // all reduce to get total source energy to make correct number of
    // particles on each rank
    double global_source_energy = mesh.get_total_photon_E();
    // #ifdef caliper_FOUND
    //   CALI_MARK_BEGIN("MPI Reduce energy distribution");
    // #else 
    //   driver.start_timer("MPI Reduce energy distribution");
    // #endif

    MPI_Allreduce(MPI_IN_PLACE, &global_source_energy, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    // #ifdef caliper_FOUND
    //   CALI_MARK_END("MPI Reduce energy distribution");
    // #else 
    //   driver.stop_timer("MPI Reduce energy distribution");
    // #endif

    // #ifdef caliper_FOUND
    //   CALI_MARK_BEGIN("Struct Init");
    // #else 
    //   driver.start_timer("Struct Init");
    // #endif

    // make gpu setup object, may want to source on GPU later so make it before sourcing here
    GPU_Setup<Census_T> gpu_setup(rank, n_ranks, imc_parameters.get_use_gpu_transporter_flag(), mesh.get_cells(), n_user_photons);

    // #ifdef caliper_FOUND
    //   CALI_MARK_END("Struct Init");
    // #else 
    //   driver.stop_timer("Struct Init");
    // #endif


    #ifdef caliper_FOUND
      CALI_MARK_BEGIN("Source Gen");
    #else
      Timer t_source;
      t_source.start_timer("source gen");
    #endif

    make_photons<Census_T>(imc_state.get_dt(), mesh, rank, imc_state.get_step(), seed, n_user_photons, global_source_energy, gpu_setup);
    auto &all_photons = gpu_setup.get_census_photons();
    imc_state.set_pre_census_E(get_photon_list_census_E(all_photons));

    #ifdef caliper_FOUND
      CALI_MARK_END("Source Gen");
    #else
      t_source.stop_timer("source gen");
      if (rank ==0)
        std::cout<<"source time: "<<t_source.get_time("source")<<std::endl;
    #endif

    MPI_Barrier(MPI_COMM_WORLD);

    imc_state.set_transported_particles(all_photons.size());

    // #ifdef caliper_FOUND
    //   CALI_MARK_BEGIN("Printing");
    // #endif

    imc_state.print_memory_estimate(rank, n_ranks, mesh.get_n_local_cells(), all_photons.size());

    // #ifdef caliper_FOUND
    //   CALI_MARK_END("Printing");
    // #endif

    // add barrier here to make sure the transport timer starts at roughly the same time
    
    // #ifdef caliper_FOUND
    //   CALI_MARK_BEGIN("MPI_Barrier");
    // #else
    //   Timer transport;
    //   transport.start_timer("MPI_Barrier");
    // #endif
    
    MPI_Barrier(MPI_COMM_WORLD);

    // #ifdef caliper_FOUND
    //   CALI_MARK_END("MPI_Barrier");
    // #else
    //   transport.stop_timer("MPI_Barrier");
    // #endif

    #ifdef caliper_FOUND
      CALI_MARK_BEGIN("Transport Call");
    #else
      transport.start_timer("Transport Call");
    #endif

      particle_pass_transport(mesh, gpu_setup, imc_parameters, mpi_info, mpi_types, imc_state, mctr, abs_E, track_E, all_photons);
    
    #ifdef caliper_FOUND
      CALI_MARK_END("Transport Call");
    #else
      transport.stop_timer("Transport Call");
    #endif

    #ifdef caliper_FOUND
      CALI_MARK_BEGIN("Material Update");
    #else
      transport.start_timer("Material Update");
    #endif
    mesh.update_temperature(abs_E, track_E, imc_state);
    #ifdef caliper_FOUND
      CALI_MARK_END("Material Update");
    #else
      transport.stop_timer("Material Update");
    #endif

    // #ifdef caliper_FOUND
    //   CALI_MARK_BEGIN("Printing");
    // #endif
    // update time for next step
    imc_state.print_conservation(imc_parameters.get_dd_mode());
    // #ifdef caliper_FOUND
    //   CALI_MARK_END("Printing");
    // #endif

    // write SILO file if it's enabled and it's the right cycle
    if (imc_parameters.get_write_silo_flag() &&
        !(imc_state.get_step() % imc_parameters.get_output_frequency())) {
      // write SILO file
      constexpr bool replicated_flag = false;
      double fake_mpi_runtime = 0.0;

      #ifdef caliper_FOUND
        CALI_MARK_BEGIN("Write SILO");
      #else
        transport.start_timer("Write SILO");
      #endif
      write_silo(mesh, imc_state.get_time(), imc_state.get_step(),
                 imc_state.get_rank_transport_runtime(), fake_mpi_runtime, rank,
                 n_ranks, replicated_flag);
      #ifdef caliper_FOUND
        CALI_MARK_END("Write SILO");
      #else
        driver.stop_timer("Write SILO");
      #endif
    }

    imc_state.next_time_step();
  }
  #ifdef caliper_FOUND
    CALI_MARK_END("Time Step Loop");
  #else
    driver.stop_timer("Time Step Loop");
  #endif
  }

#endif // particle_pass_driver_h_

//---------------------------------------------------------------------------//
// end of particle_pass_driver.h
//---------------------------------------------------------------------------//
