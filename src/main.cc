//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   main.cc
 * \author Alex Long
 * \date   July 24 2014
 * \brief  Reads input file, sets up mesh and runs transport
 * \note   Copyright (C) 2017 Los Alamos National Security, LLC.
 *         All rights reserved
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <string>
#include <sys/time.h>
#include <time.h>
#include <vector>

#include "config.h"
#include "constants.h"
#include "imc_parameters.h"
#include "imc_state.h"
#include "info.h"
#include "input.h"
#include "mesh.h"
#include "mpi_types.h"
#include "particle_pass_driver.h"
#include "replicated_driver.h"
#include "timer.h"

using Constants::PARTICLE_PASS;
using Constants::REPLICATED;
using Constants::SOA;
using Constants::AOS;
using std::cout;
using std::endl;
using std::string;
using std::vector;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  // check to see if number of arguments is correct
  if (argc != 2) {
    cout << "Usage: BRANSON <path_to_input_file>" << endl;
    exit(EXIT_FAILURE);
  }

  // wrap main loop scope so objcts are destroyed before mpi_finalize is called
  {
    // get MPI parmeters and set them in mpi_info
    const Info mpi_info;
    if (mpi_info.get_rank() == 0) {
      cout << "-------- Branson, a massively parallel proxy app for Implicit "
              "Monte Carlo ------"
           << endl;
      cout << "-------- Authors: Alex Long (along@lanl.gov), Kendra Long (keadyk@lanl.gov), "
           << "Kelly Thompson (kgt@lanl.gov), and Joseph Farmer (University of Notre Dame)"
              "--------"
           << endl;
      cout << "-------- Version: 0.83"
              "----------------------------------------------------------"
           << endl
           << endl;
      cout << " Branson compiled on: " << mpi_info.get_machine_name() << endl;
      cout << "THIS IS THE CORRECT VERSION, THIS SHOULD PRODUCE MANY CALI ANNOTATIONS\n" ; 
    }

    // timing
    Timer timers;

    #ifdef caliper_FOUND
      CALI_MARK_BEGIN("Total");
    #else 
      timers.start_timer("Total");
    #endif

    // make MPI types object
    MPI_Types mpi_types;

    // get input object from filename
    std::string filename(argv[1]);
    #ifdef caliper_FOUND
      CALI_MARK_BEGIN("Input");
    #else
      timers.start_timer("input");
    #endif

    Input input(filename, mpi_types);
    if (mpi_info.get_rank() == 0)
      input.print_problem_info();

    #ifdef caliper_FOUND
      CALI_MARK_END("Input");
    #else
      timers.stop_timer("input");
    #endif

#ifdef caliper_FOUND
    MPI_Comm adiak_mpi_comm = MPI_COMM_WORLD;
    void* adiak_mpi_comm_ptr = &adiak_mpi_comm;
    adiak::init(adiak_mpi_comm_ptr);
    adiak::collect_all();
#endif

#ifdef USE_GPU
#ifdef USE_UMPIRE
    makeUmpireDevicePool(input.get_umpire_device_pool_size());
#endif
#endif

    // IMC paramters setup
    IMC_Parameters imc_p(input);

    // IMC state setup
    IMC_State imc_state(input, mpi_info.get_rank());

    // make mesh from input object
    #ifdef caliper_FOUND
      CALI_MARK_BEGIN("Mesh setup");
    #else
      timers.start_timer("mesh setup");
    #endif

    Mesh mesh(input, mpi_types, mpi_info, imc_p);
    mesh.initialize_physical_properties(input);

    //timers.stop_timer("mesh setup");
    #ifdef caliper_FOUND
      CALI_MARK_END("Mesh setup");
    #else
      timers.start_timer("mesh setup");
    #endif

    MPI_Barrier(MPI_COMM_WORLD);
    // print_MPI_out(mesh, rank, n_rank);

    // set the number of threads, it will be used by both replicated and particle passing methods
#ifdef USE_OPENMP
    omp_set_num_threads(input.get_n_omp_threads());
#endif

    //--------------------------------------------------------------------------//
    // TRT PHYSICS CALCULATION
    //--------------------------------------------------------------------------//

    if (input.get_dd_mode() == PARTICLE_PASS) {
      if( input.get_particle_storage() == AOS) {
        
        #ifdef caliper_FOUND
          CALI_MARK_BEGIN("Particle Pass Driver AOS");
        #else
          timers.start_timer("particle pass driver aos");
        #endif

        imc_particle_pass_driver<std::vector<Photon>>(mesh, imc_state, imc_p, mpi_types, mpi_info);
        
        #ifdef caliper_FOUND
          CALI_MARK_END("Particle Pass Driver AOS");
        #else
          timers.stop_timer("particle pass driver aos");
        #endif
      }
      else if(input.get_particle_storage() == SOA) {

        #ifdef caliper_FOUND
          CALI_MARK_BEGIN("Particle Pass Driver SOA");
        #else
          timers.start_timer("particle pass driver soa");
        #endif
        
        imc_particle_pass_driver<PhotonArray>(mesh, imc_state, imc_p, mpi_types, mpi_info);
        
        #ifdef caliper_FOUND
          CALI_MARK_END("Particle Pass Driver SOA");
        #else
          timers.stop_timer("particle pass driver soa");
        #endif      
      }
      else {
        cout << "Driver for array currently not supported" << endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (input.get_dd_mode() == REPLICATED) {
      if(input.get_particle_storage() == AOS) {
        timers.start_timer("replicated aos");

        #ifdef caliper_FOUND
          CALI_MARK_BEGIN("Replicated Driver AOS");
        #else
          timers.start_timer("replicated driver aos");
        #endif

        imc_replicated_driver<std::vector<Photon>>(mesh, imc_state, imc_p, mpi_types, mpi_info);
        
        #ifdef caliper_FOUND
          CALI_MARK_END("Replicated Driver AOS");
        #else
          timers.stop_timer("replicated driver aos");
        #endif

      }
      else if( input.get_particle_storage() == SOA) {
        
        #ifdef caliper_FOUND
          CALI_MARK_BEGIN("Replicated Driver SOA");
        #else
          timers.start_timer("replicated driver soa");
        #endif

        imc_replicated_driver<PhotonArray>(mesh, imc_state, imc_p, mpi_types, mpi_info);
        
        #ifdef caliper_FOUND
          CALI_MARK_END("Replicated Driver SOA");
        #else
          timers.stop_timer("replicated driver soa");
        #endif

      }
      else {
        cout << "Driver for array currently not supported" << endl;
        exit(EXIT_FAILURE);
      }
    }
    else {
      cout << "Driver for DD transport method currently not supported" << endl;
      exit(EXIT_FAILURE);
    }

    #ifdef caliper_FOUND
          CALI_MARK_END("Total");
    #else
          timers.stop_timer("Total");
    #endif

#ifdef caliper_FOUND
    adiak::fini();
#endif

    if (mpi_info.get_rank() == 0) {
      cout << "****************************************";
      cout << "****************************************" << endl;
      imc_state.print_simulation_footer(input.get_dd_mode());
      timers.print_timers();
      cout<<"Total transport: "<<imc_state.get_total_transport_time()<<endl;
      cout<<"Photons Per Second (FOM): "<<
        imc_state.get_photons_per_second_fom(imc_p.get_n_user_photons())<<endl;
#ifdef USE_GPU
#ifdef USE_UMPIRE
      cout<<"Umpire device memory pool size: "<<input.get_umpire_device_pool_size()<<" GB"<<endl;
      cout<<"Umpire device memory high water mark: "<<getDeviceMemoryHighWatermark()<<" GB"<<endl;
#ifdef caliper_FOUND
      adiak::value("umpire_device_pool_size", input.get_umpire_device_pool_size());
      adiak::value("umpire_device_high_water_mark", getDeviceMemoryHighWatermark());
#endif
#endif
#endif

#ifdef caliper_FOUND
    adiak::value("num_particles", imc_p.get_n_user_photons());
    adiak::value("fom", imc_state.get_photons_per_second_fom(imc_p.get_n_user_photons()));
    adiak::value("n_groups", BRANSON_N_GROUPS);
    adiak::value("particle_storage", (input.get_particle_storage() == Constants::AOS) ? "AOS" : "SOA");
    adiak::value("decomposition_mode", (input.get_dd_mode() == Constants::REPLICATED) ? "PARTICLE_PASS" : "REPLICATED");
    adiak::value("particle_algorithm", (input.get_particle_algorithm() == Constants::EVENT) ? "EVENT" : "HISTORY");
#endif
    }

  } // end main loop scope, objects destroyed here

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();
}
//---------------------------------------------------------------------------//
// end of main.cc
//---------------------------------------------------------------------------//
