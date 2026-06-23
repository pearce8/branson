#ifndef transport_replicated_h_
#define transport_replicated_h_

#include <algorithm>
#include <functional>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <vector>
#include <type_traits> // Required for std::is_same_v

#include "config.h"
#include "RNG.h"
#include "constants.h"
#include "gpu_setup.h"
#include "info.h"
#include "imc_parameters.h"
#include "mesh.h"
#include "message_counter.h"
#include "post_process_functions.h"
#include "history_based_transport.h"
#include "event_based_transport.h"
#include "photon.h"
#include "photon_array.h" // Include PhotonArray
#include "transport_mode_wrapper.h"

// Add function to calculate memory usage for both approaches
template<typename T>
void print_memory_footprint(const T& particle_container, const std::string& container_type) {
  size_t total_bytes = 0;
  size_t num_particles = particle_container.size();

  if (num_particles == 0) {
      std::cout << container_type << " Memory footprint: 0 particles, 0 bytes" << std::endl;
      return;
  }

  if constexpr (std::is_same_v<T, std::vector<Photon>>) {
    // AoS memory calculation
    total_bytes = num_particles * sizeof(Photon); // Simpler calculation using sizeof class
  }
  else if constexpr (std::is_same_v<T, PhotonArray>) {
    // SoA memory calculation
    total_bytes =
      num_particles * sizeof(uint32_t) +         // cell_ID vector
      num_particles * sizeof(uint32_t) +         // group vector
      num_particles * sizeof(uint32_t) +         // source_type vector
      num_particles * sizeof(unsigned char) +     // descriptors vector
      num_particles * sizeof(std::array<double, 3>) + // position vector
      num_particles * sizeof(std::array<double, 3>) + // angle vector
      num_particles * sizeof(double) +           // E vector
      num_particles * sizeof(double) +           // E0 vector
      num_particles * sizeof(double) +           // life_dx vector
      num_particles * sizeof(RNG);               // rng vector
  } else {
      std::cout << "Unknown particle container type for memory calculation." << std::endl;
      return;
  }


  double total_mb = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
  std::cout << container_type << " Memory footprint:" << std::endl;
  std::cout << "  Total particles: " << num_particles << std::endl;
  std::cout << "  Total bytes: " << total_bytes << std::endl;
  std::cout << "  Total MB: " << total_mb << std::endl;
  std::cout << "  Bytes per particle: " << static_cast<double>(total_bytes) / num_particles << std::endl;
}

template <typename Census_T>
void replicated_transport(const Mesh& mesh, const GPU_Setup<Census_T>& gpu_setup, IMC_State& imc_state,
  std::vector<double>& rank_abs_E, std::vector<double>& rank_track_E, Census_T& all_photons,
  const IMC_Parameters &imc_parameters) {
  using std::cout;
  using std::endl;
  using std::vector;

  // Print total memory footprint information
  //cout << "\n=== Total Particle Memory Usage ===" << endl;
  //print_memory_footprint(all_photons, std::is_same_v<Census_T, std::vector<Photon>> ? "AoS (vector<Photon>)" : "SoA (PhotonArray)");

  // Print theoretical batch memory calculation once at the start (for CPU event-based)
  if (imc_parameters.get_transport_algorithm() == Constants::EVENT) {
      auto event_batch_size = imc_parameters.get_event_batch_size();
      size_t batch_memory = 0;
      if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
        batch_memory = event_batch_size * sizeof(Photon);
        cout << "\n=== AoS CPU Event Batch Memory Estimate ===" << endl;
      }
      else if constexpr (std::is_same_v<Census_T, PhotonArray>) {
        batch_memory =
          event_batch_size * sizeof(uint32_t) +          // cell_ID vector
          event_batch_size * sizeof(uint32_t) +          // group vector
          event_batch_size * sizeof(uint32_t) +          // source_type vector
          event_batch_size * sizeof(unsigned char) +      // descriptors vector
          event_batch_size * sizeof(std::array<double, 3>) + // position vector
          event_batch_size * sizeof(std::array<double, 3>) + // angle vector
          event_batch_size * sizeof(double) +            // E vector
          event_batch_size * sizeof(double) +            // E0 vector
          event_batch_size * sizeof(double) +            // life_dx vector
          event_batch_size * sizeof(RNG);                // rng vector
        cout << "\n=== SoA CPU Event Batch Memory Estimate ===" << endl;
      }
       if (event_batch_size > 0) {
            double batch_mb = static_cast<double>(batch_memory) / (1024.0 * 1024.0);
            cout << "Batch size: " << event_batch_size << " particles" << endl;
            cout << "Batch memory: " << batch_memory << " bytes (" << batch_mb << " MB)" << endl;
            cout << "Bytes per particle: " << static_cast<double>(batch_memory) / event_batch_size << endl;
       }
  }


  // is the GPU even available?
#ifdef USE_GPU
  constexpr bool gpu_available = true;
#else
  constexpr bool gpu_available = false;
#endif

  double census_E = 0.0;
  double exit_E = 0.0;
  double next_dt = imc_state.get_next_dt(); //! Set for census photons
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // print warning message if GPU transport is requested but not available
  if (rank == 0 && gpu_setup.use_gpu_transporter() && !gpu_available) {
    std::cout << "WARNING: use_gpu_transporter set to true but GPU kernel not available,";
    std::cout << " running transport on CPU" << std::endl;
  }

  // timing
  
  //Timer t_transport;
  //t_transport.start_timer("timestep transport");

  Timer t_transport;

  //------------------------------------------------------------------------//
  // main transport loop
  //------------------------------------------------------------------------//
  
  vector<Cell_Tally> cell_tallies(mesh.get_n_local_cells()); // Initialize tallies (zeroed)

  uint32_t rank_cell_offset{ 0 }; // no offset in replicated mesh
  std::vector<std::vector<Photon>> null_send_list(0); // not used in replicated mode

  #ifdef caliper_FOUND
    CALI_MARK_BEGIN("Batch Transport");
  #else
    t_transport.start_timer("batch transport");
  #endif

  auto [batch_complete, batch_exit_E, batch_census_E] = batch_transport(next_dt, gpu_available, gpu_setup, imc_parameters, rank_cell_offset, mesh, all_photons, null_send_list, cell_tallies, t_transport);
  auto n_complete = batch_complete;
  census_E += batch_census_E;
  exit_E += batch_exit_E;

  // record time of transport work for this rank
 #ifdef caliper_FOUND
    CALI_MARK_END("Batch Transport");
  #else
    t_transport.stop_timer("Batch Transport");
  #endif

  // copy cell tallies back out to rank_abs_E and rank_track_E
  // This should happen regardless of CPU/GPU or algorithm, using the final cell_tallies state.

  #ifdef caliper_FOUND
    CALI_MARK_BEGIN("Tally Copies");
  #else
    t_transport.start_timer("tally copies");
  #endif

  for (size_t i = 0; i < cell_tallies.size();++i) {
    rank_abs_E[i] = cell_tallies[i].get_abs_E();
    rank_track_E[i] = cell_tallies[i].get_track_E();
  }

  #ifdef caliper_FOUND
    CALI_MARK_END("Tally Copies");
  #else
    t_transport.stop_timer("tally copies");
  #endif

  // wait for all ranks to finish
  MPI_Barrier(MPI_COMM_WORLD);

  // set diagnostic quantities
  imc_state.set_exit_E(exit_E);
  imc_state.set_post_census_E(census_E);
  imc_state.set_rank_transport_runtime(
    t_transport.get_time("transport"));

  // remove everything but photons marked census
  remove_inactive_photons(all_photons);

  imc_state.set_census_size(all_photons.size());
}

#endif // def transport_replicated_h_
