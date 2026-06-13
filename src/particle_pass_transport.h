//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   particle_pass_transport.h
 * \author Alex Long
 * \date   December 1 2015
 * \brief  IMC transport with particle passing method
 * \note   Copyright (C) 2017 Los Alamos National Security, LLC.
 *         All rights reserved
 */
//---------------------------------------------------------------------------//

#ifndef particle_pass_transport_h_
#define particle_pass_transport_h_

#include <algorithm>
#include <functional>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <vector>

#include "config.h"
#include "post_process_functions.h"
#include "history_based_transport.h"
#include "gpu_setup.h"
#include "buffer.h"
#include "constants.h"
#include "info.h"
#include "mesh.h"
#include "message_counter.h"
#include "mpi_types.h"
#include "photon.h"
#include "sampling_functions.h"


template <typename Census_T>
void particle_pass_transport(
    const Mesh &mesh, const GPU_Setup<Census_T> &gpu_setup, const IMC_Parameters &imc_parameters, const Info &mpi_info, const MPI_Types &mpi_types,
    IMC_State &imc_state, Message_Counter &mctr, std::vector<double> &rank_abs_E, std::vector<double> &rank_track_E, Census_T &all_photons) {
  using std::vector;

  // is the GPU even available?
  #ifdef USE_GPU
  constexpr bool gpu_available = true;
  #else
  constexpr bool gpu_available = false;
  #endif

  int rank = mpi_info.get_rank();

  // print warning message if GPU transport is requested but not available
  bool use_gpu = gpu_setup.use_gpu_transporter() && gpu_available;
  if(rank==0 && gpu_setup.use_gpu_transporter() && !gpu_available) {
    std::cout<<"WARNING: use_gpu_transporter set to true but GPU kernel not available,";
    std::cout<<" running transport on CPU"<<std::endl;
  }

  double census_E = 0.0;
  double exit_E = 0.0;
  double next_dt = imc_state.get_next_dt(); //! Set for census photons

  // timing
  // #ifdef caliper_FOUND
  //   CALI_MARK_BEGIN("Timestep_transport");
  // #else
  Timer t_transport;
  t_transport.start_timer("timestep_transport");
  // #endif

  // Number of particles to run between MPI communication
  const uint32_t dd_batch_size = imc_parameters.get_dd_batch_size();

  // Preferred size of MPI message
  const uint32_t max_buffer_size = imc_parameters.get_particle_message_size();
  MPI_Datatype MPI_Particle = mpi_types.get_particle_type();

  // get global photon count
  uint64_t n_local = all_photons.size();
  uint64_t n_global;
  uint64_t last_global_complete_count = 0;

  MPI_Allreduce(&n_local, &n_global, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);

  // This flag indicates that send processing is needed for target rank
  vector<vector<Photon>> send_list;
  vector<Cell_Tally> cell_tallies(mesh.get_n_local_cells());

  // Completion count request made flag
  bool req_made = false;
  int recv_allreduce_flag;

  // get adjacent processor map (off_rank_id -> adjacent_proc_number)
  auto adjacent_procs = mesh.get_proc_adjacency_list();
  uint32_t n_adjacent = adjacent_procs.size();
  // messsage requests for photon sends and receives
  MPI_Request *phtn_recv_request = new MPI_Request[n_adjacent];
  MPI_Request *phtn_send_request = new MPI_Request[n_adjacent];
  // message request for non-blocking allreduce
  MPI_Request completion_request;
  // make a send/receive particle buffer for each adjacent processor
  vector<Buffer<Photon>> phtn_recv_buffer(n_adjacent);
  vector<Buffer<Photon>> phtn_send_buffer(n_adjacent);

  // Post receives for photons from adjacent sub-domains
  {
    uint32_t i_b; // buffer index
    int adj_rank; // adjacent rank
    for (auto const &it : adjacent_procs) {
      adj_rank = it.first;
      i_b = it.second;
      // push back send and receive lists
      vector<Photon> empty_phtn_vec;
      send_list.push_back(empty_phtn_vec);
      // make receive buffer the appropriate size
      phtn_recv_buffer[i_b].resize(max_buffer_size);
      MPI_Irecv(phtn_recv_buffer[i_b].get_buffer(), max_buffer_size,
                MPI_Particle, adj_rank, Constants::photon_tag, MPI_COMM_WORLD,
                &phtn_recv_request[i_b]);
      mctr.n_receives_posted++;
      phtn_recv_buffer[i_b].set_awaiting();
    } // end loop over adjacent processors
  }

  //------------------------------------------------------------------------//
  // main transport loop
  //------------------------------------------------------------------------//

  Census_T phtn_recv_list; //!< Photons from received messages

  uint64_t n_complete = 0; //!< Completed histories, regardless of origin
  //! Send and receive buffers for complete count
  uint64_t s_global_complete, r_global_complete;
  const uint32_t rank_cell_offset{mesh.get_rank_cell_offset(rank)};

  //------------------------------------------------------------------------//
  // on GPU, transport all photons from source
  //------------------------------------------------------------------------//
  
  #ifdef caliper_FOUND
    CALI_MARK_BEGIN("Batch Transport");
  #else
    Timer transport;
    transport.start_timer("Batch Transport");
  #endif

  bool local_work_done = false;
  if(use_gpu) {
    auto [batch_complete, batch_exit_E, batch_census_E] = batch_transport(next_dt, gpu_available, gpu_setup, imc_parameters, rank_cell_offset, mesh, all_photons, send_list, cell_tallies, t_transport);
    n_complete += batch_complete;
    exit_E += batch_exit_E;
    census_E += batch_census_E;
    local_work_done = true;
    // condense census right now?

  }

  // particles that reach census from comm need a place to live
  Census_T commed_census_particles;

  //------------------------------------------------------------------------//
  // process photon send and receives
  //------------------------------------------------------------------------//
  size_t batch_end =0;

  while (last_global_complete_count != n_global) {

    // on the CPU, allow interleaving of computation and communication with dd_batch_size and
    // the particle message size
    if(!use_gpu) {
      size_t batch_start = batch_end;
      batch_end = std::min(batch_start + dd_batch_size, all_photons.size());
      if (batch_start != batch_end) {
        if (batch_end == all_photons.size()) {
          local_work_done = true;
        }
        if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
          // Create a sub-vector for the batch (requires copy)
          std::vector<Photon> batch_photons(all_photons.begin() + batch_start,
                                            all_photons.begin() + batch_end);
          auto [batch_complete, batch_exit_E, batch_census_E] = batch_transport(next_dt, gpu_available, gpu_setup, imc_parameters, rank_cell_offset, mesh, batch_photons, send_list, cell_tallies, t_transport);
          // copy batch back into all_photons
          std::copy(batch_photons.begin(), batch_photons.end(), all_photons.begin() + batch_start);
          n_complete += batch_complete;
          exit_E += batch_exit_E;
          census_E += batch_census_E;
        } else if constexpr (std::is_same_v<Census_T, PhotonArray>) {
          // Get a sub-batch (creates a copy)
          auto batch_photons = all_photons.get_sub_batch(batch_start, batch_end);
          auto [batch_complete, batch_exit_E, batch_census_E] = batch_transport(next_dt, gpu_available, gpu_setup, imc_parameters, rank_cell_offset, mesh, batch_photons, send_list, cell_tallies, t_transport);
          all_photons.update_from_sub_batch(batch_photons, batch_start);
          n_complete += batch_complete;
          exit_E += batch_exit_E;
          census_E += batch_census_E;
        } else {
          std::cout << "Unsupported particle container for CPU DD transport." << std::endl;
          exit(EXIT_FAILURE);
        }
      }
    }
    
    #ifdef caliper_FOUND
      CALI_MARK_END("Batch Transport");
    #else
      transport.stop_timer("Batch Transport");
    #endif

    int recv_req_flag;
    int recv_count; // recieve count is 32 bit

    MPI_Status recv_status;
    uint32_t i_b; // buffer index
    int adj_rank; // adjacent rank
    for (auto const &it : adjacent_procs) {
      adj_rank = it.first;
      i_b = it.second;

      // test completion of send buffer
      if (phtn_send_buffer[i_b].sent()) {
        int send_req_flag;
        MPI_Test(&phtn_send_request[i_b], &send_req_flag, MPI_STATUS_IGNORE);
        if (send_req_flag) {
          phtn_send_buffer[i_b].reset();
          mctr.n_sends_completed++;
        }
      }

      // send full photon buffers if send_list has some photons in it
      //if (phtn_send_buffer[i_b].empty() && !send_list[i_b].empty()) {
      if (phtn_send_buffer[i_b].empty() && (send_list[i_b].size() == max_buffer_size || (!send_list[i_b].empty() && local_work_done == true))) {
        const uint32_t n_photons_to_send = (send_list[i_b].size() <= max_buffer_size) ?
            send_list[i_b].size() : max_buffer_size;
        vector<Photon>::iterator copy_start = send_list[i_b].begin();
        vector<Photon>::iterator copy_end = send_list[i_b].begin() + n_photons_to_send;
        vector<Photon> send_now_list(copy_start, copy_end);
        send_list[i_b].erase(copy_start, copy_end);
        phtn_send_buffer[i_b].fill(send_now_list);
        MPI_Isend(phtn_send_buffer[i_b].get_buffer(), n_photons_to_send, MPI_Particle, adj_rank,
          Constants::photon_tag, MPI_COMM_WORLD, &phtn_send_request[i_b]);
        phtn_send_buffer[i_b].set_sent();
        // update counters
         mctr.n_particles_sent += n_photons_to_send;
         mctr.n_sends_posted++;
         mctr.n_particle_messages++;
      }

      // process receive buffer
      if (phtn_recv_buffer[i_b].awaiting()) {
        MPI_Test(&phtn_recv_request[i_b], &recv_req_flag, &recv_status);
        if (recv_req_flag) {
          const vector<Photon> &receive_list =
              phtn_recv_buffer[i_b].get_object();
          // only push the number of received photons onto the recv_list
          MPI_Get_count(&recv_status, MPI_Particle, &recv_count);
          for (uint32_t i = 0; i < uint32_t(recv_count); ++i)
            phtn_recv_list.push_back(receive_list[i]);
          phtn_recv_buffer[i_b].reset();
          // post receive again, don't resize--it's already set to maximum
          MPI_Irecv(phtn_recv_buffer[i_b].get_buffer(), max_buffer_size,
                    MPI_Particle, adj_rank, Constants::photon_tag, MPI_COMM_WORLD,
                    &phtn_recv_request[i_b]);
          phtn_recv_buffer[i_b].set_awaiting();
          mctr.n_receives_completed++;
          mctr.n_receives_posted++;
        }
      }
    } // end loop over adjacent processors

    if(!phtn_recv_list.empty()) {
      #ifdef caliper_FOUND
        CALI_MARK_BEGIN("Received Batch Transport");
      #else
        t.start_timer("Received Batch Transport");
      #endif
      auto [batch_complete, batch_exit_E, batch_census_E] = batch_transport(next_dt, gpu_available, gpu_setup, imc_parameters, rank_cell_offset, mesh, phtn_recv_list, send_list, cell_tallies, t_transport);
      n_complete += batch_complete;
      exit_E += batch_exit_E;
      census_E += batch_census_E;

      // remove everything but photons marked census (off proc handled in batch transport above)
      remove_inactive_photons(phtn_recv_list);
      join_photon_arrays(commed_census_particles, phtn_recv_list);

      #ifdef caliper_FOUND
        CALI_MARK_END("Received Batch Transport");
      #else
        transport.stop_timer("Received Batch Transport");
      #endif
    }

    phtn_recv_list.clear();

    if (!req_made) {
      s_global_complete = n_complete;
      MPI_Iallreduce(&s_global_complete, &r_global_complete, 1,
                     MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD,
                     &completion_request);
      req_made = true;
    } else {
      MPI_Test(&completion_request, &recv_allreduce_flag, MPI_STATUS_IGNORE);
      if (recv_allreduce_flag) {
        last_global_complete_count = r_global_complete;
        s_global_complete = n_complete;
        if (last_global_complete_count != n_global) {
          MPI_Iallreduce(&s_global_complete, &r_global_complete, 1,
                         MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD,
                         &completion_request);
        }
      }
    }

  } // end while

  // record time of transport work for this rank
  t_transport.stop_timer("timestep_transport");

  // wait for all ranks to finish then send empty photon messages, do this because it's possible
  // for a rank to receive the empty message while it's still in the transport loop. In that case, it will post a
  // receive again, which will never have a matching send
  MPI_Barrier(MPI_COMM_WORLD);

  // finish off posted photon receives
  {
    vector<Photon> one_photon(1);
    int adj_rank; // adjacent rank
    for (auto const &it : adjacent_procs) {
      adj_rank = it.first;
      // send one photon vector to finish off receives, these photons will not be processed by the
      // receiving ranks (all ranks are out of transport)
      MPI_Send(one_photon.data(), 1, MPI_Particle, adj_rank, Constants::photon_tag, MPI_COMM_WORLD);
      mctr.n_sends_posted++;
      mctr.n_sends_completed++;
    } // end loop over adjacent processors
  }

  // wait for receive requests

  #ifdef caliper_FOUND
    CALI_MARK_BEGIN("MPI Receives");
  #else
    transport.start_timer("MPI Receives");
  #endif
  for (uint32_t i_b = 0; i_b < n_adjacent; ++i_b) {
    MPI_Wait(&phtn_recv_request[i_b], MPI_STATUS_IGNORE);
    mctr.n_receives_completed++;
  }
  #ifdef caliper_FOUND
    CALI_MARK_END("MPI Receives");
  #else
    transport.stop_timer("MPI Receives");
  #endif

  MPI_Barrier(MPI_COMM_WORLD);

  //std::sort(census_list.begin(), census_list.end());

  // all ranks have now finished transport
  delete[] phtn_recv_request;
  delete[] phtn_send_request;

  #ifdef caliper_FOUND
    CALI_MARK_BEGIN("Tally Copies");
  #else
    transport.start_timer("Tally Copies");
  #endif
  // copy cell tallies back out to rank_abs_E and rank_track_E
  for (size_t i = 0; i<cell_tallies.size();++i) {
    rank_abs_E[i] = cell_tallies[i].get_abs_E();
    rank_track_E[i] = cell_tallies[i].get_track_E();
  }

  #ifdef caliper_FOUND
    CALI_MARK_END("Tally Copies");
  #else
    transport.stop_timer("Tally Copies");
  #endif

  // remove everything but photons marked census from the initial source (no commed particles)
  remove_inactive_photons(all_photons);
  // add in the commed photons that reached census
  join_photon_arrays(all_photons, commed_census_particles);

  // set diagnostic quantities
  imc_state.set_exit_E(exit_E);
  imc_state.set_post_census_E(census_E);
  imc_state.set_census_size(all_photons.size());
  imc_state.set_network_message_counts(mctr);
  imc_state.set_rank_transport_runtime(t_transport.get_time("timestep_transport"));
}

#endif // def particle_pass_transport_h_
//---------------------------------------------------------------------------//
// end of transport_particle_pass.h
//---------------------------------------------------------------------------//
