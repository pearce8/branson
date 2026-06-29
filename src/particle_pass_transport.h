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
#include "photon_array.h"
#include "sampling_functions.h"

inline void reserve_particle_pass_container(std::vector<Photon> &container,
                                            const uint64_t capacity) {
  container.reserve(static_cast<size_t>(capacity));
}

inline void reserve_particle_pass_container(PhotonArray &container,
                                            const uint64_t capacity) {
  container.reserve(static_cast<size_t>(capacity));
}

inline void fill_particle_pass_batch(PhotonArray &batch, const PhotonArray &source,
                                     const size_t batch_start,
                                     const size_t batch_end) {
  const size_t batch_size = batch_end - batch_start;
  batch.resize(batch_size);
  std::copy(source.cell_ID.begin() + batch_start, source.cell_ID.begin() + batch_end,
            batch.cell_ID.begin());
  std::copy(source.group.begin() + batch_start, source.group.begin() + batch_end,
            batch.group.begin());
  std::copy(source.source_type.begin() + batch_start,
            source.source_type.begin() + batch_end, batch.source_type.begin());
  std::copy(source.descriptors.begin() + batch_start,
            source.descriptors.begin() + batch_end, batch.descriptors.begin());
  std::copy(source.pos.begin() + batch_start, source.pos.begin() + batch_end,
            batch.pos.begin());
  std::copy(source.angle.begin() + batch_start, source.angle.begin() + batch_end,
            batch.angle.begin());
  std::copy(source.E.begin() + batch_start, source.E.begin() + batch_end,
            batch.E.begin());
  std::copy(source.E0.begin() + batch_start, source.E0.begin() + batch_end,
            batch.E0.begin());
  std::copy(source.life_dx.begin() + batch_start,
            source.life_dx.begin() + batch_end, batch.life_dx.begin());
  std::copy(source.rng.begin() + batch_start, source.rng.begin() + batch_end,
            batch.rng.begin());
}

template <typename Census_T>
struct ParticlePassScratch {
  std::vector<std::vector<Photon>> send_list;
  std::vector<size_t> send_list_offset;
  std::vector<Cell_Tally> cell_tallies;
  std::vector<MPI_Request> phtn_recv_request;
  std::vector<MPI_Request> phtn_send_request;
  std::vector<Buffer<Photon>> phtn_recv_buffer;
  std::vector<Buffer<Photon>> phtn_send_buffer;
  Census_T phtn_recv_list;
  Census_T commed_census_particles;
  std::vector<Photon> aos_batch_photons;
  PhotonArray soa_batch_photons;
  std::vector<Photon> one_photon;

  ParticlePassScratch(const uint32_t n_local_cells, const uint32_t n_adjacent,
                      const uint32_t max_buffer_size, const uint32_t dd_batch_size,
                      const uint64_t particle_capacity)
      : send_list(n_adjacent), send_list_offset(n_adjacent, 0),
        cell_tallies(n_local_cells),
        phtn_recv_request(n_adjacent), phtn_send_request(n_adjacent),
        phtn_recv_buffer(n_adjacent), phtn_send_buffer(n_adjacent), one_photon(1) {
    for (uint32_t i = 0; i < n_adjacent; ++i) {
      send_list[i].reserve(max_buffer_size);
      phtn_recv_buffer[i].resize(max_buffer_size);
      phtn_send_buffer[i].get_object_ref().reserve(max_buffer_size);
    }
    reserve_particle_pass_container(phtn_recv_list, particle_capacity);
    reserve_particle_pass_container(commed_census_particles, particle_capacity);
    aos_batch_photons.reserve(dd_batch_size);
    soa_batch_photons.reserve(dd_batch_size);
  }

  void reset_timestep() {
    for (size_t i = 0; i < send_list.size(); ++i) {
      auto &queue = send_list[i];
      queue.clear();
      send_list_offset[i] = 0;
    }
    std::fill(cell_tallies.begin(), cell_tallies.end(), Cell_Tally{});
    for (auto &buffer : phtn_recv_buffer)
      buffer.reset();
    for (auto &buffer : phtn_send_buffer)
      buffer.reset();
    phtn_recv_list.clear();
    commed_census_particles.clear();
    aos_batch_photons.clear();
    soa_batch_photons.clear();
  }
};


template <typename Census_T>
void particle_pass_transport(
    const Mesh &mesh, const GPU_Setup<Census_T> &gpu_setup, const IMC_Parameters &imc_parameters, const Info &mpi_info, const MPI_Types &mpi_types,
    IMC_State &imc_state, Message_Counter &mctr, std::vector<double> &rank_abs_E, std::vector<double> &rank_track_E, Census_T &all_photons,
    ParticlePassScratch<Census_T> &scratch) {
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
  Timer t_transport;
  t_transport.start_timer("timestep_transport");

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

  // get adjacent processor map (off_rank_id -> adjacent_proc_number)
  auto adjacent_procs = mesh.get_proc_adjacency_list();
  uint32_t n_adjacent = adjacent_procs.size();
  scratch.reset_timestep();
  auto &send_list = scratch.send_list;
  auto &send_list_offset = scratch.send_list_offset;
  auto &cell_tallies = scratch.cell_tallies;

  // Completion count request made flag
  bool req_made = false;
  int recv_allreduce_flag;
  // messsage requests for photon sends and receives
  MPI_Request *phtn_recv_request = scratch.phtn_recv_request.data();
  MPI_Request *phtn_send_request = scratch.phtn_send_request.data();
  // message request for non-blocking allreduce
  MPI_Request completion_request;
  // make a send/receive particle buffer for each adjacent processor
  auto &phtn_recv_buffer = scratch.phtn_recv_buffer;
  auto &phtn_send_buffer = scratch.phtn_send_buffer;

  // Post receives for photons from adjacent sub-domains
  {
    uint32_t i_b; // buffer index
    int adj_rank; // adjacent rank
    for (auto const &it : adjacent_procs) {
      adj_rank = it.first;
      i_b = it.second;
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

  Census_T &phtn_recv_list = scratch.phtn_recv_list; //!< Photons from received messages

  uint64_t n_complete = 0; //!< Completed histories, regardless of origin
  //! Send and receive buffers for complete count
  uint64_t s_global_complete, r_global_complete;
  const uint32_t rank_cell_offset{mesh.get_rank_cell_offset(rank)};

  //------------------------------------------------------------------------//
  // on GPU, transport all photons from source
  //------------------------------------------------------------------------//
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
  Census_T &commed_census_particles = scratch.commed_census_particles;

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
          auto &batch_photons = scratch.aos_batch_photons;
          batch_photons.clear();
          batch_photons.insert(batch_photons.end(), all_photons.begin() + batch_start,
                               all_photons.begin() + batch_end);
          auto [batch_complete, batch_exit_E, batch_census_E] = batch_transport(next_dt, gpu_available, gpu_setup, imc_parameters, rank_cell_offset, mesh, batch_photons, send_list, cell_tallies, t_transport);
          // copy batch back into all_photons
          std::copy(batch_photons.begin(), batch_photons.end(), all_photons.begin() + batch_start);
          n_complete += batch_complete;
          exit_E += batch_exit_E;
          census_E += batch_census_E;
        } else if constexpr (std::is_same_v<Census_T, PhotonArray>) {
          auto &batch_photons = scratch.soa_batch_photons;
          fill_particle_pass_batch(batch_photons, all_photons, batch_start, batch_end);
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
      const size_t pending_send_count = send_list[i_b].size() - send_list_offset[i_b];
      if (phtn_send_buffer[i_b].empty() &&
          (pending_send_count == max_buffer_size ||
           (pending_send_count > 0 && local_work_done == true))) {
        const uint32_t n_photons_to_send =
            (pending_send_count <= max_buffer_size) ? pending_send_count
                                                    : max_buffer_size;
        vector<Photon>::iterator copy_start =
            send_list[i_b].begin() + send_list_offset[i_b];
        vector<Photon>::iterator copy_end = copy_start + n_photons_to_send;
        auto &send_buffer_object = phtn_send_buffer[i_b].get_object_ref();
        send_buffer_object.clear();
        send_buffer_object.insert(send_buffer_object.end(), copy_start, copy_end);
        send_list_offset[i_b] += n_photons_to_send;
        if (send_list_offset[i_b] == send_list[i_b].size()) {
          send_list[i_b].clear();
          send_list_offset[i_b] = 0;
        } else {
          const size_t pending_after_send =
              send_list[i_b].size() - send_list_offset[i_b];
          // Compact occasionally so the consumed prefix does not grow without
          // paying the cost of front-erase on every send.
          if (send_list_offset[i_b] >= max_buffer_size &&
              send_list_offset[i_b] >= pending_after_send) {
            auto remaining_begin =
                send_list[i_b].begin() + send_list_offset[i_b];
            std::move(remaining_begin, send_list[i_b].end(),
                      send_list[i_b].begin());
            send_list[i_b].resize(pending_after_send);
            send_list_offset[i_b] = 0;
          }
        }
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
      auto [batch_complete, batch_exit_E, batch_census_E] = batch_transport(next_dt, gpu_available, gpu_setup, imc_parameters, rank_cell_offset, mesh, phtn_recv_list, send_list, cell_tallies, t_transport);
      n_complete += batch_complete;
      exit_E += batch_exit_E;
      census_E += batch_census_E;

      // remove everything but photons marked census (off proc handled in batch transport above)
      remove_inactive_photons(phtn_recv_list);
      join_photon_arrays(commed_census_particles, phtn_recv_list);
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
    auto &one_photon = scratch.one_photon;
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
  for (uint32_t i_b = 0; i_b < n_adjacent; ++i_b) {
    MPI_Wait(&phtn_recv_request[i_b], MPI_STATUS_IGNORE);
    mctr.n_receives_completed++;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  //std::sort(census_list.begin(), census_list.end());

  // copy cell tallies back out to rank_abs_E and rank_track_E
  for (size_t i = 0; i<cell_tallies.size();++i) {
    rank_abs_E[i] = cell_tallies[i].get_abs_E();
    rank_track_E[i] = cell_tallies[i].get_track_E();
  }

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
