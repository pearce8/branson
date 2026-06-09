//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   event_based_transport.h
 * \author Joseph Farmer, Alex Long
 * \date   August 1 2024
 * \brief  IMC transport with an event-based algorithm (CPU and GPU)
 * \note   Copyright (C) 2024 Triad National Security, LLC.
 *         All rights reserved
 */
 //---------------------------------------------------------------------------//
#ifndef event_based_transport_h_
#define event_based_transport_h_

#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <fstream>
#include <string>
#include <array>

#include "config.h"
#include "RNG.h"
#include "cell_tally.h"
#include "constants.h"
#include "photon.h"
#include "photon_array.h"
#include "sampling_functions.h"

//----------------------------------------------------------------------------//
// GPU Specific Includes and Typedefs                                         //
//----------------------------------------------------------------------------//
#ifdef USE_GPU

enum GPUEventType { GPU_SCATTER, GPU_BOUNDARY, GPU_CENSUS, GPU_KILLED, GPU_PASS, GPU_EXIT, GPU_BOUND }; // Match Constants if possible

#endif // USE_GPU

//----------------------------------------------------------------------------//
// CPU Event-Based Transport Structures and Functions                         //
//----------------------------------------------------------------------------//

enum EventType { SCATTER, BOUNDARY, CENSUS, KILLED }; // CPU Event Type

struct Event {
  size_t photon_index;
  EventType type;
  double distance;
};

// PhotonTrackingData is specific to the  CPU event-based approach
struct PhotonTrackingData {
  double initial_angle_x;
  double final_angle_x;
  double time_in_cell;
  size_t num_interactions;
  bool entered_domain;
  bool exited_vacuum;
};

void save_tracking_data(const std::vector<PhotonTrackingData>& tracking_data,
  const std::string& filename) {
  std::ofstream outfile(filename);
  outfile << "initial_angle_x,final_angle_x,time_in_cell,num_interactions,exited_vacuum\n";
  for (const auto& data : tracking_data) {
    if (data.entered_domain) {
      outfile << data.initial_angle_x << ","
        << data.final_angle_x << ","
        << data.time_in_cell << ","
        << data.num_interactions << ","
        << data.exited_vacuum << "\n";
    }
  }
  outfile.close();
}

//----------------------------------------------------------------------------//
// CPU Event-Based Transport - SoA (PhotonArray)                              //
//----------------------------------------------------------------------------//
inline void precompute_data(const uint32_t rank_cell_offset,
  const PhotonArray& photon_array,
  const Cell* cells,
  const size_t* active_photons,
  size_t active_count,
  std::vector<double>& sigma_s,
  std::vector<double>& sigma_a,
  std::vector<double>& f,
  std::vector<double>& total_sigma_s,
  std::vector<uint32_t>& local_cell_indices) {
#pragma omp simd
  for (size_t i = 0; i < active_count; ++i) {
    size_t photon_index = active_photons[i];
    local_cell_indices[i] = photon_array.cell_ID[photon_index] - rank_cell_offset;
    const Cell* cell = &cells[local_cell_indices[i]];
    sigma_s[i] = cell->get_op_s(photon_array.group[photon_index]);
    sigma_a[i] = cell->get_op_a(photon_array.group[photon_index]);
    f[i] = cell->get_f();
    total_sigma_s[i] = (1.0 - f[i]) * sigma_a[i] + sigma_s[i];
  }
}

inline void calculate_distances(PhotonArray& photon_array,
  const Cell* cells,
  const size_t* active_photons,
  size_t active_count,
  const std::vector<double>& total_sigma_s,
  const std::vector<uint32_t>& local_cell_indices,
  std::vector<Event>& events) {
#pragma omp simd
  for (size_t i = 0; i < active_count; ++i) {
    size_t photon_index = active_photons[i];
    const Cell* cell = &cells[local_cell_indices[i]];

    double dist_to_scatter = 1.0e100;
    if (total_sigma_s[i] > 0.0) {
      double rn = photon_array.rng[photon_index].generate_random_number();
      dist_to_scatter = -std::log(rn) / total_sigma_s[i];
    }

    uint32_t surface_cross = 0;
    double dist_to_boundary = cell->get_distance_to_boundary(
      photon_array.pos[photon_index],
      photon_array.angle[photon_index],
      surface_cross);

    double dist_to_census = photon_array.life_dx[photon_index];
    double final_dist = std::min(dist_to_scatter, std::min(dist_to_boundary, dist_to_census));

    events[i].distance = final_dist;
    events[i].photon_index = photon_index;

    if (final_dist == dist_to_scatter)       events[i].type = SCATTER;
    else if (final_dist == dist_to_boundary) events[i].type = BOUNDARY;
    else                                     events[i].type = CENSUS;
  }
}

// Helper to get local cell indices for a subset of photons
inline void get_subset_local_indices(const PhotonArray& photon_array,
                                     const size_t* subset_photon_indices,
                                     size_t subset_count,
                                     uint32_t rank_cell_offset,
                                     std::vector<uint32_t>& subset_local_indices) {
    subset_local_indices.resize(subset_count);
    #pragma omp simd
    for (size_t i = 0; i < subset_count; ++i) {
        size_t photon_index = subset_photon_indices[i];
        subset_local_indices[i] = photon_array.cell_ID[photon_index] - rank_cell_offset;
    }
}


inline void process_scatter_events(PhotonArray& photon_array,
  const std::vector<double>& sigma_s,
  const std::vector<double>& total_sigma_s,
  const std::vector<uint32_t>& local_cell_indices,
  const Cell* cells,
  const size_t* scatter_photon_indices,
  size_t scatter_count,
  const std::vector<EmissionGroupData>& emission_groups,
  std::vector<PhotonTrackingData>& tracking_data) // Tracking data only for CPU
{
#pragma omp simd
  for (size_t i = 0; i < scatter_count; ++i) {
    size_t photon_index = scatter_photon_indices[i];
    uint32_t local_idx = local_cell_indices[i];
    RNG& rng = photon_array.rng[photon_index];
    tracking_data[photon_index].num_interactions += 1;
    photon_array.angle[photon_index] = get_uniform_angle(rng);
    photon_array.descriptors[photon_index] = static_cast<unsigned char>(Constants::SCATTER);
    if (rng.generate_random_number() > (sigma_s[i] / total_sigma_s[i])) {
      photon_array.group[photon_index] =
        sample_emission_group(rng, emission_groups[local_idx]);
      // chance of more intensive scatter
      if (rng.generate_random_number() <= Constants::intensive_scatter_fraction) {
        auto group = photon_array.group[photon_index];
        // get a frequency (faux multigroup so just sample from wide spectrum)
        double freq = Constants::lower_frequency_bound + static_cast<double>(group)/static_cast<double>(BRANSON_N_GROUPS)*Constants::delta_frequency_bounds;
        auto angle = photon_array.angle[photon_index];
        auto new_energy_angle = intensive_scatter(cells[local_idx].get_T_e(), freq, angle, rng);
        photon_array.angle[photon_index] = new_energy_angle.second;
      }
    }
  }
}

inline void process_boundary_events(PhotonArray& photon_array,
  const size_t* boundary_photon_indices,
  const Cell* cells,
  uint32_t rank_cell_offset,
  size_t boundary_count,
  std::vector<PhotonTrackingData>& tracking_data) { // Tracking data only for CPU
#pragma omp simd
  for (size_t i = 0; i < boundary_count; ++i) {
    size_t photon_index = boundary_photon_indices[i];
    uint32_t local_cell_idx = photon_array.cell_ID[photon_index] - rank_cell_offset;
    const Cell* cell = &cells[local_cell_idx];

    uint32_t surface_cross = 0;
    cell->get_distance_to_boundary(photon_array.pos[photon_index],
      photon_array.angle[photon_index],
      surface_cross);
    auto boundary_type = cell->get_bc(surface_cross);

    if (boundary_type == Constants::ELEMENT) {
      photon_array.cell_ID[photon_index] = cell->get_next_cell(surface_cross);
      photon_array.descriptors[photon_index] = static_cast<unsigned char>(Constants::BOUND); // Still active
    }
    else if (boundary_type == Constants::PROCESSOR) {
      photon_array.cell_ID[photon_index] = cell->get_next_cell(surface_cross);
      photon_array.descriptors[photon_index] = static_cast<unsigned char>(Constants::PASS); // Inactive
    }
    else if (boundary_type == Constants::VACUUM ||
      boundary_type == Constants::SOURCE) {
      photon_array.descriptors[photon_index] = static_cast<unsigned char>(Constants::EXIT); // Inactive
      if (boundary_type == Constants::VACUUM) {
        tracking_data[photon_index].exited_vacuum = true;
      }
    }
    else { // REFLECT
      int reflect_dim = surface_cross / 2;
      photon_array.angle[photon_index][reflect_dim] =
        -photon_array.angle[photon_index][reflect_dim];
      photon_array.descriptors[photon_index] = static_cast<unsigned char>(Constants::BOUND); // Still active
    }
  }
}

inline void process_census_events(PhotonArray& photon_array,
  const size_t* census_photon_indices,
  size_t census_count) {
#pragma omp simd
  for (size_t i = 0; i < census_count; ++i) {
    size_t photon_index = census_photon_indices[i];
    photon_array.descriptors[photon_index] = static_cast<unsigned char>(Constants::CENSUS); // Inactive
  }
}

inline void update_photon_state(PhotonArray& photon_array,
  Cell_Tally* cell_tallies,
  const std::vector<Event>& events, // Contains distance and type for active photons
  const std::vector<double>& sigma_a, // Corresponds to active photons
  const std::vector<double>& f,       // Corresponds to active photons
  const std::vector<uint32_t>& local_cell_indices, // Corresponds to active photons
  size_t active_count,
  std::vector<size_t>& scatter_indices, // Output: indices of photons that will scatter
  std::vector<size_t>& boundary_indices,// Output: indices of photons that will hit boundary
  std::vector<size_t>& census_indices,  // Output: indices of photons that will hit census
  std::vector<size_t>& killed_indices,  // Output: indices of photons that were killed
  size_t& scatter_count,                // Output count
  size_t& boundary_count,               // Output count
  size_t& census_count,                 // Output count
  size_t& killed_count,                 // Output count
  std::vector<PhotonTrackingData>& tracking_data) // Tracking data only for CPU
{
  // Reset counts
  scatter_count = 0;
  boundary_count = 0;
  census_count = 0;
  killed_count = 0;

  std::vector<double> absorbed_Es(active_count);
#pragma omp simd
  for (size_t i = 0; i < active_count; ++i) {
    double distance = events[i].distance;
    size_t photon_idx = events[i].photon_index; // Original index
    absorbed_Es[i] = photon_array.E[photon_idx] *
      (1.0 - std::exp(-sigma_a[i] * f[i] * distance));
  }

  // Accumulate tallies - cannot SIMD easily
  for (size_t i = 0; i < active_count; ++i) {
    uint32_t local_idx = local_cell_indices[i];
    double track_contrib = (sigma_a[i] > 0.0 && f[i] > 0.0) ? absorbed_Es[i] / (sigma_a[i] * f[i]) : 0.0;
    cell_tallies[local_idx].accumulate_absorbed_E(absorbed_Es[i]);
    cell_tallies[local_idx].accumulate_track_E(track_contrib);
  }

#pragma omp simd
  for (size_t i = 0; i < active_count; ++i) {
    const Event& event = events[i];
    size_t photon_index = event.photon_index; // Original index
    double distance = event.distance;

    // Update energy and position
    photon_array.E[photon_index] -= absorbed_Es[i];
    photon_array.pos[photon_index][0] += photon_array.angle[photon_index][0] * distance;
    photon_array.pos[photon_index][1] += photon_array.angle[photon_index][1] * distance;
    photon_array.pos[photon_index][2] += photon_array.angle[photon_index][2] * distance;
    photon_array.life_dx[photon_index] -= distance;
    tracking_data[photon_index].time_in_cell += distance / Constants::c;

    // Check for kill based on energy cutoff
    if (photon_array.E[photon_index] / photon_array.E0[photon_index] < Constants::cutoff_fraction) {

      cell_tallies[local_cell_indices[i]].accumulate_absorbed_E(photon_array.E[photon_index]);
      photon_array.descriptors[photon_index] = static_cast<unsigned char>(Constants::KILLED);
      // This photon will be added to killed_indices below
    }

  }

   // Separate loop to fill event index vectors - cannot SIMD easily due to conditional increments
   // and potential race conditions on counts
  for (size_t i = 0; i < active_count; ++i) {
      const Event& event = events[i];
      size_t photon_index = event.photon_index; // Original index

      // Check descriptor first (set above if killed by energy cutoff)
      if (photon_array.descriptors[photon_index] == static_cast<unsigned char>(Constants::KILLED)) {
          killed_indices[killed_count++] = photon_index;
      } else {
          // Assign to event lists based on the calculated event type
          switch (event.type) {
          case SCATTER:
              scatter_indices[scatter_count++] = photon_index;
              break;
          case BOUNDARY:
              boundary_indices[boundary_count++] = photon_index;
              break;
          case CENSUS:
              census_indices[census_count++] = photon_index;
              break;
          default: // Should not happen
              break;
          }
      }
  }
}

// CPU Main loop for SoA
void cpu_event_transport_photons(const uint32_t rank_cell_offset,
  PhotonArray& photon_array, const std::vector<Cell>& cells, std::vector<Cell_Tally>& cell_tallies,
  int n_omp_threads, // n_omp_threads currently unused in this fine-grained version
  const std::vector<EmissionGroupData>& emission_groups)
{
  const size_t maxPhotons = photon_array.size();
  if (maxPhotons == 0) return;

  // Tracking data is specific to this CPU implementation
  std::vector<PhotonTrackingData> tracking_data(maxPhotons);
  for (size_t i = 0; i < maxPhotons; ++i) {
    tracking_data[i].initial_angle_x = photon_array.angle[i][0];
    tracking_data[i].final_angle_x = photon_array.angle[i][0];
    tracking_data[i].time_in_cell = 0.0;
    tracking_data[i].num_interactions = 0;
    tracking_data[i].entered_domain = (photon_array.angle[i][0] > 0);
    tracking_data[i].exited_vacuum = false;
  }

  // Allocate vectors for event processing
  std::vector<size_t> scatter_indices(maxPhotons),
    boundary_indices(maxPhotons),
    census_indices(maxPhotons),
    killed_indices(maxPhotons),
    active_photons_indices(maxPhotons); // Stores original indices of active photons

  std::vector<Event> events(maxPhotons); // Stores event info for active photons
  std::vector<uint32_t> local_cell_indices(maxPhotons); // Stores local cell index for active photons
  std::vector<double> sigma_s(maxPhotons), sigma_a(maxPhotons), f(maxPhotons),
    total_sigma_s(maxPhotons);

  // Initialize active photons list with original indices 0 to N-1
  std::iota(active_photons_indices.begin(), active_photons_indices.end(), 0);
  size_t active_count = maxPhotons;

  const Cell* cells_ptr = cells.data();
  Cell_Tally* tallies_ptr = cell_tallies.data();

  while (active_count > 0) {
    // Resize intermediate vectors to current active count for efficiency
    events.resize(active_count);
    local_cell_indices.resize(active_count);
    sigma_s.resize(active_count);
    sigma_a.resize(active_count);
    f.resize(active_count);
    total_sigma_s.resize(active_count);

    size_t scatter_count = 0;
    size_t boundary_count = 0;
    size_t census_count = 0;
    size_t killed_count = 0;

    // 1. Precompute physics data for active photons
    //    active_photons_indices.data() points to the original indices
    precompute_data(rank_cell_offset, photon_array, cells_ptr,
      active_photons_indices.data(), active_count,
      sigma_s, sigma_a, f, total_sigma_s, local_cell_indices);

    // 2. Calculate distances and event types for active photons
    calculate_distances(photon_array, cells_ptr,
      active_photons_indices.data(), active_count,
      total_sigma_s, local_cell_indices, events);
      // events[i] now corresponds to the photon active_photons_indices[i]

    // 3. Update photon state, tally energy, check for kills, and classify into event lists
    //    Resize event index vectors before passing them
    scatter_indices.resize(active_count);
    boundary_indices.resize(active_count);
    census_indices.resize(active_count);
    killed_indices.resize(active_count);

    update_photon_state(photon_array, tallies_ptr, events,
      sigma_a, f, local_cell_indices,
      active_count,
      scatter_indices, boundary_indices, census_indices, killed_indices, // Output lists
      scatter_count, boundary_count, census_count, killed_count, // Output counts
      tracking_data);

    // 4. Process events using the filled index vectors and counts

    if (scatter_count > 0) {
        std::vector<double> scatter_sigma_s(scatter_count);
        std::vector<double> scatter_total_sigma_s(scatter_count);
        std::vector<uint32_t> scatter_local_indices(scatter_count);
        std::vector<double> subset_sigma_s(scatter_count);
        std::vector<double> subset_total_sigma_s(scatter_count);
        std::vector<uint32_t> subset_local_indices(scatter_count);
        // Find the mapping
        std::vector<size_t> active_to_subset_map(maxPhotons, maxPhotons); // Map original index to position in active list
        for(size_t i=0; i<active_count; ++i) active_to_subset_map[active_photons_indices[i]] = i;

        for(size_t i=0; i<scatter_count; ++i) {
            size_t original_index = scatter_indices[i];
            size_t active_list_pos = active_to_subset_map[original_index];
            if (active_list_pos < active_count) { // Should always be true
                 subset_sigma_s[i] = sigma_s[active_list_pos];
                 subset_total_sigma_s[i] = total_sigma_s[active_list_pos];
                 subset_local_indices[i] = local_cell_indices[active_list_pos];
            } else { /* handle error */ }
        }

       process_scatter_events(photon_array, subset_sigma_s, subset_total_sigma_s,
         subset_local_indices, cells_ptr,
         scatter_indices.data(), scatter_count, // Pass original indices
         emission_groups, tracking_data);
    }
    if (boundary_count > 0) {
      process_boundary_events(photon_array, boundary_indices.data(),
        cells_ptr, rank_cell_offset, boundary_count, tracking_data);
    }
    if (census_count > 0) {
      process_census_events(photon_array, census_indices.data(), census_count);
    }

    // 5. Filter active photons for the next iteration
    // Create the next list of active photon indices
    std::vector<size_t> next_active_photons_indices;
    next_active_photons_indices.reserve(active_count); // Reserve space

    for (size_t i = 0; i < active_count; ++i) {
        size_t index = active_photons_indices[i]; // Get original index
        unsigned char desc = photon_array.descriptors[index];
        // Keep if descriptor indicates it's still active in this domain
        if (desc == static_cast<unsigned char>(Constants::BOUND) ||
            desc == static_cast<unsigned char>(Constants::SCATTER))
        {
            next_active_photons_indices.push_back(index);
        } else {
             // Update final angle on termination if needed by tracking
             tracking_data[index].final_angle_x = photon_array.angle[index][0];
        }
    }
    active_photons_indices = std::move(next_active_photons_indices); // Replace old list
    active_count = active_photons_indices.size(); // Update active count

  } // End while(active_count > 0)

  save_tracking_data(tracking_data, "photon_tracking_data_soa_cpu.csv");
}


//----------------------------------------------------------------------------//
// CPU Event-Based Transport - AoS (std::vector<Photon>)                      //
//----------------------------------------------------------------------------//

// Precompute data for AOS (Array of Structures)
inline void precompute_data(const uint32_t rank_cell_offset, const std::vector<Photon>& photon_array, const Cell* cells, const size_t* active_photons, size_t active_count, std::vector<double>& sigma_s, std::vector<double>& sigma_a, std::vector<double>& f, std::vector<double>& total_sigma_s, std::vector<uint32_t>& local_cell_indices) {
#pragma omp simd
  for (size_t i = 0; i < active_count; ++i) {
    size_t photon_index = active_photons[i];
    const auto& phtn = photon_array[photon_index];
    local_cell_indices[i] = phtn.get_cell() - rank_cell_offset;
    const Cell* cell = &cells[local_cell_indices[i]];
    sigma_s[i] = cell->get_op_s(phtn.get_group());
    sigma_a[i] = cell->get_op_a(phtn.get_group());
    f[i] = cell->get_f();
    total_sigma_s[i] = (1.0 - f[i]) * sigma_a[i] + sigma_s[i];
  }
}

// Calculate distances for AOS
inline void calculate_distances(std::vector<Photon>& photon_array, const Cell* cells, const size_t* active_photons, size_t active_count, const std::vector<double>& total_sigma_s, const std::vector<uint32_t>& local_cell_indices, std::vector<Event>& events) {
#pragma omp simd
  for (size_t i = 0; i < active_count; ++i) {
    size_t photon_index = active_photons[i];
    auto& phtn = photon_array[photon_index];
    Cell const* cell = &cells[local_cell_indices[i]];
    auto& rng = phtn.get_rng();
    uint32_t surface_cross = 0;
    const double dist_to_scatter = (total_sigma_s[i] > 0.0) ? -log(rng.generate_random_number()) / total_sigma_s[i] : 1e100;
    const double dist_to_boundary = cell->get_distance_to_boundary(phtn.get_position(), phtn.get_angle(), surface_cross);
    const double dist_to_census = phtn.get_distance_remaining();
    events[i].distance = std::min(dist_to_scatter, std::min(dist_to_boundary, dist_to_census));
    events[i].photon_index = photon_index; // Store original index

    if (events[i].distance == dist_to_scatter)
      events[i].type = SCATTER;
    else if (events[i].distance == dist_to_boundary)
      events[i].type = BOUNDARY;
    else
      events[i].type = CENSUS;
  }
}

// Process scatter events for AOS
inline void process_scatter_events(std::vector<Photon>& photon_array,
    const std::vector<double>& sigma_s,
    const std::vector<double>& total_sigma_s,
    const std::vector<uint32_t>& local_cell_indices,
    const Cell* cells,
    const size_t* scatter_indices,
    size_t scatter_count,
    const std::vector<EmissionGroupData>& emission_groups,
    std::vector<PhotonTrackingData>& tracking_data)
{
#pragma omp simd
  for (size_t i = 0; i < scatter_count; ++i) {
    size_t photon_index = scatter_indices[i];
    uint32_t local_idx = local_cell_indices[i];
    auto& phtn = photon_array[photon_index];
    RNG& rng = phtn.get_rng();
    tracking_data[photon_index].num_interactions += 1;
    phtn.set_angle(get_uniform_angle(rng));
    phtn.set_descriptor(Constants::SCATTER);

    if (rng.generate_random_number() > (sigma_s[i] / total_sigma_s[i])) {
      phtn.set_group(sample_emission_group(rng, emission_groups[local_idx]));
      // chance of more intensive scatter
      if (rng.generate_random_number() <= Constants::intensive_scatter_fraction) {
        auto group = phtn.get_group();
        // get a frequency (faux multigroup so just sample from wide spectrum)
        double freq = Constants::lower_frequency_bound + static_cast<double>(group)/static_cast<double>(BRANSON_N_GROUPS)*Constants::delta_frequency_bounds;
        auto angle = phtn.get_angle();
        auto new_energy_angle = intensive_scatter(cells[local_idx].get_T_e(), freq, angle, rng);
        phtn.set_angle(new_energy_angle.second);
      }
    }
  }
}

// Process boundary events for AOS
inline void process_boundary_events(std::vector<Photon>& photon_array,
const size_t* boundary_indices, // Original indices
const Cell* cells,
uint32_t rank_cell_offset,
size_t boundary_count,
std::vector<PhotonTrackingData>& tracking_data)
{
#pragma omp simd
for (size_t i = 0; i < boundary_count; ++i) {
size_t photon_index = boundary_indices[i];
auto& phtn = photon_array[photon_index];
uint32_t local_cell_index = phtn.get_cell() - rank_cell_offset;
Cell const* cell = &cells[local_cell_index];
uint32_t surface_cross = 0;
// Recalculate surface_cross
cell->get_distance_to_boundary(phtn.get_position(), phtn.get_angle(), surface_cross);
auto boundary_event = cell->get_bc(surface_cross);
if (boundary_event == Constants::ELEMENT) {
  phtn.set_cell(cell->get_next_cell(surface_cross));
  phtn.set_descriptor(Constants::BOUND); // Active
}
else if (boundary_event == Constants::PROCESSOR) {
  phtn.set_cell(cell->get_next_cell(surface_cross));
  phtn.set_descriptor(Constants::PASS); // Inactive
}
else if (boundary_event == Constants::VACUUM || boundary_event == Constants::SOURCE) {
  phtn.set_descriptor(Constants::EXIT); // Inactive
   if (boundary_event == Constants::VACUUM) {
    tracking_data[photon_index].exited_vacuum = true;
  }
}
else { // REFLECT
  phtn.reflect(surface_cross);
  phtn.set_descriptor(Constants::BOUND); // Active
}
}
}

// Process census events for AOS
inline void process_census_events(std::vector<Photon>& photon_array,
const size_t* census_indices, // Original indices
size_t census_count)
{
#pragma omp simd // Safe for simple assignment
for (size_t i = 0; i < census_count; ++i) {
photon_array[census_indices[i]].set_descriptor(Constants::CENSUS); // Inactive
}
}

// Update photon state for AOS
inline void update_photon_state(std::vector<Photon>& photon_array,
Cell_Tally* cell_tallies,
const std::vector<Event>& events, // Corresponds to active photons
const std::vector<double>& sigma_a, // Corresponds to active photons
const std::vector<double>& f, // Corresponds to active photons
const std::vector<uint32_t>& local_cell_indices, // Corresponds to active photons
size_t active_count,
std::vector<size_t>& scatter_indices, // Output list of original indices
std::vector<size_t>& boundary_indices,// Output list of original indices
std::vector<size_t>& census_indices,  // Output list of original indices
std::vector<size_t>& killed_indices,  // Output list of original indices
size_t& scatter_count, size_t& boundary_count, size_t& census_count, size_t& killed_count, // Output counts
std::vector<PhotonTrackingData>& tracking_data)
{
// Reset counts
scatter_count = 0;
boundary_count = 0;
  census_count = 0;
  killed_count = 0;

  std::vector<double> absorbed_Es(active_count);
#pragma omp simd
  for (size_t i = 0; i < active_count; ++i) {
    size_t photon_idx = events[i].photon_index;
    absorbed_Es[i] = photon_array[photon_idx].get_E() * (1.0 - exp(-sigma_a[i] * f[i] * events[i].distance));
  }

  // Accumulate tallies - cannot SIMD easily
  for (size_t i = 0; i < active_count; ++i) {
    uint32_t local_idx = local_cell_indices[i];
    double track_contrib = (sigma_a[i] > 0.0 && f[i] > 0.0) ? absorbed_Es[i] / (sigma_a[i] * f[i]) : 0.0;
    cell_tallies[local_idx].accumulate_absorbed_E(absorbed_Es[i]);
    cell_tallies[local_idx].accumulate_track_E(track_contrib);
  }

#pragma omp simd
  for (size_t i = 0; i < active_count; ++i) {
    const Event& event = events[i];
    size_t photon_index = event.photon_index;
    auto& phtn = photon_array[photon_index];
    double distance = event.distance;

    phtn.set_E(phtn.get_E() - absorbed_Es[i]);
    phtn.move(distance);
    tracking_data[photon_index].time_in_cell += distance / Constants::c;

    // Check for kill based on energy cutoff FIRST
    if (phtn.below_cutoff(Constants::cutoff_fraction)) {
      cell_tallies[local_cell_indices[i]].accumulate_absorbed_E(phtn.get_E()); // Tally remaining E
      phtn.set_descriptor(Constants::KILLED);
      // This photon will be added to killed_indices below
    }
    // else: Event type determines the next state (descriptor set in process_* functions later)
  }

  // Separate loop to fill event index vectors - cannot SIMD easily
  for (size_t i = 0; i < active_count; ++i) {
      const Event& event = events[i];
      size_t photon_index = event.photon_index; // Original index
      auto& phtn = photon_array[photon_index];
      // Check descriptor first (set above if killed by energy cutoff)
      if (phtn.get_descriptor() == Constants::KILLED) {
          killed_indices[killed_count++] = photon_index;
      } else {
          // Assign to event lists based on the calculated event type
          switch (event.type) {
          case SCATTER:
              scatter_indices[scatter_count++] = photon_index;
              break;
          case BOUNDARY:
              boundary_indices[boundary_count++] = photon_index;
              break;
          case CENSUS:
              census_indices[census_count++] = photon_index;
              break;
          default: // Should not happen
              break;
          }
      }
  }
}

// Main CPU event transport function for AOS
void cpu_event_transport_photons(const uint32_t rank_cell_offset, std::vector<Photon>& photon_array, const std::vector<Cell>& cells, std::vector<Cell_Tally>& cell_tallies, int n_omp_threads, const std::vector<EmissionGroupData>& emission_groups) {

  const size_t maxPhotons = photon_array.size();
   if (maxPhotons == 0) return;

  // Tracking data specific to CPU implementation
  std::vector<PhotonTrackingData> tracking_data(maxPhotons);
   for (size_t i = 0; i < maxPhotons; ++i) {
    tracking_data[i].initial_angle_x = photon_array[i].get_angle()[0];
    tracking_data[i].final_angle_x = photon_array[i].get_angle()[0]; // Initialize
    tracking_data[i].time_in_cell = 0.0;
    tracking_data[i].num_interactions = 0;
    tracking_data[i].entered_domain = (photon_array[i].get_angle()[0] > 0); // Example condition
    tracking_data[i].exited_vacuum = false;
  }

  std::vector<size_t> scatter_indices(maxPhotons), boundary_indices(maxPhotons), census_indices(maxPhotons), killed_indices(maxPhotons), active_photons_indices(maxPhotons);
  std::vector<Event> events(maxPhotons);
  std::vector<uint32_t> local_cell_indices(maxPhotons);
  std::vector<double> sigma_s(maxPhotons), sigma_a(maxPhotons), f(maxPhotons), total_sigma_s(maxPhotons);

  std::iota(active_photons_indices.begin(), active_photons_indices.end(), 0);
  size_t active_count = maxPhotons;

  const Cell* cells_ptr = cells.data();
  Cell_Tally* tallies_ptr = cell_tallies.data();

  while (active_count > 0) {
    // Resize intermediate vectors
    events.resize(active_count);
    local_cell_indices.resize(active_count);
    sigma_s.resize(active_count);
    sigma_a.resize(active_count);
    f.resize(active_count);
    total_sigma_s.resize(active_count);

    size_t scatter_count = 0, boundary_count = 0, census_count = 0, killed_count = 0;

    // 1. Precompute
    precompute_data(rank_cell_offset, photon_array, cells_ptr, active_photons_indices.data(), active_count, sigma_s, sigma_a, f, total_sigma_s, local_cell_indices);

    // 2. Calculate distances
    calculate_distances(photon_array, cells_ptr, active_photons_indices.data(), active_count, total_sigma_s, local_cell_indices, events);

    // 3. Update state and classify
    scatter_indices.resize(active_count);
    boundary_indices.resize(active_count);
    census_indices.resize(active_count);
    killed_indices.resize(active_count);
    update_photon_state(photon_array, tallies_ptr, events, sigma_a, f, local_cell_indices, active_count, scatter_indices, boundary_indices, census_indices, killed_indices, scatter_count, boundary_count, census_count, killed_count, tracking_data);

    // 4. Process events
    if (scatter_count > 0) {
        // Rebuild subset physics data for scatter events
        std::vector<double> subset_sigma_s(scatter_count);
        std::vector<double> subset_total_sigma_s(scatter_count);
        std::vector<uint32_t> subset_local_indices(scatter_count);
        std::vector<size_t> active_to_subset_map(maxPhotons, maxPhotons);
        for(size_t i=0; i<active_count; ++i) active_to_subset_map[active_photons_indices[i]] = i;
        for(size_t i=0; i<scatter_count; ++i) {
            size_t original_index = scatter_indices[i];
            size_t active_list_pos = active_to_subset_map[original_index];
            if (active_list_pos < active_count) {
                 subset_sigma_s[i] = sigma_s[active_list_pos];
                 subset_total_sigma_s[i] = total_sigma_s[active_list_pos];
                 subset_local_indices[i] = local_cell_indices[active_list_pos];
            } else { /* error */ }
        }
        process_scatter_events(photon_array, subset_sigma_s, subset_total_sigma_s, subset_local_indices, cells_ptr, scatter_indices.data(), scatter_count, emission_groups, tracking_data);
    }
     if (boundary_count > 0) {
        process_boundary_events(photon_array, boundary_indices.data(), cells_ptr, rank_cell_offset, boundary_count, tracking_data);
    }
    if (census_count > 0) {
        process_census_events(photon_array, census_indices.data(), census_count);
    }

    // 5. Compact active list
    std::vector<size_t> next_active_photons_indices;
    next_active_photons_indices.reserve(active_count);
    for (size_t i = 0; i < active_count; ++i) {
        size_t index = active_photons_indices[i];
        Constants::event_type desc = photon_array[index].get_descriptor();
        if (desc == Constants::BOUND || desc == Constants::SCATTER) {
            next_active_photons_indices.push_back(index);
        } else {
            tracking_data[index].final_angle_x = photon_array[index].get_angle()[0]; // Update final angle
        }
    }
    active_photons_indices = std::move(next_active_photons_indices);
    active_count = active_photons_indices.size();
  }
   save_tracking_data(tracking_data, "photon_tracking_data_aos_cpu.csv");
}


//----------------------------------------------------------------------------//
// GPU Event-Based Transport Implementation                                   //
//----------------------------------------------------------------------------//
#ifdef USE_GPU

struct EventInfo
{
  uint32_t photon_idx;
  uint32_t active_idx;
};

// Kernel to reset atomic counters
GPU_KERNEL void reset_atomic_counters_kernel(unsigned int* counters, int num_counters) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_counters) {
        counters[idx] = 0;
    }
}

//----------------------------------------------------------------------------//
// SoA GPU Event Kernels                                                      //
//----------------------------------------------------------------------------//



// Function to perform atomic addition within a warp
// __device__ inline void warp_atomic_add(double *address, uint32_t cell_idx, double val) {
//     // Get the mask of active threads in the warp
//     const unsigned int active_mask = __activemask();
//     // Get the mask of threads with matching cell_idx
//     const unsigned int group_mask = __match_any_sync(active_mask, cell_idx);
//     // Perform warp-level reduction sum for the matching threads
//     double subgroup_sum = warp_reduce_sum(val, group_mask);
//     // Check if the current thread is the leader of the matching group
//     // Determine if the current thread is the leader of its group

//     // Calculate the lane ID within the warp (0-31)
//     unsigned int lane_id = threadIdx.x % 32;

//     // Create a mask with only the bit corresponding to this thread's lane ID set to 1
//     unsigned int thread_mask = 1u << lane_id;

//     // Check if this thread's bit is set in the group_mask
//     // If true, this thread is part of the group with matching cell_idx
//     bool is_thread_in_group = (group_mask & thread_mask) != 0;

//     // Find the lane ID of the first active thread in the group
//     // __ffs finds the 1-based index of the least significant set bit, so we subtract 1
//     unsigned int first_active_lane = __ffs(group_mask) - 1;

//     // Determine if this thread is the leader:
//     // It must be part of the group and have the lowest lane ID among group members
//     const bool is_leader = is_thread_in_group && (lane_id == first_active_lane);

//     // If the current thread is the leader, perform atomic addition
//     if (is_leader) {
//         atomicAdd(address, subgroup_sum);
//     }
// }

GPU_KERNEL void precompute_data_gpu(const uint32_t rank_cell_offset,
  const uint32_t* active_indices,
  uint32_t active_count,
  const uint32_t* group,
  const uint32_t* cell_ID,
  const Cell *cells,
  double* sigma_s,
  double* sigma_a,
  double* f,
  double* total_sigma_s,
  uint32_t* local_cell_indices) {

    uint32_t active_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (active_idx >= active_count) return;

    uint32_t photon_idx = active_indices[active_idx];
    // RNG& rng = rng_array[photon_idx];
    uint32_t g = group[photon_idx];
    uint32_t local_idx = cell_ID[photon_idx] - rank_cell_offset;
    const Cell* cell = &cells[local_idx];
    local_cell_indices[active_idx] = local_idx;
    sigma_s[active_idx] = cell->get_op_s(g);
    sigma_a[active_idx] = cell->get_op_a(g);
    f[active_idx] = cell->get_f();
    total_sigma_s[active_idx] = (1 - f[active_idx]) * sigma_a[active_idx] + sigma_s[active_idx];
}


GPU_KERNEL void process_transport_step_kernel_soa(
    const uint32_t rank_cell_offset,
    uint32_t* cell_ID, uint32_t* group, unsigned char* descriptors,
    std::array<double, 3>* pos, std::array<double, 3>* angle,
    double* E, const double* E0, double* life_dx, RNG* rng_array,
    const Cell* cells,
    Cell_Tally* cell_tallies,
    const uint32_t* active_indices,
    uint32_t active_count,
    GPUEventType* final_event_types,
    unsigned int* event_counters,
    uint32_t* event_queue_positions,
    const double* sigma_s, const double* sigma_a, const double* f,
    const double* total_sigma_s, const uint32_t* local_cell_indices
)
  {

    uint32_t active_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (active_idx >= active_count) return;

    uint32_t photon_idx = active_indices[active_idx];
    RNG& rng = rng_array[photon_idx];

    // --- 1. calculate Event Distances

    uint32_t local_cell_index = local_cell_indices[active_idx];
    const Cell* cell = &cells[local_cell_index];

    const double sigma_a_val = sigma_a[active_idx];
    const double f_val = f[active_idx];
    const double total_sigma_s_val = total_sigma_s[active_idx];

    const double dist_to_scatter = (total_sigma_s_val > 0.0) ?
      -log(rng.generate_random_number()) / total_sigma_s_val : 1.0e100;

    uint32_t surface_cross = 0;
    const double dist_to_boundary = cell->get_distance_to_boundary(
        pos[photon_idx], angle[photon_idx], surface_cross);
    const double dist_to_census = life_dx[photon_idx];

    double distance = dist_to_scatter;
    GPUEventType event_type = GPU_SCATTER;

    if (dist_to_boundary < distance) {
      distance = dist_to_boundary;
      event_type = GPU_BOUNDARY;
    }
    if (dist_to_census < distance) {
      distance = dist_to_census;
      event_type = GPU_CENSUS;
    }

    // --- 2. Update state tally

    const double current_E = E[photon_idx];
    const double absorbed_E = current_E * (1.0 - exp(-sigma_a_val * f_val * distance));
    const double track_E_contrib = (sigma_a_val > 0.0 && f_val > 0.0) ? absorbed_E / (sigma_a_val * f_val) : 0.0;



    // Update photon state
    double next_E = current_E - absorbed_E;
    E[photon_idx] = next_E;
    pos[photon_idx][0] += angle[photon_idx][0] * distance;
    pos[photon_idx][1] += angle[photon_idx][1] * distance;
    pos[photon_idx][2] += angle[photon_idx][2] * distance;
    life_dx[photon_idx] -= distance;


    // Check energy cutoff
    if (next_E / E0[photon_idx] < Constants::cutoff_fraction) {
        // atomicAdd(&cell_tallies[local_cell_index].abs_E, next_E); // Tally remaining E
        E[photon_idx] = 0.0;
        event_type = GPU_KILLED;
        descriptors[photon_idx] = static_cast<unsigned char>(Constants::KILLED); // Set final descriptor
    }

    final_event_types[active_idx] = event_type;

    // unsigned int cell_mask = __match_any_sync(warp_mask(), local_cell_index);

    // double warp_total_abs_E = warp_reduce_sum(absorbed_E + (event_type == GPU_KILLED ? next_E : 0.0));
    // double warp_total_track_E = warp_reduce_sum(track_E_contrib);

    warp_atomic_add(&cell_tallies[local_cell_index].abs_E, local_cell_index, absorbed_E + (event_type == GPU_KILLED ? next_E : 0.0));
    warp_atomic_add(&cell_tallies[local_cell_index].track_E, local_cell_index, track_E_contrib);

    // // if (lane_id == (__ffs(cell_mask) -1)) {
    // if (lane_id == 0) {
    //   atomicAdd(&cell_tallies[local_cell_index].abs_E, warp_total_abs_E);
    //   atomicAdd(&cell_tallies[local_cell_index].track_E, warp_total_track_E);
    // }

    // atomicAdd(&cell_tallies[local_cell_index].abs_E, absorbed_E + (event_type == GPU_KILLED ? next_E : 0.0));
    // atomicAdd(&cell_tallies[local_cell_index].track_E, track_E_contrib);

    //  increment the counter for the determined event type and get queue position
    // unsigned int queue_idx = 0;
    // switch(event_type) {
    //     case GPU_SCATTER:  queue_idx = 0; break;
    //     case GPU_BOUNDARY: queue_idx = 1; break;
    //     case GPU_CENSUS:   queue_idx = 2; break;
    //     case GPU_KILLED:   queue_idx = 3; break;
    //     default: break;
    // }
    // event_queue_positions[active_idx] = atomicAdd(&event_counters[queue_idx], 1);
    unsigned int position = 0;
    warp_atomic_inc_ballot(&event_counters[0], event_type == GPU_SCATTER, position);
    warp_atomic_inc_ballot(&event_counters[1], event_type == GPU_BOUNDARY, position);
    warp_atomic_inc_ballot(&event_counters[2], event_type == GPU_CENSUS, position);
    warp_atomic_inc_ballot(&event_counters[3], event_type == GPU_KILLED, position);
    event_queue_positions[active_idx] = position;
  }

GPU_KERNEL void partition_photons_kernel_soa(
    const uint32_t* active_indices,
    uint32_t active_count,
    const GPUEventType* final_event_types,
    const uint32_t* event_queue_positions,
    EventInfo* scatter_info,
    EventInfo* boundary_info,
    EventInfo* census_info,
    EventInfo* killed_info)
{
    uint32_t active_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (active_idx >= active_count) return;

    EventInfo info = {active_indices[active_idx], active_idx};
    GPUEventType event_type = final_event_types[active_idx];
    uint32_t queue_pos = event_queue_positions[active_idx];

    switch(event_type) {
        case GPU_SCATTER:  scatter_info[queue_pos] = info; break;
        case GPU_BOUNDARY: boundary_info[queue_pos] = info; break;
        case GPU_CENSUS:   census_info[queue_pos] = info; break;
        case GPU_KILLED:   killed_info[queue_pos] = info; break;
        default: break;
    }
}

GPU_KERNEL void process_scatter_kernel_soa(
    uint32_t* group, unsigned char* descriptors, std::array<double, 3>* angle, RNG* rng_array,
    const uint32_t* cell_ID,
    const Cell* cells,
    const EmissionGroupData* emission_groups,
    const EventInfo* scatter_info,
    const unsigned int* event_counters,
    const uint32_t rank_cell_offset,
    const double* sigma_s, const double* sigma_a, const double* f,
    const double* total_sigma_s, const uint32_t* local_cell_indices)
{
    uint32_t scatter_count = event_counters[0];
    uint32_t scatter_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (scatter_idx >= scatter_count) return;

    const EventInfo &info = scatter_info[scatter_idx];
    uint32_t photon_idx = info.photon_idx;
    uint32_t active_idx = info.active_idx;
    RNG& rng = rng_array[photon_idx];
    uint32_t local_cell_index = local_cell_indices[active_idx];
    const EmissionGroupData* emission_data = &emission_groups[local_cell_index];

    const double sigma_s_val = sigma_s[active_idx];
    const double total_sigma_s_val = total_sigma_s[active_idx];

    angle[photon_idx] = get_uniform_angle(rng); // Update angle array
    if (total_sigma_s_val > 0.0 && rng.generate_random_number() > (sigma_s_val / total_sigma_s_val)) {
        group[photon_idx] = sample_emission_group(rng, *emission_data); // Update group array
    }
    descriptors[photon_idx] = static_cast<unsigned char>(Constants::SCATTER); // Update descriptor array
}


GPU_KERNEL void process_boundary_kernel_soa(
    uint32_t* cell_ID, unsigned char* descriptors, std::array<double, 3>* angle,
    const std::array<double, 3>* pos,
    const Cell* cells,
    const EventInfo* boundary_info,
    const unsigned int* event_counters,
    const uint32_t rank_cell_offset,
    const double* sigma_s, const double* sigma_a, const double* f,
    const double* total_sigma_s, const uint32_t* local_cell_indices)
{
    uint32_t boundary_count = event_counters[1];
    uint32_t boundary_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (boundary_idx >= boundary_count) return;

    const EventInfo &info = boundary_info[boundary_idx];
    uint32_t photon_idx = info.photon_idx;
    uint32_t active_idx = info.active_idx;
    // RNG& rng = rng_array[photon_idx];
    uint32_t local_cell_index = local_cell_indices[active_idx];
    const Cell* cell = &cells[local_cell_index];
    // const EmissionGroupData* emission_data = &emission_groups[local_cell_index];

    uint32_t surface_cross = 0;
    cell->get_distance_to_boundary(pos[photon_idx], angle[photon_idx], surface_cross);

    auto boundary_event = cell->get_bc(surface_cross);
    uint32_t next_cell_id = cell->get_next_cell(surface_cross);
    if (boundary_event == Constants::ELEMENT) {
        cell_ID[photon_idx] = next_cell_id;
        descriptors[photon_idx] = static_cast<unsigned char>(Constants::BOUND);
    } else if (boundary_event == Constants::PROCESSOR) {
        cell_ID[photon_idx] = next_cell_id;
        descriptors[photon_idx] = static_cast<unsigned char>(Constants::PASS);
    } else if (boundary_event == Constants::VACUUM || boundary_event == Constants::SOURCE) {
        descriptors[photon_idx] = static_cast<unsigned char>(Constants::EXIT);
    } else { // REFLECT
        int reflect_dim = surface_cross / 2;
        angle[photon_idx][reflect_dim] = -angle[photon_idx][reflect_dim];
        descriptors[photon_idx] = static_cast<unsigned char>(Constants::BOUND);
    }
}

GPU_KERNEL void process_census_kernel_soa(
    unsigned char* descriptors,
    const EventInfo* census_info,
    const unsigned int* event_counters)
{
    uint32_t census_count = event_counters[2];
    uint32_t census_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (census_idx >= census_count) return;

    const EventInfo &info = census_info[census_idx];
    uint32_t photon_idx = info.photon_idx;
    descriptors[photon_idx] = static_cast<unsigned char>(Constants::CENSUS);
}

GPU_KERNEL void compact_active_list_kernel_soa(
    const unsigned char* descriptors,
    const uint32_t* current_active_indices,
    uint32_t current_active_count,
    uint32_t* next_active_indices,      // Output active list
    unsigned int* next_active_count_atomic) // Atomic counter for the new size
{
    uint32_t active_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (active_idx >= current_active_count) return;

    uint32_t photon_idx = current_active_indices[active_idx];
    unsigned char desc = descriptors[photon_idx];

    if (desc == static_cast<unsigned char>(Constants::BOUND) ||
        desc == static_cast<unsigned char>(Constants::SCATTER))
    {
        uint32_t write_pos = atomicAdd(next_active_count_atomic, 1);
        next_active_indices[write_pos] = photon_idx;
    }
}

//----------------------------------------------------------------------------//
// GPU Transport Function (SoA) - Event Based                                 //
//----------------------------------------------------------------------------//
void gpu_event_transport_photons(const uint32_t rank_cell_offset,
    PhotonArray &cpu_photons, const Cell *device_cells_ptr,
    std::vector<Cell_Tally> &cpu_cell_tallies,
    const std::vector<EmissionGroupData>& emission_groups) // Pass host emission data
{
  Timer t_transport;
  t_transport.start_timer("soa_gpu_event_transport_photons");
  uint32_t n_photons = static_cast<uint32_t>(cpu_photons.size());
   if (n_photons == 0) return;

  size_t n_cells = cpu_cell_tallies.size();
  size_t n_emission_groups = emission_groups.size();

  cudaError_t err;

  // --- Allocate GPU Memory ---
  // SoA Photon Data Arrays
  uint32_t* d_cell_ID;
  uint32_t* d_group;
  uint32_t* d_source_type;
  unsigned char* d_descriptors;
  std::array<double, 3>* d_pos;
  std::array<double, 3>* d_angle;
  double* d_E;
  double* d_E0;
  double* d_life_dx;
  RNG* d_rng;
  Cell_Tally* d_cell_tallies;
  EmissionGroupData* d_emission_groups;
  uint32_t *d_active_indices_1;
  uint32_t *d_active_indices_2;
  GPUEventType *d_final_event_types;
  double *d_event_distances;
  unsigned int *d_event_counters;
  uint32_t *d_event_queue_positions;
  EventInfo *d_scatter_info;
  EventInfo *d_boundary_info;
  EventInfo *d_census_info;
  EventInfo *d_killed_info;
  double *d_sigma_s;
  double *d_sigma_a;
  double *d_f;
  double *d_total_sigma_s;
  uint32_t *d_local_cell_indices;
  unsigned int *d_next_active_count_atomic;

  // Malloc SoA arrays
  err = cudaMalloc((void**)&d_cell_ID, n_photons * sizeof(uint32_t)); Insist(!err, "SoA GPU malloc failed: cell_ID");
  err = cudaMalloc((void**)&d_group, n_photons * sizeof(uint32_t)); Insist(!err, "SoA GPU malloc failed: group");
  err = cudaMalloc((void**)&d_source_type, n_photons * sizeof(uint32_t)); Insist(!err, "SoA GPU malloc failed: source_type");
  err = cudaMalloc((void**)&d_descriptors, n_photons * sizeof(unsigned char)); Insist(!err, "SoA GPU malloc failed: descriptors");
  err = cudaMalloc((void**)&d_pos, n_photons * sizeof(std::array<double, 3>)); Insist(!err, "SoA GPU malloc failed: pos");
  err = cudaMalloc((void**)&d_angle, n_photons * sizeof(std::array<double, 3>)); Insist(!err, "SoA GPU malloc failed: angle");
  err = cudaMalloc((void**)&d_E, n_photons * sizeof(double)); Insist(!err, "SoA GPU malloc failed: E");
  err = cudaMalloc((void**)&d_E0, n_photons * sizeof(double)); Insist(!err, "SoA GPU malloc failed: E0");
  err = cudaMalloc((void**)&d_life_dx, n_photons * sizeof(double)); Insist(!err, "SoA GPU malloc failed: life_dx");
  err = cudaMalloc((void**)&d_rng, n_photons * sizeof(RNG)); Insist(!err, "SoA GPU malloc failed: rng");

  // Malloc other data
  err = cudaMalloc((void**)&d_cell_tallies, n_cells * sizeof(Cell_Tally)); Insist(!err, "SoA GPU malloc failed: cell_tallies");
  err = cudaMalloc((void**)&d_emission_groups, n_emission_groups * sizeof(EmissionGroupData)); Insist(!err, "SoA GPU malloc failed: emission_groups");

  // Malloc event processing data
  err = cudaMalloc((void **)&d_active_indices_1, sizeof(uint32_t) * n_photons); Insist(!err, "GPU SoA Malloc: d_active_indices_1");
  err = cudaMalloc((void **)&d_active_indices_2, sizeof(uint32_t) * n_photons); Insist(!err, "GPU SoA Malloc: d_active_indices_2");
  err = cudaMalloc((void **)&d_final_event_types, sizeof(GPUEventType) * n_photons); Insist(!err, "GPU SoA Malloc: d_event_types_1");
  err = cudaMalloc((void **)&d_event_distances, sizeof(double) * n_photons); Insist(!err, "GPU SoA Malloc: d_event_distances");
  err = cudaMalloc((void **)&d_event_queue_positions, sizeof(uint32_t) * n_photons); Insist(!err, "GPU SoA Malloc: d_event_queue_positions");
  const int NUM_EVENT_TYPES = 4;
  err = cudaMalloc((void **)&d_event_counters, sizeof(unsigned int) * NUM_EVENT_TYPES); Insist(!err, "GPU SoA Malloc: d_event_counters");
  err = cudaMalloc((void **)&d_next_active_count_atomic, sizeof(unsigned int)); Insist(!err, "GPU SoA Malloc: d_next_active_count_atomic");
  err = cudaMalloc((void **)&d_scatter_info, sizeof(EventInfo) * n_photons); Insist(!err, "GPU SoA Malloc: d_scatter_indices");
  err = cudaMalloc((void **)&d_boundary_info, sizeof(EventInfo) * n_photons); Insist(!err, "GPU SoA Malloc: d_boundary_indices");
  err = cudaMalloc((void **)&d_census_info, sizeof(EventInfo) * n_photons); Insist(!err, "GPU SoA Malloc: d_census_indices");
  err = cudaMalloc((void **)&d_killed_info, sizeof(EventInfo) * n_photons); Insist(!err, "GPU SoA Malloc: d_killed_indices");
  err = cudaMalloc((void **)&d_sigma_s, sizeof(double) * n_photons); Insist(!err, "GPU SoA Malloc: d_event_distances");
  err = cudaMalloc((void **)&d_sigma_a, sizeof(double) * n_photons); Insist(!err, "GPU SoA Malloc: d_event_distances");
  err = cudaMalloc((void **)&d_total_sigma_s, sizeof(double) * n_photons); Insist(!err, "GPU SoA Malloc: d_event_distances");
  err = cudaMalloc((void **)&d_f, sizeof(double) * n_photons); Insist(!err, "GPU SoA Malloc: d_event_distances");
  err = cudaMalloc((void **)&d_local_cell_indices, sizeof(uint32_t) * n_photons); Insist(!err, "GPU SoA Malloc: d_event_distances");

  // --- Copy Initial Data H2D ---
  // Copy SoA arrays
  err = cudaMemcpy(d_cell_ID, cpu_photons.cell_ID.data(), n_photons * sizeof(uint32_t), cudaMemcpyHostToDevice); Insist(!err, "SoA GPU copy failed: cell_ID");
  err = cudaMemcpy(d_group, cpu_photons.group.data(), n_photons * sizeof(uint32_t), cudaMemcpyHostToDevice); Insist(!err, "SoA GPU copy failed: group");
  err = cudaMemcpy(d_source_type, cpu_photons.source_type.data(), n_photons * sizeof(uint32_t), cudaMemcpyHostToDevice); Insist(!err, "SoA GPU copy failed: source_type");
  err = cudaMemcpy(d_descriptors, cpu_photons.descriptors.data(), n_photons * sizeof(unsigned char), cudaMemcpyHostToDevice); Insist(!err, "SoA GPU copy failed: descriptors");
  err = cudaMemcpy(d_pos, cpu_photons.pos.data(), n_photons * sizeof(std::array<double, 3>), cudaMemcpyHostToDevice); Insist(!err, "SoA GPU copy failed: pos");
  err = cudaMemcpy(d_angle, cpu_photons.angle.data(), n_photons * sizeof(std::array<double, 3>), cudaMemcpyHostToDevice); Insist(!err, "SoA GPU copy failed: angle");
  err = cudaMemcpy(d_E, cpu_photons.E.data(), n_photons * sizeof(double), cudaMemcpyHostToDevice); Insist(!err, "SoA GPU copy failed: E");
  err = cudaMemcpy(d_E0, cpu_photons.E0.data(), n_photons * sizeof(double), cudaMemcpyHostToDevice); Insist(!err, "SoA GPU copy failed: E0");
  err = cudaMemcpy(d_life_dx, cpu_photons.life_dx.data(), n_photons * sizeof(double), cudaMemcpyHostToDevice); Insist(!err, "SoA GPU copy failed: life_dx");
  err = cudaMemcpy(d_rng, cpu_photons.rng.data(), n_photons * sizeof(RNG), cudaMemcpyHostToDevice); Insist(!err, "SoA GPU copy failed: rng");
  err = cudaMemcpy(d_cell_tallies, cpu_cell_tallies.data(), n_cells * sizeof(Cell_Tally), cudaMemcpyHostToDevice); Insist(!err, "SoA GPU copy failed: cell_tallies");
  err = cudaMemcpy(d_emission_groups, emission_groups.data(), n_emission_groups * sizeof(EmissionGroupData), cudaMemcpyHostToDevice); Insist(!err, "SoA GPU copy failed: emission_groups");
  std::vector<uint32_t> h_initial_indices(n_photons);
  std::iota(h_initial_indices.begin(), h_initial_indices.end(), 0);
  err = cudaMemcpy(d_active_indices_1, h_initial_indices.data(), sizeof(uint32_t) * n_photons, cudaMemcpyHostToDevice); Insist(!err, "GPU SoA Memcpy H2D: d_active_indices_1");
  std::vector<unsigned int> h_zero_counters(NUM_EVENT_TYPES, 0);
  err = cudaMemcpy(d_event_counters, h_zero_counters.data(), sizeof(unsigned int) * NUM_EVENT_TYPES, cudaMemcpyHostToDevice); Insist(!err, "GPU SoA Memcpy H2D: d_event_counters");


  // --- Event Loop ---
  uint32_t current_active_count = n_photons;
  uint32_t* d_current_active_indices = d_active_indices_1;
  uint32_t* d_next_active_indices = d_active_indices_2;
  // GPUEventType* d_current_event_types = d_final_event_types;

  int n_threads = Constants::n_threads_per_block;

  t_transport.start_timer("soa kernel");
  while (current_active_count > 0) {
    int n_blocks = (current_active_count + n_threads - 1) / n_threads;

    // 1. Reset event counters
    reset_atomic_counters_kernel<<<(NUM_EVENT_TYPES + n_threads -1)/n_threads, n_threads>>>(d_event_counters, NUM_EVENT_TYPES);
    // err = cudaGetLastError(); Insist(!err, "GPU SoA Kernel Launch Error: reset_atomic_counters");

    precompute_data_gpu<<<n_blocks, n_threads>>>(
      rank_cell_offset, d_current_active_indices, current_active_count,
      d_group, d_cell_ID, device_cells_ptr, d_sigma_s, d_sigma_a, d_f, d_total_sigma_s, d_local_cell_indices);

    // 2. Calculate distances and initial event types, update state, tally, check kills, classify, and get queue positions
    process_transport_step_kernel_soa<<<n_blocks, n_threads>>>(
        rank_cell_offset,
        d_cell_ID, d_group, d_descriptors, d_pos, d_angle, d_E, d_E0, d_life_dx, // Photon data
        d_rng, device_cells_ptr, d_cell_tallies,
        d_current_active_indices, current_active_count,
        d_final_event_types, d_event_counters, d_event_queue_positions,
        d_sigma_s, d_sigma_a, d_f,
        d_total_sigma_s, d_local_cell_indices);
    // err = cudaGetLastError(); Insist(!err, "GPU SoA Kernel Launch Error: update_state_tally_classify");

    // 3. Place original indices into event-specific lists
    partition_photons_kernel_soa<<<n_blocks, n_threads>>>(
        d_current_active_indices, current_active_count,
        d_final_event_types, d_event_queue_positions,
        d_scatter_info, d_boundary_info, d_census_info, d_killed_info);
    // err = cudaGetLastError(); Insist(!err, "GPU SoA Kernel Launch Error: partition_photons");

    // 4. Launch kernels, process events
    int max_blocks = (n_photons + n_threads - 1) / n_threads;
    // if (scatter_count > 0) {
    //     int scatter_blocks = (scatter_count + n_threads - 1) / n_threads;
    process_scatter_kernel_soa<<<max_blocks, n_threads>>>(
        d_group, d_descriptors, d_angle, d_rng, d_cell_ID, // Photon data
        device_cells_ptr, d_emission_groups, // Other data
        d_scatter_info, d_event_counters, rank_cell_offset,
        d_sigma_s, d_sigma_a, d_f,
        d_total_sigma_s, d_local_cell_indices);
    // err = cudaGetLastError(); Insist(!err, "GPU SoA Kernel Launch Error: process_scatter");
    // }
    // if (boundary_count > 0) {
    //     int boundary_blocks = (boundary_count + n_threads - 1) / n_threads;
    process_boundary_kernel_soa<<<max_blocks, n_threads>>>(
        d_cell_ID, d_descriptors, d_angle, d_pos, // Photon data
        device_cells_ptr, // Other data
        d_boundary_info, d_event_counters, rank_cell_offset,
        d_sigma_s, d_sigma_a, d_f,
        d_total_sigma_s, d_local_cell_indices);
    // err = cudaGetLastError(); Insist(!err, "GPU SoA Kernel Launch Error: process_boundary");
    // }
    // if (census_count > 0) {
    //     int census_blocks = (census_count + n_threads - 1) / n_threads;
    process_census_kernel_soa<<<max_blocks, n_threads>>>(
        d_descriptors, d_census_info, d_event_counters);
    // err = cudaGetLastError(); Insist(!err, "GPU SoA Kernel Launch Error: process_census");
    // }

    // 5. Compact active list for next iteration
    err = cudaMemset(d_next_active_count_atomic, 0, sizeof(unsigned int)); Insist(!err, "GPU SoA Memset Error: d_next_active_count_atomic");
    compact_active_list_kernel_soa<<<n_blocks, n_threads>>>(
        d_descriptors, d_current_active_indices, current_active_count,
        d_next_active_indices, d_next_active_count_atomic);
    // err = cudaGetLastError(); Insist(!err, "GPU SoA Kernel Launch Error: compact_active_list");

    // Get the new active count
    unsigned int h_next_active_count = 0;
    err = cudaMemcpy(&h_next_active_count, d_next_active_count_atomic, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    // Insist(!err, "GPU SoA Memcpy D2H Error: d_next_active_count_atomic");
    current_active_count = h_next_active_count;

    // Swap active list pointers
    std::swap(d_current_active_indices, d_next_active_indices);

  } // End while(current_active_count > 0)


  std::cout<<"finished while"<<std::endl;
  auto sync_err = cudaDeviceSynchronize();
  t_transport.stop_timer("soa kernel");
  Insist(!sync_err, "error in synchronize");
  std::cout<<"finished while post sync"<<std::endl;

  // --- Copy Results Back ---
  err = cudaMemcpy(cpu_photons.cell_ID.data(), d_cell_ID, n_photons * sizeof(uint32_t), cudaMemcpyDeviceToHost); Insist(!err, "SoA GPU copy back failed: cell_ID");
  err = cudaMemcpy(cpu_photons.group.data(), d_group, n_photons * sizeof(uint32_t), cudaMemcpyDeviceToHost); Insist(!err, "SoA GPU copy back failed: group");
  err = cudaMemcpy(cpu_photons.source_type.data(), d_source_type, n_photons * sizeof(uint32_t), cudaMemcpyDeviceToHost); Insist(!err, "SoA GPU copy back failed: source_type");
  err = cudaMemcpy(cpu_photons.descriptors.data(), d_descriptors, n_photons * sizeof(unsigned char), cudaMemcpyDeviceToHost); Insist(!err, "SoA GPU copy back failed: descriptors");
  err = cudaMemcpy(cpu_photons.pos.data(), d_pos, n_photons * sizeof(std::array<double, 3>), cudaMemcpyDeviceToHost); Insist(!err, "SoA GPU copy back failed: pos");
  err = cudaMemcpy(cpu_photons.angle.data(), d_angle, n_photons * sizeof(std::array<double, 3>), cudaMemcpyDeviceToHost); Insist(!err, "SoA GPU copy back failed: angle");
  err = cudaMemcpy(cpu_photons.E.data(), d_E, n_photons * sizeof(double), cudaMemcpyDeviceToHost); Insist(!err, "SoA GPU copy back failed: E");
  err = cudaMemcpy(cpu_photons.E0.data(), d_E0, n_photons * sizeof(double), cudaMemcpyDeviceToHost); Insist(!err, "SoA GPU copy back failed: E0");
  err = cudaMemcpy(cpu_photons.life_dx.data(), d_life_dx, n_photons * sizeof(double), cudaMemcpyDeviceToHost); Insist(!err, "SoA GPU copy back failed: life_dx");
  err = cudaMemcpy(cpu_photons.rng.data(), d_rng, n_photons * sizeof(RNG), cudaMemcpyDeviceToHost); Insist(!err, "SoA GPU copy back failed: rng");
  err = cudaMemcpy(cpu_cell_tallies.data(), d_cell_tallies, n_cells * sizeof(Cell_Tally), cudaMemcpyDeviceToHost); Insist(!err, "SoA GPU copy back failed: cell_tallies");

  // --- Free GPU Memory ---
  auto free_err = cudaFree(d_cell_ID);
  if (free_err) std::cout<<"Error freeing d_cell_ID"<<std::endl;
  free_err = cudaFree(d_group);
  if (free_err) std::cout<<"Error freeing d_group"<<std::endl;
  free_err = cudaFree(d_source_type);
  if (free_err) std::cout<<"Error freeing d_source_type"<<std::endl;
  free_err = cudaFree(d_descriptors);
  if (free_err) std::cout<<"Error freeing d_descriptors"<<std::endl;
  free_err = cudaFree(d_pos);
  if (free_err) std::cout<<"Error freeing d_pos"<<std::endl;
  free_err = cudaFree(d_angle);
  if (free_err) std::cout<<"Error freeing d_angle"<<std::endl;
  free_err = cudaFree(d_E);
  if (free_err) std::cout<<"Error freeing d_E"<<std::endl;
  free_err = cudaFree(d_E0);
  if (free_err) std::cout<<"Error freeing d_E0"<<std::endl;
  free_err = cudaFree(d_life_dx);
  if (free_err) std::cout<<"Error freeing d_life_dx"<<std::endl;
  free_err = cudaFree(d_rng);
  if (free_err) std::cout<<"Error freeing d_rng"<<std::endl;
  free_err = cudaFree(d_cell_tallies);
  if (free_err) std::cout<<"Error freeing d_cell_tallies"<<std::endl;
  free_err = cudaFree(d_emission_groups);
  if (free_err) std::cout<<"Error freeing d_emission_groups"<<std::endl;
  free_err = cudaFree(d_active_indices_1);
  if (free_err) std::cout<<"Error freeing d_active_indices_1"<<std::endl;
  free_err = cudaFree(d_active_indices_2);
  if (free_err) std::cout<<"Error freeing d_active_indices_2"<<std::endl;
  free_err = cudaFree(d_final_event_types);
  if (free_err) std::cout<<"Error freeing d_final_event_types"<<std::endl;
  free_err = cudaFree(d_event_distances);
  if (free_err) std::cout<<"Error freeing d_event_distances"<<std::endl;
  free_err = cudaFree(d_event_counters);
  if (free_err) std::cout<<"Error freeing d_event_counters"<<std::endl;
  free_err = cudaFree(d_event_queue_positions);
  if (free_err) std::cout<<"Error freeing d_event_queue_positions"<<std::endl;
  free_err = cudaFree(d_scatter_info);
  if (free_err) std::cout<<"Error freeing d_scatter_info"<<std::endl;
  free_err = cudaFree(d_boundary_info);
  if (free_err) std::cout<<"Error freeing d_boundary_info"<<std::endl;
  free_err = cudaFree(d_census_info);
  if (free_err) std::cout<<"Error freeing d_census_info"<<std::endl;
  free_err = cudaFree(d_killed_info);
  if (free_err) std::cout<<"Error freeing d_killed_info"<<std::endl;
  free_err = cudaFree(d_sigma_s);
  if (free_err) std::cout<<"Error freeing d_sigma_s"<<std::endl;
  free_err = cudaFree(d_sigma_a);
  if (free_err) std::cout<<"Error freeing d_sigma_a"<<std::endl;
  free_err = cudaFree(d_total_sigma_s);
  if (free_err) std::cout<<"Error freeing d_total_sigma_s"<<std::endl;
  free_err = cudaFree(d_f);
  if (free_err) std::cout<<"Error freeing d_f"<<std::endl;
  free_err = cudaFree(d_next_active_count_atomic);
  if (free_err) std::cout<<"Error freeing d_next_avtive_count_atomic"<<std::endl;
  free_err = cudaFree(d_local_cell_indices);
  if (free_err) std::cout<<"Error freeing d_local_cell_indices"<<std::endl;
  std::cout<<"about to exit event transport loop"<<std::endl;
  t_transport.stop_timer("soa_gpu_event_transport_photons");
}


//----------------------------------------------------------------------------//
// AoS GPU Event Kernels                                                      //
//----------------------------------------------------------------------------//

__global__ void calculate_events_kernel_aos(
    const uint32_t rank_cell_offset,
    Photon* all_photons,
    const Cell* cells,
    const uint32_t* active_indices,
    uint32_t active_count,
    GPUEventType* event_types,      // Event type for each active photon
    double* event_distances)        // Event distance for each active photon
{
    uint32_t active_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (active_idx >= active_count) return;

    uint32_t photon_idx = active_indices[active_idx];
    Photon& phtn = all_photons[photon_idx];
    RNG& rng = phtn.get_rng();

    uint32_t local_cell_index = phtn.get_cell() - rank_cell_offset;
    // Add bounds check if necessary, assume valid for now
    const Cell* cell = &cells[local_cell_index];

    const double sigma_s = cell->get_op_s(phtn.get_group());
    const double sigma_a = cell->get_op_a(phtn.get_group());
    const double f = cell->get_f();
    const double total_sigma_s = (1.0 - f) * sigma_a + sigma_s;

    const double dist_to_scatter = (total_sigma_s > 0.0) ?
      -log(rng.generate_random_number()) / total_sigma_s : 1.0e100;

    uint32_t surface_cross = 0; // Output param, value not needed here
    const double dist_to_boundary = cell->get_distance_to_boundary(
        phtn.get_position(), phtn.get_angle(), surface_cross);
    const double dist_to_census = phtn.get_distance_remaining();

    const double dist = min(dist_to_scatter, min(dist_to_boundary, dist_to_census));
    event_distances[active_idx] = dist;

    if (dist == dist_to_scatter)       event_types[active_idx] = GPU_SCATTER;
    else if (dist == dist_to_boundary) event_types[active_idx] = GPU_BOUNDARY;
    else                               event_types[active_idx] = GPU_CENSUS;
}


__global__ void update_state_tally_and_classify_kernel_aos(
    const uint32_t rank_cell_offset,
    Photon* all_photons,
    const Cell* cells,
    Cell_Tally* cell_tallies,
    const uint32_t* active_indices,
    uint32_t active_count,
    const GPUEventType* event_types_in, // Event type from calculate kernel
    const double* event_distances,
    GPUEventType* event_types_out,      // Final event type
    unsigned int* event_counters,       // Atomic counters for scatter, boundary, census, kill
    uint32_t* event_queue_positions)    //  position in the specific event queue
{
    uint32_t active_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (active_idx >= active_count) return;

    uint32_t photon_idx = active_indices[active_idx];
    Photon& phtn = all_photons[photon_idx];
    double distance = event_distances[active_idx];
    GPUEventType event_type = event_types_in[active_idx]; // Initial event type

    uint32_t local_cell_index = phtn.get_cell() - rank_cell_offset;
    const Cell* cell = &cells[local_cell_index];
    const double sigma_a = cell->get_op_a(phtn.get_group());
    const double f = cell->get_f();

    // Calculate and tally absorbed energy
    const double absorbed_E = phtn.get_E() * (1.0 - exp(-sigma_a * f * distance));
    const double track_E_contrib = (sigma_a > 0.0 && f > 0.0) ? absorbed_E / (sigma_a * f) : 0.0;

    atomicAdd(&cell_tallies[local_cell_index].abs_E, absorbed_E);
    atomicAdd(&cell_tallies[local_cell_index].track_E, track_E_contrib);

    // Update photon state
    phtn.set_E(phtn.get_E() - absorbed_E);
    phtn.move(distance);

    // Check for energy kill
    if (phtn.below_cutoff(Constants::cutoff_fraction)) {
        atomicAdd(&cell_tallies[local_cell_index].abs_E, phtn.get_E()); // Tally remaining E
        phtn.set_E(0.0); // Zero out energy
        event_type = GPU_KILLED;
        phtn.set_descriptor(Constants::KILLED); // Set final descriptor
    }

    // Store final event type for partitioning
    event_types_out[active_idx] = event_type;

    //  increment the counter for the determined event type and get queue position
    // Map GPUEventType enum to counter index (e.g., SCATTER=0, BOUNDARY=1, CENSUS=2, KILLED=3)
    unsigned int queue_idx = 0; // Default or invalid
    switch(event_type) {
        case GPU_SCATTER:  queue_idx = 0; break;
        case GPU_BOUNDARY: queue_idx = 1; break;
        case GPU_CENSUS:   queue_idx = 2; break;
        case GPU_KILLED:   queue_idx = 3; break;
        default: break; // Should not happen for these types
    }
    //  increment the counter and store the previous value
    event_queue_positions[active_idx] = atomicAdd(&event_counters[queue_idx], 1);
}

__global__ void partition_photons_kernel_aos(
    const uint32_t* active_indices,       // List of active photon original indices
    uint32_t active_count,
    const GPUEventType* final_event_types,// Final event type for each active photon
    const uint32_t* event_queue_positions,// Position within the event queue
    uint32_t* scatter_indices,            // Original indices of scatter photons
    uint32_t* boundary_indices,           // Original indices of boundary photons
    uint32_t* census_indices,             // Original indices of census photons
    uint32_t* killed_indices)             // Original indices of killed photons
{
    uint32_t active_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (active_idx >= active_count) return;

    uint32_t photon_idx = active_indices[active_idx]; // Original photon index
    GPUEventType event_type = final_event_types[active_idx];
    uint32_t queue_pos = event_queue_positions[active_idx];

    // Write the original photon index into the correct event list
    switch(event_type) {
        case GPU_SCATTER:  scatter_indices[queue_pos] = photon_idx; break;
        case GPU_BOUNDARY: boundary_indices[queue_pos] = photon_idx; break;
        case GPU_CENSUS:   census_indices[queue_pos] = photon_idx; break;
        case GPU_KILLED:   killed_indices[queue_pos] = photon_idx; break;
        default: break; // Should not happen
    }
}


__global__ void process_scatter_kernel_aos(
    Photon* all_photons,
    const Cell* cells,
    const EmissionGroupData* emission_groups,
    const uint32_t* scatter_indices,
    uint32_t scatter_count,
    const uint32_t rank_cell_offset)
{
    uint32_t scatter_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (scatter_idx >= scatter_count) return;

    uint32_t photon_idx = scatter_indices[scatter_idx];
    Photon& phtn = all_photons[photon_idx];
    RNG& rng = phtn.get_rng();

    uint32_t local_cell_index = phtn.get_cell() - rank_cell_offset;
    const Cell* cell = &cells[local_cell_index];
    const EmissionGroupData* emission_data = &emission_groups[local_cell_index];

    const double sigma_s = cell->get_op_s(phtn.get_group());
    const double sigma_a = cell->get_op_a(phtn.get_group());
    const double f = cell->get_f();
    const double total_sigma_s = (1.0 - f) * sigma_a + sigma_s; // Recalculate needed data

    phtn.set_angle(get_uniform_angle(rng));
    if (total_sigma_s > 0.0 && rng.generate_random_number() > (sigma_s / total_sigma_s)) {
        phtn.set_group(sample_emission_group(rng, *emission_data));
    }
    phtn.set_descriptor(Constants::SCATTER); // Mark as scattered (still active for next round)
}

__global__ void process_boundary_kernel_aos(
    Photon* all_photons,
    const Cell* cells,
    const uint32_t* boundary_indices,
    uint32_t boundary_count,
    const uint32_t rank_cell_offset)
{
    uint32_t boundary_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (boundary_idx >= boundary_count) return;

    uint32_t photon_idx = boundary_indices[boundary_idx];
    Photon& phtn = all_photons[photon_idx];

    uint32_t local_cell_index = phtn.get_cell() - rank_cell_offset;
    const Cell* cell = &cells[local_cell_index];

    uint32_t surface_cross = 0;
    cell->get_distance_to_boundary(phtn.get_position(), phtn.get_angle(), surface_cross); // Recalculate surface

    auto boundary_event = cell->get_bc(surface_cross);

    if (boundary_event == Constants::ELEMENT) {
        phtn.set_cell(cell->get_next_cell(surface_cross));
        phtn.set_descriptor(Constants::BOUND); // Still active
    } else if (boundary_event == Constants::PROCESSOR) {
        phtn.set_cell(cell->get_next_cell(surface_cross)); // Set global ID for post-processing
        phtn.set_descriptor(Constants::PASS); // Inactive
    } else if (boundary_event == Constants::VACUUM || boundary_event == Constants::SOURCE) {
        phtn.set_descriptor(Constants::EXIT); // Inactive
    } else { // REFLECT
        phtn.reflect(surface_cross);
        phtn.set_descriptor(Constants::BOUND); // Still active
    }
}

__global__ void process_census_kernel_aos(
    Photon* all_photons,
    const uint32_t* census_indices,
    uint32_t census_count)
{
    uint32_t census_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (census_idx >= census_count) return;

    uint32_t photon_idx = census_indices[census_idx];
    all_photons[photon_idx].set_descriptor(Constants::CENSUS);
}

__global__ void compact_active_list_kernel_aos(
    const Photon* all_photons,          // Read descriptors
    const uint32_t* current_active_indices, // Input active list
    uint32_t current_active_count,
    uint32_t* next_active_indices,      // Output active list
    unsigned int* next_active_count_atomic) // Atomic counter for the new size
{
    uint32_t active_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (active_idx >= current_active_count) return;

    uint32_t photon_idx = current_active_indices[active_idx];
    Constants::event_type desc = all_photons[photon_idx].get_descriptor();

    // Check if the photon should remain active for the next iteration
    if (desc == Constants::BOUND || desc == Constants::SCATTER) {
        //  get the position in the output array and increment the counter
        uint32_t write_pos = atomicAdd(next_active_count_atomic, 1);
        // Write the original index to the new active list
        next_active_indices[write_pos] = photon_idx;
    }
}



//----------------------------------------------------------------------------//
// GPU Transport Function (AoS) - Event Based                                 //
//----------------------------------------------------------------------------//
void gpu_event_transport_photons(const uint32_t rank_cell_offset,
    std::vector<Photon> &cpu_photons, const Cell *device_cells_ptr,
    std::vector<Cell_Tally> &cpu_cell_tallies,
    const std::vector<EmissionGroupData>& emission_groups)
{
  Timer t_transport;
  t_transport.start_timer("aos_gpu_event_transport_photons");
  uint32_t n_photons = static_cast<uint32_t>(cpu_photons.size());
  if (n_photons == 0) return;

  size_t n_cells = cpu_cell_tallies.size();
  size_t n_emission_groups = emission_groups.size();
  if (n_cells != n_emission_groups) {
      std::cerr << "Error: Mismatch between cell tally count and emission group count." << std::endl;
      return;
  }

  cudaError_t err;

  // --- Allocate GPU Memory ---
  Photon *d_photons;
  Cell_Tally *d_cell_tallies;
  EmissionGroupData *d_emission_groups;
  uint32_t *d_active_indices_1, *d_active_indices_2; //  buffers for active list
  GPUEventType *d_event_types_1, *d_event_types_2; // for event types
  double *d_event_distances;
  unsigned int *d_event_counters; // Atomic counters (scatter, boundary, census, kill)
  uint32_t *d_event_queue_positions;
  uint32_t *d_scatter_indices, *d_boundary_indices, *d_census_indices, *d_killed_indices;
  unsigned int *d_next_active_count_atomic; // Single atomic counter for compaction

  // Photon data
  err = cudaMalloc((void **)&d_photons, sizeof(Photon) * n_photons); Insist(!err, "GPU AoS Malloc: d_photons");
  err = cudaMemcpy(d_photons, cpu_photons.data(), sizeof(Photon) * n_photons, cudaMemcpyHostToDevice); Insist(!err, "GPU AoS Memcpy H2D: d_photons");

  // Tallies
  err = cudaMalloc((void **)&d_cell_tallies, sizeof(Cell_Tally) * n_cells); Insist(!err, "GPU AoS Malloc: d_cell_tallies");
  err = cudaMemcpy(d_cell_tallies, cpu_cell_tallies.data(), sizeof(Cell_Tally) * n_cells, cudaMemcpyHostToDevice); Insist(!err, "GPU AoS Memcpy H2D: d_cell_tallies");

  // Emission Groups
  err = cudaMalloc((void **)&d_emission_groups, sizeof(EmissionGroupData) * n_emission_groups); Insist(!err, "GPU AoS Malloc: d_emission_groups");
  err = cudaMemcpy(d_emission_groups, emission_groups.data(), sizeof(EmissionGroupData) * n_emission_groups, cudaMemcpyHostToDevice); Insist(!err, "GPU AoS Memcpy H2D: d_emission_groups");

  // Active Indices (initialize with 0, 1, ..., n_photons-1 on host first)
  std::vector<uint32_t> h_initial_indices(n_photons);
  std::iota(h_initial_indices.begin(), h_initial_indices.end(), 0);
  err = cudaMalloc((void **)&d_active_indices_1, sizeof(uint32_t) * n_photons); Insist(!err, "GPU AoS Malloc: d_active_indices_1");
  err = cudaMalloc((void **)&d_active_indices_2, sizeof(uint32_t) * n_photons); Insist(!err, "GPU AoS Malloc: d_active_indices_2");
  err = cudaMemcpy(d_active_indices_1, h_initial_indices.data(), sizeof(uint32_t) * n_photons, cudaMemcpyHostToDevice); Insist(!err, "GPU AoS Memcpy H2D: d_active_indices_1");

  // Intermediate Event Data
  err = cudaMalloc((void **)&d_event_types_1, sizeof(GPUEventType) * n_photons); Insist(!err, "GPU AoS Malloc: d_event_types_1");
  err = cudaMalloc((void **)&d_event_types_2, sizeof(GPUEventType) * n_photons); Insist(!err, "GPU AoS Malloc: d_event_types_2");
  err = cudaMalloc((void **)&d_event_distances, sizeof(double) * n_photons); Insist(!err, "GPU AoS Malloc: d_event_distances");
  err = cudaMalloc((void **)&d_event_queue_positions, sizeof(uint32_t) * n_photons); Insist(!err, "GPU AoS Malloc: d_event_queue_positions");

  // Event Counters (host init to 0, then copy)
  const int NUM_EVENT_TYPES = 4; // Scatter, Boundary, Census, Kill
  std::vector<unsigned int> h_zero_counters(NUM_EVENT_TYPES, 0);
  err = cudaMalloc((void **)&d_event_counters, sizeof(unsigned int) * NUM_EVENT_TYPES); Insist(!err, "GPU AoS Malloc: d_event_counters");
  err = cudaMemcpy(d_event_counters, h_zero_counters.data(), sizeof(unsigned int) * NUM_EVENT_TYPES, cudaMemcpyHostToDevice); Insist(!err, "GPU AoS Memcpy H2D: d_event_counters");
  err = cudaMalloc((void **)&d_next_active_count_atomic, sizeof(unsigned int)); Insist(!err, "GPU AoS Malloc: d_next_active_count_atomic");

  // Event-Specific Index Lists
  err = cudaMalloc((void **)&d_scatter_indices, sizeof(uint32_t) * n_photons); Insist(!err, "GPU AoS Malloc: d_scatter_indices");
  err = cudaMalloc((void **)&d_boundary_indices, sizeof(uint32_t) * n_photons); Insist(!err, "GPU AoS Malloc: d_boundary_indices");
  err = cudaMalloc((void **)&d_census_indices, sizeof(uint32_t) * n_photons); Insist(!err, "GPU AoS Malloc: d_census_indices");
  err = cudaMalloc((void **)&d_killed_indices, sizeof(uint32_t) * n_photons); Insist(!err, "GPU AoS Malloc: d_killed_indices");

  // --- Event Loop ---
  uint32_t current_active_count = n_photons;
  uint32_t* d_current_active_indices = d_active_indices_1;
  uint32_t* d_next_active_indices = d_active_indices_2;
  GPUEventType* d_current_event_types = d_event_types_1; // Used for final type storage

  int n_threads = Constants::n_threads_per_block;

  t_transport.start_timer("aos kernel");
  while (current_active_count > 0) {
    int n_blocks = (current_active_count + n_threads - 1) / n_threads;

    // 1. Calculate distances and initial event types
    calculate_events_kernel_aos<<<n_blocks, n_threads>>>(
        rank_cell_offset, d_photons, device_cells_ptr,
        d_current_active_indices, current_active_count,
        d_current_event_types,
        d_event_distances);
    err = cudaGetLastError(); Insist(!err, "GPU AoS Kernel Launch Error: calculate_events");

    // 2. Reset event counters
    reset_atomic_counters_kernel<<<(NUM_EVENT_TYPES + n_threads -1)/n_threads, n_threads>>>(d_event_counters, NUM_EVENT_TYPES);
    err = cudaGetLastError(); Insist(!err, "GPU AoS Kernel Launch Error: reset_atomic_counters");

    // 3. Update state, tally, check kills, classify, and get queue positions
    update_state_tally_and_classify_kernel_aos<<<n_blocks, n_threads>>>(
        rank_cell_offset, d_photons, device_cells_ptr, d_cell_tallies,
        d_current_active_indices, current_active_count,
        d_current_event_types, // Input: initial types
        d_event_distances,
        d_current_event_types, // (overwrites initial)
        d_event_counters, d_event_queue_positions);
    err = cudaGetLastError(); Insist(!err, "GPU AoS Kernel Launch Error: update_state_tally_classify");

    // 4. Place original indices into event-specific lists
    partition_photons_kernel_aos<<<n_blocks, n_threads>>>(
        d_current_active_indices, current_active_count,
        d_current_event_types,
        d_event_queue_positions,
        d_scatter_indices, d_boundary_indices, d_census_indices, d_killed_indices);
    err = cudaGetLastError(); Insist(!err, "GPU AoS Kernel Launch Error: partition_photons");

    // 5. Get event counts (copy counters back to host)
    std::vector<unsigned int> h_event_counts(NUM_EVENT_TYPES);
    err = cudaMemcpy(h_event_counts.data(), d_event_counters, sizeof(unsigned int) * NUM_EVENT_TYPES, cudaMemcpyDeviceToHost);
    Insist(!err, "GPU AoS Memcpy D2H Error: d_event_counters");
    uint32_t scatter_count = h_event_counts[0];
    uint32_t boundary_count = h_event_counts[1];
    uint32_t census_count = h_event_counts[2];

    // 6. Process Events (launch kernels only if counts > 0)
    if (scatter_count > 0) {
        int scatter_blocks = (scatter_count + n_threads - 1) / n_threads;
        process_scatter_kernel_aos<<<scatter_blocks, n_threads>>>(
            d_photons, device_cells_ptr, d_emission_groups,
            d_scatter_indices, scatter_count, rank_cell_offset);
        err = cudaGetLastError(); Insist(!err, "GPU AoS Kernel Launch Error: process_scatter");
    }
    if (boundary_count > 0) {
        int boundary_blocks = (boundary_count + n_threads - 1) / n_threads;
        process_boundary_kernel_aos<<<boundary_blocks, n_threads>>>(
            d_photons, device_cells_ptr,
            d_boundary_indices, boundary_count, rank_cell_offset);
        err = cudaGetLastError(); Insist(!err, "GPU AoS Kernel Launch Error: process_boundary");
    }
    if (census_count > 0) {
        int census_blocks = (census_count + n_threads - 1) / n_threads;
        process_census_kernel_aos<<<census_blocks, n_threads>>>(
            d_photons, d_census_indices, census_count);
        err = cudaGetLastError(); Insist(!err, "GPU AoS Kernel Launch Error: process_census");
    }

    // 7. Compact active list for next iteration
    // Reset the atomic counter for the next active list size
    err = cudaMemset(d_next_active_count_atomic, 0, sizeof(unsigned int)); Insist(!err, "GPU AoS Memset Error: d_next_active_count_atomic");

    compact_active_list_kernel_aos<<<n_blocks, n_threads>>>(
        d_photons, d_current_active_indices, current_active_count,
        d_next_active_indices, d_next_active_count_atomic);
    err = cudaGetLastError(); Insist(!err, "GPU AoS Kernel Launch Error: compact_active_list");

    // Get the new active count
    // D2H
    unsigned int h_next_active_count = 0;
    err = cudaMemcpy(&h_next_active_count, d_next_active_count_atomic, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    Insist(!err, "GPU AoS Memcpy D2H Error: d_next_active_count_atomic");
    current_active_count = h_next_active_count;

    std::swap(d_current_active_indices, d_next_active_indices);
    // std::swap(d_current_event_types, d_next_event_types);

  } // End while(current_active_count > 0)

  auto sync_error = cudaDeviceSynchronize(); // Ensure all kernels are finished before copy back
  t_transport.stop_timer("aos kernel");
  Insist(!sync_error, "Error in synchronize");

  // --- Copy Results Back ---
  err = cudaMemcpy(cpu_photons.data(), d_photons, sizeof(Photon) * n_photons, cudaMemcpyDeviceToHost); Insist(!err, "GPU AoS Memcpy D2H Error: d_photons final");
  err = cudaMemcpy(cpu_cell_tallies.data(), d_cell_tallies, sizeof(Cell_Tally) * n_cells, cudaMemcpyDeviceToHost); Insist(!err, "GPU AoS Memcpy D2H Error: d_cell_tallies final");

  // --- Free GPU Memory ---
  auto free_err = cudaFree(d_photons);
  if (free_err) std::cout<<"Error freeing d_photons"<<std::endl;
  free_err = cudaFree(d_cell_tallies);
  if (free_err) std::cout<<"Error freeing d_cell_tallies"<<std::endl;
  free_err = cudaFree(d_emission_groups);
  if (free_err) std::cout<<"Error freeing d_emission_groups"<<std::endl;
  free_err = cudaFree(d_active_indices_1);
  if (free_err) std::cout<<"Error freeing d_active_indices_1"<<std::endl;
  free_err = cudaFree(d_active_indices_2);
  if (free_err) std::cout<<"Error freeing d_active_indices_2"<<std::endl;
  free_err = cudaFree(d_event_types_1);
  if (free_err) std::cout<<"Error freeing d_event_types_1"<<std::endl;
  free_err = cudaFree(d_event_types_2);
  if (free_err) std::cout<<"Error freeing d_event_types_2"<<std::endl;
  free_err = cudaFree(d_event_distances);
  if (free_err) std::cout<<"Error freeing d_event_distance"<<std::endl;
  free_err = cudaFree(d_event_counters);
  if (free_err) std::cout<<"Error freeing d_event_counters"<<std::endl;
  free_err = cudaFree(d_event_queue_positions);
  if (free_err) std::cout<<"Error freeing d_event_queue_positions"<<std::endl;
  free_err = cudaFree(d_scatter_indices);
  if (free_err) std::cout<<"Error freeing d_scatter_indices"<<std::endl;
  free_err = cudaFree(d_boundary_indices);
  if (free_err) std::cout<<"Error freeing d_boundary_indices"<<std::endl;
  free_err = cudaFree(d_census_indices);
  if (free_err) std::cout<<"Error freeing d_census_indices"<<std::endl;
  free_err = cudaFree(d_killed_indices);
  if (free_err) std::cout<<"Error freeing d_killed_indices"<<std::endl;
  free_err = cudaFree(d_next_active_count_atomic);
  if (free_err) std::cout<<"Error freeing d_next_active_count_atomic"<<std::endl;
  t_transport.stop_timer("aos_gpu_event_transport_photons");
}

#endif // USE_GPU

#endif // event_based_transport_h_
//----------------------------------------------------------------------------//
// end of event_based_transport.h
//----------------------------------------------------------------------------//
