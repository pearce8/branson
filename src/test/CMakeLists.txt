# Unit tests
cmake_minimum_required (VERSION 3.11)

#
# force mpiexec to be srun in cray environments
# can't run as root on a cray so don't try to match
# mpirun's allow-run-as-root and oversubscribe
#
message( STATUS "Looking to see if we are building in a Cray Environment...")
if( DEFINED ENV{CRAYPE_VERSION} )
  set(MPIEXEC_EXECUTABLE srun)
set( MPIEXEC_POSTFLAGS "--cpu-bind=none" CACHE STRING
  "extra mpirun flags (list)." FORCE)
  message("In Cray Environment, MPI executable is: ${MPIEXEC_EXECUTABLE}")
else()
# allow run as root
if( "$ENV{GITLAB_CI}" STREQUAL "true" OR "$ENV{TRAVIS}" STREQUAL "true")
  message("In CI mode, allow running mpiexec as root")
  set(runasroot "--allow-run-as-root --oversubscribe")
endif()
# set extra flags for mpiexec in testing
set( MPIEXEC_POSTFLAGS "-bind-to none ${runasroot}" CACHE STRING
  "extra mpirun flags (list)." FORCE)
endif()



string( REPLACE " " ";" mpiexec_postflags_list "${MPIEXEC_POSTFLAGS}" )

#------------------------------------------------------------------------------#
# Helper for building and registering tests.
#
# add_branson_test(
#   SOURCE  "foo.cc"
#   PE_LIST "1;4" )

function( add_branson_test )
  cmake_parse_arguments(
    abt
    ""
    "SOURCE"
    "PE_LIST"
    ${ARGV} )
  string( REPLACE ".cc" "_exe" build_target ${abt_SOURCE} )
  string( REPLACE ".cc" ""     ctest_name   ${abt_SOURCE} )

  # Build the test
  add_executable( ${build_target} ${abt_SOURCE} )
  target_include_directories( ${build_target} PRIVATE
    $<BUILD_INTERFACE:${BRANSON_BINARY_DIR}>
    ${branson_mpi_include_dirs} )
  target_link_libraries( ${build_target} PUBLIC ${branson_deps} )

  # Register the test command with ctest.
  foreach( np ${abt_PE_LIST} )
    add_test(
      NAME    ${ctest_name}_${np}pe
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
              ${mpiexec_postflags_list}
              $<TARGET_FILE:${build_target}> )
    set_tests_properties( ${ctest_name}_${np}pe PROPERTIES PROCESSORS ${np} )
  endforeach()
endfunction()

function( add_branson_gpu_test )
  cmake_parse_arguments(
    abt
    ""
    "SOURCE"
    "PE_LIST"
    ${ARGV} )
  string( REPLACE ".cc" "_exe" build_target ${abt_SOURCE} )
  string( REPLACE ".cc" ""     ctest_name   ${abt_SOURCE} )

  # Build the test
  add_executable( ${build_target} ${abt_SOURCE} )
  target_include_directories( ${build_target} PRIVATE
    $<BUILD_INTERFACE:${BRANSON_BINARY_DIR}>
    ${branson_mpi_include_dirs} )
  target_link_libraries( ${build_target} PUBLIC ${branson_deps} )

  if("${GPU_DBS_STRING}" STREQUAL "CUDA" )
    message("Setting CUDA compiler options")
    #set_target_properties(BRANSON PROPERTIES CUDA_ARCHITECTURES "90") # H100
    #set_target_properties(BRANSON PROPERTIES CUDA_ARCHITECTURES "70") # V100
    set_target_properties(${build_target} PROPERTIES CUDA_ARCHITECTURES "80") # A100
    set_target_properties(${build_target} PROPERTIES CUDA_STANDARD 17)
    #string(APPEND CMAKE_CUDA_FLAGS " -g --expt-relaxed-constexpr --fmad=false -lineinfo")
    #string(APPEND CMAKE_CUDA_FLAGS " --expt-extended-lambda"  )
    set_source_files_properties(${abt_SOURCE} PROPERTIES LANGUAGE CUDA)
  elseif("${GPU_DBS_STRING}" STREQUAL "HIP" )
    message("Setting HIP compiler options")
    #if(CMAKE_HIP_COMPILER MATCHES "CC")
    #  set(CMAKE_HIP_FLAGS "-x hip")
    #endif()
    set_target_properties(${build_target} PROPERTIES HIP_ARCHITECTURES "gfx942") # MI300
    set_target_properties(${build_target} PROPERTIES HIP_STANDARD 17)
    #string(APPEND CMAKE_CXX_FLAGS " -D__HIP_PLATFORM_HCC__= -D__HIP_PLATFORM_AMD__=")
    #string(APPEND CMAKE_CXX_FLAGS " -Wno-switch-default")
    set_source_files_properties(${abt_SOURCE} PROPERTIES LANGUAGE HIP)
  else()
    message("GPU Options: Not a GPU build or GPU_DBS_STRING not recognized")
  endif()

  # Register the test command with ctest.
  foreach( np ${abt_PE_LIST} )
    add_test(
      NAME    ${ctest_name}_${np}pe
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
              ${mpiexec_postflags_list}
              $<TARGET_FILE:${build_target}> )
    set_tests_properties( ${ctest_name}_${np}pe PROPERTIES PROCESSORS ${np} )
  endforeach()
endfunction()

#------------------------------------------------------------------------------#
# 1 PE tests

set(branson_test_sources_1pe
  test_buffer.cc
  test_cell.cc
  test_proto_cell.cc
  test_counter_rng.cc
  test_input.cc
  test_mesh.cc
  test_mpi_types.cc
  test_sampling_functions.cc
  test_imc_parameters.cc
 )
foreach( btest ${branson_test_sources_1pe} )
  add_branson_test( SOURCE  ${btest} PE_LIST "1" )
endforeach()

#------------------------------------------------------------------------------#
# Other tests

if(${USE_GPU})
add_branson_gpu_test( SOURCE test_warp_reductions.cc      PE_LIST "1" )
endif()
add_branson_test( SOURCE test_imc_state.cc      PE_LIST "2" )
add_branson_test( SOURCE test_photon.cc      PE_LIST "2" )

#------------------------------------------------------------------------------#
# copy these input files for Input, IMC_State, Mesh and write_silo tests

set( inputfiles
  simple_input.xml
  large_particle_input.xml
  three_region_mesh_input.xml )
foreach( ifile ${inputfiles} )
  configure_file(${ifile} ${CMAKE_CURRENT_BINARY_DIR}/${ifile} COPYONLY)
endforeach()

#------------------------------------------------------------------------------#
# write_silo function test
if (VIZ_LIBRARIES_FOUND)
  add_branson_test( SOURCE test_write_silo.cc PE_LIST "2")
endif()

#------------------------------------------------------------------------------#
# End test/CMakeLists.txt
#------------------------------------------------------------------------------#
