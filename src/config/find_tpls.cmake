#-----------------------------*-cmake-*----------------------------------------#
# file   src/config/find_tpls.cmake
# author Kelly Thompson <kgt@lanl.gov>
# date   Tuesday, Aug 14, 2018, 15:24 pm
# brief  Look for third party libraries like metis
# note   Copyright (C) 2018 Los Alamos National Security, LLC.
#        All rights reserved.
#------------------------------------------------------------------------------#

include( FeatureSummary )

# ------------------------------------------------------------------------------------------------ #
# Query CPU topology
#
# Returns:
#
# * MPI_CORES_PER_CPU
# * MPI_CPUS_PER_NODE
# * MPI_PHYSICAL_CORES
# * MPI_MAX_NUMPROCS_PHYSICAL
# * MPI_HYPERTHREADING
#
# See also:
#
# * Try running 'lstopo' for a graphical view of the local topology or 'lscpu' for a text version.
# * EAP's flags can be found in Test.rh/General/run_job.pl (look for $other_args).  In particular,
#   it may be useful to examine EAP's options for srun or aprun.
# ------------------------------------------------------------------------------------------------
# #
macro(query_topology)

  # These cmake commands, while useful, don't provide the topology detail that we are interested in
  # (i.e. number of sockets per node). We could use the results of these queries to know if
  # hyper-threading is enabled (if logical != physical cores)
  #
  # * cmake_host_system_information(RESULT MPI_PHYSICAL_CORES QUERY NUMBER_OF_PHYSICAL_CORES)
  # * cmake_host_system_information(RESULT MPI_LOGICAL_CORES  QUERY NUMBER_OF_LOGICAL_CORES)

  # start with default values
  set(MPI_CORES_PER_CPU 4)
  set(MPI_PHYSICAL_CORES 1)

  if(SITENAME MATCHES "RZNevada"
     OR SITENAME MATCHES "RZVernal"
     OR SITENAME MATCHES "RZAdams")
    set(MPI_CORES_PER_CPU 2)
    set(MPI_PHYSICAL_CORES 64)
    set(MPIEXEC_MAX_NUMPROCS
        64
        CACHE STRING "Max procs on node." FORCE)
  elseif(SITENAME MATCHES "RZAdams")
    set(MPI_CORES_PER_CPU 2)
    set(MPI_PHYSICAL_CORES 96)
    set(MPIEXEC_MAX_NUMPROCS
        96
        CACHE STRING "Max procs on node." FORCE)
  elseif(EXISTS "/proc/cpuinfo")
    # read the system's cpuinfo...
    file(READ "/proc/cpuinfo" cpuinfo_data)
    string(REGEX REPLACE "\n" ";" cpuinfo_data "${cpuinfo_data}")
    foreach(line ${cpuinfo_data})
      if("${line}" MATCHES "cpu cores")
        string(REGEX REPLACE ".* ([0-9]+).*" "\\1" MPI_CORES_PER_CPU "${line}")
      elseif("${line}" MATCHES "physical id")
        string(REGEX REPLACE ".* ([0-9]+).*" "\\1" tmp "${line}")
        if(${tmp} GREATER ${MPI_PHYSICAL_CORES})
          set(MPI_PHYSICAL_CORES ${tmp})
        endif()
      endif()
    endforeach()
    # correct 0-based indexing
    math(EXPR MPI_PHYSICAL_CORES "${MPI_PHYSICAL_CORES} + 1")
  endif()

  math(EXPR MPI_CPUS_PER_NODE "${MPIEXEC_MAX_NUMPROCS} / ${MPI_CORES_PER_CPU}")
  set(MPI_CPUS_PER_NODE
      ${MPI_CPUS_PER_NODE}
      CACHE STRING "Number of multi-core CPUs per node" FORCE)
  set(MPI_CORES_PER_CPU
      ${MPI_CORES_PER_CPU}
      CACHE STRING "Number of cores per cpu" FORCE)

  #
  # Check for hyper-threading - This is important for reserving threads for OpenMP tests...
  #
  math(EXPR MPI_MAX_NUMPROCS_PHYSICAL "${MPI_PHYSICAL_CORES} * ${MPI_CORES_PER_CPU}")
  if("${MPI_MAX_NUMPROCS_PHYSICAL}" STREQUAL "${MPIEXEC_MAX_NUMPROCS}")
    set(MPI_HYPERTHREADING
        "OFF"
        CACHE BOOL "Are we using hyper-threading?" FORCE)
  else()
    set(MPI_HYPERTHREADING
        "ON"
        CACHE BOOL "Are we using hyper-threading?" FORCE)
  endif()
endmacro()

# cmake-lint: disable=R0912,R0915,W0106
#
# * too many branches
# * function too long

# ------------------------------------------------------------------------------------------------ #
# Set MPI flavor and vendor version
#
# Returns (as cache variables)
#
# * MPI_VERSION
# * MPI_FLAVOR = {openmpi, mpich, cray, spectrum, mvapich2, intel}
#
# ------------------------------------------------------------------------------------------------ #
function(setMPIflavorVer)

  # First attempt to determine MPI flavor -- scape flavor from full path (this ususally works for
  # HPC or systems with modules)
  if(CMAKE_CXX_COMPILER_WRAPPER STREQUAL CrayPrgEnv OR "$ENV{LOADEDMODULES}" MATCHES "cray-mpich")
    set(MPI_FLAVOR "cray")
  elseif(
    "${MPIEXEC_EXECUTABLE}" MATCHES "openmpi"
    OR "${MPIEXEC_EXECUTABLE}" MATCHES "smpi"
    OR ("${MPIEXEC_EXECUTABLE}" MATCHES "srun" AND "${MPI_C_COMPILER}" MATCHES "openmpi"))
    set(MPI_FLAVOR "openmpi")
  elseif("${MPIEXEC_EXECUTABLE}" MATCHES "mpich" OR "${MPI_C_HEADER_DIR}" MATCHES "mpich")
    set(MPI_FLAVOR "mpich")
  elseif("${MPIEXEC_EXECUTABLE}" MATCHES "impi" OR "${MPIEXEC_EXECUTABLE}" MATCHES "clusterstudio")
    set(MPI_FLAVOR "intel")
  elseif("${MPIEXEC_EXECUTABLE}" MATCHES "mvapich2")
    set(MPI_FLAVOR "mvapich2")
  elseif(
    "${MPIEXEC_EXECUTABLE}" MATCHES "spectrum-mpi"
    OR "${MPIEXEC_EXECUTABLE}" MATCHES "lrun"
    OR "${MPIEXEC_EXECUTABLE}" MATCHES "jsrun")
    set(MPI_FLAVOR "spectrum")
  endif()

  if(CMAKE_CXX_COMPILER_WRAPPER STREQUAL CrayPrgEnv)
    if(DEFINED ENV{CRAY_MPICH2_VER})
      set(MPI_VERSION $ENV{CRAY_MPICH2_VER})
    endif()
  elseif(DEFINED ENV{LMOD_MPI_VERSION})
    # Toss3 with srun
    string(REGEX REPLACE "-[a-z0-9]+" "" MPI_VERSION "$ENV{LMOD_MPI_VERSION}")
  elseif(DEFINED ENV{LMOD_FAMILY_MPI_VERSION})
    # ATS-2 with lrun
    string(REGEX REPLACE "-[a-z0-9]+" "" MPI_VERSION "$ENV{LMOD_FAMILY_MPI_VERSION}")
  else()
    execute_process(
      COMMAND ${MPIEXEC_EXECUTABLE} --version
      OUTPUT_VARIABLE DBS_MPI_VER_OUT
      ERROR_VARIABLE DBS_MPI_VER_ERR)
    set(DBS_MPI_VER "${DBS_MPI_VER_OUT}${DBS_MPI_VER_ERR}")
    string(REPLACE "\n" ";" TEMP ${DBS_MPI_VER})
    foreach(line ${TEMP})
      # extract the version...
      if(${line} MATCHES "Version"
         OR ${line} MATCHES "OpenRTE"
         OR ${line} MATCHES "Open MPI"
         OR ${line} MATCHES "Spectrum MPI")
        set(DBS_MPI_VER "${line}")
        if("${DBS_MPI_VER}" MATCHES "[0-9]+[.][0-9]+[.][0-9]+")
          string(REGEX REPLACE ".*([0-9]+)[.]([0-9]+)[.]([0-9]+).*" "\\1" DBS_MPI_VER_MAJOR
                               ${DBS_MPI_VER})
          string(REGEX REPLACE ".*([0-9]+)[.]([0-9]+)[.]([0-9]+).*" "\\2" DBS_MPI_VER_MINOR
                               ${DBS_MPI_VER})
          string(REGEX REPLACE ".*([0-9]+)[.]([0-9]+)[.]([0-9]+).*" "\\3" DBS_MPI_VER_PATCH
                               ${DBS_MPI_VER})
          set(MPI_VERSION "${DBS_MPI_VER_MAJOR}.${DBS_MPI_VER_MINOR}.${DBS_MPI_VER_PATCH}")
        elseif("${DBS_MPI_VER}" MATCHES "[0-9]+[.][0-9]+")
          string(REGEX REPLACE ".*([0-9]+)[.]([0-9]+).*" "\\1" DBS_MPI_VER_MAJOR ${DBS_MPI_VER})
          string(REGEX REPLACE ".*([0-9]+)[.]([0-9]+).*" "\\2" DBS_MPI_VER_MINOR ${DBS_MPI_VER})
          set(MPI_VERSION "${DBS_MPI_VER_MAJOR}.${DBS_MPI_VER_MINOR}")
        endif()
      endif()

      # if needed, make a 2nd pass at identifying the MPI flavor
      if(NOT DEFINED MPI_FLAVOR)
        if("${line}" MATCHES "HYDRA")
          set(MPI_FLAVOR "mpich")
        elseif("${line}" MATCHES "OpenRTE")
          set(MPI_FLAVOR "openmpi")
        elseif("${line}" MATCHES "intel-mpi" OR "${line}" MATCHES "Intel[(]R[)] MPI Library")
          set(MPI_FLAVOR "intel")
        endif()
      endif()

      # Once we have the needed information stop parsing...
      if(DEFINED MPI_FLAVOR AND DEFINED MPI_VERSION)
        break()
      endif()
    endforeach()

  endif()

  # if the FindMPI.cmake module didn't set the version, then try to do so here.
  if(NOT DEFINED MPI_VERSION AND DEFINED MPI_C_VERSION)
    set(MPI_VERSION ${MPI_C_VERSION})
  endif()

  # Return the discovered values to the calling scope
  if(DEFINED MPI_FLAVOR)
    set(MPI_FLAVOR
        "${MPI_FLAVOR}"
        CACHE STRING "Vendor brand of MPI" FORCE)
  endif()
  if(DEFINED MPI_VERSION)
    set(MPI_VERSION
        "${MPI_VERSION}"
        CACHE STRING "Vendor version of MPI" FORCE)
  endif()

endfunction()

# ------------------------------------------------------------------------------------------------ #
# Setup OpenMPI
# ------------------------------------------------------------------------------------------------ #
macro(setupOpenMPI)

  # sanity check, these OpenMPI flags (below) require version >= 1.8
  if(MPI_VERSION VERSION_LESS 1.8)
    message(FATAL_ERROR "OpenMPI version < 1.8 found.")
  endif()

  # Find cores/cpu, cpu/node, hyper-threading
  query_topology()

  # Extra options provided from the environment or by cmake
  if(DEFINED ENV{MPIEXEC_PREFLAGS})
    set(MPIEXEC_PREFLAGS "$ENV{MPIEXEC_PREFLAGS}")
  endif()

  if("${MPIEXEC_EXECUTABLE}" MATCHES "srun")
    set(preflags " --overlap") # -N 1 --cpu_bind=verbose,cores
    set(MPIEXEC_PREFLAGS ${preflags})
    set(MPIEXEC_PREFLAGS_PERFBENCH ${preflags})
    set(MPIEXEC_OMP_PREFLAGS "${MPIEXEC_PREFLAGS} -c ${MPI_CORES_PER_CPU}")
  else()
    # Notes:
    #
    # * For PERFBENCH that use Quo, we need '--map-by socket:SPAN' instead of '-bind-to none'.  The
    #   'bind-to none' is required to pack a node.
    # * Adding '--debug-daemons' is often requested by the OpenMPI dev team in conjunction with
    #   'export OMPI_MCA_btl_base_verbose=100' to obtain debug traces from openmpi.
    set(MPIEXEC_PREFLAGS_PERFBENCH "${MPIEXEC_PREFLAGS} --map-by socket:SPAN")
    if(NOT MPIEXEC_PREFLAGS MATCHES " -bind-to none")
      string(APPEND MPIEXEC_PREFLAGS " -bind-to none")
    endif()
    # Setup for OMP plus MPI
    if((NOT APPLE)
       AND (NOT MPIEXEC_OMP_PREFLAGS MATCHES "--map-by ppr")
       AND (MPI_VERSION VERSION_LESS 5.0))
      # -bind-to fails on OSX, See #691 OpenMPI version 5.0+ doesn't use this ppr syntax.
      set(MPIEXEC_OMP_PREFLAGS
          "${MPIEXEC_PREFLAGS} --map-by ppr:${MPI_CORES_PER_CPU}:socket --report-bindings")
    endif()

    # Spectrum-MPI on darwin
    #
    # Limit communication to on-node via '-intra sm' or 'intra vader'
    # https://www.ibm.com/support/knowledgecenter/SSZTET_EOS/eos/guide_101.pdf
    if("${MPIEXEC_EXECUTABLE}" MATCHES "smpi" AND NOT MPIEXEC_PREFLAGS MATCHES "-intra sm")
      string(REPLACE "-bind-to none" "-bind-to core" MPIEXEC_PREFLAGS ${MPIEXEC_PREFLAGS})
      # string(REPLACE "-bind-to none" "-bind-to core" MPIEXEC_OMP_PREFLAGS ${MPIEXEC_OMP_PREFLAGS})
      set(smpi-sm-only "-intra sm -aff off --report-bindings")
      string(APPEND MPIEXEC_PREFLAGS " ${smpi-sm-only}")
      string(APPEND MPIEXEC_OMP_PREFLAGS " ${smpi-sm-only}")
      unset(smpi-sm-only)
    endif()
  endif()

  # Cache the result
  set(MPIEXEC_PREFLAGS
      "${MPIEXEC_PREFLAGS}"
      CACHE STRING "extra mpirun flags (list)." FORCE)
  set(MPIEXEC_PREFLAGS_PERFBENCH
      "${MPIEXEC_PREFLAGS_PERFBENCH}"
      CACHE STRING "extra mpirun flags (list)." FORCE)
  set(MPIEXEC_OMP_PREFLAGS
      "${MPIEXEC_OMP_PREFLAGS}"
      CACHE STRING "extra mpirun flags (list)." FORCE)

  mark_as_advanced(MPI_CPUS_PER_NODE MPI_CORES_PER_CPU MPI_PHYSICAL_CORES MPI_MAX_NUMPROCS_PHYSICAL
                   MPI_HYPERTHREADING)

endmacro()

macro(setupTPLs)

  ##############################################################################
  # MPI
  ##############################################################################
  if( NOT TARGET MPI::MPI_C )
    #message(STATUS "Looking for MPI...")
    #find_package(MPI QUIET REQUIRED)
    #if( ${MPI_C_FOUND} )
    #  message(STATUS "Looking for MPI...${MPIEXEC}")
    #else()
    #  message(STATUS "Looking for MPI...not found")
    #endif()

    #set_package_properties( MPI PROPERTIES
    #  URL "http://www.open-mpi.org/"
    #  DESCRIPTION "A High Performance Message Passing Library"
    #  TYPE REQUIRED
    #  PURPOSE "A parallel communication library is required in BRANSON.")
    message(STATUS "Looking for MPI...")

    # Preserve data that may already be set.
    if(DEFINED ENV{MPIRUN})
      set(MPIEXEC_EXECUTABLE
          $ENV{MPIRUN}
          CACHE STRING "Program to execute MPI parallel programs.")
    elseif(DEFINED ENV{MPIEXEC_EXECUTABLE})
      set(MPIEXEC_EXECUTABLE
          $ENV{MPIEXEC_EXECUTABLE}
          CACHE STRING "Program to execute MPI parallel programs.")
    elseif(DEFINED ENV{MPIEXEC})
      set(MPIEXEC_EXECUTABLE
          $ENV{MPIEXEC}
          CACHE STRING "Program to execute MPI parallel programs.")
    endif()

    # If this is a Cray system and the Cray MPI compile wrappers are used, or if this is CTS-1 with
    # Toss3, then do some special setup:

    if(CMAKE_CXX_COMPILER_WRAPPER MATCHES CrayPrgEnv OR IS_DIRECTORY "/usr/projects/hpcsoft/toss3/")
      if(NOT EXISTS ${MPIEXEC_EXECUTABLE})
        find_program(MPIEXEC_EXECUTABLE flux) # 1st option is flux
        if(EXISTS ${MPIEXEC_EXECUTABLE})
          execute_process(
            COMMAND ${MPIEXEC_EXECUTABLE} jobs
            RESULT_VARIABLE fluxfailure
            OUTPUT_QUIET ERROR_QUIET)
          if(NOT "${fluxfailure}" STREQUAL "0")
            unset(MPIEXEC_EXECUTABLE CACHE)
          endif()
        endif()
        if(NOT EXISTS ${MPIEXEC_EXECUTABLE})
          find_program(MPIEXEC_EXECUTABLE srun) # fall back to srun
        endif()
      endif()
      if(MPIEXEC_EXECUTABLE MATCHES "flux")
        set(MPIEXEC_NUMPROC_FLAG run;-n)
      else()
        set(MPIEXEC_NUMPROC_FLAG "-n")
      endif()
      set(MPIEXEC_NUMPROC_FLAG
          "${MPIEXEC_NUMPROC_FLAG}"
          CACHE STRING "mpirun flag used to specify the number of processors to use")
      set(MPIEXEC_EXECUTABLE
          ${MPIEXEC_EXECUTABLE}
          CACHE STRING "Program to execute MPI parallel programs." FORCE)

    elseif(DEFINED ENV{SYS_TYPE} AND "$ENV{SYS_TYPE}" MATCHES "ppc64le_ib") # ATS-2
      if(NOT EXISTS ${MPIEXEC_EXECUTABLE})
        find_program(MPIEXEC_EXECUTABLE lrun)
      endif()
      set(MPIEXEC_EXECUTABLE
          ${MPIEXEC_EXECUTABLE}
          CACHE STRING "Program to execute MPI parallel programs." FORCE)
      set(MPIEXEC_NUMPROC_FLAG
          "--np"
          CACHE STRING "mpirun flag used to specify the number of processors to use" FORCE)
    endif()

    # Call the standard CMake FindMPI macro.
    find_package(MPI QUIET)

    # Try to discover the MPI flavor and the vendor version. Returns MPI_VERSION, MPI_FLAVOR as
    # cache variables
    setmpiflavorver()

    # Set additional flags, environments that are MPI vendor specific.
    if("${MPI_FLAVOR}" MATCHES "openmpi")
      setupopenmpi()
    elseif("${MPI_FLAVOR}" MATCHES "mpich")
      setupmpichmpi()
    elseif("${MPI_FLAVOR}" MATCHES "intel")
      setupintelmpi()
    elseif("${MPI_FLAVOR}" MATCHES "spectrum")
      setupspectrummpi()
    elseif("${MPI_FLAVOR}" MATCHES "cray")
      setupCrayMPI()
    else()
      message(
        FATAL_ERROR
          "
The Draco build system doesn't know how to configure the build for
  MPIEXEC_EXECUTABLE         = ${MPIEXEC_EXECUTABLE}
  DBS_MPI_VER                = ${DBS_MPI_VER}
  CMAKE_CXX_COMPILER_WRAPPER = ${CMAKE_CXX_COMPILER_WRAPPER}")
    endif()

    # Mark some of the variables created by the above logic as 'advanced' so that they do not show
    # up in the 'simple' ccmake view.
    mark_as_advanced(MPI_EXTRA_LIBRARY MPI_LIBRARY)

    message(STATUS "Looking for MPI.......found ${MPIEXEC_EXECUTABLE}")

    # Sanity Checks for DRACO_C4==MPI
    if("${MPI_CORES_PER_CPU}x" STREQUAL "x")
      message(FATAL_ERROR "setupMPILibrariesUnix:: MPI_CORES_PER_CPU " "is not set!")
    endif()

  set_package_properties(
    MPI PROPERTIES
    URL "http://www.open-mpi.org/"
    DESCRIPTION "A High Performance Message Passing Library"
    TYPE RECOMMENDED
    PURPOSE "If not available, all Draco components will be built as scalar applications.")

  mark_as_advanced(MPIEXEC_OMP_PREFLAGS MPI_LIBRARIES)

  endif()

  ##############################################################################
  # OpenMP
  ##############################################################################
  message(STATUS "Looking for Threads...")
  find_package(Threads QUIET)
  if(Threads_FOUND)
    message(STATUS "Looking for Threads...found")
  else()
    message(STATUS "Looking for Threads...not found")
  endif()

  message(STATUS "Looking for OpenMP...")
  if(DEFINED USE_OPENMP)
    # no-op (use defined value, -DUSE_OPENMP=<OFF|ON>,  instead of attempting to guess)
  else()
    # Assume we want to use it if it is found.
    set(USE_OPENMP ON)
  endif()
  set(USE_OPENMP
      ${USE_OPENMP}
      CACHE BOOL "Enable OpenMP threading support if detected." FORCE)

  # Find package if desired:
  if(USE_OPENMP)
    find_package(OpenMP QUIET)
  else()
    set(OpenMP_FOUND FALSE)
  endif()

  if(OpenMP_FOUND)
    # [2022-10-27 KT] cmake/3.22 doesn't report OpenMP_C_VERSION for nvc++. Fake it for now.
    if("${OpenMP_C_VERSION}x" STREQUAL "x" AND CMAKE_CXX_COMPILER_ID MATCHES "NVHPC")
      set(OpenMP_C_VERSION
          "5.0"
          CACHE BOOL "OpenMP version." FORCE)
      set(OpenMP_FOUND TRUE)
    endif()
    message(STATUS "Looking for OpenMP... ${OpenMP_C_FLAGS} (supporting the ${OpenMP_C_VERSION} "
                   "standard)")
    if(OpenMP_C_VERSION VERSION_LESS 3.0)
      message(STATUS "OpenMP standard support is too old (< 3.0). Disabling OpenMP build features.")
      set(OpenMP_FOUND FALSE)
      set(OpenMP_C_FLAGS
          ""
          CACHE BOOL "OpenMP disabled (too old)." FORCE)
    endif()
    set(OpenMP_FOUND
        ${OpenMP_FOUND}
        CACHE BOOL "Is OpenMP available?" FORCE)
  else()
    if(USE_OPENMP)
      # Not detected, though desired.
      message(STATUS "Looking for OpenMP... not found")
    else()
      # Detected, but not desired.
      message(STATUS "Looking for OpenMP... found, but disabled for this build")
    endif()
  endif()
  message("CXX OPEN MP FLAGS: ${OpenMP_CXX_FLAGS}")

  ##############################################################################
  # metis
  # Load modules for metis to get correct environment variables
  ##############################################################################
  if( NOT TARGET METIS::metis )

    message( STATUS "Looking for METIS..." )
    find_package( METIS QUIET)
    if( METIS_FOUND )
      message( STATUS "Looking for METIS.....found ${METIS_LIBRARY}" )
    else()
      message( STATUS "Looking for METIS.....not found" )
    endif()

    set_package_properties( METIS PROPERTIES
      DESCRIPTION "METIS"
      TYPE OPTIONAL
      URL "http://glaros.dtc.umn.edu/gkhome/metis/metis/overview"
      PURPOSE "METIS is a set of serial programs for partitioning graphs,
   partitioning finite element meshes, and producing fill reducing orderings for
   sparse matrices.")

  endif()

  ##############################################################################
  # Silo and HDF5 libraries
  # Load modules for hdf5 and solo to get correct environment variables
  # use find package
  ##############################################################################

  if( NOT HDF5_FOUND )

    message( STATUS "Looking for HDF5..." )
    find_package( HDF5 QUIET )
    if( HDF5_FOUND )
      list(GET HDF5_LIBRARIES 0 hdf5lib)
      message( STATUS "Looking for HDF5..found ${hdf5lib}" )
      unset(hdf5lib)
    else()
      message( STATUS "Looking for HDF5..not found" )
    endif()

    set_package_properties( HDF5 PROPERTIES
      DESCRIPTION "HDF5 is a data model, library, and file format for storing
   and managing data. It supports an unlimited variety of datatypes, and is
   designed for flexible and efficient I/O and for high volume and complex
   data."
      TYPE OPTIONAL
      URL "https://support.hdfgroup.org/HDF5/"
      PURPOSE "Provides optional visualization support for Branson." )

  endif()

  if( HDF5_FOUND AND NOT TARGET Silo::silo )

    message( STATUS "Looking for Silo..." )
    find_package( Silo QUIET )
    if( Silo_FOUND )
      message( STATUS "Looking for Silo..found ${Silo_LIBRARY}" )
    else()
      message( STATUS "Looking for Silo..not found" )
    endif()

    set_package_properties( Silo PROPERTIES
      DESCRIPTION "Silo is a library for reading and writing a wide variety of
   scientific data to binary, disk files."
      TYPE OPTIONAL
      URL "http://wci.llnl.gov/simulation/computer-codes/silo"
      PURPOSE "Provides optional visualization support for Branson.")

  endif()

  if (HDF5_FOUND AND Silo_FOUND)
    set(VIZ_LIBRARIES_FOUND TRUE)
  else ()
    message(STATUS "Optional visualization libraries not loaded...skipping")
  endif ()

endmacro()

# ------------------------------------------------------------------------------------------------ #
# Setup Cray MPI and wrappers
# ------------------------------------------------------------------------------------------------ #
macro(setupCrayMPI)

  query_topology()

  # salloc/sbatch options:
  #
  # * -N      limit job to a single node.
  # * --gres=craynetwork:0 This option allows more than one srun to be running at the same time on
  #   the Cray. There are 4 gres “tokens” available. If unspecified, each srun invocation will
  #   consume all of them. Setting the value to 0 means consume none and allow the user to run as
  #   many concurrent jobs as there are cores available on the node. This should only be specified
  #   on the salloc/sbatch command.  Gabe doesn't recommend this option for regression testing.
  # * --vm-overcommit=disable|enable Do not allow overcommit of heap resources.
  # * -p knl    Limit allocation to KNL nodes.
  #
  # srun options:
  #
  # * --cpu_bind=verbose,cores Bind MPI ranks to cores and print a summary of binding when run
  # * --exclusive This option will keep concurrent jobs from running on the same cores. If you want
  #   to background tasks to have them run simultaneously, this option is required to be set or they
  #   will stomp on the same cores.
  # * --overlap Allow steps to overlap each other on the CPUs. By default steps do not share CPUs
  #   with other parallel steps.
  # * --hint=nomultithread Disable use of hyperthreads (use only physical cores)

  set(preflags " ") # -N 1 --cpu_bind=verbose,cores
  if(NOT MPIEXEC_EXECUTABLE MATCHES "flux")
    if(DEFINED ENV{PE_PRODUCT_LIST} AND "$ENV{PE_PRODUCT_LIST}" MATCHES "CRAYPE_X86_SPR")
      # ATS-3
      string(APPEND preflags " --overlap --cpu-bind=none --hint=nomultithread")
    else()
      string(APPEND preflags " --gres=craynetwork:0 --overlap")
    endif()
  endif()
  set(MPIEXEC_PREFLAGS
      ${preflags}
      CACHE STRING "extra mpirun flags (list)." FORCE)
  # consider adding '-m=cyclic'
  set(MPIEXEC_PREFLAGS_PERFBENCH
      ${preflags}
      CACHE STRING "extra mpirun flags (list)." FORCE)
  set(MPIEXEC_OMP_PREFLAGS
      "${MPIEXEC_PREFLAGS} -c ${MPI_CORES_PER_CPU}"
      CACHE STRING "extra mpirun flags (list)." FORCE)
endmacro()


#------------------------------------------------------------------------------#
# End find_tpls.cmake
#------------------------------------------------------------------------------#
