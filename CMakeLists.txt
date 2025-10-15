cmake_minimum_required(VERSION 3.18)

set(ROOT_FOLDER "${CMAKE_CURRENT_LIST_DIR}")

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

# Dependencies
include(FetchContent)
FetchContent_Declare(Eigen EXCLUDE_FROM_ALL
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
  SOURCE_DIR "${ROOT_FOLDER}/deps/eigen-src"
)
FetchContent_Declare(nanobind
	GIT_REPOSITORY https://github.com/wjakob/nanobind.git
	GIT_TAG 116e098cfa96effca2a54e32e0ce5b93abe25393
	SOURCE_DIR "${ROOT_FOLDER}/deps/nanobind-src"
)
FetchContent_Declare(fabsim
	GIT_REPOSITORY https://github.com/DavidJourdan/fabsim.git
	GIT_TAG 54322da6ce1f51bb1e08c65d102be6737a7d0eda
	SOURCE_DIR "${ROOT_FOLDER}/deps/fabsim-src"
)
FetchContent_Declare(optim
	GIT_REPOSITORY https://github.com/DavidJourdan/optim.git
	GIT_TAG 2a2d48917a9b8d588e4174ad879212748ae5bfe4
	SOURCE_DIR "${ROOT_FOLDER}/deps/optim-src"
)
FetchContent_Declare(TinyAD
  GIT_REPOSITORY https://github.com/patr-schm/TinyAD.git
  GIT_TAG 81fab13c3884b787c0f03bbbbb95b0f794e54434
  SOURCE_DIR "${ROOT_FOLDER}/deps/TinyAD-src"
)
FetchContent_Declare(libigl
  URL https://github.com/libigl/libigl/archive/refs/tags/v2.6.0.zip
  SOURCE_DIR "${ROOT_FOLDER}/deps/libigl-src"
)
FetchContent_MakeAvailable(eigen nanobind fabsim optim TinyAD libigl)

igl_include(restricted triangle)

nanobind_add_module(fabsim_py
	FiberStress.cpp
	newton.cpp
	distance.cpp
	main.cpp
	data_to_mesh.cpp
	MuscleTissueElement.cpp
	MuscleTissueModel.cpp)
target_link_libraries(fabsim_py PRIVATE Eigen3::Eigen fabsim optim TinyAD igl::core igl_restricted::triangle)

set_target_properties(fabsim_py PROPERTIES
                             CXX_STANDARD 20
                             CXX_STANDARD_REQUIRED ON
                             CXX_EXTENSIONS ON)

set(CMAKE_INSTALL_PREFIX ${ROOT_FOLDER})
install(TARGETS fabsim_py
  LIBRARY 
  DESTINATION "${ROOT_FOLDER}")