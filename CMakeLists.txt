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
	GIT_TAG 8d7f1ee0621c17fa370b704b2100ffa0243d5bfb
	SOURCE_DIR "${ROOT_FOLDER}/deps/nanobind-src"
)
FetchContent_Declare(fabsim
	GIT_REPOSITORY https://github.com/DavidJourdan/fabsim.git
	GIT_TAG a653337e2eba15bd7fe17391648c08ea3493f464
	SOURCE_DIR "${ROOT_FOLDER}/deps/fabsim-src"
)
FetchContent_Declare(optim
	GIT_REPOSITORY https://github.com/DavidJourdan/optim.git
	GIT_TAG ada03c97f6334622a6e6657ff5cb2f599830a979
	SOURCE_DIR "${ROOT_FOLDER}/deps/optim-src"
)
FetchContent_MakeAvailable(eigen nanobind fabsim optim)

nanobind_add_module(fabsim_py
	PillarModel.cpp
	FiberStress.cpp
	ActinBundle.cpp
	main.cpp)
target_link_libraries(fabsim_py PRIVATE Eigen3::Eigen fabsim optim)

set_target_properties(fabsim_py PROPERTIES
                             CXX_STANDARD 20
                             CXX_STANDARD_REQUIRED OFF
                             CXX_EXTENSIONS ON)

set(CMAKE_INSTALL_PREFIX ${ROOT_FOLDER})
install(TARGETS fabsim_py
  LIBRARY 
  DESTINATION "${ROOT_FOLDER}")