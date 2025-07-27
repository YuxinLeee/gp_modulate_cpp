cmake_minimum_required(VERSION 3.10)
project(locally_modulated_ds)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 可选：把你的可执行文件都丢到 bin 里
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../bin)

# -------------------------------------------------------------------
#                强制使用 Homebrew 的 Eigen3
# -------------------------------------------------------------------
# Homebrew 在 Intel Mac 上默认安装到 /usr/local/opt/eigen，
# M1/M2 上通常是 /opt/homebrew/opt/eigen

# find_package(Eigen3 3.4 REQUIRED CONFIG
#   PATHS /opt/homebrew/share/eigen3/cmake
#   NO_DEFAULT_PATH
# )


set(Eigen3_INCLUDE_DIRS "/opt/homebrew/include/"
    CACHE PATH "Homebrew Eigen3 include directory" FORCE
)
message(STATUS ">>> Using Eigen3 from: ${Eigen3_DIR}")
message(STATUS ">>> Eigen3 include dirs: ${Eigen3_INCLUDE_DIRS}")

# 查找 matio 库
find_package(PkgConfig REQUIRED)
pkg_check_modules(MATIO REQUIRED matio)
message(STATUS ">>> Matio include dirs: ${MATIO_INCLUDE_DIRS}")
message(STATUS ">>> Matio libraries: ${MATIO_LIBRARIES}")

# -------------------------------------------------------------------
#                  剩下的 include & sources 配置
# -------------------------------------------------------------------
include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/../gaussian-process-regression-master/include
  ${CMAKE_SOURCE_DIR}/../lpvDS/include
  ${CMAKE_SOURCE_DIR}/../mathlib/include
  ${Eigen3_INCLUDE_DIRS}
  ${MATIO_INCLUDE_DIRS}
)

# 收集源码
file(GLOB LMDS_SOURCES   ${CMAKE_SOURCE_DIR}/src/*.cpp)
file(GLOB GP_SOURCES     ${CMAKE_SOURCE_DIR}/../gaussian-process-regression-master/src/*.cpp)
file(GLOB GP_SOURCES     ${CMAKE_SOURCE_DIR}/../mathlib/src/*.cpp)

# 设置 lpvDS 源码目录
set(LPVDS_SRC_DIR ${CMAKE_SOURCE_DIR}/../lpvDS/src/lpvDS)

# 收集 lpvDS 源码
set(LPVDS_SOURCES
  ${LPVDS_SRC_DIR}/lpvDS.cpp
  ${LPVDS_SRC_DIR}/utils.cpp
  ${LPVDS_SRC_DIR}/GMM.cpp
)

# 最终可执行文件
add_executable(lmds_demo
  ${LMDS_SOURCES}
  ${GP_SOURCES}
  ${LPVDS_SOURCES}
)

# 链接库
target_link_libraries(lmds_demo PRIVATE ${MATIO_LIBRARIES})
target_link_directories(lmds_demo PRIVATE ${MATIO_LIBRARY_DIRS})