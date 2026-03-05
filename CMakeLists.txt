cmake_minimum_required(VERSION 3.14)
project(LatLibrary)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找Python和pybind11 - 添加 Interpreter 组件
find_package(Python3 3.11 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# 设置输出目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# 1. 构建原始库（所有源文件）
file(GLOB SRC_FILES src/*.cpp)

# 创建原始库
add_library(lat_core SHARED ${SRC_FILES})
target_include_directories(lat_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 2. 构建Python绑定模块
# 创建Python模块
pybind11_add_module(lattice_env 
    pybind/pybind_wrapper.cpp
    pybind/PyLatticeEnv.cpp
    pybind/RL_ENUM_Wrapper.cpp
)

# 链接依赖 - 使用正确的目标名称
target_link_libraries(lattice_env PRIVATE 
    lat_core
    pybind11::module
    Python3::Module
)

# 包含目录
target_include_directories(lattice_env PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/pybind
)

# 3. 可执行文件（如果需要）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test.cpp")
    add_executable(test_exec ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)
    target_link_libraries(test_exec lat_core)
    enable_testing()
    add_test(NAME RunTestExec COMMAND test_exec)
endif()

# 安装目标（可选）
install(TARGETS lat_core DESTINATION lib)
install(TARGETS lattice_env DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)