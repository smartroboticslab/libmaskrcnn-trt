# SPDX-FileCopyrightText: 2019 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

macro(find_library_create_target target_name lib libtype hints)
    message(STATUS "========================= Importing and creating target ${target_name} ==========================")
    message(STATUS "Looking for library ${lib}")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        find_library(${lib}_LIB_PATH ${lib}${TRT_DEBUG_POSTFIX} HINTS ${hints} NO_DEFAULT_PATH)
    endif()
    find_library(${lib}_LIB_PATH ${lib} HINTS ${hints} NO_DEFAULT_PATH)
    find_library(${lib}_LIB_PATH ${lib})
    message(STATUS "Library that was found ${${lib}_LIB_PATH}")
    add_library(${target_name} ${libtype} IMPORTED)
    set_property(TARGET ${target_name} PROPERTY IMPORTED_LOCATION ${${lib}_LIB_PATH})
    message(STATUS "==========================================================================================")
endmacro()
