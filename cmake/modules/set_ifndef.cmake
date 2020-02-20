# SPDX-FileCopyrightText: 2019 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

function (set_ifndef variable value)
  if(NOT DEFINED ${variable})
    set(${variable} ${value} PARENT_SCOPE)
  endif()
endfunction()
