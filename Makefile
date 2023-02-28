# SPDX-FileCopyrightText: 2021-2023 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2021-2023 Sotiris Papatheodorou
# SPDX-License-Identifier: Apache-2.0
.POSIX:

CMAKE_BUILD_TYPE = Release

all:
	mkdir -p build
	cd build && cmake -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) ..
	cmake --build build

clean:
	rm -rf build

download-model:
	wget https://www.doc.ic.ac.uk/~sleutene/software/mrcnn_nchw.uff
