# SPDX-FileCopyrightText: 2021-2023 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2021-2023 Sotiris Papatheodorou
# SPDX-License-Identifier: Apache-2.0

.PHONY: release
release:
	mkdir -p build/release
	cd build/release && cmake -DCMAKE_BUILD_TYPE=Release ../..
	$(MAKE) -C build/release $(MFLAGS)

.PHONY: relwithdebinfo
relwithdebinfo:
	mkdir -p build/relwithdebinfo
	cd build/relwithdebinfo && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ../..
	$(MAKE) -C build/relwithdebinfo $(MFLAGS)

.PHONY: debug
debug:
	mkdir -p build/debug
	cd build/debug && cmake -DCMAKE_BUILD_TYPE=Debug ../..
	$(MAKE) -C build/debug $(MFLAGS)

.PHONY: download-model
download-model:
	wget https://www.doc.ic.ac.uk/~sleutene/software/mrcnn_nchw.uff

.PHONY: clean
clean:
	rm -rf build
