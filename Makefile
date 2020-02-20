# SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
# SPDX-License-Identifier: Apache-2.0

.PHONY: release
release:
	mkdir -p build/release
	cd build/release && cmake -DCMAKE_BUILD_TYPE=Release $(CMAKE_ARGUMENTS) ../..
	$(MAKE) -C build/release $(MFLAGS)

.PHONY: relwithdebinfo
relwithdebinfo:
	mkdir -p build/relwithdebinfo
	cd build/relwithdebinfo && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo $(CMAKE_ARGUMENTS) ../..
	$(MAKE) -C build/relwithdebinfo $(MFLAGS)

.PHONY: debug
debug:
	mkdir -p build/debug
	cd build/debug && cmake -DCMAKE_BUILD_TYPE=Debug $(CMAKE_ARGUMENTS) ../..
	$(MAKE) -C build/debug $(MFLAGS)

.PHONY: clean
clean:
	rm -rf build

