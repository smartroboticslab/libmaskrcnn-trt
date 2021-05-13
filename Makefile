# SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
# SPDX-License-Identifier: Apache-2.0

# Installation directories
PREFIX ?= /usr/local
_INSTDIR = $(DESTDIR)$(PREFIX)
HEADERDIR ?= $(_INSTDIR)/include
LIBDIR ?= $(_INSTDIR)/lib
DATADIR ?= $(_INSTDIR)/share/
# Pass the prefix to CMake so it can generate the pkg-config file.
CMAKE_ARGUMENTS += -DMAKE_PREFIX=$(PREFIX)



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

.PHONY: download-model
download-model:
	wget https://www.doc.ic.ac.uk/~sleutene/software/mrcnn_nchw.uff

.PHONY: clean
clean:
	rm -rf build

.PHONY: install
install:
	install -D -m 644 build/release/libmaskrcnn-trt.a $(LIBDIR)
	install -D -m 644 -t $(HEADERDIR)/maskrcnn_trt/ \
		include/maskrcnn_trt/buffers.hpp \
		include/maskrcnn_trt/common.hpp \
		include/maskrcnn_trt/detection.hpp \
		include/maskrcnn_trt/filesystem.hpp \
		include/maskrcnn_trt/half.hpp \
		include/maskrcnn_trt/logger.hpp \
		include/maskrcnn_trt/logging.hpp \
		include/maskrcnn_trt/maskrcnn.hpp \
		include/maskrcnn_trt/maskrcnn_config.hpp
	install -D -m 644 -t $(DATADIR)/pkgconfig/ build/release/maskrcnn-trt.pc

.PHONY: uninstall
uninstall:
	rm $(DATADIR)/pkgconfig/maskrcnn-trt.pc
	rm $(LIBDIR)/libmaskrcnn-trt.a \
		$(HEADERDIR)/maskrcnn_trt/buffers.hpp \
		$(HEADERDIR)/maskrcnn_trt/common.hpp \
		$(HEADERDIR)/maskrcnn_trt/detection.hpp \
		$(HEADERDIR)/maskrcnn_trt/filesystem.hpp \
		$(HEADERDIR)/maskrcnn_trt/half.hpp \
		$(HEADERDIR)/maskrcnn_trt/logger.hpp \
		$(HEADERDIR)/maskrcnn_trt/logging.hpp \
		$(HEADERDIR)/maskrcnn_trt/maskrcnn.hpp \
		$(HEADERDIR)/maskrcnn_trt/maskrcnn_config.hpp
	rmdir $(HEADERDIR)/maskrcnn_trt

