// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: Apache-2.0

#ifndef __MASKRCNN_HPP
#define __MASKRCNN_HPP

#include <NvUffParser.h>

#include "buffers.hpp"
#include "detection.hpp"
#include "maskrcnn_config.hpp"

namespace mr {
    class MaskRCNN {
        public:
            /** Initialize a network instance based on the config. In order to
             * load and build the network call MaskRCNN::build() after the
             * constructor.
             */
            MaskRCNN(const MaskRCNNConfig& config);

            /** Create a network instance based on the config. Return true on
             * success.
             */
            bool build();

            /** Run inference on an RGB image and return the resulting
             * detections. The image must be of type CV_8UC3 and in BGR order
             * (the default in OpenCV).
             */
            std::vector<Detection> infer(const cv::Mat& rgb_image);

        private:
            template <typename T>
            using NVUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

            MaskRCNNConfig config_;
            nvinfer1::Dims input_dims_;
            std::shared_ptr<nvinfer1::ICudaEngine> engine_;
            NVUniquePtr<nvinfer1::IExecutionContext> context_;
            std::unique_ptr<samplesCommon::BufferManager> buffer_manager_;

            /** Create the network from a UFF model or by deserializing it.
             */
            bool constructNetwork(nvinfer1::IBuilder&           builder,
                                  IBuilderConfig&               builder_config,
                                  nvinfer1::INetworkDefinition& network,
                                  nvuffparser::IUffParser&      parser);

            /** Resize, pad and copy the input image into the host input buffer.
             */
            void preprocessInput(const samplesCommon::BufferManager& buffer_manager,
                                 const cv::Mat&                      rgb_image);

            /** TODO
             */
            std::vector<Detection> postprocessOutput(
                    const samplesCommon::BufferManager& buffer_manager,
                    int                                 input_width,
                    int                                 input_height);
    };
} // namespace mr

#endif // __MASKRCNN_HPP

