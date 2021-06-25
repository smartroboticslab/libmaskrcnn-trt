// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: Apache-2.0

#include <opencv2/imgproc.hpp>

#include "maskrcnn_trt/maskrcnn.hpp"
#include "maskrcnn_trt/filesystem.hpp"

namespace mr {
    MaskRCNN::MaskRCNN(const MaskRCNNConfig& config)
        : config_(config)
    {
        srand((int) time(nullptr));
    }



    bool MaskRCNN::build()
    {
        initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
        auto builder = NVUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
        if (!builder) {
            return false;
        }
        NVUniquePtr<IBuilderConfig> builder_config (builder->createBuilderConfig());
        if (config_.use_fp16) {
            builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        builder_config->setMaxWorkspaceSize(config_.max_workspace_size);

        auto network = NVUniquePtr<nvinfer1::INetworkDefinition>(
                builder->createNetworkV2(static_cast<uint32_t>(nvinfer1::EngineCapability::kDEFAULT)));
        if (!network) {
            return false;
        }

        auto parser = NVUniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());
        if (!parser) {
            return false;
        }

        bool constructed = constructNetwork(*builder, *builder_config, *network, *parser);
        if (!constructed) {
            return false;
        }

        context_ = NVUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_) {
            return false;
        }

        // Create the host/device buffer manager.
        buffer_manager_ = std::make_unique<samplesCommon::BufferManager>(engine_, config_.batch_size);

        // Ensure the network has the expected number of inputs and outputs.
        assert(network->getNbInputs() == 1);
        assert(network->getNbOutputs() == 2);
        // Ensure the network input has the expected shape.
        input_dims_ = network->getInput(0)->getDimensions();
        assert(input_dims_.nbDims == 3);
        assert(input_dims_.d[0] == MaskRCNNConfig::model_input_shape[0]);
        assert(input_dims_.d[1] == MaskRCNNConfig::model_input_shape[1]);
        assert(input_dims_.d[2] == MaskRCNNConfig::model_input_shape[2]);
        // Ensure the network output has the expected shape.
        // TODO
        return true;
    }



    std::vector<Detection> MaskRCNN::infer(const cv::Mat& rgb_image)
    {
        // Ensure the network has been built before running inference.
        if (!context_) {
            gLogError << "Error: The network must be built using build() before running infer()"
                << std::endl;
            return std::vector<Detection>();
        }

        // Read the input data into the host buffer.
        preprocessInput(*buffer_manager_, rgb_image);

        // Copy image from the host input buffer to the device input buffer.
        buffer_manager_->copyInputToDevice();

        // Run and time inference.
        const bool status = context_->execute(config_.batch_size, buffer_manager_->getDeviceBindings().data());
        if (!status) {
            return std::vector<Detection>();
        }

        // Copy the detections from the device output buffers to the host output
        // buffers.
        buffer_manager_->copyOutputToHost();

        // Post-process the detections into a Detection vector.
        return postprocessOutput(*buffer_manager_, rgb_image.cols, rgb_image.rows);
    }



    bool MaskRCNN::constructNetwork(nvinfer1::IBuilder&           builder,
                                    IBuilderConfig&               builder_config,
                                    nvinfer1::INetworkDefinition& network,
                                    nvuffparser::IUffParser&      parser)
    {
        const nvinfer1::Dims3 shape (MaskRCNNConfig::model_input_shape[0],
                MaskRCNNConfig::model_input_shape[1],
                MaskRCNNConfig::model_input_shape[2]);
        parser.registerInput(MaskRCNNConfig::model_input.c_str(), shape,
                nvuffparser::UffInputOrder::kNCHW);
        parser.registerOutput(MaskRCNNConfig::model_outputs[0].c_str());
        parser.registerOutput(MaskRCNNConfig::model_outputs[1].c_str());

        auto parsed = parser.parse(config_.model_filename.c_str(), network, DataType::kFLOAT);
        if (!parsed) {
            return false;
        }

        builder.setMaxBatchSize(config_.batch_size);

        // Test if the a serialized model is available.
        if (!config_.serialized_model_filename.empty()
                && stdfs::is_regular_file(config_.serialized_model_filename)) {
            // Open the file and go to the end of the stream.
            std::ifstream f (config_.serialized_model_filename, std::ios::binary | std::ios::ate);
            if (!f.is_open()) {
                gLogError << "Error: Could not read serialized network model from "
                    << config_.serialized_model_filename << std::endl;
                return false;
            }
            // Get the stream size
            const std::streamsize s = f.tellg();
            f.seekg(0, std::ios::beg);

            std::vector<char> buffer(s);
            // TODO Check for error below: if (!f.read(...)) { return false; }?
            f.read(buffer.data(), buffer.size());
            nvinfer1::IRuntime* runtime = createInferRuntime(gLogger);
            nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(
                    reinterpret_cast<void*>(buffer.data()), buffer.size());
            delete runtime;
            if (!engine) {
                gLogError << "Error: Could not create engine from serialized network model "
                    << config_.serialized_model_filename << std::endl;
                return false;
            }
            engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(engine, samplesCommon::InferDeleter());
            gLogInfo << "Loaded serialized network model from "
                << config_.serialized_model_filename << std::endl;
        } else {
            // Build the network
            engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(builder.buildEngineWithConfig(
                        network, builder_config), samplesCommon::InferDeleter());
            if (!engine_) {
                return false;
            }

            // Serialize the network
            if (!config_.serialized_model_filename.empty()) {
                IHostMemory* serializedModel = engine_->serialize();
                std::ofstream f (config_.serialized_model_filename, std::ios::binary);
                if (f.is_open()) {
                    f.write(reinterpret_cast<char*>(serializedModel->data()), serializedModel->size());
                    delete serializedModel;
                    gLogInfo << "Saved serialized network model to "
                        << config_.serialized_model_filename << std::endl;
                } else {
                    gLogWarning << "Warning: Could not write serialized network model to "
                        << config_.serialized_model_filename << std::endl;
                }
            }
        }

        return true;
    }



    void MaskRCNN::preprocessInput(const samplesCommon::BufferManager& buffer_manager,
                                   const cv::Mat&                      rgb_image)
    {
        const int net_channels = input_dims_.d[0];
        const int net_height = input_dims_.d[1];
        const int net_width = input_dims_.d[2];
        const int net_type = CV_8UC(net_channels);
        // Ensure the input image has the same pixel type as the network.
        assert(rgb_image.type() == net_type);
        // This is the image that will be passed to the network. The zero
        // initialization is important for padding purposes.
        cv::Mat net_image (net_height, net_width, net_type, cv::Scalar(0));

        // Find the dimensions that rgb_image must be resized to so that its
        // maximum dimension is the same as the net_width (which should be the
        // same as net_height) while keeping the aspect ratio.
        const int input_max_dim = std::max(rgb_image.rows, rgb_image.cols);
        const double scaling_factor = (double) net_width / input_max_dim;
        const int input_new_width = rgb_image.cols * scaling_factor;
        const int input_new_height = rgb_image.rows * scaling_factor;

        // The coordinates in net_image where rgb_image will be copied.
        const int x_start = (net_width - input_new_width) / 2;
        const int y_start = (net_height - input_new_height) / 2;
        const int x_end = x_start + input_new_width;
        const int y_end = y_start + input_new_height;

        // Get a view of the center of the net_image where the resized rgb_image
        // will be copied into.
        cv::Mat centre_image
            = net_image(cv::Range(y_start, y_end), cv::Range(x_start, x_end));

        // Resize the input image into the centre of the network image.
        cv::resize(rgb_image, centre_image, centre_image.size());

        // Change the channel order from BGR to RGB.
        cv::cvtColor(net_image, net_image, cv::COLOR_BGR2RGB);

        // The image data must be in continuous memory.
        if (!net_image.isContinuous()) {
            net_image = net_image.clone();
        }

        // Get a pointer to the host input buffer.
        float* host_input_buffer = static_cast<float*>(buffer_manager.getHostBuffer(MaskRCNNConfig::model_input));
        // Copy the image into the host buffer. The channels are not interleaved
        // in the host buffer.
        const size_t num_pixels = net_image.total();
        const size_t num_channels = net_image.channels();
        for (int c = 0; c < num_channels; c++) {
            for (size_t p = 0; p < num_pixels; p++) {
                host_input_buffer[c * num_pixels + p] = (float) net_image.data[p * num_channels + c] - MaskRCNNConfig::network_bias[c];
            }
        }
    }



    std::vector<Detection> MaskRCNN::postprocessOutput(
            const samplesCommon::BufferManager& buffer_manager,
            int                                 input_width,
            int                                 input_height)
    {
        const void* host_detection_buffer = buffer_manager.getHostBuffer(MaskRCNNConfig::model_outputs[0]);
        const void* host_mask_buffer = buffer_manager.getHostBuffer(MaskRCNNConfig::model_outputs[1]);
        return get_detections(input_width, input_height, host_detection_buffer, host_mask_buffer);
    }
} // namespace mr

