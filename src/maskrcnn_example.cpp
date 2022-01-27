// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>

#include <opencv2/imgcodecs.hpp>

#include "maskrcnn_trt/maskrcnn.hpp"

int main(int argc, char** argv) {
    // Ensure the correct number of arguments was supplied.
    if (argc < 3) {
        std::cout << "maskrcnn-trt-example MODEL IMAGE...\n";
        return EXIT_FAILURE;
    }

    // Setup the network configuration struct.
    // mr::MaskRCNNConfig::model_filename is
    // required and should be the path to the Uff network model.
    mr::MaskRCNNConfig config;
    config.model_filename = argv[1];
    // If mr::MaskRCNNConfig::serialized_model_filename is empty the network
    // will be serialized to a device-specific format each time it is created.
    // If mr::MaskRCNNConfig::serialized_model_filename is non-empty but refers
    // to a non-existent file the device-specific serialization will be saved to
    // that file. If mr::MaskRCNNConfig::serialized_model_filename refers to an
    // existing file the device-specific serialization will be loaded from that
    // file, speeding up the network creation process.
    config.serialized_model_filename = config.model_filename + ".bin";

    // Create a network instance.
    mr::MaskRCNN network (config);
    // Load and build the network. Error messages will be printed to standard
    // error if this fails.
    if (!network.build()) {
        return EXIT_FAILURE;
    }

    // Run inference on each input image.
    for (int i = 2; i < argc; i ++) {
        // Read the input image.
        const std::string filename (argv[i]);
        const cv::Mat image = cv::imread(filename);

        // Time the inference.
        const auto t_start = std::chrono::high_resolution_clock::now();
        // Pass the image through the network and get the detections.
        const std::vector<mr::Detection> detections = network.infer(image);
        const auto t_end = std::chrono::high_resolution_clock::now();
        const float t = std::chrono::duration<float, std::milli>(t_end - t_start).count();
        std::cout << "Inference time for " << argv[i] << " was " << t << " ms\n";

        // Print the detections on standard output.
        for (const auto& detection : detections) {
            std::cout << "  " << detection << "\n";
        }

        // Visualize the detections and save them to an image file.
        const std::string vis_filename = filename + ".detections.png";
        if (cv::imwrite(vis_filename, visualize_detections(detections, image))) {
            std::cout << "Saved detection visualization in " << vis_filename << "\n";
        } else {
            std::cerr << "Error saving detection visualization in " << vis_filename << "\n";
        }
        std::cout << "\n";
    }

    return EXIT_SUCCESS;
}

