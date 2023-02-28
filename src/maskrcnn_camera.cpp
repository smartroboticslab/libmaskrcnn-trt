// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "maskrcnn_trt/maskrcnn.hpp"

int main(int argc, char** argv) {
    const char *device = "/dev/video0";
    // Ensure the correct number of arguments was supplied.
    if (argc < 2 || argc > 3) {
        std::cout << "maskrcnn-trt-camera MODEL [DEVICE]\n";
        return EXIT_FAILURE;
    } else if (argc == 3) {
        device = argv[2];
    }

    cv::VideoCapture cap(device);
    if(!cap.isOpened()) {
        std::cout << "Error opening device " << device << "\n";
        return EXIT_FAILURE;
    }

    mr::MaskRCNNConfig config;
    config.model_filename = argv[1];
    config.serialized_model_filename = config.model_filename + ".bin";

    mr::MaskRCNN network (config);
    if (!network.build()) {
        return EXIT_FAILURE;
    }

    do {
        cv::Mat image;
        cap.read(image);
        if (image.empty()) {
            cerr << "Error reading image\n";
            return EXIT_FAILURE;
        }

        const auto t_start = std::chrono::high_resolution_clock::now();
        const std::vector<mr::Detection> detections = network.infer(image);
        const auto t_end = std::chrono::high_resolution_clock::now();
        const float t = std::chrono::duration<float, std::milli>(t_end - t_start).count();
        std::cout << "Inference time was " << t << " ms\n";

        for (const auto& detection : detections) {
            std::cout << "  " << detection << "\n";
        }

        cv::imshow("Mask R-CNN", visualize_detections(detections, image));
        if (cv::waitKey(10) == 'q') {
            break;
        }
    } while (true);

    return EXIT_SUCCESS;
}

