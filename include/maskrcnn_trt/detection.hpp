// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: Apache-2.0

#ifndef __DETECTIONS_HPP
#define __DETECTIONS_HPP

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>

namespace mr {
    /** A single object detection.
     */
    struct Detection {
        /** The class ID of the detection. In the range [0-80] inclusive.
         */
        int class_id     = 0;
        /** The confidence the network assigned to this detection. In the range
         * [0-1] inclusive.
         */
        float confidence = 0.0f;
        /** The x coordinate of the top left corner of the bounding box.
         */
        float x_start    = 0.0f;
        /** The y coordinate of the top left corner of the bounding box.
         */
        float y_start    = 0.0f;
        /** The x coordinate + 1 of the bottom right corner of the bounding box.
         */
        float x_end      = 0.0f;
        /** The y coordinate + 1 of the bottom right corner of the bounding box.
         */
        float y_end      = 0.0f;
        /** The mask of the detection. Its type is CV_8UC1. In the range [0-255]
         * inclusive.
         */
        cv::Mat mask;
    };

    std::ostream& operator<<(std::ostream& os, const Detection& d);



    /** Get the detections from the host buffers and create a vector of
     * Detection structs. The input_width and input_height should be those of
     * the original input image, before preprocessing.
     *
     * \note The original Nvidia code set any mask values above
     * MaskRCNNConfig::mask_threshold to 1. Here it is left up to the user to do
     * this if needed.
     */
    std::vector<Detection> get_detections(int         input_width,
                                          int         input_height,
                                          const void* detection_buffer,
                                          const void* mask_buffer);

    /** Render the detections on the image and return the render.
     */
    cv::Mat visualize_detections(const std::vector<Detection>& detections,
                                 const cv::Mat&                image);
} // namespace mr

#endif // __DETECTIONS_HPP

