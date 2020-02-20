// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include <opencv2/imgproc.hpp>

#include "detection.hpp"
#include "maskrcnn_config.hpp"

namespace mr {
    /** A detection as stored in host memory.
     */
    struct RawDetection {
        float y_start    = 0.0f;
        float x_start    = 0.0f;
        float y_end      = 0.0f;
        float x_end      = 0.0f;
        float class_id   = 0.0f;
        float confidence = 0.0f;
    };



    /** An instance mask as stored in host memory.
     */
    typedef float RawMask[2 * MaskRCNNConfig::mask_pool_size * 2 * MaskRCNNConfig::mask_pool_size];



    std::ostream& operator<<(std::ostream& os, const Detection& d)
    {
        os << MaskRCNNConfig::class_names[d.class_id]
            << " with confidence " << d.confidence
            << ", bbox (" << d.x_start << ", " << d.y_start
            << "), (" << d.x_end << ", " << d.y_end << ")"
            << " and mask " << d.mask.cols << "x" << d.mask.rows;
        return os;
    }



    std::vector<Detection> get_detections(int         input_width,
                                          int         input_height,
                                          const void* detection_buffer,
                                          const void* mask_buffer)
    {
        std::vector<Detection> detections;

        const int net_width = MaskRCNNConfig::model_input_shape[2];
        const int net_height = MaskRCNNConfig::model_input_shape[1];
        const int max_input_dim = std::max(input_height, input_width);

        // The dimensions of the input image after it has been resized to fit
        // the network but before padding.
        const int input_new_width = input_width * net_width / max_input_dim;
        const int input_new_height = input_height * net_height / max_input_dim;
        // TODO clean up
        // keep accurary from (float) to (int), then to float
        float window_x = (1.0f - (float) input_new_width / net_width) / 2.0f;
        float window_y = (1.0f - (float) input_new_height / net_height) / 2.0f;
        float window_width = (float) input_new_width / net_width;
        float window_height = (float) input_new_height / net_height;
        float final_ratio_x = (float) input_width / window_width;
        float final_ratio_y = (float) input_height / window_height;

        // There is no image offset since we assume a batch size of 1 so
        // inference is run on a single image.
        const RawDetection* raw_detections = reinterpret_cast<const RawDetection*>(detection_buffer);
        const RawMask* raw_masks = reinterpret_cast<const RawMask*>(mask_buffer);
        // Loop over all possible detections.
        for (int d = 0; d < MaskRCNNConfig::detection_max_instances; d++) {
            const RawDetection raw_detection = raw_detections[d];
            const int class_id = raw_detection.class_id;
            // Skip detections with invalid class IDs.
            if (class_id <= 0) {
                continue;
            }

            // TODO clean up
            const float x_start = std::clamp((raw_detection.x_start - window_x) * final_ratio_x, 0.0f, (float) input_width);
            const float y_start = std::clamp((raw_detection.y_start - window_y) * final_ratio_y, 0.0f, (float) input_height);
            const float x_end = std::clamp((raw_detection.x_end - window_x) * final_ratio_x, 0.0f, (float) input_width);
            const float y_end = std::clamp((raw_detection.y_end - window_y) * final_ratio_y, 0.0f, (float) input_height);
            // Skip detections with invalid bounding boxes.
            if (x_end <= x_start || y_end <= y_start) {
                continue;
            }

            Detection detection = {class_id, raw_detection.confidence, x_start, y_start, x_end, y_end};

            // Initialize a mask from the raw data.
            RawMask* raw_mask_data = (RawMask*) raw_masks + d * MaskRCNNConfig::num_classes + class_id;
            cv::Mat raw_mask (2 * MaskRCNNConfig::mask_pool_size, 2 * MaskRCNNConfig::mask_pool_size, CV_32FC1, raw_mask_data);
            // Convert the float mask to an int mask.
            cv::Mat int_mask;
            raw_mask.convertTo(int_mask, CV_8UC1, UINT8_MAX);
            // Resize the mask to the bounding box dimensions.
            cv::Mat box_mask;
            const int box_width = detection.x_end - detection.x_start;
            const int box_height = detection.y_end - detection.y_start;
            cv::resize(int_mask, box_mask, cv::Size(box_width, box_height));
            // Initialize a mask for the whole input image.
            detection.mask = cv::Mat(input_height, input_width, CV_8UC1, cv::Scalar(0));
            // Get the ROI of the bounding box portion of the whole image mask
            // and copy the mask there.
            cv::Rect roi (x_start, y_start, box_width, box_height);
            cv::Mat mask_roi = detection.mask(roi);
            box_mask.copyTo(mask_roi);

            detections.push_back(detection);
        }
        return detections;
    }



    cv::Mat visualize_detections(const std::vector<Detection>& detections,
                                 const cv::Mat&                image)
    {
        cv::Mat render = image.clone();
        // Overlay the detection masks first to avoid affecting the other
        // overlays.
        for (size_t i = 0; i < detections.size(); i++) {
            const Detection& d = detections[i];
            // Select the colour to use based on the class ID.
            const uint8_t* colour = MaskRCNNConfig::class_colours[d.class_id];
            const cv::Scalar cv_colour (colour[2], colour[1], colour[0]);
            // Blend the colour mask with a solid colour image.
            cv::Mat colour_image (d.mask.size(), CV_8UC3, cv_colour);
            cv::Mat blended_render;
            cv::addWeighted(render, 1.0 - MaskRCNNConfig::mask_threshold,
                    colour_image, MaskRCNNConfig::mask_threshold, 0.0, blended_render);
            // Binarize the mask to make blending simpler.
            cv::Mat binary_mask;
            cv::threshold(d.mask, binary_mask, UINT8_MAX/2.0, UINT8_MAX, cv::THRESH_BINARY);
            // Use only the masked part of the blended render.
            blended_render.copyTo(render, binary_mask);
        }
        // Then overlay the rest of the info.
        for (size_t i = 0; i < detections.size(); i++) {
            const Detection& d = detections[i];
            // Select the colour to use based on the class ID.
            const uint8_t* colour = MaskRCNNConfig::class_colours[d.class_id];
            const cv::Scalar cv_colour (colour[2], colour[1], colour[0]);
            // Draw the bounding box.
            cv::rectangle(render, cv::Point(d.x_start, d.y_start),
                    cv::Point(d.x_end - 1, d.y_end - 1), cv_colour);
            // Draw the class name and confidence.
            const std::string label = MaskRCNNConfig::class_names[d.class_id]
                + " " + std::to_string(d.confidence).substr(0, 4);
            cv::putText(render, label, cv::Point(d.x_start, d.y_start - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv_colour);
         }
        return render;
    }
} // namespace mr

