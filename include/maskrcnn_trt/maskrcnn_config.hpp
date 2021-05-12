// SPDX-FileCopyrightText: 2019 NVIDIA Corporation
// SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
// SPDX-License-Identifier: Apache-2.0

#ifndef __MASKRCNN_CONFIG_HPP
#define __MASKRCNN_CONFIG_HPP

#include <string>

namespace mr {
    /** Constant and runtime parameters of Mask RCNN. Used to initialize an
     * instance of MaskRCNN.
     */
    struct MaskRCNNConfig {
        /** The filename of the Uff model to use.
         */
        std::string model_filename;
        /** The filename of a serialized version of the model. If the supplied
         * file does not exist it will be created.
         */
        std::string serialized_model_filename;
        /** Run the network in 16-bit float mode.
         */
        bool use_fp16 = false;
        /** Use up to 2 GiB of VRAM for the workspace by default.
         */
        size_t max_workspace_size = (1ULL << 31);



        // Constant network parameters, do not change //////////////////////////
        // Number of inputs in a batch.
        static constexpr int batch_size = 1;

        // Pooled ROIs.
        static constexpr int pool_size = 7;
        static constexpr int mask_pool_size = 14;

        // Threshold to determine the mask area out of final convolution output.
        static constexpr float mask_threshold = 0.5f;

        // Bounding box refinement standard deviation for RPN and final
        // detections.
        static constexpr float rpn_bbox_std_dev[4] = {0.1f, 0.1f, 0.2f, 0.2f};
        static constexpr float bbox_std_dev[4] = {0.1f, 0.1f, 0.2, 0.2f};

        // Max number of final detections.
        static constexpr int detection_max_instances = 100;

        // Minimum probability value to accept a detected instance. ROIs below
        // this threshold are skipped.
        static constexpr float detection_min_confidence = 0.7f;

        // Non-maximum suppression threshold for detection.
        static constexpr float detection_nms_threshold = 0.3f;

        // The strides of each layer of the FPN Pyramid. These values are based
        // on a Resnet101 backbone.
        static constexpr float backbone_strides[5] = {4, 8, 16, 32, 64};

        // Size of the fully-connected layers in the classification graph.
        static constexpr int fpn_classif_fc_layers_size = 1024;

        // Size of the top-down layers used to build the feature pyramid.
        static constexpr int top_down_pyramid_size = 256;

        // Number of classification classes (including the background). COCO has
        // 80 classes.
        static constexpr int num_classes = 1 + 80;

        // Length of square anchor side in pixels
        static constexpr float rpn_anchor_scales[5] = {32, 64, 128, 256, 512};

        // Ratios of anchors at each cell (width/height)
        // A value of 1 represents a square anchor, and 0.5 is a wide anchor
        static constexpr float rpn_anchor_ratios[3] = {0.5f, 1.0f, 2.0f};

        // Anchor stride
        // If 1 then anchors are created for each cell in the backbone feature map.
        // If 2, then anchors are created for every other cell, and so on.
        static constexpr int rpn_anchor_stride = 1;

        // Although the Python impementation uses 6000, TRT fails if this number
        // is larger than MAX_TOPK_K defined in engine/checkMacros.h
        static constexpr int max_pre_nms_results = 1024;

        // Non-max suppression threshold to filter RPN proposals.
        // You can increase this during training to generate more propsals.
        static constexpr float rpn_nms_threshold = 0.7f;

        // ROIs kept after non-maximum suppression (training and inference)
        static constexpr int post_nms_rois_inference = 1000;

        // The network bias needs to be subtracted from each input image pixel.
        static constexpr float network_bias[3] = {123.7f, 116.8f, 103.9f};

        // The name of the input tensor.
        static const std::string model_input;
        // The shape of the input image (channels, height, width).
        static constexpr int model_input_shape[3] = {3, 1024, 1024};
        // The names of the output tensors.
        static const std::string model_outputs[2];
        // The shape of the detection information.
        static constexpr int model_detection_shape[2] = {MaskRCNNConfig::detection_max_instances, 6};
        // The shape of the detection masks.
        static constexpr int model_mask_shape[4] = {MaskRCNNConfig::detection_max_instances, MaskRCNNConfig::num_classes, 2 * mask_pool_size, 2 * mask_pool_size};

        // COCO Class names
        static const std::string class_names[num_classes];
        static constexpr uint8_t class_colours[num_classes][3] = {
            {0x00, 0x00, 0x00},
            {0xae, 0xc7, 0xe8},
            {0x70, 0x80, 0x90},
            {0x98, 0xdf, 0x8a},
            {0xc5, 0xb0, 0xd5},
            {0xff, 0x7f, 0x0e},
            {0xd6, 0x27, 0x28},
            {0x1f, 0x77, 0xb4},
            {0xbc, 0xbd, 0x22},
            {0xff, 0x98, 0x96},
            {0x2c, 0xa0, 0x2c},
            {0xe3, 0x77, 0xc2},
            {0xde, 0x9e, 0xd6},
            {0x94, 0x67, 0xbd},
            {0x8c, 0xa2, 0x52},
            {0x84, 0x3c, 0x39},
            {0x9e, 0xda, 0xe5},
            {0x9c, 0x9e, 0xde},
            {0xe7, 0x96, 0x9c},
            {0x63, 0x79, 0x39},
            {0x8c, 0x56, 0x4b},
            {0xdb, 0xdb, 0x8d},
            {0xd6, 0x61, 0x6b},
            {0xce, 0xdb, 0x9c},
            {0xe7, 0xba, 0x52},
            {0x39, 0x3b, 0x79},
            {0xa5, 0x51, 0x94},
            {0xad, 0x49, 0x4a},
            {0xb5, 0xcf, 0x6b},
            {0x52, 0x54, 0xa3},
            {0xbd, 0x9e, 0x39},
            {0xc4, 0x9c, 0x94},
            {0xf7, 0xb6, 0xd2},
            {0x6b, 0x6e, 0xcf},
            {0xff, 0xbb, 0x78},
            {0xc7, 0xc7, 0xc7},
            {0x8c, 0x6d, 0x31},
            {0xe7, 0xcb, 0x94},
            {0xce, 0x6d, 0xbd},
            {0x17, 0xbe, 0xcf},
            {0xae, 0xc7, 0xe8},
            {0x70, 0x80, 0x90},
            {0x98, 0xdf, 0x8a},
            {0xc5, 0xb0, 0xd5},
            {0xff, 0x7f, 0x0e},
            {0xd6, 0x27, 0x28},
            {0x1f, 0x77, 0xb4},
            {0xbc, 0xbd, 0x22},
            {0xff, 0x98, 0x96},
            {0x2c, 0xa0, 0x2c},
            {0xe3, 0x77, 0xc2},
            {0xde, 0x9e, 0xd6},
            {0x94, 0x67, 0xbd},
            {0x8c, 0xa2, 0x52},
            {0x84, 0x3c, 0x39},
            {0x9e, 0xda, 0xe5},
            {0x9c, 0x9e, 0xde},
            {0xe7, 0x96, 0x9c},
            {0x63, 0x79, 0x39},
            {0x8c, 0x56, 0x4b},
            {0xdb, 0xdb, 0x8d},
            {0xd6, 0x61, 0x6b},
            {0xce, 0xdb, 0x9c},
            {0xe7, 0xba, 0x52},
            {0x39, 0x3b, 0x79},
            {0xa5, 0x51, 0x94},
            {0xad, 0x49, 0x4a},
            {0xb5, 0xcf, 0x6b},
            {0x52, 0x54, 0xa3},
            {0xbd, 0x9e, 0x39},
            {0xc4, 0x9c, 0x94},
            {0xf7, 0xb6, 0xd2},
            {0x6b, 0x6e, 0xcf},
            {0xff, 0xbb, 0x78},
            {0xc7, 0xc7, 0xc7},
            {0x8c, 0x6d, 0x31},
            {0xe7, 0xcb, 0x94},
            {0xce, 0x6d, 0xbd},
            {0x17, 0xbe, 0xcf},
            {0xae, 0xc7, 0xe8},
            {0x70, 0x80, 0x90}};
    };
} // namespace mr

#endif // __MASKRCNN_CONFIG_HPP

