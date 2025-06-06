/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.facelandmarker

/**
 * 注意力状态枚举
 */
enum class AttentionState {
    ATTENTIVE,                    // 专注
    DISTRACTED_LOOKING_AWAY,      // 分心看向别处
    DROWSY_FATIGUED,             // 困倦疲劳
    YAWNING,                     // 打哈欠
    THINKING_CONCENTRATING,       // 思考专注
    CONFUSED,                    // 困惑
    UNKNOWN                      // 未知
}

/**
 * 表情状态枚举
 */
enum class ExpressionState {
    NEUTRAL,          // 中性表情
    SMILING,          // 微笑
    LAUGHING,         // 大笑
    SURPRISED,        // 吃惊
    CONFUSED,         // 困惑
    CONCENTRATED,     // 专注思考
    BORED,           // 无聊
    FRUSTRATED,      // 沮丧
    EXCITED,         // 兴奋
    UNKNOWN          // 未知表情
}

/**
 * Blendshape特征数据类
 */
data class BlendshapeFeatures(
    val eyeBlinkLeft: Float = 0f,
    val eyeBlinkRight: Float = 0f,
    val eyeLookDownLeft: Float = 0f,
    val eyeLookDownRight: Float = 0f,
    val eyeLookUpLeft: Float = 0f,
    val eyeLookUpRight: Float = 0f,
    val eyeLookOutLeft: Float = 0f,
    val eyeLookOutRight: Float = 0f,
    val eyeSquintLeft: Float = 0f,
    val eyeSquintRight: Float = 0f,
    val eyeWideLeft: Float = 0f,
    val eyeWideRight: Float = 0f,
    val browDownLeft: Float = 0f,
    val browDownRight: Float = 0f,
    val browInnerUp: Float = 0f,
    val browOuterUpLeft: Float = 0f,
    val browOuterUpRight: Float = 0f,
    val jawOpen: Float = 0f,
    val mouthSmileLeft: Float = 0f,
    val mouthSmileRight: Float = 0f,
    val mouthPressLeft: Float = 0f,
    val mouthPressRight: Float = 0f,
    val mouthPucker: Float = 0f,
    val mouthShrugLower: Float = 0f,
    val mouthShrugUpper: Float = 0f,
    // 新增表情相关特征
    val cheekSquintLeft: Float = 0f,
    val cheekSquintRight: Float = 0f,
    val mouthFrownLeft: Float = 0f,
    val mouthFrownRight: Float = 0f,
    val mouthRollLower: Float = 0f,
    val mouthRollUpper: Float = 0f,
    val noseSneerLeft: Float = 0f,
    val noseSneerRight: Float = 0f
)

/**
 * 表情分析结果
 */
data class ExpressionResult(
    val primaryExpression: ExpressionState,
    val confidence: Float,
    val secondaryExpression: ExpressionState? = null,
    val intensity: Float = 0f, // 表情强度 0.0-1.0
    val timestamp: Long = System.currentTimeMillis()
)

/**
 * 注意力特征数据类
 */
data class AttentionFeatures(
    val headPoseYaw: Float = 0f,
    val headPosePitch: Float = 0f,
    val headPoseRoll: Float = 0f,
    val leftEyeEAR: Float = 0f,
    val rightEyeEAR: Float = 0f,
    val averageEAR: Float = 0f,
    val blendshapes: BlendshapeFeatures = BlendshapeFeatures(),
    val isHeadPoseAttentive: Boolean = false,
    val areEyesOpen: Boolean = false,
    val confidence: Float = 0f,
    val timestamp: Long = System.currentTimeMillis()
)

/**
 * 综合分析结果（注意力 + 表情）
 */
data class ComprehensiveAnalysisResult(
    val attentionResult: AttentionResult,
    val expressionResult: ExpressionResult,
    val overallEngagement: Float, // 综合参与度 0.0-1.0
    val timestamp: Long = System.currentTimeMillis()
)

/**
 * 用户基准数据类
 */
data class UserBaseline(
    val avgEAR: Float = 0.25f,
    val avgYawRange: Float = 30f,
    val avgPitchRange: Float = 20f,
    val calibrationCount: Int = 0,
    val isCalibrated: Boolean = false
)

/**
 * 眼部关键点索引
 */
data class EyeIndices(
    val p1: Int, // 外角
    val p2: Int, // 上眼睑点1
    val p3: Int, // 上眼睑点2
    val p4: Int, // 内角
    val p5: Int, // 下眼睑点1
    val p6: Int  // 下眼睑点2
)

/**
 * 注意力检测结果
 */
data class AttentionResult(
    val state: AttentionState,
    val confidence: Float,
    val features: AttentionFeatures,
    val timestamp: Long = System.currentTimeMillis()
) 