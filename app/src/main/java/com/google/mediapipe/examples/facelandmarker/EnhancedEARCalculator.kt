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

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.*

/**
 * 增强的EAR（Eye Aspect Ratio）算法
 * 包含3D距离计算、头部倾斜修正和多关键点优化
 */
class EnhancedEARCalculator {
    
    companion object {
        // MediaPipe 468 Face Landmark 眼部关键点索引
        // 左眼（从人物视角）
        val LEFT_EYE_INDICES = EyeIndices(
            p1 = 362, // 外角
            p2 = 385, // 上眼睑点1 
            p3 = 387, // 上眼睑点2
            p4 = 263, // 内角
            p5 = 373, // 下眼睑点1
            p6 = 380  // 下眼睑点2
        )
        
        // 右眼（从人物视角）
        val RIGHT_EYE_INDICES = EyeIndices(
            p1 = 133, // 外角
            p2 = 158, // 上眼睑点1
            p3 = 160, // 上眼睑点2
            p4 = 33,  // 内角
            p5 = 144, // 下眼睑点1
            p6 = 153  // 下眼睑点2
        )
        
        // 扩展的眼部关键点（用于更精确的EAR计算）
        val LEFT_EYE_EXTENDED = listOf(362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382)
        val RIGHT_EYE_EXTENDED = listOf(133, 173, 157, 158, 159, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155)
        
        // 平滑滤波参数
        private const val SMOOTHING_ALPHA = 0.3f
        private const val MIN_EAR_VALUE = 0.05f
        private const val MAX_EAR_VALUE = 0.5f
    }
    
    // 历史EAR值用于平滑
    private var leftEyeEARHistory = CircularBuffer<Float>(5)
    private var rightEyeEARHistory = CircularBuffer<Float>(5)
    
    /**
     * 计算增强的双眼EAR
     */
    fun calculateEnhancedEAR(
        landmarks: List<NormalizedLandmark>,
        headPoseYaw: Float = 0f,
        headPosePitch: Float = 0f,
        headPoseRoll: Float = 0f
    ): Triple<Float, Float, Float> { // 返回 (左眼EAR, 右眼EAR, 平均EAR)
        
        if (landmarks.size <= maxOf(
                LEFT_EYE_EXTENDED.maxOrNull() ?: 0,
                RIGHT_EYE_EXTENDED.maxOrNull() ?: 0
            )) {
            return Triple(0f, 0f, 0f)
        }
        
        // 计算基础EAR
        val leftEAR = calculateAdvancedEAR(landmarks, LEFT_EYE_INDICES, LEFT_EYE_EXTENDED)
        val rightEAR = calculateAdvancedEAR(landmarks, RIGHT_EYE_INDICES, RIGHT_EYE_EXTENDED)
        
        // 应用头部姿态修正
        val correctedLeftEAR = applyHeadPoseCorrection(leftEAR, headPoseYaw, headPosePitch, headPoseRoll, isLeftEye = true)
        val correctedRightEAR = applyHeadPoseCorrection(rightEAR, headPoseYaw, headPosePitch, headPoseRoll, isLeftEye = false)
        
        // 平滑滤波
        val smoothedLeftEAR = applySmoothingFilter(correctedLeftEAR, leftEyeEARHistory)
        val smoothedRightEAR = applySmoothingFilter(correctedRightEAR, rightEyeEARHistory)
        
        // 计算加权平均EAR（考虑头部旋转对左右眼的不同影响）
        val averageEAR = calculateWeightedAverageEAR(smoothedLeftEAR, smoothedRightEAR, headPoseYaw)
        
        return Triple(smoothedLeftEAR, smoothedRightEAR, averageEAR)
    }
    
    /**
     * 高级EAR计算，使用多个关键点
     */
    private fun calculateAdvancedEAR(
        landmarks: List<NormalizedLandmark>,
        basicIndices: EyeIndices,
        extendedIndices: List<Int>
    ): Float {
        // 基础6点EAR
        val basicEAR = calculateBasicEAR(landmarks, basicIndices)
        
        // 扩展多点EAR（使用更多关键点提高精度）
        val extendedEAR = calculateExtendedEAR(landmarks, extendedIndices)
        
        // 加权组合
        return basicEAR * 0.7f + extendedEAR * 0.3f
    }
    
    /**
     * 基础6点EAR计算
     */
    private fun calculateBasicEAR(landmarks: List<NormalizedLandmark>, indices: EyeIndices): Float {
        val p1 = landmarks[indices.p1] // 外角
        val p2 = landmarks[indices.p2] // 上眼睑点1
        val p3 = landmarks[indices.p3] // 上眼睑点2
        val p4 = landmarks[indices.p4] // 内角
        val p5 = landmarks[indices.p5] // 下眼睑点1
        val p6 = landmarks[indices.p6] // 下眼睑点2
        
        // 计算3D距离
        val verticalDist1 = calculate3DDistance(p2, p6)
        val verticalDist2 = calculate3DDistance(p3, p5)
        val horizontalDist = calculate3DDistance(p1, p4)
        
        // 避免除零
        if (horizontalDist < 0.001f) return 0f
        
        return (verticalDist1 + verticalDist2) / (2.0f * horizontalDist)
    }
    
    /**
     * 扩展多点EAR计算
     */
    private fun calculateExtendedEAR(landmarks: List<NormalizedLandmark>, indices: List<Int>): Float {
        if (indices.size < 8) return 0f
        
        // 将关键点分为上下两组
        val upperPoints = indices.take(indices.size / 2).map { landmarks[it] }
        val lowerPoints = indices.drop(indices.size / 2).map { landmarks[it] }
        
        // 计算上下眼睑的平均距离
        var totalVerticalDist = 0f
        var count = 0
        
        for (i in upperPoints.indices) {
            if (i < lowerPoints.size) {
                totalVerticalDist += calculate3DDistance(upperPoints[i], lowerPoints[i])
                count++
            }
        }
        
        if (count == 0) return 0f
        
        val avgVerticalDist = totalVerticalDist / count
        
        // 计算眼睛水平长度（外角到内角）
        val horizontalDist = if (indices.size >= 2) {
            calculate3DDistance(landmarks[indices.first()], landmarks[indices[indices.size / 2]])
        } else {
            1f
        }
        
        return if (horizontalDist > 0.001f) avgVerticalDist / horizontalDist else 0f
    }
    
    /**
     * 计算3D距离（考虑Z轴深度）
     */
    private fun calculate3DDistance(point1: NormalizedLandmark, point2: NormalizedLandmark): Float {
        val dx = point1.x() - point2.x()
        val dy = point1.y() - point2.y()
        val dz = (point1.z() - point2.z()) * 0.1f // Z轴权重较小，因为主要是2D分析
        
        return sqrt(dx * dx + dy * dy + dz * dz)
    }
    
    /**
     * 应用头部姿态修正
     */
    private fun applyHeadPoseCorrection(
        ear: Float,
        yaw: Float,
        pitch: Float,
        roll: Float,
        isLeftEye: Boolean
    ): Float {
        // 将角度转换为弧度
        val yawRad = Math.toRadians(yaw.toDouble()).toFloat()
        val pitchRad = Math.toRadians(pitch.toDouble()).toFloat()
        val rollRad = Math.toRadians(roll.toDouble()).toFloat()
        
        // 头部旋转对EAR的影响系数
        var correctionFactor = 1.0f
        
        // Yaw旋转修正（左右转头）
        if (isLeftEye) {
            // 左眼在头部向右转时更容易被遮挡
            correctionFactor *= (1.0f + abs(yawRad) * 0.3f)
        } else {
            // 右眼在头部向左转时更容易被遮挡
            correctionFactor *= (1.0f + abs(yawRad) * 0.3f)
        }
        
        // Pitch旋转修正（上下点头）
        // 低头时眼睛显得更小，抬头时显得更大
        correctionFactor *= (1.0f + pitchRad * 0.2f)
        
        // Roll旋转修正（歪头）
        // 歪头对EAR的影响相对较小
        correctionFactor *= (1.0f + abs(rollRad) * 0.1f)
        
        // 限制修正范围
        correctionFactor = correctionFactor.coerceIn(0.5f, 2.0f)
        
        return (ear * correctionFactor).coerceIn(MIN_EAR_VALUE, MAX_EAR_VALUE)
    }
    
    /**
     * 应用平滑滤波
     */
    private fun applySmoothingFilter(currentEAR: Float, history: CircularBuffer<Float>): Float {
        history.add(currentEAR)
        
        val earList = history.getAll()
        if (earList.size < 2) return currentEAR
        
        // 指数加权移动平均
        var smoothedEAR = currentEAR
        for (i in earList.size - 2 downTo 0) {
            smoothedEAR = SMOOTHING_ALPHA * earList[i] + (1 - SMOOTHING_ALPHA) * smoothedEAR
        }
        
        return smoothedEAR.coerceIn(MIN_EAR_VALUE, MAX_EAR_VALUE)
    }
    
    /**
     * 计算加权平均EAR
     */
    private fun calculateWeightedAverageEAR(leftEAR: Float, rightEAR: Float, headYaw: Float): Float {
        // 根据头部旋转调整左右眼权重
        val yawRad = Math.toRadians(headYaw.toDouble()).toFloat()
        
        // 头部向右转时，右眼权重增加；向左转时，左眼权重增加
        val leftWeight = (cos(yawRad) + 1.0f) / 2.0f
        val rightWeight = 1.0f - leftWeight
        
        return leftEAR * leftWeight + rightEAR * rightWeight
    }
    
    /**
     * 检测眨眼事件
     */
    fun detectBlink(currentEAR: Float, threshold: Float): Boolean {
        // 简单的阈值检测
        return currentEAR < threshold
    }
    
    /**
     * 高级眨眼检测（基于EAR变化率）
     */
    fun detectBlinkAdvanced(
        leftEAR: Float,
        rightEAR: Float,
        threshold: Float,
        changeRateThreshold: Float = 0.05f
    ): Pair<Boolean, Float> { // 返回 (是否眨眼, 置信度)
        
        val avgEAR = (leftEAR + rightEAR) / 2.0f
        
        // 检查是否低于阈值
        val belowThreshold = avgEAR < threshold
        
        // 计算EAR变化率
        val leftHistory = leftEyeEARHistory.getAll()
        val rightHistory = rightEyeEARHistory.getAll()
        
        var changeRate = 0f
        if (leftHistory.size >= 2 && rightHistory.size >= 2) {
            val leftChange = abs(leftHistory.last() - leftHistory[leftHistory.size - 2])
            val rightChange = abs(rightHistory.last() - rightHistory[rightHistory.size - 2])
            changeRate = (leftChange + rightChange) / 2.0f
        }
        
        // 结合阈值检测和变化率检测
        val isBlink = belowThreshold && changeRate > changeRateThreshold
        
        // 计算置信度
        val confidence = if (isBlink) {
            val thresholdConfidence = (threshold - avgEAR) / threshold
            val changeConfidence = min(changeRate / (changeRateThreshold * 2), 1.0f)
            (thresholdConfidence * 0.6f + changeConfidence * 0.4f).coerceIn(0f, 1f)
        } else {
            0f
        }
        
        return Pair(isBlink, confidence)
    }
    
    /**
     * 重置历史数据
     */
    fun reset() {
        leftEyeEARHistory.clear()
        rightEyeEARHistory.clear()
    }
    
    /**
     * 获取EAR统计信息
     */
    fun getEARStats(): String {
        val leftHistory = leftEyeEARHistory.getAll()
        val rightHistory = rightEyeEARHistory.getAll()
        
        return if (leftHistory.isNotEmpty() && rightHistory.isNotEmpty()) {
            """
                Left EAR: avg=${String.format("%.3f", leftHistory.average())}, current=${String.format("%.3f", leftHistory.last())}
                Right EAR: avg=${String.format("%.3f", rightHistory.average())}, current=${String.format("%.3f", rightHistory.last())}
                History size: L=${leftHistory.size}, R=${rightHistory.size}
            """.trimIndent()
        } else {
            "No EAR data available"
        }
    }
} 