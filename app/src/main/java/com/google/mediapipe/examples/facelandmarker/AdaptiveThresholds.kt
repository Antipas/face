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

import android.content.Context
import android.content.SharedPreferences
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * 自适应阈值管理器
 * 根据用户特征和环境条件动态调整检测阈值
 */
class AdaptiveThresholds(private val context: Context) {
    
    companion object {
        // 默认阈值
        private const val DEFAULT_EAR_THRESHOLD = 0.11f
        private const val DEFAULT_YAW_THRESHOLD = 25f
        private const val DEFAULT_PITCH_THRESHOLD = 20f
        private const val DEFAULT_ROLL_THRESHOLD = 15f
        
        // Blendshape阈值
        private const val DEFAULT_YAWN_THRESHOLD = 0.5f
        private const val DEFAULT_BLINK_THRESHOLD = 0.5f
        private const val DEFAULT_LOOK_SIDE_THRESHOLD = 0.45f
        private const val DEFAULT_LOOK_DOWN_THRESHOLD = 0.6f
        private const val DEFAULT_LOOK_UP_THRESHOLD = 0.5f
        private const val DEFAULT_BROW_DOWN_THRESHOLD = 0.35f
        private const val DEFAULT_EYE_SQUINT_THRESHOLD = 0.4f
        
        // 校准参数
        private const val MIN_CALIBRATION_SAMPLES = 30
        private const val ADAPTATION_RATE = 0.1f
        
        // SharedPreferences键名
        private const val PREFS_NAME = "adaptive_thresholds"
        private const val KEY_USER_BASELINE = "user_baseline"
        private const val KEY_EAR_THRESHOLD = "ear_threshold"
        private const val KEY_YAW_THRESHOLD = "yaw_threshold"
        private const val KEY_PITCH_THRESHOLD = "pitch_threshold"
    }
    
    private val sharedPreferences: SharedPreferences = 
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    
    // 当前阈值
    var earThreshold = DEFAULT_EAR_THRESHOLD
        private set
    var yawThreshold = DEFAULT_YAW_THRESHOLD
        private set
    var pitchThreshold = DEFAULT_PITCH_THRESHOLD
        private set
    var rollThreshold = DEFAULT_ROLL_THRESHOLD
        private set
    
    // Blendshape阈值
    var yawnThreshold = DEFAULT_YAWN_THRESHOLD
        private set
    var blinkThreshold = DEFAULT_BLINK_THRESHOLD
        private set
    var lookSideThreshold = DEFAULT_LOOK_SIDE_THRESHOLD
        private set
    var lookDownThreshold = DEFAULT_LOOK_DOWN_THRESHOLD
        private set
    var lookUpThreshold = DEFAULT_LOOK_UP_THRESHOLD
        private set
    var browDownThreshold = DEFAULT_BROW_DOWN_THRESHOLD
        private set
    var eyeSquintThreshold = DEFAULT_EYE_SQUINT_THRESHOLD
        private set
    
    // 用户基准数据
    private var userBaseline = UserBaseline()
    
    // 校准数据收集
    private val calibrationEARSamples = mutableListOf<Float>()
    private val calibrationYawSamples = mutableListOf<Float>()
    private val calibrationPitchSamples = mutableListOf<Float>()
    
    // 环境适应因子
    private var lightingFactor = 1.0f
    private var distanceFactor = 1.0f
    
    init {
        loadSavedThresholds()
    }
    
    /**
     * 开始用户校准过程
     */
    fun startCalibration() {
        calibrationEARSamples.clear()
        calibrationYawSamples.clear()
        calibrationPitchSamples.clear()
    }
    
    /**
     * 添加校准样本（用户正常专注状态下的数据）
     */
    fun addCalibrationSample(features: AttentionFeatures) {
        if (features.areEyesOpen && features.isHeadPoseAttentive) {
            calibrationEARSamples.add(features.averageEAR)
            calibrationYawSamples.add(abs(features.headPoseYaw))
            calibrationPitchSamples.add(abs(features.headPosePitch))
        }
    }
    
    /**
     * 完成校准并更新用户基准
     */
    fun finishCalibration(): Boolean {
        if (calibrationEARSamples.size < MIN_CALIBRATION_SAMPLES) {
            return false
        }
        
        // 计算用户基准值
        val avgEAR = calibrationEARSamples.average().toFloat()
        val avgYawRange = calibrationYawSamples.average().toFloat()
        val avgPitchRange = calibrationPitchSamples.average().toFloat()
        
        userBaseline = UserBaseline(
            avgEAR = avgEAR,
            avgYawRange = avgYawRange,
            avgPitchRange = avgPitchRange,
            calibrationCount = calibrationEARSamples.size,
            isCalibrated = true
        )
        
        // 更新阈值
        updateThresholdsFromBaseline()
        saveThresholds()
        
        return true
    }
    
    /**
     * 根据用户基准更新阈值
     */
    private fun updateThresholdsFromBaseline() {
        if (!userBaseline.isCalibrated) return
        
        // EAR阈值设为用户平均值的70%
        earThreshold = (userBaseline.avgEAR * 0.7f).coerceIn(0.08f, 0.20f)
        
        // 头部姿态阈值设为用户平均范围的120%
        yawThreshold = (userBaseline.avgYawRange * 1.2f).coerceIn(15f, 40f)
        pitchThreshold = (userBaseline.avgPitchRange * 1.2f).coerceIn(10f, 30f)
        
        // 根据用户EAR特征调整其他阈值
        val earRatio = userBaseline.avgEAR / DEFAULT_EAR_THRESHOLD
        blinkThreshold = (DEFAULT_BLINK_THRESHOLD * earRatio).coerceIn(0.3f, 0.8f)
    }
    
    /**
     * 在线自适应调整（运行时动态优化）
     */
    @Suppress("UNUSED_PARAMETER")
    fun adaptOnline(attentionFeatures: AttentionFeatures, actualState: AttentionState, predictedState: AttentionState) {
        if (!userBaseline.isCalibrated) return
        
        // 如果预测错误，微调阈值
        if (actualState != predictedState) {
            when {
                // 误判为困倦，应该提高EAR阈值
                predictedState == AttentionState.DROWSY_FATIGUED && actualState == AttentionState.ATTENTIVE -> {
                    earThreshold = min(earThreshold * (1 + ADAPTATION_RATE), 0.20f)
                }
                // 漏判困倦，应该降低EAR阈值
                predictedState == AttentionState.ATTENTIVE && actualState == AttentionState.DROWSY_FATIGUED -> {
                    earThreshold = max(earThreshold * (1 - ADAPTATION_RATE), 0.08f)
                }
                // 误判为分心，应该放宽头部姿态阈值
                predictedState == AttentionState.DISTRACTED_LOOKING_AWAY && actualState == AttentionState.ATTENTIVE -> {
                    yawThreshold = min(yawThreshold * (1 + ADAPTATION_RATE), 40f)
                    pitchThreshold = min(pitchThreshold * (1 + ADAPTATION_RATE), 30f)
                }
            }
            saveThresholds() // 保存更新后的阈值
        }
    }
    
    /**
     * 根据环境光照调整阈值
     */
    fun adjustForLighting(brightness: Float) {
        // brightness范围：0.0-1.0
        lightingFactor = when {
            brightness < 0.3f -> 1.2f // 暗光环境，放宽阈值
            brightness > 0.8f -> 0.9f // 强光环境，收紧阈值
            else -> 1.0f // 正常光照
        }
        
        applyEnvironmentalFactors()
    }
    
    /**
     * 根据人脸距离调整阈值
     */
    fun adjustForDistance(faceSize: Float) {
        // faceSize为归一化的人脸大小（0.0-1.0）
        distanceFactor = when {
            faceSize < 0.3f -> 1.3f // 距离远，放宽阈值
            faceSize > 0.7f -> 0.8f // 距离近，收紧阈值
            else -> 1.0f // 正常距离
        }
        
        applyEnvironmentalFactors()
    }
    
    /**
     * 应用环境因子
     */
    private fun applyEnvironmentalFactors() {
        // 环境因子通过 getAdjusted* 方法动态应用
        // 避免直接修改基础阈值，保持校准结果的完整性
    }
    
    /**
     * 获取环境调整后的EAR阈值
     */
    fun getAdjustedEARThreshold(): Float {
        return earThreshold * lightingFactor * distanceFactor
    }
    
    /**
     * 获取环境调整后的头部姿态阈值
     */
    fun getAdjustedYawThreshold(): Float {
        return yawThreshold * lightingFactor * distanceFactor
    }
    
    fun getAdjustedPitchThreshold(): Float {
        return pitchThreshold * lightingFactor * distanceFactor
    }
    
    /**
     * 重置为默认阈值
     */
    fun resetToDefaults() {
        earThreshold = DEFAULT_EAR_THRESHOLD
        yawThreshold = DEFAULT_YAW_THRESHOLD
        pitchThreshold = DEFAULT_PITCH_THRESHOLD
        rollThreshold = DEFAULT_ROLL_THRESHOLD
        
        yawnThreshold = DEFAULT_YAWN_THRESHOLD
        blinkThreshold = DEFAULT_BLINK_THRESHOLD
        lookSideThreshold = DEFAULT_LOOK_SIDE_THRESHOLD
        lookDownThreshold = DEFAULT_LOOK_DOWN_THRESHOLD
        lookUpThreshold = DEFAULT_LOOK_UP_THRESHOLD
        browDownThreshold = DEFAULT_BROW_DOWN_THRESHOLD
        eyeSquintThreshold = DEFAULT_EYE_SQUINT_THRESHOLD
        
        userBaseline = UserBaseline()
        
        clearSavedThresholds()
    }
    
    /**
     * 保存阈值到SharedPreferences
     */
    private fun saveThresholds() {
        with(sharedPreferences.edit()) {
            putFloat(KEY_EAR_THRESHOLD, earThreshold)
            putFloat(KEY_YAW_THRESHOLD, yawThreshold)
            putFloat(KEY_PITCH_THRESHOLD, pitchThreshold)
            
            // 保存用户基准数据
            putFloat("baseline_ear", userBaseline.avgEAR)
            putFloat("baseline_yaw", userBaseline.avgYawRange)
            putFloat("baseline_pitch", userBaseline.avgPitchRange)
            putInt("baseline_count", userBaseline.calibrationCount)
            putBoolean("baseline_calibrated", userBaseline.isCalibrated)
            
            apply()
        }
    }
    
    /**
     * 从SharedPreferences加载阈值
     */
    private fun loadSavedThresholds() {
        earThreshold = sharedPreferences.getFloat(KEY_EAR_THRESHOLD, DEFAULT_EAR_THRESHOLD)
        yawThreshold = sharedPreferences.getFloat(KEY_YAW_THRESHOLD, DEFAULT_YAW_THRESHOLD)
        pitchThreshold = sharedPreferences.getFloat(KEY_PITCH_THRESHOLD, DEFAULT_PITCH_THRESHOLD)
        
        // 加载用户基准数据
        userBaseline = UserBaseline(
            avgEAR = sharedPreferences.getFloat("baseline_ear", 0.25f),
            avgYawRange = sharedPreferences.getFloat("baseline_yaw", 30f),
            avgPitchRange = sharedPreferences.getFloat("baseline_pitch", 20f),
            calibrationCount = sharedPreferences.getInt("baseline_count", 0),
            isCalibrated = sharedPreferences.getBoolean("baseline_calibrated", false)
        )
    }
    
    /**
     * 清除保存的阈值
     */
    private fun clearSavedThresholds() {
        sharedPreferences.edit().clear().apply()
    }
    
    /**
     * 获取校准进度
     */
    fun getCalibrationProgress(): Float {
        return calibrationEARSamples.size.toFloat() / MIN_CALIBRATION_SAMPLES
    }
    
    /**
     * 是否已完成用户校准
     */
    fun isUserCalibrated(): Boolean = userBaseline.isCalibrated
    
    /**
     * 获取当前阈值信息（用于调试）
     */
    fun getThresholdInfo(): String {
        return """
            EAR: ${String.format("%.3f", getAdjustedEARThreshold())}
            Yaw: ${String.format("%.1f", getAdjustedYawThreshold())}°
            Pitch: ${String.format("%.1f", getAdjustedPitchThreshold())}°
            Light Factor: ${String.format("%.2f", lightingFactor)}
            Distance Factor: ${String.format("%.2f", distanceFactor)}
            User Calibrated: ${userBaseline.isCalibrated}
        """.trimIndent()
    }
} 