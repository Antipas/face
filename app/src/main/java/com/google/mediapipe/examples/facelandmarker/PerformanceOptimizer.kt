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

import android.os.SystemClock
import kotlin.math.abs

/**
 * 性能优化管理器
 * 负责控制计算频率、缓存结果、优化资源使用
 */
class PerformanceOptimizer {
    
    companion object {
        // 性能控制参数
        private const val TARGET_FPS = 15 // 目标帧率（降低以节省资源）
        private const val MIN_FRAME_INTERVAL_MS = 1000L / TARGET_FPS // 66.7ms
        private const val UI_UPDATE_INTERVAL_MS = 200L // UI更新间隔
        private const val CACHE_VALIDITY_MS = 100L // 缓存有效期
        
        // 计算优化阈值
        private const val SIGNIFICANT_CHANGE_THRESHOLD = 0.05f // 显著变化阈值
        private const val SKIP_FRAME_THRESHOLD = 3 // 跳帧阈值
    }
    
    // 帧率控制
    private var lastProcessTime = 0L
    private var lastUIUpdateTime = 0L
    private var frameSkipCounter = 0
    
    // 新增：实际帧间隔跟踪
    private val frameIntervals = CircularBuffer<Long>(10)
    private var actualFrameInterval = MIN_FRAME_INTERVAL_MS
    
    // 结果缓存
    private var cachedStatusText: String = ""
    private var cachedStatusTime = 0L
    private var lastAttentionState: AttentionState? = null
    private var lastExpressionState: ExpressionState? = null
    private var lastEngagement = 0f
    
    // 性能监控
    private val processingTimes = CircularBuffer<Long>(10)
    private var averageProcessingTime = 0L
    
    // 新增：跳帧统计
    private var totalFrames = 0L
    private var skippedFrames = 0L
    
    /**
     * 检查是否应该处理当前帧
     */
    fun shouldProcessFrame(): Boolean {
        val currentTime = SystemClock.uptimeMillis()
        totalFrames++
        
        // 基本帧率控制 - 严格执行目标帧率
        val timeSinceLastFrame = currentTime - lastProcessTime
        if (timeSinceLastFrame < MIN_FRAME_INTERVAL_MS) {
            // 帧率控制生效 - 跳过此帧
            skippedFrames++
            return false
        }
        
        // 记录实际帧间隔
        if (lastProcessTime > 0) {
            frameIntervals.add(timeSinceLastFrame)
            val intervals = frameIntervals.getAll()
            actualFrameInterval = if (intervals.isNotEmpty()) {
                intervals.average().toLong()
            } else MIN_FRAME_INTERVAL_MS
        }
        
        // 动态跳帧（根据处理负载调整）
        if (averageProcessingTime > MIN_FRAME_INTERVAL_MS * 0.8f) {
            frameSkipCounter++
            if (frameSkipCounter < SKIP_FRAME_THRESHOLD) {
                lastProcessTime = currentTime // 重要：更新时间以维持间隔控制
                skippedFrames++
                return false
            }
            frameSkipCounter = 0
        }
        
        lastProcessTime = currentTime
        return true
    }
    
    /**
     * 检查是否应该更新UI
     */
    fun shouldUpdateUI(): Boolean {
        val currentTime = SystemClock.uptimeMillis()
        return currentTime - lastUIUpdateTime >= UI_UPDATE_INTERVAL_MS
    }
    
    /**
     * 记录UI更新时间
     */
    fun markUIUpdated() {
        lastUIUpdateTime = SystemClock.uptimeMillis()
    }
    
    /**
     * 检查状态是否有显著变化
     */
    fun hasSignificantChange(
        currentAttention: AttentionState,
        currentExpression: ExpressionState,
        currentEngagement: Float
    ): Boolean {
        // 状态改变
        if (lastAttentionState != currentAttention || 
            lastExpressionState != currentExpression) {
            return true
        }
        
        // 参与度显著变化
        if (abs(currentEngagement - lastEngagement) > SIGNIFICANT_CHANGE_THRESHOLD) {
            return true
        }
        
        return false
    }
    
    /**
     * 获取缓存的状态文本（如果仍然有效）
     */
    fun getCachedStatusText(): String? {
        val currentTime = SystemClock.uptimeMillis()
        return if (currentTime - cachedStatusTime < CACHE_VALIDITY_MS) {
            cachedStatusText
        } else null
    }
    
    /**
     * 缓存状态文本
     */
    fun cacheStatusText(
        text: String,
        attentionState: AttentionState,
        expressionState: ExpressionState,
        engagement: Float
    ) {
        cachedStatusText = text
        cachedStatusTime = SystemClock.uptimeMillis()
        lastAttentionState = attentionState
        lastExpressionState = expressionState
        lastEngagement = engagement
    }
    
    /**
     * 记录处理时间
     */
    fun recordProcessingTime(startTime: Long) {
        val processingTime = SystemClock.uptimeMillis() - startTime
        processingTimes.add(processingTime)
        
        // 更新平均处理时间
        val times = processingTimes.getAll()
        averageProcessingTime = if (times.isNotEmpty()) {
            times.average().toLong()
        } else 0L
    }
    
    /**
     * 获取性能统计信息
     */
    fun getPerformanceStats(): PerformanceStats {
        val times = processingTimes.getAll()
        val frameSkipPercentage = if (totalFrames > 0) {
            (skippedFrames * 100f / totalFrames)
        } else 0f
        
        return PerformanceStats(
            averageProcessingTime = averageProcessingTime,
            maxProcessingTime = times.maxOrNull() ?: 0L,
            minProcessingTime = times.minOrNull() ?: 0L,
            targetFPS = TARGET_FPS,
            actualFPS = if (actualFrameInterval > 0) {
                (1000f / actualFrameInterval).toInt()
            } else 0,
            frameSkipRate = frameSkipCounter,
            frameSkipPercentage = frameSkipPercentage,
            totalFrames = totalFrames,
            processedFrames = totalFrames - skippedFrames
        )
    }
    
    /**
     * 重置性能统计
     */
    fun reset() {
        processingTimes.clear()
        frameIntervals.clear()
        averageProcessingTime = 0L
        actualFrameInterval = MIN_FRAME_INTERVAL_MS
        frameSkipCounter = 0
        totalFrames = 0L
        skippedFrames = 0L
        cachedStatusText = ""
        cachedStatusTime = 0L
        lastAttentionState = null
        lastExpressionState = null
        lastEngagement = 0f
        lastProcessTime = 0L
        lastUIUpdateTime = 0L
    }
    
    /**
     * 动态调整目标帧率（根据设备性能）
     */
    fun adjustTargetFPS(batteryLevel: Float, thermalState: Int) {
        // 根据电池电量和热状态动态调整
        // 这里可以实现更复杂的自适应逻辑
    }
}

/**
 * 性能统计数据类
 */
data class PerformanceStats(
    val averageProcessingTime: Long,
    val maxProcessingTime: Long,
    val minProcessingTime: Long,
    val targetFPS: Int,
    val actualFPS: Int,
    val frameSkipRate: Int,
    val frameSkipPercentage: Float,
    val totalFrames: Long,
    val processedFrames: Long
) 