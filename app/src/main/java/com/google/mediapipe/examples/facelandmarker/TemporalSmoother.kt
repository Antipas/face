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

import java.util.*
import kotlin.math.exp

/**
 * 循环缓冲区实现
 */
class CircularBuffer<T>(private val capacity: Int) {
    private val buffer = arrayOfNulls<Any>(capacity)
    private var head = 0
    private var size = 0
    
    fun add(item: T) {
        buffer[head] = item
        head = (head + 1) % capacity
        if (size < capacity) size++
    }
    
    fun getAll(): List<T> {
        val result = mutableListOf<T>()
        for (i in 0 until size) {
            val index = (head - size + i + capacity) % capacity
            @Suppress("UNCHECKED_CAST")
            result.add(buffer[index] as T)
        }
        return result
    }
    
    fun getLast(): T? {
        return if (size > 0) {
            val index = (head - 1 + capacity) % capacity
            @Suppress("UNCHECKED_CAST")
            buffer[index] as T
        } else null
    }
    
    fun size() = size
    fun isEmpty() = size == 0
    fun isFull() = size == capacity
    
    fun clear() {
        head = 0
        size = 0
        Arrays.fill(buffer, null)
    }
}

/**
 * 时序平滑滤波器
 * 用于减少单帧误判，提升注意力检测的稳定性
 */
class TemporalSmoother {
    companion object {
        private const val WINDOW_SIZE = 7 // 7帧历史
        private const val MIN_CONFIDENCE_THRESHOLD = 0.6f
        private const val STATE_CHANGE_THRESHOLD = 0.7f
    }
    
    // 历史数据缓冲区
    private val attentionHistory = CircularBuffer<AttentionResult>(WINDOW_SIZE)
    private val earHistory = CircularBuffer<Float>(WINDOW_SIZE)
    private val headPoseHistory = CircularBuffer<Triple<Float, Float, Float>>(WINDOW_SIZE)
    
    // 指数加权移动平均权重（最新帧权重最高）
    private val weights = floatArrayOf(0.05f, 0.10f, 0.15f, 0.20f, 0.25f, 0.25f)
    
    // 状态持续计数器
    private var currentStateCount = 0
    private var lastStableState = AttentionState.UNKNOWN
    
    /**
     * 添加新的注意力检测结果并返回平滑后的结果
     */
    fun addAndSmooth(result: AttentionResult): AttentionResult {
        attentionHistory.add(result)
        earHistory.add(result.features.averageEAR)
        headPoseHistory.add(Triple(
            result.features.headPoseYaw,
            result.features.headPosePitch,
            result.features.headPoseRoll
        ))
        
        val smoothedState = smoothAttentionState(result)
        val smoothedFeatures = smoothFeatures(result.features)
        val smoothedConfidence = calculateSmoothedConfidence()
        
        return AttentionResult(
            state = smoothedState,
            confidence = smoothedConfidence,
            features = smoothedFeatures,
            timestamp = result.timestamp
        )
    }
    
    /**
     * 平滑注意力状态
     */
    private fun smoothAttentionState(currentResult: AttentionResult): AttentionState {
        val history = attentionHistory.getAll()
        if (history.size < 3) {
            return currentResult.state
        }
        
        // 计算各状态的加权投票
        val stateVotes = mutableMapOf<AttentionState, Float>()
        
        for (i in history.indices) {
            val weight = if (i < weights.size) weights[i] else weights.last()
            val state = history[i].state
            stateVotes[state] = (stateVotes[state] ?: 0f) + weight
        }
        
        // 找到得票最高的状态
        val winningState = stateVotes.maxByOrNull { it.value }?.key ?: currentResult.state
        val winningVoteRatio = (stateVotes[winningState] ?: 0f) / stateVotes.values.sum()
        
        // 状态变化的稳定性检查
        if (winningState != lastStableState) {
            currentStateCount = 1
            
            // 需要达到一定置信度才能改变状态
            if (winningVoteRatio >= STATE_CHANGE_THRESHOLD) {
                lastStableState = winningState
                return winningState
            } else {
                // 置信度不够，保持原状态
                return if (lastStableState != AttentionState.UNKNOWN) lastStableState else winningState
            }
        } else {
            currentStateCount++
            return winningState
        }
    }
    
    /**
     * 平滑特征值
     */
    private fun smoothFeatures(currentFeatures: AttentionFeatures): AttentionFeatures {
        val smoothedEAR = smoothEAR()
        val (smoothedYaw, smoothedPitch, smoothedRoll) = smoothHeadPose()
        
        return currentFeatures.copy(
            averageEAR = smoothedEAR,
            leftEyeEAR = smoothedEAR, // 简化处理
            rightEyeEAR = smoothedEAR,
            headPoseYaw = smoothedYaw,
            headPosePitch = smoothedPitch,
            headPoseRoll = smoothedRoll,
            areEyesOpen = smoothedEAR > 0.15f // 使用平滑后的EAR判断
        )
    }
    
    /**
     * 平滑EAR值
     */
    private fun smoothEAR(): Float {
        val earList = earHistory.getAll()
        if (earList.isEmpty()) return 0f
        
        // 指数加权移动平均
        var weightedSum = 0f
        var totalWeight = 0f
        
        for (i in earList.indices) {
            val weight = if (i < weights.size) weights[i] else weights.last()
            weightedSum += earList[i] * weight
            totalWeight += weight
        }
        
        return if (totalWeight > 0) weightedSum / totalWeight else earList.last()
    }
    
    /**
     * 平滑头部姿态
     */
    private fun smoothHeadPose(): Triple<Float, Float, Float> {
        val poseList = headPoseHistory.getAll()
        if (poseList.isEmpty()) return Triple(0f, 0f, 0f)
        
        var weightedYaw = 0f
        var weightedPitch = 0f
        var weightedRoll = 0f
        var totalWeight = 0f
        
        for (i in poseList.indices) {
            val weight = if (i < weights.size) weights[i] else weights.last()
            val (yaw, pitch, roll) = poseList[i]
            
            weightedYaw += yaw * weight
            weightedPitch += pitch * weight
            weightedRoll += roll * weight
            totalWeight += weight
        }
        
        return if (totalWeight > 0) {
            Triple(
                weightedYaw / totalWeight,
                weightedPitch / totalWeight,
                weightedRoll / totalWeight
            )
        } else {
            poseList.last()
        }
    }
    
    /**
     * 计算平滑后的置信度
     */
    private fun calculateSmoothedConfidence(): Float {
        val history = attentionHistory.getAll()
        if (history.isEmpty()) return 0f
        
        // 基于状态一致性计算置信度
        val currentState = history.lastOrNull()?.state ?: AttentionState.UNKNOWN
        val consistentCount = history.count { it.state == currentState }
        val consistencyRatio = consistentCount.toFloat() / history.size
        
        // 结合原始置信度和一致性
        val avgOriginalConfidence = history.map { it.confidence }.average().toFloat()
        
        return (consistencyRatio * 0.7f + avgOriginalConfidence * 0.3f).coerceIn(0f, 1f)
    }
    
    /**
     * 重置平滑器状态
     */
    fun reset() {
        attentionHistory.clear()
        earHistory.clear()
        headPoseHistory.clear()
        currentStateCount = 0
        lastStableState = AttentionState.UNKNOWN
    }
    
    /**
     * 获取当前状态的稳定性（连续帧数）
     */
    fun getCurrentStateStability(): Int = currentStateCount
    
    /**
     * 判断当前状态是否稳定
     */
    fun isCurrentStateStable(): Boolean = currentStateCount >= 3
} 