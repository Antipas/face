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

import java.util.concurrent.ConcurrentLinkedQueue

/**
 * 内存池管理器
 * 减少频繁对象创建和GC压力
 */
class MemoryPool {
    
    companion object {
        private const val MAX_POOL_SIZE = 10
    }
    
    // StringBuilder对象池
    private val stringBuilderPool = ConcurrentLinkedQueue<StringBuilder>()
    
    // FloatArray对象池（用于计算缓存）
    private val floatArrayPool = ConcurrentLinkedQueue<FloatArray>()
    
    // AttentionFeatures对象池
    private val attentionFeaturesPool = ConcurrentLinkedQueue<AttentionFeatures>()
    
    /**
     * 获取StringBuilder对象
     */
    fun acquireStringBuilder(): StringBuilder {
        return stringBuilderPool.poll()?.apply { 
            setLength(0) // 清空内容
        } ?: StringBuilder(512) // 预分配容量
    }
    
    /**
     * 归还StringBuilder对象
     */
    fun releaseStringBuilder(sb: StringBuilder) {
        if (stringBuilderPool.size < MAX_POOL_SIZE) {
            sb.setLength(0) // 清空
            stringBuilderPool.offer(sb)
        }
    }
    
    /**
     * 获取FloatArray对象（用于计算缓存）
     */
    fun acquireFloatArray(size: Int): FloatArray {
        return floatArrayPool.poll()?.takeIf { it.size >= size } 
            ?: FloatArray(size)
    }
    
    /**
     * 归还FloatArray对象
     */
    fun releaseFloatArray(array: FloatArray) {
        if (floatArrayPool.size < MAX_POOL_SIZE) {
            array.fill(0f) // 清零
            floatArrayPool.offer(array)
        }
    }
    
    /**
     * 安全的字符串构建操作
     */
    inline fun <T> withStringBuilder(block: (StringBuilder) -> T): T {
        val sb = acquireStringBuilder()
        try {
            return block(sb)
        } finally {
            releaseStringBuilder(sb)
        }
    }
    
    /**
     * 安全的数组操作
     */
    inline fun <T> withFloatArray(size: Int, block: (FloatArray) -> T): T {
        val array = acquireFloatArray(size)
        try {
            return block(array)
        } finally {
            releaseFloatArray(array)
        }
    }
    
    /**
     * 清理对象池
     */
    fun cleanup() {
        stringBuilderPool.clear()
        floatArrayPool.clear()
        attentionFeaturesPool.clear()
    }
    
    /**
     * 获取内存池统计信息
     */
    fun getStats(): MemoryPoolStats {
        return MemoryPoolStats(
            stringBuilderPoolSize = stringBuilderPool.size,
            floatArrayPoolSize = floatArrayPool.size,
            maxPoolSize = MAX_POOL_SIZE
        )
    }
}

/**
 * 内存池统计数据
 */
data class MemoryPoolStats(
    val stringBuilderPoolSize: Int,
    val floatArrayPoolSize: Int,
    val maxPoolSize: Int
)

/**
 * 优化的字符串格式化工具
 */
object OptimizedFormatter {
    
    // 预定义格式化缓存
    private val percentageCache = Array(101) { i -> "${i}%" }
    private val confidenceCache = Array(101) { i -> String.format("%.0f", i.toFloat()) + "%" }
    
    /**
     * 优化的百分比格式化
     */
    fun formatPercentage(value: Float): String {
        val intValue = (value * 100).toInt().coerceIn(0, 100)
        return percentageCache[intValue]
    }
    
    /**
     * 优化的置信度格式化
     */
    fun formatConfidence(value: Float): String {
        val intValue = (value * 100).toInt().coerceIn(0, 100)
        return confidenceCache[intValue]
    }
    
    /**
     * 避免装箱的浮点数比较
     */
    fun isSignificantChange(oldValue: Float, newValue: Float, threshold: Float = 0.05f): Boolean {
        return kotlin.math.abs(oldValue - newValue) > threshold
    }
} 