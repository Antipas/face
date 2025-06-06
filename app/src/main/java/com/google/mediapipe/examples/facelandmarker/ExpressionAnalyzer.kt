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

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * 表情识别分析器
 * 基于MediaPipe面部Blendshape特征识别各种表情状态
 */
class ExpressionAnalyzer {
    
    companion object {
        // 表情识别阈值
        private const val SMILE_THRESHOLD = 0.3f
        private const val LAUGH_THRESHOLD = 0.6f
        private const val SURPRISE_THRESHOLD = 0.4f
        private const val CONFUSED_THRESHOLD = 0.35f
        private const val CONCENTRATED_THRESHOLD = 0.25f
        private const val BORED_THRESHOLD = 0.2f
        private const val FRUSTRATED_THRESHOLD = 0.3f
        private const val EXCITED_THRESHOLD = 0.5f
        
        // 表情强度计算权重
        private const val INTENSITY_SMOOTHING = 0.7f
        private const val MIN_CONFIDENCE = 0.4f
    }
    
    // 历史表情数据用于平滑
    val expressionHistory = CircularBuffer<ExpressionResult>(5)
    private var lastIntensity = 0f
    
    /**
     * 分析表情状态
     */
    fun analyzeExpression(blendshapes: BlendshapeFeatures): ExpressionResult {
        val candidateExpressions = mutableListOf<Pair<ExpressionState, Float>>()
        
        // 1. 检测微笑和大笑
        val smileScore = analyzeSmile(blendshapes)
        if (smileScore.second > SMILE_THRESHOLD) {
            if (smileScore.second > LAUGH_THRESHOLD) {
                candidateExpressions.add(ExpressionState.LAUGHING to smileScore.second)
            } else {
                candidateExpressions.add(ExpressionState.SMILING to smileScore.second)
            }
        }
        
        // 2. 检测吃惊
        val surpriseScore = analyzeSurprise(blendshapes)
        if (surpriseScore.second > SURPRISE_THRESHOLD) {
            candidateExpressions.add(ExpressionState.SURPRISED to surpriseScore.second)
        }
        
        // 3. 检测困惑
        val confusedScore = analyzeConfusion(blendshapes)
        if (confusedScore.second > CONFUSED_THRESHOLD) {
            candidateExpressions.add(ExpressionState.CONFUSED to confusedScore.second)
        }
        
        // 4. 检测专注思考
        val concentratedScore = analyzeConcentration(blendshapes)
        if (concentratedScore.second > CONCENTRATED_THRESHOLD) {
            candidateExpressions.add(ExpressionState.CONCENTRATED to concentratedScore.second)
        }
        
        // 5. 检测无聊
        val boredScore = analyzeBoredom(blendshapes)
        if (boredScore.second > BORED_THRESHOLD) {
            candidateExpressions.add(ExpressionState.BORED to boredScore.second)
        }
        
        // 6. 检测沮丧
        val frustratedScore = analyzeFrustration(blendshapes)
        if (frustratedScore.second > FRUSTRATED_THRESHOLD) {
            candidateExpressions.add(ExpressionState.FRUSTRATED to frustratedScore.second)
        }
        
        // 7. 检测兴奋
        val excitedScore = analyzeExcitement(blendshapes)
        if (excitedScore.second > EXCITED_THRESHOLD) {
            candidateExpressions.add(ExpressionState.EXCITED to excitedScore.second)
        }
        
        // 选择最佳表情
        val result = selectBestExpression(candidateExpressions)
        
        // 应用时序平滑
        val smoothedResult = applySmoothingFilter(result)
        
        expressionHistory.add(smoothedResult)
        return smoothedResult
    }
    
    /**
     * 分析微笑
     */
    private fun analyzeSmile(bs: BlendshapeFeatures): Pair<ExpressionState, Float> {
        val mouthSmile = (bs.mouthSmileLeft + bs.mouthSmileRight) / 2.0f
        val cheekSquint = (bs.cheekSquintLeft + bs.cheekSquintRight) / 2.0f
        val eyeSquint = (bs.eyeSquintLeft + bs.eyeSquintRight) / 2.0f
        
        // 综合评分：嘴角上扬 + 脸颊上提 + 轻微眯眼
        val smileScore = mouthSmile * 0.6f + cheekSquint * 0.3f + eyeSquint * 0.1f
        
        return ExpressionState.SMILING to smileScore
    }
    
    /**
     * 分析吃惊
     */
    private fun analyzeSurprise(bs: BlendshapeFeatures): Pair<ExpressionState, Float> {
        val eyeWide = (bs.eyeWideLeft + bs.eyeWideRight) / 2.0f
        val browUp = (bs.browInnerUp + bs.browOuterUpLeft + bs.browOuterUpRight) / 3.0f
        val jawOpen = bs.jawOpen
        
        // 吃惊特征：眼睛睁大 + 眉毛上提 + 可能嘴巴张开
        val surpriseScore = eyeWide * 0.4f + browUp * 0.4f + jawOpen * 0.2f
        
        return ExpressionState.SURPRISED to surpriseScore
    }
    
    /**
     * 分析困惑
     */
    private fun analyzeConfusion(bs: BlendshapeFeatures): Pair<ExpressionState, Float> {
        val browDown = (bs.browDownLeft + bs.browDownRight) / 2.0f
        val browInnerUp = bs.browInnerUp
        val eyeSquint = (bs.eyeSquintLeft + bs.eyeSquintRight) / 2.0f
        val mouthPress = (bs.mouthPressLeft + bs.mouthPressRight) / 2.0f
        
        // 困惑特征：皱眉 + 内眉上提 + 轻微眯眼 + 抿嘴
        val confusedScore = browDown * 0.3f + browInnerUp * 0.3f + eyeSquint * 0.2f + mouthPress * 0.2f
        
        return ExpressionState.CONFUSED to confusedScore
    }
    
    /**
     * 分析专注思考
     */
    private fun analyzeConcentration(bs: BlendshapeFeatures): Pair<ExpressionState, Float> {
        val browDown = (bs.browDownLeft + bs.browDownRight) / 2.0f
        val eyeSquint = (bs.eyeSquintLeft + bs.eyeSquintRight) / 2.0f
        val mouthPress = (bs.mouthPressLeft + bs.mouthPressRight) / 2.0f
        val mouthPucker = bs.mouthPucker
        
        // 专注思考：轻微皱眉 + 眯眼 + 嘴唇紧闭
        val concentratedScore = browDown * 0.3f + eyeSquint * 0.3f + mouthPress * 0.2f + mouthPucker * 0.2f
        
        return ExpressionState.CONCENTRATED to concentratedScore
    }
    
    /**
     * 分析无聊
     */
    private fun analyzeBoredom(bs: BlendshapeFeatures): Pair<ExpressionState, Float> {
        val eyeBlink = (bs.eyeBlinkLeft + bs.eyeBlinkRight) / 2.0f
        val mouthFrown = (bs.mouthFrownLeft + bs.mouthFrownRight) / 2.0f
        val browDown = (bs.browDownLeft + bs.browDownRight) / 2.0f
        val eyeLookDown = (bs.eyeLookDownLeft + bs.eyeLookDownRight) / 2.0f
        
        // 无聊特征：频繁眨眼 + 嘴角下垂 + 轻微皱眉 + 眼睛向下看
        val boredScore = eyeBlink * 0.25f + mouthFrown * 0.25f + browDown * 0.25f + eyeLookDown * 0.25f
        
        return ExpressionState.BORED to boredScore
    }
    
    /**
     * 分析沮丧
     */
    private fun analyzeFrustration(bs: BlendshapeFeatures): Pair<ExpressionState, Float> {
        val browDown = (bs.browDownLeft + bs.browDownRight) / 2.0f
        val mouthFrown = (bs.mouthFrownLeft + bs.mouthFrownRight) / 2.0f
        val eyeSquint = (bs.eyeSquintLeft + bs.eyeSquintRight) / 2.0f
        val noseSneer = (bs.noseSneerLeft + bs.noseSneerRight) / 2.0f
        
        // 沮丧特征：深度皱眉 + 嘴角下垂 + 眯眼 + 鼻子皱起
        val frustratedScore = browDown * 0.3f + mouthFrown * 0.3f + eyeSquint * 0.2f + noseSneer * 0.2f
        
        return ExpressionState.FRUSTRATED to frustratedScore
    }
    
    /**
     * 分析兴奋
     */
    private fun analyzeExcitement(bs: BlendshapeFeatures): Pair<ExpressionState, Float> {
        val mouthSmile = (bs.mouthSmileLeft + bs.mouthSmileRight) / 2.0f
        val eyeWide = (bs.eyeWideLeft + bs.eyeWideRight) / 2.0f
        val browUp = (bs.browOuterUpLeft + bs.browOuterUpRight) / 2.0f
        val jawOpen = bs.jawOpen
        
        // 兴奋特征：大笑 + 眼睛睁大 + 眉毛上提 + 嘴巴张开
        val excitedScore = mouthSmile * 0.4f + eyeWide * 0.2f + browUp * 0.2f + jawOpen * 0.2f
        
        return ExpressionState.EXCITED to excitedScore
    }
    
    /**
     * 选择最佳表情
     */
    private fun selectBestExpression(candidates: List<Pair<ExpressionState, Float>>): ExpressionResult {
        if (candidates.isEmpty()) {
            return ExpressionResult(
                primaryExpression = ExpressionState.NEUTRAL,
                confidence = 0.8f,
                intensity = 0.1f
            )
        }
        
        // 按置信度排序
        val sortedCandidates = candidates.sortedByDescending { it.second }
        val bestCandidate = sortedCandidates.first()
        
        // 计算表情强度
        val intensity = calculateIntensity(bestCandidate.second)
        
        // 确定次要表情
        val secondaryExpression = if (sortedCandidates.size > 1 && 
            sortedCandidates[1].second > MIN_CONFIDENCE) {
            sortedCandidates[1].first
        } else null
        
        return ExpressionResult(
            primaryExpression = bestCandidate.first,
            confidence = bestCandidate.second,
            secondaryExpression = secondaryExpression,
            intensity = intensity
        )
    }
    
    /**
     * 计算表情强度
     */
    private fun calculateIntensity(score: Float): Float {
        // 非线性映射，使强度更加明显
        val normalizedScore = score.coerceIn(0f, 1f)
        val intensity = normalizedScore * normalizedScore // 平方映射
        
        // 时序平滑
        lastIntensity = INTENSITY_SMOOTHING * lastIntensity + (1 - INTENSITY_SMOOTHING) * intensity
        
        return lastIntensity.coerceIn(0f, 1f)
    }
    
    /**
     * 应用时序平滑滤波
     */
    private fun applySmoothingFilter(currentResult: ExpressionResult): ExpressionResult {
        val history = expressionHistory.getAll()
        if (history.isEmpty()) return currentResult
        
        // 检查表情一致性
        val recentExpression = history.lastOrNull()?.primaryExpression
        
        // 如果表情变化太快，降低置信度
        val stabilityFactor = if (recentExpression == currentResult.primaryExpression) {
            1.0f
        } else {
            0.7f // 表情变化时降低置信度
        }
        
        return currentResult.copy(
            confidence = (currentResult.confidence * stabilityFactor).coerceIn(0f, 1f)
        )
    }
    
    /**
     * 获取表情历史统计
     */
    fun getExpressionStats(): String {
        val history = expressionHistory.getAll()
        if (history.isEmpty()) return "No expression data"
        
        val expressionCounts = history.groupingBy { it.primaryExpression }.eachCount()
        val avgConfidence = history.map { it.confidence }.average()
        val avgIntensity = history.map { it.intensity }.average()
        
        return """
            Expression History (${history.size} samples):
            ${expressionCounts.entries.joinToString(", ") { "${it.key}: ${it.value}" }}
            Avg Confidence: ${String.format("%.2f", avgConfidence)}
            Avg Intensity: ${String.format("%.2f", avgIntensity)}
        """.trimIndent()
    }
    
    /**
     * 重置分析器状态
     */
    fun reset() {
        expressionHistory.clear()
        lastIntensity = 0f
    }
    
    /**
     * 判断表情是否为积极表情
     */
    fun isPositiveExpression(expression: ExpressionState): Boolean {
        return when (expression) {
            ExpressionState.SMILING, 
            ExpressionState.LAUGHING, 
            ExpressionState.EXCITED -> true
            else -> false
        }
    }
    
    /**
     * 判断表情是否为消极表情
     */
    fun isNegativeExpression(expression: ExpressionState): Boolean {
        return when (expression) {
            ExpressionState.FRUSTRATED,
            ExpressionState.BORED,
            ExpressionState.CONFUSED -> true
            else -> false
        }
    }
    
    /**
     * 获取表情的学习状态指示
     */
    fun getLearningStateFromExpression(expression: ExpressionState): String {
        return when (expression) {
            ExpressionState.CONCENTRATED -> "深度学习中"
            ExpressionState.CONFUSED -> "遇到困难"
            ExpressionState.EXCITED -> "学习兴趣高"
            ExpressionState.BORED -> "需要激发兴趣"
            ExpressionState.FRUSTRATED -> "需要帮助"
            ExpressionState.SMILING -> "学习愉快"
            ExpressionState.SURPRISED -> "有新发现"
            ExpressionState.NEUTRAL -> "平静学习"
            else -> "状态未知"
        }
    }
} 