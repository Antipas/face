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
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.annotation.VisibleForTesting
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import kotlin.math.abs
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.pow
import kotlin.math.sqrt

class FaceLandmarkerHelper(
    var minFaceDetectionConfidence: Float = DEFAULT_FACE_DETECTION_CONFIDENCE,
    var minFaceTrackingConfidence: Float = DEFAULT_FACE_TRACKING_CONFIDENCE,
    var minFacePresenceConfidence: Float = DEFAULT_FACE_PRESENCE_CONFIDENCE,
    var maxNumFaces: Int = DEFAULT_NUM_FACES,
    var currentDelegate: Int = DELEGATE_CPU,
    var runningMode: RunningMode = RunningMode.IMAGE,
    val context: Context,
    // this listener is only used when running in RunningMode.LIVE_STREAM
    val faceLandmarkerHelperListener: LandmarkerListener? = null
) {

    // For this example this needs to be a var so it can be reset on changes.
    // If the Face Landmarker will not change, a lazy val would be preferable.
    private var faceLandmarker: FaceLandmarker? = null

    // 新增：优化算法组件
    private val temporalSmoother = TemporalSmoother()
    private val adaptiveThresholds = AdaptiveThresholds(context)
    private val enhancedEARCalculator = EnhancedEARCalculator()
    private val expressionAnalyzer = ExpressionAnalyzer() // 新增表情分析器
    
    // 新增：性能优化组件
    private val performanceOptimizer = PerformanceOptimizer()
    private val memoryPool = MemoryPool()
    
    // 新增：帧计数器用于控制日志频率
    private var frameCounter = 0L
    
    // 用于环境适应的变量
    private var currentBrightness = 0.5f
    private var currentFaceSize = 0.5f

    // 这些索引基于 MediaPipe 468 Face Landmark model 的常见映射 (针对人物的左眼)
    // !! 强烈建议根据你使用的具体模型版本和文档进行核对和调整 !!
    private val LEFT_EYE_UPPER_LID_INDEX = 386 // 人物左眼上眼睑中点 (近似)
    private val LEFT_EYE_LOWER_LID_INDEX = 374 // 人物左眼下眼睑中点 (近似)

    // 眼睛闭合阈值 (基于归一化y坐标的差值) - !! 需要实验调整 !!
    private val EYE_CLOSED_THRESHOLD = 0.02f


    // --- 新增 EAR 算法相关的常量 (人物左眼) ---
    // !! 强烈建议根据你使用的具体模型版本和文档进行核对和调整 !!
    private val LEFT_EYE_P1_INDEX = 362 // 外角
    private val LEFT_EYE_P2_INDEX = 385 // 上眼睑点1
    private val LEFT_EYE_P3_INDEX = 387 // 上眼睑点2
    private val LEFT_EYE_P4_INDEX = 263 // 内角
    private val LEFT_EYE_P5_INDEX = 373 // 下眼睑点1
    private val LEFT_EYE_P6_INDEX = 380 // 下眼睑点2

    private val RIGHT_EYE_P1_INDEX = 133 // 外角 (通常是133)
    private val RIGHT_EYE_P2_INDEX = 158 // 上眼睑点1 (近似)
    private val RIGHT_EYE_P3_INDEX = 160 // 上眼睑点2 (近似)
    private val RIGHT_EYE_P4_INDEX = 33  // 内角 (通常是33)
    private val RIGHT_EYE_P5_INDEX = 144 // 下眼睑点1 (近似)
    private val RIGHT_EYE_P6_INDEX = 153 // 下眼睑点2 (近似)


    // EAR 闭眼阈值 - !! 需要通过实验和校准来确定 !!
    // 对于归一化坐标，典型的EAR值范围：睁眼时约 0.25-0.35，闭眼时 < 0.1 或更低
    private val EAR_CLOSED_THRESHOLD = 0.11f // 初始猜测值，需要调整
    // --- 结束新增常量 ---

    init {
        setupFaceLandmarker()
    }

    fun clearFaceLandmarker() {
        faceLandmarker?.close()
        faceLandmarker = null
        // 清理性能优化组件
        memoryPool.cleanup()
        performanceOptimizer.reset()
    }

    // Return running status of FaceLandmarkerHelper
    fun isClose(): Boolean {
        return faceLandmarker == null
    }

    // Initialize the Face landmarker using current settings on the
    // thread that is using it. CPU can be used with Landmarker
    // that are created on the main thread and used on a background thread, but
    // the GPU delegate needs to be used on the thread that initialized the
    // Landmarker
    fun setupFaceLandmarker() {
        // Set general face landmarker options
        val baseOptionBuilder = BaseOptions.builder()

        // Use the specified hardware for running the model. Default to CPU
        when (currentDelegate) {
            DELEGATE_CPU -> {
                baseOptionBuilder.setDelegate(Delegate.CPU)
            }
            DELEGATE_GPU -> {
                baseOptionBuilder.setDelegate(Delegate.GPU)
            }
        }

        baseOptionBuilder.setModelAssetPath(MP_FACE_LANDMARKER_TASK)

        // Check if runningMode is consistent with faceLandmarkerHelperListener
        when (runningMode) {
            RunningMode.LIVE_STREAM -> {
                if (faceLandmarkerHelperListener == null) {
                    throw IllegalStateException(
                        "faceLandmarkerHelperListener must be set when runningMode is LIVE_STREAM."
                    )
                }
            }
            else -> {
                // no-op
            }
        }

        try {
            val baseOptions = baseOptionBuilder.build()
            // Create an option builder with base options and specific
            // options only use for Face Landmarker.
            val optionsBuilder =
                FaceLandmarker.FaceLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setMinFaceDetectionConfidence(minFaceDetectionConfidence)
                    .setMinTrackingConfidence(minFaceTrackingConfidence)
                    .setMinFacePresenceConfidence(minFacePresenceConfidence)
                    .setNumFaces(maxNumFaces)
                    .setOutputFaceBlendshapes(true)
                    .setOutputFacialTransformationMatrixes(true)
                    .setRunningMode(runningMode)

            // The ResultListener and ErrorListener only use for LIVE_STREAM mode.
            if (runningMode == RunningMode.LIVE_STREAM) {
                optionsBuilder
                    .setResultListener(this::returnLivestreamResult)
                    .setErrorListener(this::returnLivestreamError)
            }

            val options = optionsBuilder.build()
            faceLandmarker =
                FaceLandmarker.createFromOptions(context, options)
        } catch (e: IllegalStateException) {
            faceLandmarkerHelperListener?.onError(
                "Face Landmarker failed to initialize. See error logs for " +
                        "details"
            )
            Log.e(
                TAG, "MediaPipe failed to load the task with error: " + e
                    .message
            )
        } catch (e: RuntimeException) {
            // This occurs if the model being used does not support GPU
            faceLandmarkerHelperListener?.onError(
                "Face Landmarker failed to initialize. See error logs for " +
                        "details", GPU_ERROR
            )
            Log.e(
                TAG,
                "Face Landmarker failed to load model with error: " + e.message
            )
        }
    }

    // Convert the ImageProxy to MP Image and feed it to FacelandmakerHelper.
    fun detectLiveStream(
        imageProxy: ImageProxy,
        isFrontCamera: Boolean
    ) {
        if (runningMode != RunningMode.LIVE_STREAM) {
            throw IllegalArgumentException(
                "Attempting to call detectLiveStream" +
                        " while not using RunningMode.LIVE_STREAM"
            )
        }
        val frameTime = SystemClock.uptimeMillis()

        // Copy out RGB bits from the frame to a bitmap buffer
        val bitmapBuffer =
            Bitmap.createBitmap(
                imageProxy.width,
                imageProxy.height,
                Bitmap.Config.ARGB_8888
            )
        imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
        imageProxy.close()

        val matrix = Matrix().apply {
            // Rotate the frame received from the camera to be in the same direction as it'll be shown
            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())

            // flip image if user use front camera
            if (isFrontCamera) {
                postScale(
                    -1f,
                    1f,
                    imageProxy.width.toFloat(),
                    imageProxy.height.toFloat()
                )
            }
        }
        val rotatedBitmap = Bitmap.createBitmap(
            bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
            matrix, true
        )

        // Convert the input Bitmap object to an MPImage object to run inference
        val mpImage = BitmapImageBuilder(rotatedBitmap).build()

        detectAsync(mpImage, frameTime)
    }

    // Run face face landmark using MediaPipe Face Landmarker API
    @VisibleForTesting
    fun detectAsync(mpImage: MPImage, frameTime: Long) {
        faceLandmarker?.detectAsync(mpImage, frameTime)
        // As we're using running mode LIVE_STREAM, the landmark result will
        // be returned in returnLivestreamResult function
    }

    // Accepts the URI for a video file loaded from the user's gallery and attempts to run
    // face landmarker inference on the video. This process will evaluate every
    // frame in the video and attach the results to a bundle that will be
    // returned.
    fun detectVideoFile(
        videoUri: Uri,
        inferenceIntervalMs: Long
    ): VideoResultBundle? {
        if (runningMode != RunningMode.VIDEO) {
            throw IllegalArgumentException(
                "Attempting to call detectVideoFile" +
                        " while not using RunningMode.VIDEO"
            )
        }

        // Inference time is the difference between the system time at the start and finish of the
        // process
        val startTime = SystemClock.uptimeMillis()

        var didErrorOccurred = false

        // Load frames from the video and run the face landmarker.
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(context, videoUri)
        val videoLengthMs =
            retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLong()

        // Note: We need to read width/height from frame instead of getting the width/height
        // of the video directly because MediaRetriever returns frames that are smaller than the
        // actual dimension of the video file.
        val firstFrame = retriever.getFrameAtTime(0)
        val width = firstFrame?.width
        val height = firstFrame?.height

        // If the video is invalid, returns a null detection result
        if ((videoLengthMs == null) || (width == null) || (height == null)) return null

        // Next, we'll get one frame every frameInterval ms, then run detection on these frames.
        val resultList = mutableListOf<FaceLandmarkerResult>()
        val numberOfFrameToRead = videoLengthMs.div(inferenceIntervalMs)

        for (i in 0..numberOfFrameToRead) {
            val timestampMs = i * inferenceIntervalMs // ms

            retriever
                .getFrameAtTime(
                    timestampMs * 1000, // convert from ms to micro-s
                    MediaMetadataRetriever.OPTION_CLOSEST
                )
                ?.let { frame ->
                    // Convert the video frame to ARGB_8888 which is required by the MediaPipe
                    val argb8888Frame =
                        if (frame.config == Bitmap.Config.ARGB_8888) frame
                        else frame.copy(Bitmap.Config.ARGB_8888, false)

                    // Convert the input Bitmap object to an MPImage object to run inference
                    val mpImage = BitmapImageBuilder(argb8888Frame).build()

                    // Run face landmarker using MediaPipe Face Landmarker API
                    faceLandmarker?.detectForVideo(mpImage, timestampMs)
                        ?.let { detectionResult ->
                            resultList.add(detectionResult)
                        } ?: {
                        didErrorOccurred = true
                        faceLandmarkerHelperListener?.onError(
                            "ResultBundle could not be returned" +
                                    " in detectVideoFile"
                        )
                    }
                }
                ?: run {
                    didErrorOccurred = true
                    faceLandmarkerHelperListener?.onError(
                        "Frame at specified time could not be" +
                                " retrieved when detecting in video."
                    )
                }
        }

        retriever.release()

        val inferenceTimePerFrameMs =
            (SystemClock.uptimeMillis() - startTime).div(numberOfFrameToRead)

        return if (didErrorOccurred) {
            null
        } else {
            VideoResultBundle(resultList, inferenceTimePerFrameMs, height, width)
        }
    }

    // Accepted a Bitmap and runs face landmarker inference on it to return
    // results back to the caller
    fun detectImage(image: Bitmap): ResultBundle? {
        if (runningMode != RunningMode.IMAGE) {
            throw IllegalArgumentException(
                "Attempting to call detectImage" +
                        " while not using RunningMode.IMAGE"
            )
        }


        // Inference time is the difference between the system time at the
        // start and finish of the process
        val startTime = SystemClock.uptimeMillis()

        // Convert the input Bitmap object to an MPImage object to run inference
        val mpImage = BitmapImageBuilder(image).build()

        // Run face landmarker using MediaPipe Face Landmarker API
        faceLandmarker?.detect(mpImage)?.also { landmarkResult ->
            val inferenceTimeMs = SystemClock.uptimeMillis() - startTime
            return ResultBundle(
                landmarkResult,
                inferenceTimeMs,
                image.height,
                image.width
            )
        }

        // If faceLandmarker?.detect() returns null, this is likely an error. Returning null
        // to indicate this.
        faceLandmarkerHelperListener?.onError(
            "Face Landmarker failed to detect."
        )
        return null
    }

    private fun returnLivestreamResult(
        result: FaceLandmarkerResult, // 你需要分析这个 'result' 对象
        input: MPImage
    ) {
        // 性能控制：首先检查是否应该处理当前帧（在任何处理之前）
        if (!performanceOptimizer.shouldProcessFrame()) {
            return
        }
        
        if (result.faceLandmarks().size > 0) { // 确保检测到了面部
            val startTime = SystemClock.uptimeMillis()
            frameCounter++ // 增加帧计数器
            
            val finishTimeMs = SystemClock.uptimeMillis()
            val inferenceTime = finishTimeMs - result.timestampMs()

            // 在这里或通过 listener.onResults 传递后处理结果
            // --- 开始优化后的注意力判断逻辑 ---

            // 示例：获取面部界标
            val faceLandmarks = result.faceLandmarks().get(0) // 假设只处理第一个检测到的人脸

            // --- 开始优化后的注意力判断逻辑 ---
            var headPoseYaw = 0f
            var headPosePitch = 0f
            var headPoseRoll = 0f
            var isHeadPoseConsideredAttentive = false
            
            // 1. 头部姿态分析（与原来相同）
            if (result.facialTransformationMatrixes().isPresent
                && result.facialTransformationMatrixes().get().isNotEmpty()) {
                val matrixValues = result.facialTransformationMatrixes().get()[0]

                // 假设 matrixValues 是列主序的 4x4 矩阵
                // R = [ m0, m4, m8  ]
                //     [ m1, m5, m9  ]
                //     [ m2, m6, m10 ]

                val r00 = matrixValues[0]
                val r01 = matrixValues[4]
                val r02 = matrixValues[8]
                val r12 = matrixValues[9]
                val r22 = matrixValues[10]

                // 计算欧拉角 (近似，单位：弧度)
                val pitchRad = atan2(-r12, r22) // Pitch (绕X轴旋转，点头)
                val yawRad = asin(r02)          // Yaw (绕Y轴旋转，摇头)
                val rollRad = atan2(-r01, r00)  // Roll (绕Z轴旋转，歪头)

                // 转换为角度（使用预计算常数提升性能）
                headPosePitch = pitchRad * RAD_TO_DEG
                headPoseYaw = yawRad * RAD_TO_DEG
                headPoseRoll = rollRad * RAD_TO_DEG

                // 使用自适应阈值判断头部姿态
                val yawThreshold = adaptiveThresholds.getAdjustedYawThreshold()
                val pitchThreshold = adaptiveThresholds.getAdjustedPitchThreshold()
                
                if (abs(headPoseYaw) <= yawThreshold && abs(headPosePitch) <= pitchThreshold) {
                    isHeadPoseConsideredAttentive = true
                }

                // 优化：条件日志输出，减少I/O开销
                if (BuildConfig.DEBUG && frameCounter % 10 == 0L) {
                    Log.d(TAG, "Head Pose: Pitch=${String.format("%.2f", headPosePitch)}, Yaw=${String.format("%.2f", headPoseYaw)}, Roll=${String.format("%.2f", headPoseRoll)}")
                    Log.d(TAG, "Adaptive Thresholds: Yaw=${String.format("%.2f", yawThreshold)}, Pitch=${String.format("%.2f", pitchThreshold)}")
                }

            } else {
                Log.d(TAG, "Facial transformation matrix not available for head pose analysis.")
                isHeadPoseConsideredAttentive = false // 无法判断或默认不专注
            }

            // 2. 使用增强的EAR算法进行眼睛状态分析
            val (leftEyeEAR, rightEyeEAR, averageEAR) = enhancedEARCalculator.calculateEnhancedEAR(
                faceLandmarks, headPoseYaw, headPosePitch, headPoseRoll
            )
            
            // 使用自适应阈值判断眼睛状态
            val earThreshold = adaptiveThresholds.getAdjustedEARThreshold()
            val areEyesOpenBasedOnEar = averageEAR >= earThreshold
            
            // 优化：条件日志输出，减少I/O开销
            if (BuildConfig.DEBUG && frameCounter % 10 == 0L) {
                Log.d(TAG, "Enhanced EAR - Left: ${String.format("%.4f", leftEyeEAR)}, Right: ${String.format("%.4f", rightEyeEAR)}, Avg: ${String.format("%.4f", averageEAR)}")
                Log.d(TAG, "Adaptive EAR Threshold: ${String.format("%.4f", earThreshold)}")
            }

            // 3. 提取所有相关的 Blendshape 特征（与原来相同）
            var blendshapeFeatures = BlendshapeFeatures() // Initialize with defaults
            if (result.faceBlendshapes().isPresent && result.faceBlendshapes().get().isNotEmpty()) {
                val blendshapesCategories = result.faceBlendshapes().get()[0] // Get categories for the first face
                blendshapeFeatures = extractBlendshapeFeatures(blendshapesCategories)
                Log.i(TAG, "Blendshapes raw: JawOpen=${String.format("%.2f",blendshapeFeatures.jawOpen)}, BlinkL=${String.format("%.2f",blendshapeFeatures.eyeBlinkLeft)}")
            }

            // 4. 构建注意力特征对象
            val attentionFeatures = AttentionFeatures(
                headPoseYaw = headPoseYaw,
                headPosePitch = headPosePitch,
                headPoseRoll = headPoseRoll,
                leftEyeEAR = leftEyeEAR,
                rightEyeEAR = rightEyeEAR,
                averageEAR = averageEAR,
                blendshapes = blendshapeFeatures,
                isHeadPoseAttentive = isHeadPoseConsideredAttentive,
                areEyesOpen = areEyesOpenBasedOnEar,
                confidence = 0.8f, // 初始置信度，后续会由时序平滑器调整
                timestamp = System.currentTimeMillis()
            )

            // 5. 使用优化的规则引擎分析注意力状态
            var currentAttentionState = analyzeAttentionState(attentionFeatures)
            
            // 6. 新增：分析表情状态
            val expressionResult = expressionAnalyzer.analyzeExpression(blendshapeFeatures)

            // 7. 创建原始检测结果
            val rawAttentionResult = AttentionResult(
                state = currentAttentionState,
                confidence = 0.8f,
                features = attentionFeatures
            )

            // 8. 应用时序平滑
            val smoothedResult = temporalSmoother.addAndSmooth(rawAttentionResult)
            
            // 9. 创建综合分析结果
            val comprehensiveResult = createComprehensiveResult(smoothedResult, expressionResult)

            // 10. 如果正在校准，添加校准样本
            if (!adaptiveThresholds.isUserCalibrated() && 
                smoothedResult.state == AttentionState.ATTENTIVE && 
                smoothedResult.confidence > 0.7f) {
                adaptiveThresholds.addCalibrationSample(smoothedResult.features)
            }

            // 11. 更新人脸大小用于距离适应
            updateEnvironmentParams(currentBrightness, calculateFaceSize(faceLandmarks))

            // 优化：条件日志输出主要状态信息，减少I/O开销
            if (BuildConfig.DEBUG && frameCounter % 5 == 0L) {
                Log.i(TAG, "原始状态: $currentAttentionState")
                Log.i(TAG, "平滑后状态: ${smoothedResult.state} (置信度: ${String.format("%.2f", smoothedResult.confidence)})")
                Log.i(TAG, "表情状态: ${expressionResult.primaryExpression} (强度: ${String.format("%.2f", expressionResult.intensity)})")
                Log.i(TAG, "学习状态: ${expressionAnalyzer.getLearningStateFromExpression(expressionResult.primaryExpression)}")
                Log.i(TAG, "综合参与度: ${String.format("%.2f", comprehensiveResult.overallEngagement)}")
                Log.i(TAG, "状态稳定性: ${temporalSmoother.getCurrentStateStability()}帧")
            }

            // 性能优化：检查是否需要更新UI和状态是否有显著变化
            val shouldUpdateUI = performanceOptimizer.shouldUpdateUI() && 
                performanceOptimizer.hasSignificantChange(
                    smoothedResult.state,
                    expressionResult.primaryExpression,
                    comprehensiveResult.overallEngagement
                )

            if (shouldUpdateUI) {
                // 构建状态信息字符串并发送到UI
                val statusText = buildOptimizedStatusText(
                    smoothedResult, 
                    expressionResult, 
                    comprehensiveResult,
                    isHeadPoseConsideredAttentive,
                    areEyesOpenBasedOnEar
                )
                faceLandmarkerHelperListener?.onStatusUpdate(statusText)
                performanceOptimizer.markUIUpdated()
            } else {
                // 使用缓存的状态文本
                performanceOptimizer.getCachedStatusText()?.let { cachedText ->
                    faceLandmarkerHelperListener?.onStatusUpdate(cachedText)
                }
            }

            // 优化：条件输出调试统计信息，减少I/O开销
            if (BuildConfig.DEBUG && frameCounter % 15 == 0L) {
                Log.d(TAG, adaptiveThresholds.getThresholdInfo())
                Log.d(TAG, enhancedEARCalculator.getEARStats())
                Log.d(TAG, expressionAnalyzer.getExpressionStats())
                val perfStats = performanceOptimizer.getPerformanceStats()
                Log.d(TAG, "Performance: AvgTime=${perfStats.averageProcessingTime}ms, " +
                          "ActualFPS=${perfStats.actualFPS}/${perfStats.targetFPS}, " +
                          "Skip=${String.format("%.1f", perfStats.frameSkipPercentage)}% " +
                          "(${perfStats.processedFrames}/${perfStats.totalFrames})")
            }

            // 通过回调传递综合结果
            faceLandmarkerHelperListener?.onResults(
                ResultBundle(
                    result,
                    inferenceTime,
                    input.height,
                    input.width
                )
            )
            
            // 传递综合分析结果
            faceLandmarkerHelperListener?.onComprehensiveResults(comprehensiveResult)
            
            // 记录处理时间用于性能监控
            performanceOptimizer.recordProcessingTime(startTime)
        } else {
            faceLandmarkerHelperListener?.onEmpty()
        }
    }

    /**
     * 优化的注意力状态分析（使用自适应阈值）
     */
    private fun analyzeAttentionState(features: AttentionFeatures): AttentionState {
        val blendshapeFeatures = features.blendshapes

        // 使用自适应阈值
        val yawnThreshold = adaptiveThresholds.yawnThreshold
        val lookSideThreshold = adaptiveThresholds.lookSideThreshold
        val lookDownThreshold = adaptiveThresholds.lookDownThreshold
        val lookUpThreshold = adaptiveThresholds.lookUpThreshold
        val browDownThreshold = adaptiveThresholds.browDownThreshold
        val eyeSquintThreshold = adaptiveThresholds.eyeSquintThreshold
        val blinkThreshold = adaptiveThresholds.blinkThreshold

        // 优先级高的状态先判断
        // a. 打哈欠
        if (blendshapeFeatures.jawOpen > yawnThreshold) {
            return AttentionState.YAWNING
        }
        // b. 明显分心 - 看别处
        else if (abs(features.headPoseYaw) > adaptiveThresholds.getAdjustedYawThreshold() ||
            blendshapeFeatures.eyeLookOutLeft > lookSideThreshold ||
            blendshapeFeatures.eyeLookOutRight > lookSideThreshold) {
            return AttentionState.DISTRACTED_LOOKING_AWAY
        }
        // c. 明显分心 - 持续向下看
        else if ((blendshapeFeatures.eyeLookDownLeft > lookDownThreshold || 
                 blendshapeFeatures.eyeLookDownRight > lookDownThreshold) && 
                 features.headPosePitch < -15f) {
            return AttentionState.DISTRACTED_LOOKING_AWAY
        }
        // d. 困倦 (基于EAR和Blendshape的眨眼)
        else if (!features.areEyesOpen && 
                (blendshapeFeatures.eyeBlinkLeft > blinkThreshold || 
                 blendshapeFeatures.eyeBlinkRight > blinkThreshold)) {
            return AttentionState.DROWSY_FATIGUED
        }
        // e. 基础的头部姿态和眼睛睁开作为"专注"的底线
        else if (features.isHeadPoseAttentive && features.areEyesOpen) {
            if ((blendshapeFeatures.browDownLeft > browDownThreshold || 
                 blendshapeFeatures.browDownRight > browDownThreshold) &&
                (blendshapeFeatures.eyeSquintLeft > eyeSquintThreshold || 
                 blendshapeFeatures.eyeSquintRight > eyeSquintThreshold) &&
                (blendshapeFeatures.mouthPressLeft > 0.3f || 
                 blendshapeFeatures.mouthPressRight > 0.3f)) {
                return AttentionState.THINKING_CONCENTRATING
            } else if (blendshapeFeatures.eyeLookUpLeft > lookUpThreshold || 
                      blendshapeFeatures.eyeLookUpRight > lookUpThreshold) {
                return AttentionState.DISTRACTED_LOOKING_AWAY
            } else {
                return AttentionState.ATTENTIVE
            }
        }
        // f. 困惑
        else if (blendshapeFeatures.browDownLeft > browDownThreshold && 
                blendshapeFeatures.browInnerUp > 0.3f) {
            return AttentionState.CONFUSED
        }
        else {
            return AttentionState.UNKNOWN
        }
    }

    /**
     * 计算人脸大小（用于距离适应）
     */
    private inline fun calculateFaceSize(landmarks: List<NormalizedLandmark>): Float {
        if (landmarks.size < 17) return DEFAULT_DISTANCE // 使用预定义常数
        
        // 使用人脸轮廓计算大小
        val leftFace = landmarks[0]
        val rightFace = landmarks[16]
        val topFace = landmarks[10]
        val bottomFace = landmarks[152]
        
        val faceWidth = abs(rightFace.x() - leftFace.x())
        val faceHeight = abs(bottomFace.y() - topFace.y())
        
        // 归一化人脸大小（优化：减少除法运算）
        return ((faceWidth + faceHeight) * 0.5f).coerceIn(0.1f, 1.0f)
    }

    // 新增：开始用户校准
    fun startCalibration() {
        adaptiveThresholds.startCalibration()
        temporalSmoother.reset()
        enhancedEARCalculator.reset()
        expressionAnalyzer.reset() // 重置表情分析器
        Log.i(TAG, "开始用户校准过程")
    }
    
    // 新增：完成用户校准
    fun finishCalibration(): Boolean {
        val success = adaptiveThresholds.finishCalibration()
        Log.i(TAG, "用户校准${if (success) "成功" else "失败"}")
        return success
    }
    
    // 新增：获取校准进度
    fun getCalibrationProgress(): Float {
        return adaptiveThresholds.getCalibrationProgress()
    }
    
    // 新增：是否已校准
    fun isUserCalibrated(): Boolean {
        return adaptiveThresholds.isUserCalibrated()
    }
    
    // 新增：重置为默认阈值
    fun resetToDefaultThresholds() {
        adaptiveThresholds.resetToDefaults()
        temporalSmoother.reset()
        enhancedEARCalculator.reset()
        expressionAnalyzer.reset() // 重置表情分析器
        performanceOptimizer.reset() // 重置性能优化器
        memoryPool.cleanup() // 清理内存池
        frameCounter = 0L // 重置帧计数器
        Log.i(TAG, "重置为默认阈值")
    }

    // 新增：更新环境参数
    fun updateEnvironmentParams(brightness: Float, faceSize: Float) {
        currentBrightness = brightness
        currentFaceSize = faceSize
        adaptiveThresholds.adjustForLighting(brightness)
        adaptiveThresholds.adjustForDistance(faceSize)
    }

    // 新增：获取当前表情状态
    fun getCurrentExpressionState(): ExpressionState? {
        val history = expressionAnalyzer.expressionHistory.getAll()
        return history.lastOrNull()?.primaryExpression
    }
    
    // 新增：获取表情统计信息
    fun getExpressionStats(): String {
        return expressionAnalyzer.getExpressionStats()
    }
    
    // 新增：判断当前是否为积极表情
    fun isCurrentExpressionPositive(): Boolean {
        val currentExpression = getCurrentExpressionState()
        return currentExpression?.let { expressionAnalyzer.isPositiveExpression(it) } ?: false
    }
    
    // 新增：判断当前是否为消极表情
    fun isCurrentExpressionNegative(): Boolean {
        val currentExpression = getCurrentExpressionState()
        return currentExpression?.let { expressionAnalyzer.isNegativeExpression(it) } ?: false
    }
    
    // 新增：获取当前学习状态描述
    fun getCurrentLearningState(): String {
        val currentExpression = getCurrentExpressionState()
        return currentExpression?.let { expressionAnalyzer.getLearningStateFromExpression(it) } ?: "状态未知"
    }
    
    // 新增：获取表情变化趋势
    fun getExpressionTrend(): String {
        val history = expressionAnalyzer.expressionHistory.getAll()
        if (history.size < 3) return "数据不足"
        
        val recentExpressions = history.takeLast(3)
        val positiveCount = recentExpressions.count { expressionAnalyzer.isPositiveExpression(it.primaryExpression) }
        val negativeCount = recentExpressions.count { expressionAnalyzer.isNegativeExpression(it.primaryExpression) }
        
        return when {
            positiveCount >= 2 -> "积极趋势"
            negativeCount >= 2 -> "消极趋势"
            else -> "中性趋势"
        }
    }
    
    // 新增：重置表情分析器
    fun resetExpressionAnalyzer() {
        expressionAnalyzer.reset()
        Log.i(TAG, "表情分析器已重置")
    }

    // Return errors thrown during detection to this FaceLandmarkerHelper's
    // caller
    private fun returnLivestreamError(error: RuntimeException) {
        faceLandmarkerHelperListener?.onError(
            error.message ?: "An unknown error has occurred"
        )
    }

    /**
     * 创建综合分析结果
     */
    private fun createComprehensiveResult(
        attentionResult: AttentionResult, 
        expressionResult: ExpressionResult
    ): ComprehensiveAnalysisResult {
        // 计算综合参与度
        val overallEngagement = calculateOverallEngagement(attentionResult, expressionResult)
        
        return ComprehensiveAnalysisResult(
            attentionResult = attentionResult,
            expressionResult = expressionResult,
            overallEngagement = overallEngagement
        )
    }
    
    /**
     * 计算综合参与度分数
     */
    private fun calculateOverallEngagement(
        attentionResult: AttentionResult,
        expressionResult: ExpressionResult
    ): Float {
        // 注意力状态权重 (60%)
        val attentionWeight = 0.6f
        val attentionScore = when (attentionResult.state) {
            AttentionState.ATTENTIVE -> 1.0f
            AttentionState.THINKING_CONCENTRATING -> 0.9f
            AttentionState.CONFUSED -> 0.7f // 困惑也算一种参与
            AttentionState.YAWNING -> 0.3f
            AttentionState.DROWSY_FATIGUED -> 0.2f
            AttentionState.DISTRACTED_LOOKING_AWAY -> 0.1f
            AttentionState.UNKNOWN -> 0.5f
        }
        
        // 表情状态权重 (40%)
        val expressionWeight = 0.4f
        val expressionScore = when (expressionResult.primaryExpression) {
            ExpressionState.EXCITED -> 1.0f
            ExpressionState.CONCENTRATED -> 0.95f
            ExpressionState.SMILING -> 0.8f
            ExpressionState.SURPRISED -> 0.7f
            ExpressionState.NEUTRAL -> 0.6f
            ExpressionState.CONFUSED -> 0.5f
            ExpressionState.FRUSTRATED -> 0.3f
            ExpressionState.BORED -> 0.2f
            ExpressionState.LAUGHING -> 0.4f // 大笑可能表示分心
            ExpressionState.UNKNOWN -> 0.5f
        }
        
        // 考虑表情强度
        val intensityFactor = (expressionResult.intensity * 0.5f + 0.5f) // 0.5-1.0 range
        
        val engagementScore = (attentionScore * attentionWeight + 
                          expressionScore * expressionWeight * intensityFactor) *
                         attentionResult.confidence * expressionResult.confidence
        
        return engagementScore.coerceIn(0f, 1f)
    }

    /**
     * 构建UI状态显示文本（性能优化版本）
     */
    private fun buildOptimizedStatusText(
        attentionResult: AttentionResult,
        expressionResult: ExpressionResult,
        comprehensiveResult: ComprehensiveAnalysisResult,
        isHeadPoseAttentive: Boolean,
        areEyesOpen: Boolean
    ): String {
        return memoryPool.withStringBuilder { sb ->
            
            // 基础状态（优化：减少字符串查找）
            sb.append("🧠 注意力状态: ")
            val attentionText = when (attentionResult.state) {
                AttentionState.ATTENTIVE -> "✅ 专注"
                AttentionState.THINKING_CONCENTRATING -> "🤔 思考专注"
                AttentionState.CONFUSED -> "😕 困惑"
                AttentionState.DISTRACTED_LOOKING_AWAY -> "👀 分心看别处"
                AttentionState.DROWSY_FATIGUED -> "😴 困倦疲劳"
                AttentionState.YAWNING -> "🥱 打哈欠"
                AttentionState.UNKNOWN -> "❓ 未知"
            }
            sb.append(attentionText)
            sb.append(" (").append(OptimizedFormatter.formatConfidence(attentionResult.confidence)).append(")\n")
            
            // 表情状态（优化：减少字符串查找）
            sb.append("😊 表情状态: ")
            val expressionText = when (expressionResult.primaryExpression) {
                ExpressionState.SMILING -> "😊 微笑"
                ExpressionState.LAUGHING -> "😂 大笑"
                ExpressionState.SURPRISED -> "😲 吃惊"
                ExpressionState.CONFUSED -> "😕 困惑"
                ExpressionState.CONCENTRATED -> "🤔 专注思考"
                ExpressionState.BORED -> "😑 无聊"
                ExpressionState.FRUSTRATED -> "😤 沮丧"
                ExpressionState.EXCITED -> "🤩 兴奋"
                ExpressionState.NEUTRAL -> "😐 中性"
                ExpressionState.UNKNOWN -> "❓ 未知"
            }
            sb.append(expressionText)
            sb.append(" (强度: ").append(OptimizedFormatter.formatPercentage(expressionResult.intensity)).append(")\n")
            
            // 学习状态（优化：缓存函数调用结果）
            val learningState = expressionAnalyzer.getLearningStateFromExpression(expressionResult.primaryExpression)
            sb.append("📚 学习状态: ").append(learningState).append("\n")
            
            // 综合参与度（优化：预计算字符串）
            val engagementLevel = when {
                comprehensiveResult.overallEngagement >= 0.8f -> "🔥 非常高"
                comprehensiveResult.overallEngagement >= 0.6f -> "👍 较高"
                comprehensiveResult.overallEngagement >= 0.4f -> "📊 中等"
                comprehensiveResult.overallEngagement >= 0.2f -> "📉 较低"
                else -> "⚠️ 很低"
            }
            sb.append("📈 综合参与度: ").append(engagementLevel)
                .append(" (").append(OptimizedFormatter.formatPercentage(comprehensiveResult.overallEngagement)).append(")\n")
            
            // 细节状态（优化：减少条件判断）
            sb.append("\n📊 检测细节:\n")
            sb.append("   • 头部姿态: ").append(if (isHeadPoseAttentive) "✅ 专注" else "❌ 不专注").append("\n")
            sb.append("   • 眼睛状态: ").append(if (areEyesOpen) "👁️ 睁开" else "👁️‍🗨️ 闭合").append("\n")
            sb.append("   • 状态稳定性: ").append(temporalSmoother.getCurrentStateStability()).append("帧\n")
            
            // 表情趋势（优化：缓存调用结果）
            val expressionTrend = getExpressionTrend()
            sb.append("   • 表情趋势: ").append(expressionTrend)
            
            val statusText = sb.toString()
            
            // 缓存状态文本
            performanceOptimizer.cacheStatusText(
                statusText,
                attentionResult.state,
                expressionResult.primaryExpression,
                comprehensiveResult.overallEngagement
            )
            
            statusText
        }
    }

    // 新增：获取性能统计信息
    fun getPerformanceStats(): PerformanceStats {
        return performanceOptimizer.getPerformanceStats()
    }
    
    // 新增：获取内存池统计信息
    fun getMemoryStats(): MemoryPoolStats {
        return memoryPool.getStats()
    }

    companion object {
        const val TAG = "FaceLandmarkerHelper"
        private const val MP_FACE_LANDMARKER_TASK = "face_landmarker.task"

        // 性能优化：预计算的数学常数
        private const val RAD_TO_DEG = 57.2958f // 180/π，避免重复计算
        private const val DEFAULT_DISTANCE = 0.5f
        
        // 性能优化：Blendshape索引映射，避免字符串查找
        private val BLENDSHAPE_INDICES = mapOf(
            "eyeBlinkLeft" to 0, "eyeBlinkRight" to 1,
            "eyeLookDownLeft" to 2, "eyeLookDownRight" to 3,
            "eyeLookUpLeft" to 4, "eyeLookUpRight" to 5,
            "eyeLookOutLeft" to 6, "eyeLookOutRight" to 7,
            "eyeSquintLeft" to 8, "eyeSquintRight" to 9,
            "eyeWideLeft" to 10, "eyeWideRight" to 11,
            "browDownLeft" to 12, "browDownRight" to 13,
            "browInnerUp" to 14, "browOuterUpLeft" to 15, "browOuterUpRight" to 16,
            "jawOpen" to 17,
            "mouthSmileLeft" to 18, "mouthSmileRight" to 19,
            "mouthPressLeft" to 20, "mouthPressRight" to 21,
            "mouthPucker" to 22, "mouthShrugLower" to 23, "mouthShrugUpper" to 24,
            "cheekSquintLeft" to 25, "cheekSquintRight" to 26,
            "mouthFrownLeft" to 27, "mouthFrownRight" to 28,
            "mouthRollLower" to 29, "mouthRollUpper" to 30,
            "noseSneerLeft" to 31, "noseSneerRight" to 32
        )
        
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DEFAULT_FACE_DETECTION_CONFIDENCE = 0.5F
        const val DEFAULT_FACE_TRACKING_CONFIDENCE = 0.5F
        const val DEFAULT_FACE_PRESENCE_CONFIDENCE = 0.5F
        const val DEFAULT_NUM_FACES = 1
        const val OTHER_ERROR = 0
        const val GPU_ERROR = 1
    }

    data class ResultBundle(
        val result: FaceLandmarkerResult,
        val inferenceTime: Long,
        val inputImageHeight: Int,
        val inputImageWidth: Int,
    )

    data class VideoResultBundle(
        val results: List<FaceLandmarkerResult>,
        val inferenceTime: Long,
        val inputImageHeight: Int,
        val inputImageWidth: Int,
    )

    interface LandmarkerListener {
        fun onError(error: String, errorCode: Int = OTHER_ERROR)
        fun onResults(resultBundle: ResultBundle)
        fun onEmpty() {}
        
        // 新增：状态更新回调
        fun onStatusUpdate(statusText: String) {}
        
        // 新增：综合分析结果回调
        fun onComprehensiveResults(result: ComprehensiveAnalysisResult) {}
    }

    /**
     * 优化的Blendshape特征提取（避免字符串查找）
     */
    private fun extractBlendshapeFeatures(blendshapesCategories: List<com.google.mediapipe.tasks.components.containers.Category>): BlendshapeFeatures {
        // 性能优化：预构建数组避免重复Map查找
        val values = FloatArray(33) { 0f }
        
        // 一次遍历填充所有需要的值
        for (category in blendshapesCategories) {
            val index = BLENDSHAPE_INDICES[category.categoryName()]
            if (index != null) {
                values[index] = category.score()
            }
        }
        
        return BlendshapeFeatures(
            eyeBlinkLeft = values[0],
            eyeBlinkRight = values[1],
            eyeLookDownLeft = values[2],
            eyeLookDownRight = values[3],
            eyeLookUpLeft = values[4],
            eyeLookUpRight = values[5],
            eyeLookOutLeft = values[6],
            eyeLookOutRight = values[7],
            eyeSquintLeft = values[8],
            eyeSquintRight = values[9],
            eyeWideLeft = values[10],
            eyeWideRight = values[11],
            browDownLeft = values[12],
            browDownRight = values[13],
            browInnerUp = values[14],
            browOuterUpLeft = values[15],
            browOuterUpRight = values[16],
            jawOpen = values[17],
            mouthSmileLeft = values[18],
            mouthSmileRight = values[19],
            mouthPressLeft = values[20],
            mouthPressRight = values[21],
            mouthPucker = values[22],
            mouthShrugLower = values[23],
            mouthShrugUpper = values[24],
            cheekSquintLeft = values[25],
            cheekSquintRight = values[26],
            mouthFrownLeft = values[27],
            mouthFrownRight = values[28],
            mouthRollLower = values[29],
            mouthRollUpper = values[30],
            noseSneerLeft = values[31],
            noseSneerRight = values[32]
        )
    }
}
