# Preprocessing Steps for Brazilian Sign Language Recognition

## Overview

This is an outline of the preprocessing pipeline. The pipeline processes videos from the four different data sources (INES, SignBank, UFV, and V-Librasil) to create two types of preprocessed outputs:
1. Landmark data for the Landmark-LSTM model
2. Video data for the CNN-LSTM model

### Data Source Characteristics
- Different video dimensions and FPS across sources, and within sources
- Varying image capture conditions
- Different signer styles and speeds

### Data Organization
- Metadata stored in `data/raw/{data_source}/metadata.csv`

## Preprocessing Steps

### 1. Unify Signer Orientation
**Purpose**: Unify all signers to right-handed orientation for consistent processing.

**Process**:
- Assume all videos contain right-handed signs by default
- Record any left-handed signs in metadata
- For left-handed signs:
  - Flip video frames horizontally

**Benefits**:
- Pre-processes in subsequent steps that is based on motion measurements and pose detection will be more consistent
- We can apply data augmentation through horizontal flipping to easily include left-handed signs

### 2. Motion Detection and Trimming
**Purpose**: Remove non-informative frames before and after the actual sign performance.

**Process**:
- Apply motion detection algorithms to measure movement
- Process and combine results from multiple motion detection algorithms
- Use thresholding to determine:
  - Start frame: When significant motion begins
  - End frame: When significant motion ends
- Trim videos to the start and end of the sign performance

**Benefits**:
- Remove uninformative frames before and after the sign performance
- Focuses model on relevant sign performance

### 3. Pose Detection and Transformation
**Purpose**: Center and scale signers consistently in the frame.

**Process**:
- Detect face, pose, and hand landmarks using MediaPipe Holistic
- Use specific landmarks/measurements for:
  - Centering: Position signer in frame center
  - Scaling: Normalize signer size
- Apply transformations to:
  - Video frames
  - Landmark coordinates
- Store both original and normalized landmark data

**Benefits**:
- Consistent spatial positioning across videos means the model can focus on learning the unique features of the sign
- The positions and scales being unified means we have more control over the outcome of applying data augmentation methods
- If we reduce the dimension size, we will lose less information if the signer was scaled up (relative to the original size)

### 4. Dimension Resizing
**Purpose**: Match input requirements of respective models.

**Process**:
- Resize video frames to CNN-LSTM input dimensions
- Resize landmark coordinates to Landmark-LSTM input dimensions
- Maintain aspect ratios where possible
- Apply appropriate interpolation methods

**Benefits**:
- Consistent input dimensions for model training
- Optimized memory usage
- Preserved spatial relationships

### 5. Padding the Beginning and End of the Video
**Purpose**: Unify the real-time durations of the videos. If we had varying durations, when we sample frames, the model would have no sense of the time that has passed between frames. We have already trimmed the videos to the start and end of the sign performance, so if we pad the beginning and end equally, the sign performance will be centered in the series.

**Process**:
- Determine target sequence length
- For each video & landmark sequence:
  - Pad start and end with first and last frames respectively

**Benefits**:
- Consistent sequence lengths for model input
- Preserved temporal characteristics of signs i.e. the speed of the sign