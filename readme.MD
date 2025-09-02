# Human-Centric Video Stabilization

This project provides a complete pipeline to process a video of a person, isolate them from the background, and stabilize their position on screen. The final output includes a stabilized video with a side-by-side comparison with the original, and the stabilized data.

This implementation uses:

- **PyTorch (DeepLabv3)** for background removal.
- **MediaPipe** for human pose detection.
- **Kalman Filter** for smoothing the stabilization trajectory.
- **OpenCV** for video processing and rendering.

> Please note that the background removal is not being done currently in the code. Uncomment the required lines to run with background removal.

## Setup

1. **Clone the repository:**

    ```sh
    git clone https://github.com/IndraReddy5/Human_Centric_Video_Stabilization.git
    cd human-centric-stabilization
    ```

2. **Create a virtual environment (recommended):**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required libraries:**
    Make sure you have PyTorch installed first. You can find installation instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/). Choose the command that matches your system (CPU or GPU).

    Then, install the rest of the dependencies:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

Run the main script from the root directory of the project. You must provide the path to an input video.

```sh
python src/run.py --input_video path/to/your/video.mp4 --output_dir output --device cuda
```

### Command-Line Arguments

- `--input_video` (required): Path to the input video file.
- `--output_dir` (optional): Directory where output files will be saved. Defaults to `./output`.
- `--device` (optional): The device to use for the DeepLabv3 model. Choices are `cpu` or `cuda`. Defaults to `cpu`.

## Pipeline Explained

The process is executed in three main passes:

1. **Pass 1: Data Collection**: The pipeline first iterates through the entire video to detect the human pose in each frame using MediaPipe. The center of the left_hip and right_hip or shoulder points is chosen as a stable **anchor point**, and its raw (shaky) coordinates are stored for every frame. All detected pose keypoints are also saved.

2. **Pass 2: Trajectory Smoothing**: The raw trajectory of the anchor point is smoothed using a **Kalman Filter**. This filter predicts the person's position and corrects its prediction with the actual detected position, resulting in a smooth path that filters out high-frequency camera shake.

3. **Pass 3: Rendering**: The pipeline reads the video a final time. For each frame:

    - The background can be removed using **DeepLabv3**.
    - A transformation is calculated to move the person from their original position to the smoothed position.
    - `cv2.warpAffine` applies this transformation to the frame, effectively stabilizing the person.
    - The final stabilized video and a side-by-side comparison video are rendered.

## Outputs

- `output/stabilized_video.mp4`: The final video with the person stabilized.
- `output/comparison_video.mp4`: A side-by-side video showing the original footage next to the stabilized version.
- `output/pose_data.json`: A JSON file containing the detected pose keypoints for each frame of the video.

## Results

- The following results are acquired on a 13secs video with 25fps, 328 frames and a resolution of 1080 * 1920.

> | Time Taken| Device | Background Removal |
> |:---------:|:------:|:------------------:|
> |    24s    |  CPU   |    False           |
> |  1H 15m   |  CPU   |    True            |
> |  3m 27s   |  GPU   |    True            |

<video width="640" height="360" controls>
  <source src="./Results/without_bg.mp4" type="video/mp4">
</video>

<video width="640" height="360" controls>
  <source src="./Results/with_bg_1.mp4" type="video/mp4">
</video>

<video width="640" height="360" controls>
  <source src="./Results/with_bg_2.mp4" type="video/mp4">
</video>

## Limitations

- **Performance**: This pipeline processes the video three times and runs one / two deep learning models, so it is not real-time. Processing speed will depend heavily on the CPU/GPU and video resolution. On a standard CPU, expect a low frame rate.
- **Detection Failures**: If MediaPipe fails to detect a pose in a frame, the last known position is used. This can cause a slight drift if detection fails for several consecutive frames.
- **Single Person**: The current implementation is designed to track a single person and will anchor on the first person detected by MediaPipe.

## Future Targets

- Background construction using temporal relation.
- Auto selecting video resolution adapting to the frequent translations of the frames.
