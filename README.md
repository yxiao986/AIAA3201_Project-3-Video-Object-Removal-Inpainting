# Project 3: Video Object Removal & Inpainting

This repository contains the implementation for the Video Object Removal & Inpainting pipeline. It includes a baseline hand-crafted approach and a state-of-the-art (SOTA) AI-driven pipeline to identify, mask, and remove dynamic objects from videos, followed by background restoration.

## Project Overview

The core task of this project is to automatically remove dynamic objects (e.g., pedestrians, bicycles) from a video sequence and restore the missing background naturally. 

* **Part 1: Baseline Pipeline:** Uses **Mask R-CNN** for segmentation, **Optical Flow** for motion filtering, and **Temporal Background Propagation** with spatial fallback for inpainting.
* **Part 2: SOTA Reproduction:** Integrates foundation models. Uses **Track Anything** (Segment Anything + XMem) for zero-shot dynamic object tracking and masking, followed by **ProPainter** (Dual-domain Propagation + Sparse Transformer) for high-fidelity video inpainting.

## Environment Setup


### Setting up Part 1 (Lightweight)
```bash
# Create and activate a clean Conda environment
conda create -n cv_proj3 python=3.10 -y
conda activate cv_proj3

# Install required packages
pip install torch torchvision opencv-python numpy tqdm scikit-image
```
### Setting up Part 2 (SOTA Models)
Since Part 2 utilizes heavy models, you need to clone the official repositories and download their respective pre-trained weights.
#### 1. Clone Repositories:
```Bash
cd part2_sota
mkdir third_party
cd third_party
git clone https://github.com/gaomingqi/Track-Anything.git
git clone https://github.com/sczhou/ProPainter.git
```
#### 2. Download Weights:
**ProPainter**: Follow instructions in their official repo to download the weights (`ProPainter.pth`, `raft-things.pth`, `i3d_rgb_imagenet.pt`) and place them in `third_party/ProPainter/weights/`.

**Track Anything**: Download SAM weights (`sam_vit_b_01ec64.pth`) and XMem weights (`XMem-s012.pth`) and place them in `third_party/Track-Anything/checkpoints/`.

#### 3. Conda Environment:
```Bash
# It is recommended to use the same environment or ensure all dependencies are met
pip install -r third_party/Track-Anything/requirements.txt
pip install -r third_party/ProPainter/requirements.txt
```


## Repository Structure
```Plaintext
PROJECT3/
├── data/                       # Datasets directory
│   ├── bmx-trees/              # Raw frames for BMX sequence
│   └── ...                     # Other datasets
├── part1_baseline/             # Code for Part 1
│   ├── main.py                 # Pipeline controller
│   ├── mask_extractor.py       # Mask R-CNN & Optical Flow logic
│   └── inpainter.py            # Temporal & Spatial Inpainting
├── part2_sota/                 # Code for Part 2
│   └── main.py                 # SOTA Orchestration Script
├── third_party/                # Cloned SOTA repositories (NOT pushed to Git)
│   ├── Track-Anything/         
│   └── ProPainter/             
├── utils/
│   └── metrics.py              # Evaluation metrics (J_M, J_R, PSNR, SSIM)
└── results/                    # Output directory for generated artifacts
```
## Part 1: The Baseline Hand-crafted Approach

### Code Explanation
- `mask_extractor.py`: Handles the detection phase. It first retrieves soft masks from Mask R-CNN, thresholds them, and then compares frame $t-1$ and $t$ using optical flow to discard stationary bounding boxes.
- `inpainter.py`: Handles the restoration phase. It implements a temporal sliding window (default $\pm 15$ frames) to search for clean background pixels at the exact spatial location. If holes remain, cv2.inpaint patches them spatially.
- `main.py`: The entry point script. It uses argparse to dynamically load specific datasets, routes data through the extractor and inpainter, calculates evaluation metrics (if GT masks are provided), and saves the final frames and metrics.json.

### How to Run
Navigate to the `part1_baseline` directory before executing the commands:
```bash
cd part1_baseline
```

#### 1. Tennis Dataset (Mandatory)
   
Runs the pipeline and evaluates mask accuracy against the provided Ground Truth.
```bash
python main.py \
    --dataset_name tennis \
    --data_dir ../data/tennis \
    --gt_mask_dir ../data/tennis_mask \
    --output_base_dir ../results/part1_baseline
```

#### 2. BMX-Trees Dataset (Mandatory)
```bash
python main.py \
    --dataset_name bmx-trees \
    --data_dir ../data/bmx-trees \
    --gt_mask_dir ../data/bmx-trees_mask \
    --output_base_dir ../results/part1_baseline
```

#### 3. Wild Video Dataset (Mandatory)
For your self-captured video, there are no Ground Truth masks. Ensure your extracted frames are placed in ../data/wild_video. Omit the --gt_mask_dir argument.
```bash
python main.py \
    --dataset_name wild_video \
    --data_dir ../data/wild_video \
    --output_base_dir ../results/part1_baseline
```

#### 4. DAVIS Dataset (Optional / High-Score Attempt)
To evaluate on a specific sequence (e.g., `skate-jump`) from the standard DAVIS dataset, ensure the data is structured correctly and point the directories to the specific sequence.
```bash
python main.py \
    --dataset_name DAVIS_skate-jump \
    --data_dir ../data/DAVIS/JPEGImages/480p/skate-jump \
    --gt_mask_dir ../data/DAVIS/Annotations/480p/skate-jump \
    --output_base_dir ../results/part1_baseline
```

### Outputs & Artifacts
After running the commands, check the `../results/part1_baseline/[dataset_name]/` directory. You will find:
- `masks/`: The binary dynamic masks generated by our extractor.
- `inpainted/`: The final restored video frames.
- `metrics.json`: A JSON file containing the calculated evaluation metrics ($\mathcal{J}_M$ and $\mathcal{J}_R$) for the sequence (only generated if `--gt_mask_dir` was provided).
- 
## Part 2: SOTA Reproduction

This section utilizes foundation models to handle complex tracking scenarios (e.g., occlusion, motion blur) and performs high-fidelity video inpainting. 

To ensure the highest mask quality ($\mathcal{J}_M$ and $\mathcal{J}_R$), we employ an **Interactive-to-Automated Workflow**: User interaction is only required for the first frame via a Web UI, and the model automatically propagates the mask for the remaining frames.

### Code Architecture
- `part2_sota/main.py`: The main orchestration script that evaluates mask quality and executes the video inpainting engine.
- `third_party/Track-Anything/app.py`: The official Gradio UI used to obtain the initial high-precision prompt.
- `third_party/ProPainter/inference_propainter.py`: SOTA video inpainting engine.

### How to Run

#### Step 0: Data Preparation (Image to Video)
The Track-Anything UI requires a single video file (.mp4) for input. Use the provided script to convert the image sequence of a dataset into a video:

```bash
# For tennis dataset
python make_video.py --input_dir data/tennis --output_file data/tennis.mp4

# For bmx-trees dataset
python make_video.py --input_dir data/bmx-trees --output_file data/bmx-trees.mp4
```

#### Step 1: Interactive Mask Generation
Open a terminal and launch the Track-Anything Web UI:
```bash
cd third_party/Track-Anything
python app.py --sam_model_type vit_b --device cuda:0`
```
1. Open your browser to the local URL provided in the terminal (usually `http://127.0.0.1:12212`).
2. Upload your target video frames (e.g., `data/tennis`.
3. Click on the target object (e.g., the tennis player) in the first frame
4. Click "Add new object", then click "Tracking".
5. Once tracking is 100% complete, manually move all the generated `.png` mask files from `third_party/Track-Anything/result/mask/tennis/` to your project's target directory: `results/part2_sota/tennis/masks/`.

#### Step 2: Evaluation & Inpainting
Once the masks are saved, open a new terminal and run the main orchestration script:
```Bash
# Return to the project root
python part2_sota/main.py --dataset_name tennis --data_dir data/tennis --gt_mask_dir data/tennis_mask
```
*The pipeline will automatically apply mask dilation, compute J_M and J_R metrics, and save the final inpainted video frames to `results/part2_sota/tennis/inpainted/`.*

### ⚠️ Bug Fix in Track-Anything (app.py)

The modern `torchvision.io.write_video` function in `app.py` relies on deprecated versions of the `av` package (`<10.0.0`), which no longer have pre-compiled binaries available on PyPI, leading to Cython compilation crashes on Windows.

To resolve this without altering the environment requirements, **we reverted to the author's original (commented out) OpenCV implementation** for video generation (around line 347 in `app.py`), with an added RGB-to-BGR color channel conversion to prevent color inversion:

```python
# Reverted and patched video generation function:
def generate_video_from_frames(frames, output_path, fps=30):
    import cv2, os, np
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    height, width, _ = frames[0].shape
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (width, height))
    for frame in frames:
        # Added RGB2BGR conversion
        video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video.release()
    return output_path
```


### Outputs & Artifacts
After running the commands, check the `../results/part2_sota/[dataset_name]/` directory. You will find:
- `masks/`: The binary dynamic masks generated by our extractor.
- `inpainted/`: The final restored video frames.
- `metrics.json`: A JSON file containing the calculated evaluation metrics ($\mathcal{J}_M$ and $\mathcal{J}_R$) for the sequence (only generated if `--gt_mask_dir` was provided).

## Part 3: Exploration - Generative Video Inpainting

In this exploration phase, we attempt to address the limitations of pure propagation-based inpainting models (like ProPainter) by introducing Generative AI (Stable Diffusion). The core idea is to generatively repair missing keyframes and use the "Injection Trick" to propagate these high-fidelity hallucinations to adjacent frames.

We designed a **Dual-Track Evaluation Pipeline**:
1. **Qualitative Track (Dynamic Mask):** Uses GT dynamic masks to evaluate visual harmony.
2. **Quantitative Track (Stationary Mask):** Uses synthetically generated random stationary masks on clean videos to force "information starvation" and evaluate structural similarity (SSIM) and peak signal-to-noise ratio (PSNR) against Ground Truth.

### Code Explanation
- `part3_exploration/main.py`: The orchestrator script. It provides arguments to seamlessly switch between Qualitative and Quantitative evaluation tracks, manages isolated workspaces, and extracts specific frames for direct metric comparison.
- `utils/diffusion_utils.py`: Contains the `run_sd_inpainting` function. It executes the Stable Diffusion pipeline and implements **Pre-Generation Dilation** (to avoid motion blur residues) and **Gaussian Feathering Blending** to soften the hard edges of generative patches.
- `utils/mask_utils.py`: Generates the random stationary masks (combinations of strokes and circles) used exclusively for the quantitative evaluation track.

### How to Run
Navigate to the project root directory before executing.

**1. Qualitative Evaluation (Visual Testing with Dynamic Masks)**
For `tennis`:
```bash
python part3_exploration/main.py \
  --dataset_name tennis \
  --prompt "a clean tennis court, clay ground" \
  --gt_data_dir data/tennis \
  --gt_mask_dir data/tennis_mask
```

**2. Quantitative Evaluation (Metric Testing with Stationary Masks)**
```Bash
python part3_exploration/main.py \
  --dataset_name tennis_clean \
  --prompt "a clean tennis court, clay ground" \
  --clean_data_dir data/tennis_clean
  ```

### Outputs & Artifacts
All results for Part 3 are cleanly organized under the `results/part3_evaluation/[dataset_name]/ directory`:

`qualitative/`: Contains the .mp4 video outputs for both Baseline (Pure ProPainter) and Ours (SD + ProPainter).

`quantitative/`: Contains generated `stationary_masks/`, raw extracted video frames for visual auditing (`extracted_baseline_frames/`, `extracted_generative_frames/`), and the final metrics.json comparing SSIM and PSNR.

### Results & Comparison Analysis
Surprisingly, our empirical results indicate that introducing Stable Diffusion via simple keyframe injection does not yield quantitative or qualitative improvements over the Baseline ProPainter.

**Quantitative Results (Stationary Mask Evaluation)**
| Dataset | Method | PSNR | SSIM |
| :--- | :--- | :--- | :--- |
| **Tennis** | Baseline (Pure ProPainter) | **23.61** | **0.7922** |
| | Ours (SD + ProPainter) | 22.57 | 0.7540 |
| **BMX-Trees** | Baseline (Pure ProPainter) | **24.99** | **0.8342** |
| | Ours (SD + ProPainter) | 22.79 | 0.7671 |

### Advanced Exploration: Video Diffusion with DiffuEraser (Ultimate SOTA)

To push the boundaries of our video inpainting pipeline, we integrated **DiffuEraser**, a state-of-the-art conditional video diffusion model. Unlike traditional propagation methods or 2D SD injection, DiffuEraser leverages Temporal Attention and a specialized BrushNet to deeply understand semantics and generate highly coherent backgrounds, solving complex occlusion scenarios.

#### 1. Setup & Weights Configuration

To ensure a smooth and error-free evaluation experience for the TAs, avoiding complex HuggingFace network dependencies and hard-coded path issues, we have bundled all necessary pre-trained weights.

**Step 1: Clone the Official Repository**
```bash
cd third_party
git clone [https://github.com/lixiaowen-xw/diffueraser.git](https://github.com/lixiaowen-xw/diffueraser.git) DiffuEraser
cd ..
```
Step 2: Quick Weights Setup (⚡ Highly Recommended)
Please download  weights.zip follow the instruction

To ensure the native DiffuEraser code runs flawlessly, your directory structure must look exactly like this:
```Plaintext
third_party/DiffuEraser/
├── weights/
│   ├── diffuEraser/              # Core BrushNet & UNet configurations
│   ├── PCM_Weights/              # Phased Consistency Model (Acceleration LoRA)
│   ├── propainter/               # ProPainter core weights
│   ├── sd-vae-ft-mse/            # VAE model
│   └── stable-diffusion-v1-5/    # SD 1.5 Base Model
├── run_diffueraser.py
└── ...
```

2. How to Run DiffuEraser (Dual-Track Evaluation)
We have seamlessly integrated DiffuEraser into our unified evaluation orchestrator (main.py). By passing both --gt_data_dir and --clean_data_dir, the pipeline will automatically perform both qualitative generation and quantitative metric testing (PSNR & SSIM) in a single run.

Execution Command (Example on BMX-Trees):
```bash
python part3_exploration/main.py \
  --dataset_name bmx-trees \
  --prompt "a clean graffiti wall" \
  --gt_data_dir data/bmx-trees \
  --gt_mask_dir data/bmx-trees_mask \
  --clean_data_dir data/bmx-trees \
  --method diffueraser
```

**Quantitative Results (Stationary Mask Evaluation)**
| Dataset | Method | PSNR | SSIM |
| :--- | :--- | :--- | :--- |
| **Tennis** | Baseline (Pure ProPainter) | **24.37** | **0.7973** |
| | Ours (SD + ProPainter) | 20.45 | 0.6758 |
| **BMX-Trees** | Baseline (Pure ProPainter) | **25.76** | **0.8544** |
| | Ours (SD + ProPainter) | 19.66 | 0.5609 |