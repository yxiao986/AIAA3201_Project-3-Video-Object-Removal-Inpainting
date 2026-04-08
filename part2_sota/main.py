import os
import sys
import glob
import cv2
import json
import argparse
import subprocess
from tqdm import tqdm

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import evaluate_mask_quality

def parse_args():
    parser = argparse.ArgumentParser(description="Run Part 2 SOTA Pipeline (Automated)")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--gt_mask_dir", type=str, default=None)
    parser.add_argument("--output_base_dir", type=str, default="../results/part2_sota")
    parser.add_argument("--track_prompt", type=str, default="auto", 
                        help="Format: 'x1,y1,x2,y2'. Use 'auto' for default center box.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Define output paths
    dataset_out_dir = os.path.join(args.output_base_dir, args.dataset_name)
    mask_out_dir = os.path.join(dataset_out_dir, "masks")
    inpaint_out_dir = os.path.join(dataset_out_dir, "inpainted")
    
    os.makedirs(mask_out_dir, exist_ok=True)
    os.makedirs(inpaint_out_dir, exist_ok=True)

    # --- Step 1: Automated Tracking via Track Anything ---
    print(f"\n[{args.dataset_name}] Step 1: Generating masks using Track Anything API...")
    tracking_script = "../third_party/Track-Anything/run_tracking.py"
    
    # We run the script using the current python interpreter
    # We need to set PYTHONPATH so the script can find its own internal modules
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("../third_party/Track-Anything")
    
    try:
        subprocess.run([
            sys.executable, tracking_script,
            "--input_video", os.path.abspath(args.data_dir),
            "--output_dir", os.path.abspath(mask_out_dir),
            "--prompt", args.track_prompt
        ], check=True, env=env, cwd="../third_party/Track-Anything")
    except subprocess.CalledProcessError as e:
        print(f"Error during tracking: {e}")
        print("Please ensure weights are downloaded to third_party/Track-Anything/checkpoints/")
        return

    # --- Step 2: Mask Evaluation ---
    if args.gt_mask_dir:
        print(f"\n[{args.dataset_name}] Step 2: Evaluating SOTA mask quality...")
        pred_masks = [cv2.imread(p, 0) for p in sorted(glob.glob(os.path.join(mask_out_dir, "*.png")))]
        gt_masks = [cv2.imread(p, 0) for p in sorted(glob.glob(os.path.join(args.gt_mask_dir, "*.png")))]
        
        if len(pred_masks) == len(gt_masks) and len(gt_masks) > 0:
            metrics = evaluate_mask_quality(pred_masks, gt_masks)
            with open(os.path.join(dataset_out_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"Mask Evaluation Done: J_M={metrics['J_M']:.4f}")

    # --- Step 3: Video Inpainting via ProPainter ---
    print(f"\n[{args.dataset_name}] Step 3: Inpainting using ProPainter...")
    propainter_script = "../third_party/ProPainter/inference_propainter.py"
    
    try:
        subprocess.run([
            sys.executable, propainter_script,
            "--video", os.path.abspath(args.data_dir),
            "--mask", os.path.abspath(mask_out_dir),
            "--output", os.path.abspath(inpaint_out_dir)
        ], check=True, cwd="../third_party/ProPainter")
    except subprocess.CalledProcessError as e:
        print(f"Error during inpainting: {e}")

if __name__ == '__main__':
    main()