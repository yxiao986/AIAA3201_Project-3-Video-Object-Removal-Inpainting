import cv2
import os
import glob

def folder_to_video(image_folder, output_video_path, fps=30):
    # Get all PNG or JPG images and sort them by filename (ensuring 00000.png is first)
    images = sorted(glob.glob(os.path.join(image_folder, "*.png")) + 
                    glob.glob(os.path.join(image_folder, "*.jpg")))
    
    if not images:
        print(f"[Error] No images found in {image_folder}!")
        return
    
    # Read the first image to obtain the video resolution
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    
    # Initialize the OpenCV video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4 encoding
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Fusing {len(images)} images into a video...")
    for image in images:
        video.write(cv2.imread(image))
        
    video.release()
    print(f"✅ Success! Video saved to: {output_video_path}")

if __name__ == "__main__":
    # === Modify your dataset path here ===
    dataset_name = "tennis"  # Change to "bmx-trees" if running the bmx-trees dataset
    
    # Input image folder path
    input_folder = f"./data/{dataset_name}"
    # Output mp4 video path
    output_video = f"./data/{dataset_name}.mp4"
    
    folder_to_video(input_folder, output_video, fps=24)