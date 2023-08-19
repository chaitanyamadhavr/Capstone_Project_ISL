""" import os
import cv2
import shutil
from pathlib import Path


# Input and output folders
input_folder = Path("C:/Users/chait/Desktop/Lady/RUcoming-20230819T145259Z-001/RUcoming")
output_folder = Path("C:/Users/chait/Desktop/Lady/NEW")

# Recreate output folder if it already exists
if output_folder.exists():
    shutil.rmtree(output_folder)
output_folder.mkdir(parents=True)

# Process videos in the input folder and its subfolders
for subfolder in input_folder.iterdir():
    if subfolder.is_dir():
        output_subfolder = output_folder / subfolder.relative_to(input_folder)
        output_subfolder.mkdir(parents=True, exist_ok=True)
        
        # Process videos in the subfolder
        video_files = list(subfolder.glob("*"))
        num_videos = len(video_files)

        for i, video_file in enumerate(video_files):
            # Generate output filename as sequential numbers
            output_filename = f"{i + 1}.mp4"
            output_path = output_subfolder / output_filename
            
            # Use OpenCV to convert the video format
            video = cv2.VideoCapture(str(video_file))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (680, 544))
                    
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                        
                cropped_frame = frame[:, 140:-140]
                resized_frame = cv2.resize(cropped_frame, (680, 544))
                out.write(resized_frame)
                    
            video.release()
            out.release()

# Print completion message
print("Video processing finished!")
 """

import os
import cv2
import shutil
from pathlib import Path

# Input and output folders
input_folder = Path("C:/Users/chait/Desktop/Lady/RUcoming-20230819T145259Z-001")
output_folder = Path("C:/Users/chait/Desktop/Lady/one")

# Recreate output folder if it already exists
if output_folder.exists():
    shutil.rmtree(output_folder)
output_folder.mkdir(parents=True)

# Process videos in the input folder and its subfolders
for subfolder in input_folder.iterdir():
    if subfolder.is_dir():
        output_subfolder = output_folder / subfolder.relative_to(input_folder)
        output_subfolder.mkdir(parents=True, exist_ok=True)

        # Process videos in the subfolder
        video_files = list(subfolder.glob("*.avi"))  # Process only .avi files
        num_videos = len(video_files)

        for i, video_file in enumerate(video_files):
            # Generate output filename as sequential numbers
            output_filename = f"{i + 1}.mp4"
            output_path = output_subfolder / output_filename

            # Use OpenCV to convert the video format
            video = cv2.VideoCapture(str(video_file))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (680, 544))

            while True:
                ret, frame = video.read()
                if not ret:
                    break

                cropped_frame = frame[:, 140:-140]
                resized_frame = cv2.resize(cropped_frame, (680, 544))
                out.write(resized_frame)

            video.release()
            out.release()

# Print completion message
print("Video processing finished!")
