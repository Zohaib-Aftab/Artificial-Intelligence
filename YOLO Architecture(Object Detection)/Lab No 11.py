
"""Task No 1:
Test YOLOv5 on Sample Images for Object Detection. Objective: Use a pre-trained YOLOv5 model to detect objects in 5 sample images and visualize the results."""
import os
import cv2
import math
import time
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
from google.colab import files
from IPython.display import display, Video, Image
model = torch.hub.load(
    '/content/yolov5',
    'custom',
    path='yolov5s.pt',
    source='local',
    force_reload=False
)
model.conf = 0.25
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print('Model loaded successfully!')
print('Using device:', device)
print('Number of classes:', len(model.names))
!if [ ! -d /content/yolov5 ]; then git clone https://github.com/ultralytics/yolov5.git /content/yolov5; fi
!pip install -q -r requirements.txt
!pip install -q opencv-python-headless
print('YOLOv5 environment setup completed.')

image_dir = Path('/content/yolov5/test_images')
image_dir.mkdir(parents=True, exist_ok=True)
for old_file in image_dir.iterdir():
    if old_file.is_file() and old_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
        old_file.unlink()
print('Upload exactly 5 image files:')
uploaded_images = files.upload()
valid_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
for filename, data in uploaded_images.items():
    if Path(filename).suffix.lower() in valid_image_extensions:
        destination = image_dir / Path(filename).name
        with open(destination, 'wb') as f:
            f.write(data)
image_paths = sorted(
    p for p in image_dir.iterdir()
    if p.is_file() and p.suffix.lower() in valid_image_extensions
)
if len(image_paths) != 5:
    raise ValueError(
        f'Please provide exactly 5 images. The folder currently contains {len(image_paths)} valid image(s). '
        'Re-run this cell after ensuring there are exactly five images in test_images.'
    )
print('Total test images ready:', len(image_paths))
for p in image_paths:
    print('-', p.name)
image_results = model([str(p) for p in image_paths], size=640)
rendered_images = image_results.render()
image_output_dir = Path('/content/yolov5/detection_results')
image_output_dir.mkdir(parents=True, exist_ok=True)
for index, (image_path, rendered, detections) in enumerate(
    zip(image_paths, rendered_images, image_results.pandas().xyxy), start=1
):
    output_path = image_output_dir / f'detection_{index}.jpg'
    cv2.imwrite(str(output_path), rendered)
    print(f'\nImage {index}: {image_path.name}')
    if detections.empty:
        print('No objects detected above the confidence threshold.')
    else:
        for _, row in detections.iterrows():
            print(f"  {row['name']} — confidence: {row['confidence']:.2f}")
print('\nAnnotated images saved in:', image_output_dir)
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
for index, (image_path, rendered) in enumerate(zip(image_paths, rendered_images)):
    rgb_image = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
    axes[index].imshow(rgb_image)
    axes[index].set_title(image_path.name, fontsize=9)
    axes[index].axis('off')
plt.tight_layout()
plt.show()





"""Task No 2:
Test YOLOv5 on a 5-Second Video for Object Detection. Objective: Use a pre-trained YOLOv5 model to detect objects in a 5-second video and visualize the results."""
video_input_dir = Path('/content/yolov5/input_videos')
video_input_dir.mkdir(parents=True, exist_ok=True)
print('Upload one video file (preferably 5 seconds or longer):')
uploaded_videos = files.upload()
valid_video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.mpeg', '.mpg'}
video_paths = []
for filename, data in uploaded_videos.items():
    if Path(filename).suffix.lower() in valid_video_extensions:
        destination = video_input_dir / Path(filename).name
        with open(destination, 'wb') as f:
            f.write(data)
        video_paths.append(destination)
if len(video_paths) == 0:
    raise ValueError('No supported video was uploaded. Please upload an MP4, AVI, MOV, or MKV file.')
if len(video_paths) > 1:
    raise ValueError('Please upload only one video at a time and re-run this cell.')
video_path = video_paths[0]
print('Video selected:', video_path.name)
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError('OpenCV could not open this video. Try another video file/format.')
source_fps = cap.get(cv2.CAP_PROP_FPS)
total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if source_fps <= 0 or total_source_frames <= 0:
    cap.release()
    raise ValueError('Could not read the video FPS or frame count. Please use a valid video file.')
source_duration = total_source_frames / source_fps
frames_to_process = min(total_source_frames, int(math.ceil(5 * source_fps)))
print(f'Video FPS: {source_fps:.2f}')
print(f'Video duration: {source_duration:.2f} seconds')
print(f'Frames to process (first 5 seconds maximum): {frames_to_process}')
if source_duration < 5:
    print('NOTE: This video is shorter than 5 seconds, so all available frames will be processed.')
video_output_dir = Path('/content/yolov5/video_detection_results')
video_output_dir.mkdir(parents=True, exist_ok=True)
output_video_path = video_output_dir / 'yolov5_detected_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(
    str(output_video_path), fourcc, source_fps, (video_width, video_height)
)
if not video_writer.isOpened():
    cap.release()
    raise RuntimeError('Could not create the output video. Try another video format.')
print('Output video will be saved at:', output_video_path)
class_counts = Counter()
confidence_values = defaultdict(list)
processed_frames = 0
total_detections = 0
sampled_annotated_frames = []
sample_frame_indices = set()
if frames_to_process > 0:
    sample_frame_indices = set(
        int(x) for x in torch.linspace(0, frames_to_process - 1, steps=min(5, frames_to_process)).tolist()
    )
start_time = time.perf_counter()
try:
    for frame_index in range(frames_to_process):
        success, frame = cap.read()
        if not success:
            print(f'Video ended or frame could not be read at frame {frame_index}.')
            break
        frame_result = model(frame, size=640)
        annotated_frame = frame_result.render()[0]
        video_writer.write(annotated_frame)
        frame_detections = frame_result.pandas().xyxy[0]
        total_detections += len(frame_detections)
        for _, row in frame_detections.iterrows():
            class_name = str(row['name'])
            class_counts[class_name] += 1
            confidence_values[class_name].append(float(row['confidence']))
        if frame_index in sample_frame_indices:
            sampled_annotated_frames.append((frame_index, annotated_frame.copy()))
        processed_frames += 1
finally:
    cap.release()
    video_writer.release()
processing_seconds = time.perf_counter() - start_time
processing_fps = processed_frames / processing_seconds if processing_seconds > 0 else 0
if processed_frames == 0:
    raise RuntimeError('No video frames were processed. Please check the input video.')
print('\nVideo detection completed!')
print('Frames processed:', processed_frames)
print(f'Video time processed: {processed_frames / source_fps:.2f} seconds')
print(f'Processing time: {processing_seconds:.2f} seconds')
print(f'Processing speed: {processing_fps:.2f} frames/second')
print('Total detections across all processed frames:', total_detections)
print('Annotated video saved at:', output_video_path)
if sampled_annotated_frames:
    fig, axes = plt.subplots(1, len(sampled_annotated_frames), figsize=(20, 5))
    if len(sampled_annotated_frames) == 1:
        axes = [axes]
    for ax, (frame_index, annotated_frame) in zip(axes, sampled_annotated_frames):
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb_frame)
        ax.set_title(f'Frame {frame_index}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
display(Video(str(output_video_path), embed=True, width=720))
if class_counts:
    print('Detected classes (number of detections across all frames):')
    for class_name, count in class_counts.most_common():
        average_confidence = sum(confidence_values[class_name]) / len(confidence_values[class_name])
        print(f'- {class_name}: {count} detection(s), average confidence = {average_confidence:.2f}')
else:
    print('No objects were detected above the confidence threshold.')