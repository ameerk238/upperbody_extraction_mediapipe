import ffmpeg
import numpy as np
import mediapipe as mp
import os
import cv2
import psutil
import argparse
from multiprocessing import Pool
from math import ceil

class VideoWriter:
    def __init__(self, output_path, fps, frame_size):
        self.output_path = output_path
        self.video_writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            frame_size
        )
        print(f"Writing to: {output_path}")
        
    def write_frame(self, frame):
        if frame is not None:
            self.video_writer.write(frame)
        
    def release(self):
        if self.video_writer:
            self.video_writer.release()

class VideoProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def check_face_pose_correspondence(self, frame_rgb):
        # Get face detections
        face_results = self.mp_face.process(frame_rgb)
        if not face_results.detections:
            return False, None, None
            
        # Find a face with the right size
        valid_face = None
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box
            bbox_area = bbox.width * bbox.height * 100
            
            # Check if face is right size (1.80% - 4.20%)
            if 1.80 <= bbox_area <= 4.20:
                valid_face = detection
                break
        
        if not valid_face:
            return False, None, None
        
        # Check pose
        pose_results = self.mp_pose.process(frame_rgb)
        if not pose_results.pose_landmarks:
            return False, None, None
            
        # Verify nose landmark is within face bbox
        face_bbox = valid_face.location_data.relative_bounding_box
        nose_landmark = pose_results.pose_landmarks.landmark[0]
        if not (face_bbox.xmin < nose_landmark.x < face_bbox.xmin + face_bbox.width and 
                face_bbox.ymin < nose_landmark.y < face_bbox.ymin + face_bbox.height):
            return False, None, None
            
        return True, valid_face, pose_results.pose_landmarks

    def validate_landmarks(self, landmarks):
        # Key points for upper body
        key_points = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 19, 20]
        visibility_threshold = 0.5
        
        for idx in key_points:
            landmark = landmarks.landmark[idx]
            if landmark.visibility < visibility_threshold:
                return False
            if not (0.1 < landmark.x < 0.9 and 0.1 < landmark.y < 0.9):
                return False
        return True

    def draw_detections(self, frame, face_detection, pose_landmarks):
        frame_height, frame_width = frame.shape[:2]
        
        # Draw face bounding box
        if face_detection:
            bbox = face_detection.location_data.relative_bounding_box
            x = int(bbox.xmin * frame_width)
            y = int(bbox.ymin * frame_height)
            w = int(bbox.width * frame_width)
            h = int(bbox.height * frame_height)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw face detection score and bbox area
            score = face_detection.score[0]
            bbox_area = bbox.width * bbox.height * 100
            cv2.putText(frame, f'Face: {score:.2f}, Area: {bbox_area:.2f}%', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw pose landmarks
        if pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

    def process_video(self, input_path, output_path):
        if not os.path.exists(input_path):
            print(f"Input video not found: {input_path}")
            return
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        video_writer = None
        
        try:
            # Get video info
            probe = ffmpeg.probe(input_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            width = int(video_info['width'])
            height = int(video_info['height'])
            fps = eval(video_info.get('r_frame_rate', '30/1'))
            
            print(f"Processing: {input_path}")
            
            # Set up ffmpeg process
            process = (
                ffmpeg
                .input(input_path)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .overwrite_output()
                .run_async(pipe_stdout=True)
            )
            
            frame_count = 0
            processed_count = 0
            
            # Process frames
            while True:
                try:
                    in_bytes = process.stdout.read(width * height * 3)
                    if not in_bytes:
                        break
                        
                    frame_rgb = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
                    frame_rgb.flags.writeable = False
                    
                    # Check for face and pose
                    has_correspondence, face_detection, pose_landmarks = self.check_face_pose_correspondence(frame_rgb)
                    
                    if has_correspondence and self.validate_landmarks(pose_landmarks):
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        
                        # Draw detections
                        self.draw_detections(frame_bgr, face_detection, pose_landmarks)
                        
                        # Initialize video writer if needed
                        if video_writer is None:
                            video_writer = VideoWriter(output_path, fps, (width, height))
                        
                        video_writer.write_frame(frame_bgr)
                        processed_count += 1
                    
                    frame_count += 1
                    
                    if frame_count % 500 == 0:
                        print(f"Processed {frame_count} frames, kept {processed_count} frames")
                    
                except Exception as e:
                    print(f"Error on frame {frame_count}: {e}")
                    break
                    
        finally:
            process.stdout.close()
            process.wait()
            if video_writer:
                video_writer.release()
            self.mp_pose.close()
            self.mp_face.close()
            print(f"Done. Total: {frame_count}, Kept: {processed_count}")

def get_optimal_processes():
    # Get available memory
    mem = psutil.virtual_memory()
    available_ram_gb = mem.available / (1024 ** 3)
    
    # Estimate memory per process
    mem_per_process_gb = 2.0
    
    # Calculate max processes
    max_by_ram = max(1, int(available_ram_gb / mem_per_process_gb))
    
    # Consider CPU cores too
    cpu_count = os.cpu_count()
    
    # Take the smaller limit
    return min(max_by_ram, cpu_count - 1)

def process_video_batch(video_batch, input_folder, output_folder):
    processor = VideoProcessor()
    for video in video_batch:
        try:
            input_path = os.path.join(input_folder, video)
            output_path = os.path.join(output_folder, f"{os.path.splitext(video)[0]}_processed.mp4")
            processor.process_video(input_path, output_path)
        except Exception as e:
            print(f"Error processing {video}: {str(e)}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Extract upper body frames from videos')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input folder with videos')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output folder for processed videos')
    parser.add_argument('--processes', '-p', type=int, default=0,
                        help='Number of processes (0=auto)')
    args = parser.parse_args()
    
    input_folder = args.input
    output_folder = args.output
    
    # Check input folder
    if not os.path.exists(input_folder):
        print(f"Input folder doesn't exist: {input_folder}")
        return
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Get video files
    video_files = [f for f in os.listdir(input_folder) 
                  if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    
    if not video_files:
        print(f"No videos found in {input_folder}")
        return
    
    # Get processes count
    num_processes = args.processes if args.processes > 0 else get_optimal_processes()
    
    # Split into batches
    batch_size = ceil(len(video_files) / num_processes)
    video_batches = [video_files[i:i + batch_size] 
                    for i in range(0, len(video_files), batch_size)]
    
    print(f"Processing {len(video_files)} videos using {num_processes} processes")
    
    # Process in parallel
    with Pool(processes=num_processes) as pool:
        pool.starmap(process_video_batch,
                    [(batch, input_folder, output_folder) 
                     for batch in video_batches])
    
    print(f"All done! Check {output_folder} for results")

if __name__ == "__main__":
    main()