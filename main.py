from Detector import *
import os
import cv2

def main():
    # Specify the input source: "webcam" for PC camera, "video" for a video file
    input_source = "webcam"  # Change to "video" for video qfile input

    # Specify the webcam index (0, 1, 2, etc.)
    webcam_index = 0  # Change this to the desired webcam indexq

    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths relative to the current directory
    videoPath = os.path.join(current_dir, "test_videos", "test1.mp4")
    configPath = os.path.join(current_dir,"model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join(current_dir, "model_data","frozen_inference_graph.pb")
    classesPath = os.path.join(current_dir, "model_data","coco.names")

    # Verify if files exist
    for path in [configPath, modelPath, classesPath]:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            return

    # Initialize the detector based on the input source
    if input_source == "video":
        if not os.path.exists(videoPath):
            print(f"Error: File not found: {videoPath}")
            return
        detector = Detector(videoPath, configPath, modelPath, classesPath)
    elif input_source == "webcam":
        detector = Detector(webcam_index, configPath, modelPath, classesPath)

    # Start the video detection
    detector.onvideo()

if __name__ == "__main__":
    main()
