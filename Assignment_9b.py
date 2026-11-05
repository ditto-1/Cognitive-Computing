import os
import shutil
import cv2
from deepface import DeepFace

class EmotionAwareAssistantModule:
    def __init__(self):
        print("Initializing Emotion-Aware Cognitive Assistant Module...")
        self._setup_local_weights()
        print("Using local OpenCV backend for emotion detection.")

    def _setup_local_weights(self):
        """Copies OpenCV's built-in Haar cascade locally so DeepFace doesn't need to download from GitHub."""
        try:
            weights_dir = os.path.expanduser("~/.deepface/weights")
            os.makedirs(weights_dir, exist_ok=True)
            
            haar_file = "haarcascade_frontalface_default.xml"
            target_path = os.path.join(weights_dir, haar_file)
            
            if not os.path.exists(target_path):
                source_path = os.path.join(cv2.data.haarcascades, haar_file)
                if os.path.exists(source_path):
                    shutil.copy(source_path, target_path)
                    print("Successfully linked local OpenCV Haar cascade weights.")
        except Exception as e:
            print(f"Weight setup note: {e}")

    def run_live_monitor(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam. Please verify device permissions or connection.")
            return

        print("Starting live monitor. Look at the camera! Press 'q' in the video window to quit.")

        frame_count = 0
        cached_emotion = "detecting..."
        assistant_strategy = "Warming up..."
        status_color = (0, 255, 255) # Yellow while loading

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam. Exiting loop.")
                break

            frame_count += 1

            # Analyze every 30 frames using the local opencv backend
            if frame_count % 30 == 0:
                try:
                    analysis = DeepFace.analyze(
                        frame, 
                        actions=['emotion'], 
                        detector_backend='opencv', 
                        enforce_detection=False, 
                        silent=True
                    )
                    
                    if isinstance(analysis, list) and len(analysis) > 0:
                        cached_emotion = analysis[0].get('dominant_emotion', 'neutral')
                    elif isinstance(analysis, dict):
                        cached_emotion = analysis.get('dominant_emotion', 'neutral')

                    # Update strategy based on detected emotion
                    if cached_emotion in ['fear', 'sad']:
                        assistant_strategy = "Reassuring Mode: Offering gentle prompts."
                        status_color = (0, 0, 255)    # Red
                    elif cached_emotion in ['surprise', 'angry']:
                        assistant_strategy = "Clarification Mode: Re-explaining step-by-step."
                        status_color = (0, 165, 255)  # Orange
                    else:
                        assistant_strategy = "Standard Assistance Mode: Normal pacing."
                        status_color = (0, 255, 0)    # Green

                except Exception as e:
                    print(f"DeepFace Analysis Error: {e}")

            # Render real-time feedback overlays onto the video stream
            cv2.putText(frame, f"Emotion: {cached_emotion.upper()}", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, f"Action: {assistant_strategy}", (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display the resulting frame
            cv2.imshow('Cognitive Assistant - Emotion & Behavior Adapter', frame)

            # Break loop when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Emotion monitoring session closed.")

if __name__ == "__main__":
    assistant_module = EmotionAwareAssistantModule()
    assistant_module.run_live_monitor()