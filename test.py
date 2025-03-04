import cv2
import picamera
import picamera.array
import numpy as np

def capture_and_display():
    """
    Capture video from Raspberry Pi Camera and display using OpenCV
    """
    # Initialize the PiCamera
    with picamera.PiCamera() as camera:
        # Camera configuration
        camera.resolution = (640, 480)  # Set resolution
        camera.framerate = 32  # Set framerate
        
        # Create a video capture object using PiRGBArray
        raw_capture = picamera.array.PiRGBArray(camera, size=(640, 480))
        
        # Allow the camera to warm up
        camera.start_preview()
        time.sleep(0.1)
        
        # Capture frames continuously
        for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
            # Get the NumPy array representing the image
            image = frame.array
            
            # Optional: Apply OpenCV processing
            # For example, convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Display the original and processed images
            cv2.imshow("Original", image)
            cv2.imshow("Grayscale", gray)
            
            # Clear the stream for the next frame
            raw_capture.truncate(0)
            
            # Exit loop if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Close all windows
        cv2.destroyAllWindows()

def main():
    # Ensure you have the required libraries installed
    # pip install opencv-python picamera numpy
    
    # Call the capture function
    capture_and_display()

if __name__ == "__main__":
    import time
    main()