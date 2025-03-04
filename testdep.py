import sys
import platform

def check_pi_dependencies():
    try:
        import picamera
        print("PiCamera library is installed successfully.")
    except ImportError:
        print("PiCamera library is not installed.")
        print("Recommended installation:")
        print("sudo apt-get install python3-picamera")
    
    try:
        import cv2
        print("OpenCV is installed successfully.")
    except ImportError:
        print("OpenCV is not installed.")
        print("Recommended installation:")
        print("sudo apt-get install python3-opencv")
    
    print("\nSystem Information:")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")

if __name__ == "__main__":
    check_pi_dependencies()