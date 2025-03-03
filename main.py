import cv2
import numpy as np
import os
import time
from datetime import datetime
import csv
import face_recognition
import RPi.GPIO as GPIO

class FaceRecognitionAttendance:
    def __init__(self, database_path="employees", attendance_log="attendance.csv"):
        self.database_path = database_path
        self.attendance_log = attendance_log
        self.known_face_encodings = []
        self.known_face_names = []
        self.led_pin = 17  # GPIO pin for success LED
        self.buzzer_pin = 18  # GPIO pin for buzzer
        
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.led_pin, GPIO.OUT)
        GPIO.setup(self.buzzer_pin, GPIO.OUT)
        
        # Create directories if they don't exist
        if not os.path.exists(database_path):
            os.makedirs(database_path)
            
        # Create attendance log if it doesn't exist
        if not os.path.exists(attendance_log):
            with open(attendance_log, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "Date", "Time"])
        
        # Load known faces
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load employee faces from the database directory"""
        print("Loading employee database...")
        
        for filename in os.listdir(self.database_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Get employee name from filename (without extension)
                name = os.path.splitext(filename)[0]
                
                # Load image and get face encoding
                image_path = os.path.join(self.database_path, filename)
                image = face_recognition.load_image_file(image_path)
                
                # Try to find a face in the image
                face_encodings = face_recognition.face_encodings(image)
                
                if len(face_encodings) > 0:
                    # Take the first face found in the image
                    encoding = face_encodings[0]
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(name)
                    print(f"Loaded: {name}")
                else:
                    print(f"Warning: No face found in {filename}")
        
        print(f"Database loaded. {len(self.known_face_names)} employees registered.")
    
    def mark_attendance(self, name):
        """Record attendance in the CSV file"""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        # Check if person already marked attendance today
        already_marked = False
        if os.path.exists(self.attendance_log):
            with open(self.attendance_log, 'r') as file:
                for line in file:
                    if name in line and date_str in line:
                        already_marked = True
                        break
        
        if not already_marked:
            with open(self.attendance_log, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([name, date_str, time_str])
            
            print(f"Attendance marked for {name} at {time_str}")
            # Visual/audio confirmation
            self.success_indication()
            return True
        else:
            print(f"{name} already marked attendance today")
            return False
    
    def success_indication(self):
        """Provide visual/audio feedback on successful recognition"""
        # Blink LED
        GPIO.output(self.led_pin, GPIO.HIGH)
        # Short beep
        GPIO.output(self.buzzer_pin, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(self.led_pin, GPIO.LOW)
        GPIO.output(self.buzzer_pin, GPIO.LOW)
    
    def run(self):
        """Main loop to capture video and process faces"""
        # Initialize camera
        camera = cv2.VideoCapture(0)  # Use 0 for default camera (Pi Camera)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Starting face recognition attendance system...")
        print("Press 'q' to quit")
        
        try:
            while True:
                # Capture frame
                ret, frame = camera.read()
                if not ret:
                    print("Failed to capture image")
                    break
                
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB
                
                # Find face locations and encodings
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                # Process each face found
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # Scale back up face locations
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    # Check if the face matches any known face
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            # Mark attendance
                            self.mark_attendance(name)
                    
                    # Draw rectangle and name
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
                
                # Display result
                cv2.imshow('Face Recognition Attendance', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Short delay to reduce CPU usage
                time.sleep(0.1)
                
        finally:
            # Cleanup
            camera.release()
            cv2.destroyAllWindows()
            GPIO.cleanup()
    
    def add_employee(self, name):
        """Add a new employee to the database"""
        # Initialize camera
        camera = cv2.VideoCapture(0)
        
        print(f"Adding new employee: {name}")
        print("Position face in front of camera")
        print("Press 'c' to capture, 'q' to cancel")
        
        try:
            while True:
                # Capture frame
                ret, frame = camera.read()
                if not ret:
                    print("Failed to capture image")
                    break
                
                # Display frame
                cv2.imshow('Add Employee', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('c'):
                    # Save the image
                    filename = os.path.join(self.database_path, f"{name}.jpg")
                    cv2.imwrite(filename, frame)
                    
                    # Check if face is detected in the saved image
                    image = face_recognition.load_image_file(filename)
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if len(face_encodings) > 0:
                        # Add to current session
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(name)
                        print(f"Successfully added {name} to database")
                        break
                    else:
                        print("No face detected! Please try again.")
                        os.remove(filename)  # Remove the file if no face detected
                
                elif key == ord('q'):
                    print("Cancelled adding employee")
                    break
        
        finally:
            camera.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create attendance system
    attendance_system = FaceRecognitionAttendance()
    
    while True:
        print("\nFace Recognition Attendance System")
        print("1. Start Attendance System")
        print("2. Add New Employee")
        print("3. View Attendance Log")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            attendance_system.run()
        elif choice == '2':
            name = input("Enter employee name: ")
            attendance_system.add_employee(name)
        elif choice == '3':
            # Display attendance log
            if os.path.exists(attendance_system.attendance_log):
                with open(attendance_system.attendance_log, 'r') as file:
                    print("\nAttendance Log:")
                    print(file.read())
            else:
                print("No attendance records found.")
        elif choice == '4':
            print("Exiting program")
            break
        else:
            print("Invalid choice. Please try again.")