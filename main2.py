import cv2
import numpy as np
import os
import time
from datetime import datetime
import csv
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from libcamera import controls

class LightweightFaceAttendance:
    def __init__(self, database_path="employees", attendance_log="attendance.csv"):
        self.database_path = database_path
        self.attendance_log = attendance_log
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.employee_ids = {}  # Map of ID to name
        self.led_pin = 17  # GPIO pin for success LED
        self.buzzer_pin = 18  # GPIO pin for buzzer
        
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.led_pin, GPIO.OUT)
        GPIO.setup(self.buzzer_pin, GPIO.OUT)
        
        # Create directories if they don't exist
        if not os.path.exists(database_path):
            os.makedirs(database_path)
            
        # Create models directory
        if not os.path.exists("models"):
            os.makedirs("models")
            
        # Create attendance log if it doesn't exist
        if not os.path.exists(attendance_log):
            with open(attendance_log, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "Date", "Time"])
        
        # Load or train the model
        self.load_or_train_model()
    
    def load_or_train_model(self):
        """Load existing model or train a new one if necessary"""
        model_path = "models/face_recognizer.yml"
        id_map_path = "models/employee_ids.csv"
        
        # Check if model and ID mapping exist
        if os.path.exists(model_path) and os.path.exists(id_map_path):
            # Load the model
            self.recognizer.read(model_path)
            
            # Load the ID to name mapping
            with open(id_map_path, 'r') as file:
                for line in file:
                    if line.strip():
                        parts = line.strip().split(',')
                        if len(parts) == 2:
                            emp_id, name = parts
                            self.employee_ids[int(emp_id)] = name
            
            print(f"Model loaded. {len(self.employee_ids)} employees registered.")
        else:
            # Train the model
            self.train_model()
    
    def train_model(self):
        """Train the face recognition model with current employee database"""
        print("Training model with employee database...")
        
        faces = []
        ids = []
        
        next_id = 1
        id_map = {}
        
        # Process each image in the database
        for filename in os.listdir(self.database_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Get employee name from filename (without extension)
                name = os.path.splitext(filename)[0]
                
                # Assign an ID
                emp_id = next_id
                next_id += 1
                id_map[emp_id] = name
                
                # Load image
                img_path = os.path.join(self.database_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                # Detect faces
                detected_faces = self.face_cascade.detectMultiScale(
                    img,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                # Add each face to training data
                for (x, y, w, h) in detected_faces:
                    faces.append(img[y:y+h, x:x+w])
                    ids.append(emp_id)
                    print(f"Processed: {name}")
                    break  # Use only the first face found in each image
        
        # Save ID mapping
        self.employee_ids = id_map
        with open("models/employee_ids.csv", 'w') as file:
            for emp_id, name in id_map.items():
                file.write(f"{emp_id},{name}\n")
        
        # Train the model
        if len(faces) > 0:
            self.recognizer.train(faces, np.array(ids))
            self.recognizer.save("models/face_recognizer.yml")
            print(f"Model trained and saved with {len(faces)} faces from {len(id_map)} employees.")
        else:
            print("No faces found in the database. Model not trained.")
    
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
        # Initialize Picamera2
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        
        print("Starting face recognition attendance system...")
        print("Press 'q' to quit")
        
        try:
            while True:
                # Capture frame
                frame = picam2.capture_array()
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                # Process each face found
                for (x, y, w, h) in faces:
                    # Draw rectangle around the face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Recognize the face
                    face_img = gray[y:y+h, x:x+w]
                    
                    try:
                        # Predict the id and confidence
                        id_num, confidence = self.recognizer.predict(face_img)
                        
                        # Lower confidence means better match (0 is perfect match)
                        if confidence < 70:  # Confidence threshold
                            name = self.employee_ids.get(id_num, "Unknown")
                            confidence_txt = f"{round(100 - confidence)}%"
                            
                            # Mark attendance
                            self.mark_attendance(name)
                        else:
                            name = "Unknown"
                            confidence_txt = f"{round(100 - confidence)}%"
                        
                        # Display name and confidence
                        cv2.putText(frame, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(frame, confidence_txt, (x+5, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1)
                    
                    except Exception as e:
                        print(f"Error during recognition: {e}")
                
                # Display result
                cv2.imshow('Face Recognition Attendance', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Short delay to reduce CPU usage
                time.sleep(0.1)
                
        finally:
            # Cleanup
            picam2.stop()
            cv2.destroyAllWindows()
            GPIO.cleanup()
    
    def add_employee(self, name):
        """Add a new employee to the database"""
        # Initialize Picamera2
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        
        print(f"Adding new employee: {name}")
        print("Position face in front of camera")
        print("Press 'c' to capture, 'q' to cancel")
        
        try:
            while True:
                # Capture frame
                frame = picam2.capture_array()
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                # Draw rectangle around detected faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Add Employee', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('c'):
                    if len(faces) > 0:
                        # Save the image
                        filename = os.path.join(self.database_path, f"{name}.jpg")
                        cv2.imwrite(filename, frame)
                        print(f"Successfully captured {name}'s face image")
                        
                        # Retrain the model with new data
                        self.train_model()
                        break
                    else:
                        print("No face detected! Please try again.")
                
                elif key == ord('q'):
                    print("Cancelled adding employee")
                    break
        
        finally:
            picam2.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create attendance system
    attendance_system = LightweightFaceAttendance()
    
    while True:
        print("\nLightweight Face Recognition Attendance System")
        print("1. Start Attendance System")
        print("2. Add New Employee")
        print("3. View Attendance Log")
        print("4. Retrain Model")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
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
            attendance_system.train_model()
        elif choice == '5':
            print("Exiting program")
            break
        else:
            print("Invalid choice. Please try again.")