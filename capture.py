import cv2
import os

# Folder to save captured images
save_folder = "sample_images"
os.makedirs(save_folder, exist_ok=True)

# Try opening the microscope camera
# If your microscope is not detected on 0, try 1 or 2
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open digital microscope/camera")
    exit()

print("Digital microscope is running...")
print("Press 'c' to capture image")
print("Press 'q' to quit")

img_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame from microscope")
        break

    cv2.imshow("Digital Microscope Live Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        img_name = os.path.join(save_folder, f"microscope_image_{img_count}.png")
        cv2.imwrite(img_name, frame)
        print(f"Saved: {img_name}")
        img_count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()