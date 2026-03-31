from flask import Flask, render_template, request, url_for
import os
import uuid
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    # Save uploaded image
    ext = os.path.splitext(file.filename)[1]
    unique_name = f"{uuid.uuid4().hex}{ext}"
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(upload_path)

    # Read image
    image = cv2.imread(upload_path)
    if image is None:
        return "Could not read uploaded image"

    output = image.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur slightly to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect dark particles
    _, thresh_dark = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)
    thresh_dark = cv2.morphologyEx(thresh_dark, cv2.MORPH_OPEN, kernel)
    thresh_dark = cv2.morphologyEx(thresh_dark, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(thresh_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    particle_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Ignore tiny noise and very large unwanted regions
        if 15 < area < 2000:
            x, y, w, h = cv2.boundingRect(cnt)

            # Ignore large border/edge detections
            if w > 5 and h > 5:
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    output,
                    "Particle",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
                particle_count += 1

    # Write count on image
    cv2.putText(
        output,
        f"Particles Count: {particle_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Save result image
    result_name = f"result_{unique_name}.jpg"
    result_path = os.path.join(app.config["RESULT_FOLDER"], result_name)
    cv2.imwrite(result_path, output)

    return render_template(
        "index.html",
        uploaded_image=url_for("static", filename=f"uploads/{unique_name}"),
        result_image=url_for("static", filename=f"results/{result_name}"),
        particle_count=particle_count
    )


if __name__ == "__main__":
    app.run(debug=True)