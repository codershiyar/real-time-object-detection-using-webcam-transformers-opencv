# 📦 Ignore unnecessary warnings from the transformers library
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 📦 Import required libraries
import cv2  # For webcam input and drawing
import torch  # For using the model
from PIL import Image  # For handling images in Python
from transformers import DetrImageProcessor, DetrForObjectDetection  # DETR model from Hugging Face

# 🧠 WHAT IS PIL?
# PIL stands for Python Imaging Library.
# It's used to open and manipulate images in Python.
# Many deep learning models (like DETR) work best with images in PIL format.

# 🧠 WHAT IS DETR?
# DETR = DEtection TRansformer
# It's an object detection model developed by Facebook AI.
# Unlike YOLO or SSD, DETR is based entirely on transformer architecture.
# It predicts objects directly without using anchor boxes or region proposals.

# 1️⃣ Load the pretrained DETR model and image processor
# These will download automatically from Hugging Face if not already cached
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model.eval()  # Set model to evaluation mode (inference only)

# 2️⃣ Open the default webcam (index 0)
camera = cv2.VideoCapture(0)

while True:
    # 3️⃣ Read a frame from the webcam
    success, frame = camera.read()
    if not success:
        break  # Stop if webcam frame can't be read

    # 4️⃣ Convert the frame from OpenCV (BGR) to PIL (RGB)
    # DETR expects images in RGB PIL format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)  # Convert NumPy array to PIL image

    # 5️⃣ Preprocess the PIL image so the model can understand it
    inputs = processor(images=pil_image, return_tensors="pt")

    # 6️⃣ Run the image through the model to get raw predictions
    with torch.no_grad():  # Disable gradient tracking for faster inference
        outputs = model(**inputs)

    # 7️⃣ Post-process the raw outputs to get boxes, labels, and scores
    height, width = pil_image.size[1], pil_image.size[0]
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([[height, width]]),
        threshold=0.8  # Minimum confidence score to show detection
    )[0]

    # 8️⃣ Loop over all detected objects
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        class_name = model.config.id2label[label.item()]

        # Only show "person" class
        if class_name != "person":
            continue

        # Get box coordinates and convert to integers
        x_min, y_min, x_max, y_max = map(int, box.tolist())

        # Draw the bounding box around the detected person
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Draw the label with confidence score
        label_text = f"{class_name} {score:.2f}"
        cv2.putText(frame, label_text, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 9️⃣ Show the annotated webcam frame in a window
    cv2.imshow("DETR - Live Person Detection", frame)

    # 🔟 Press 'q' to quit the live stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🔚 Clean up: close the camera and window
camera.release()
cv2.destroyAllWindows()
