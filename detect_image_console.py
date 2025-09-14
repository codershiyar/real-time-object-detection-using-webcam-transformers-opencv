# Ignore unnecessary warnings to keep the output clean
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import necessary libraries
from transformers import DetrImageProcessor, DetrForObjectDetection  # Hugging Face DETR model
from PIL import Image  # For opening images
import torch  # For tensors and model inference

# 1️⃣ Load the image you want to run detection on
# Make sure "img.png" is in the same folder as this script
image = Image.open("./imgs/img.png")

# 2️⃣ Load the pretrained DETR model and its image processor
# The 'no_timm' version avoids needing extra libraries like timm
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# 3️⃣ Prepare (preprocess) the image for the model
# This step converts the image into a tensor format the model understands
inputs = processor(images=image, return_tensors="pt")

# 4️⃣ Run the image through the DETR model to get predictions
# This gives us raw model outputs (logits, bounding boxes, etc.)
outputs = model(**inputs)

# 5️⃣ Post-process the model output into usable detection results
# We must tell the processor the size of the original image (height x width)
image_height, image_width = image.size[1], image.size[0]  # PIL: (width, height)
target_sizes = torch.tensor([[image_height, image_width]])

# This step gives us boxes, labels, and scores of detected objects above 90% confidence
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.9
)[0]

# 6️⃣ Loop over all detected objects and print their label and confidence score
for score, label in zip(results["scores"], results["labels"]):
    # Convert numeric label ID to a human-readable class name
    label_name = model.config.id2label[label.item()]

    # Round the confidence score to 3 decimal places for readability
    confidence = round(score.item(), 3)

    # Print the result
    print(f"Object: {label_name} | Confidence: {confidence}")
