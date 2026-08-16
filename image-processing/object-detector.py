import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import ImageDraw, ImageFont
import gradio as gr

# ---------------------------
# 1. Load model + processor once
# ---------------------------
# Loading outside the function means we download and initialize
# the model only at startup, not on every prediction.
MODEL_NAME = "facebook/detr-resnet-50"

# Processor: resizes/normalizes the image and converts detection outputs
# back into real pixel coordinates for the original image size.
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

# Model: DETR (DEtection TRansformer) - detects objects and draws boxes
# around them, along with a predicted label and confidence score.
model = AutoModelForObjectDetection.from_pretrained(MODEL_NAME)
model.eval()  # inference mode - disables dropout etc.

# Automatically use GPU if available, otherwise fall back to CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# A small color palette to cycle through so each box gets a different color.
COLORS = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
]


# ---------------------------
# 2. Define the detection logic
# ---------------------------
def detect_objects(image, confidence_threshold: float = 0.7):
    """
    Run object detection on the uploaded image and draw bounding boxes
    around every object detected above the given confidence threshold.

    Args:
        image: A PIL image uploaded by the user.
        confidence_threshold: Minimum confidence score (0-1) required
                               to keep a detected object.

    Returns:
        annotated: The image with bounding boxes drawn on it.
        summary: A text list of detected objects and their confidence scores.
    """
    # Guard against no image being uploaded.
    if image is None:
        return None, "Please upload an image."

    # Convert the PIL image into the tensor format DETR expects,
    # and move it to the same device (CPU/GPU) as the model.
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Run the model without tracking gradients (faster, less memory - we're not training).
    with torch.no_grad():
        outputs = model(**inputs)

    # DETR predicts boxes in a normalized format; this converts them
    # back to actual pixel coordinates based on the original image size.
    # image.size is (width, height), but target_sizes expects (height, width).
    target_sizes = torch.tensor([image.size[::-1]])

    # Keep only detections whose confidence is above the threshold.
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=confidence_threshold
    )[0]

    # Make a copy of the image so we can draw on it without changing the original.
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    detections = []
    # results["scores"], ["labels"], and ["boxes"] are aligned lists -
    # each index i refers to the same detected object.
    for i, (score, label_id, box) in enumerate(
        zip(results["scores"], results["labels"], results["boxes"])
    ):
        # box is [x_min, y_min, x_max, y_max] in pixel coordinates.
        box = [round(coord, 1) for coord in box.tolist()]

        # Convert the numeric label ID (e.g. 17) into a readable name (e.g. "cat").
        label = model.config.id2label[label_id.item()]
        confidence = round(score.item(), 3)

        # Cycle through the color palette so each box looks distinct.
        color = COLORS[i % len(COLORS)]

        # Draw the bounding box rectangle and a text label above it.
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0], max(box[1] - 12, 0)), f"{label} ({confidence})", fill=color)

        detections.append(f"{label}: {confidence}")

    # Build a readable summary string, or a fallback message if nothing was found.
    summary = (
        "\n".join(detections)
        if detections
        else "No objects detected above the confidence threshold."
    )
    return annotated, summary


# ---------------------------
# 3. Build the web UI with Gradio
# ---------------------------
with gr.Blocks(title="Object Detector") as demo:
    gr.Markdown("## Object Detector")
    gr.Markdown("Powered by Hugging Face DETR (facebook/detr-resnet-50)")

    with gr.Row():
        # Left side: where the user uploads their image.
        image_input = gr.Image(type="pil", label="Upload Image")
        # Right side: the same image, but with boxes drawn on it.
        image_output = gr.Image(type="pil", label="Detected Objects")

    # Lets the user control how strict detection should be.
    # Lower threshold = more (possibly less accurate) detections.
    confidence_slider = gr.Slider(
        minimum=0.1, maximum=0.95, value=0.7, step=0.05, label="Confidence Threshold"
    )
    detect_btn = gr.Button("Detect Objects")
    detections_output = gr.Textbox(label="Detections", lines=6)

    # Wire the button click to the detect_objects function:
    # takes the image + slider value as input, and updates
    # both the annotated image and the text summary.
    detect_btn.click(
        fn=detect_objects,
        inputs=[image_input, confidence_slider],
        outputs=[image_output, detections_output],
    )

# ---------------------------
# 4. Start the web server
# ---------------------------
if __name__ == "__main__":
    demo.launch()
