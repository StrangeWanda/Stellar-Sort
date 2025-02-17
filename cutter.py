import roboflow
import os
import cv2

# Authenticate with your API key
rf = roboflow.Roboflow(api_key="eupf89ahjno75Ip8E94w")

# Access your project and model version
project = rf.workspace().project("galaxy-detection")
model = project.version("3").model

# Define the image file to use for inference
image_file = "manygals.jpg"

# Run inference on the image
prediction = model.predict(image_file)

model.confidence = 0.14

# Print the JSON response
print(prediction.json())

# Display the image with predictions
prediction.plot()

output_dir = "extracted_objects"
os.makedirs(output_dir, exist_ok=True)

image = cv2.imread(image_file)

# Iterate over each prediction
for i, pred in enumerate(prediction.json()['predictions']):
    # Extract bounding box coordinates
    x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
    
    # Calculate top-left and bottom-right coordinates
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)
    
    # Crop the detected object from the image
    cropped_image = image[y1:y2, x1:x2]
    
    # Save the cropped image
    output_path = os.path.join(output_dir, f"object_{i + 1}.jpg")
    cv2.imwrite(output_path, cropped_image)
    print(f"Saved: {output_path}")

