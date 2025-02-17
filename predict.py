import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from n import Galaxy10CNN  # Import Model

# Load Model
def load_model(model_path):
    model = Galaxy10CNN(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Define Transformations (Same as Training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prediction Function
def predict(model_name, img_array):
    model = load_model(model_name)
    
    # Preprocess Image
    img_tensor = transform(img_array).unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# Example Usage
if __name__ == "__main__":
    import cv2
    model_path = "galaxy10_final.pth"  # Change if using checkpoint
    image_path = "test_image.jpg"  # Replace with your image
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    prediction = predict(model_path, img)
    print(f"Predicted class: {prediction}")
