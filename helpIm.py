import cv2
import numpy as np

def preprocess_image(pil_image):
    # Convert PIL image to NumPy array
    img = np.array(pil_image)
    
    # Convert RGB to BGR (OpenCV uses BGR by default)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Get image dimensions
    h, w, _ = img.shape

    # Find the smaller dimension and crop to a centered square
    size = min(h, w)
    start_x = (w - size) // 2
    start_y = (h - size) // 2
    cropped_img = img[start_y:start_y+size, start_x:start_x+size]

    # Resize to 207x207
    resized_img = cv2.resize(cropped_img, (207, 207), interpolation=cv2.INTER_LINEAR)

    # Downscale 3 times to 69x69
    final_img = cv2.resize(resized_img, (69, 69), interpolation=cv2.INTER_LINEAR)

    return final_img  # This is a NumPy array

def get_galaxy_type(class_number):
    # Map galaxy class numbers to their types
    galaxy_type_map = {
        0: "Irregular/Lenticular",
        1: "Elliptical",
        2: "Elliptical",
        3: "Elliptical",
        4: "Edge-on Spiral",
        5: "Edge-on Spiral",
        6: "Edge-on Spiral",
        7: "Spiral",
        8: "Spiral",
        9: "Spiral"
    }
    
    # Return the corresponding type
    return galaxy_type_map.get(class_number, "Unknown Class")

if __name__=="__main__":
    for i in range(10):
        print(f"Class {i}: {get_galaxy_type(i)}")