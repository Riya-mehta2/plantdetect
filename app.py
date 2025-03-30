from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

def detect_plant(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the HSV range for green color
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    
    # Create a mask to detect green color
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    
    # Apply morphological operations to remove small noise
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around green areas (optional visualization)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter by area to remove small regions
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box
    
    # Calculate the percentage of green in the image
    green_percentage = (np.sum(mask_cleaned > 0) / mask_cleaned.size) * 100
    
    # Save the image with bounding boxes (optional)
    output_path = image_path.replace('.jpg', '_output.jpg')
    cv2.imwrite(output_path, image)
    
    # Set a threshold for green detection (e.g., at least 5% of the image should be green)
    if green_percentage > 5:
        return f"Plant detected! Green percentage: {green_percentage:.2f}%.", output_path
    else:
        return "No plant detected.", output_path

@app.route('/detect_plant', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    
    # Call the detect_plant function
    result, output_image_path = detect_plant(file_path)
    
    return jsonify({
        'result': result,
        'output_image': output_image_path
    })

if __name__ == '__main__':
    # Ensure the uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    app.run(debug=True)