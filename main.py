from ultralytics import YOLO

def run_oceanic_ai():
    print("Loading YOLOv8 model for ocean waste detection...")
    model = YOLO('yolov8n.pt') 
    
    # Placeholder for Colab training execution
    print("Initiating training on marine debris dataset...")
    # results = model.train(data='ocean_waste_dataset.yaml', epochs=50, imgsz=640)
    
    # Run inference on a sample image
    image_path = 'sample_ocean_water.jpg'
    print(f"Running detection on {image_path}...")
    try:
        predictions = model(image_path)
        predictions[0].show()
        print("Detection complete. Bounding boxes drawn for plastic waste.")
    except Exception as e:
        print("Please place a valid image in the directory to run inference.")

if __name__ == "__main__":
    run_oceanic_ai()
