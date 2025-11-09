from ultralytics import YOLO

# YOLOv11 training for helmet detection with optimized parameters
# Using YOLOv11 nano model for better accuracy on helmet and number plate detection
model = YOLO("yolo12n.pt")  # YOLOv11 nano model

# Training parameters optimized for small object detection (helmets and number plates):
# - epochs: 200 (increased for better convergence, monitor for overfitting)
# - imgsz: 640 (higher resolution for better detection of small objects like number plates)
# - batch: 4 (keep small for CPU training, increase if you have GPU)
# - patience: 50 (stop early if no improvement to prevent overfitting)
# - workers: 0 (for CPU, or increase to 2-4 if you have multiple CPU cores)
# - lr0: 0.001 (learning rate, can be adjusted)
# - augment: True (enable data augmentation for better generalization)

model.train(
    data="coco128.yaml",
    imgsz=640,  # Higher resolution for small object detection
    batch=4,    # Keep small for CPU, increase for GPU
    epochs=500, # More epochs for better accuracy, but monitor validation loss
    workers=0,  # CPU workers
    patience=50, # Early stopping patience
    lr0=0.001,  # Learning rate
    augment=True, # Enable augmentation
    cos_lr=True, # Cosine learning rate scheduler
    save=True,   # Save checkpoints
    project="runs/detect", # Save location
    name="train_yolo12_helmet" # Run name
)
