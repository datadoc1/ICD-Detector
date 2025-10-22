# model training script (updated paths)
# Usage: python training_script.py
from ultralytics import YOLO

def main():
    """
    Train a YOLOv11 model on the cleaned project layout.
    Expected dataset layout:
      model/data/train
      model/data/val
      model/data/test
    Dataset config expected at: model/data.yaml
    """
    # 1. Load a pretrained YOLOv11s model (lightweight starter)
    model = YOLO('yolo11s.pt')
    
    # 2. Train the model.
    # Point 'data' at the reorganized config under model/
    results = model.train(
        data='model/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        project='runs/train',
        name='xray_model_v11s'
    )
    
    print("Training complete!")
    print("Final results:", results)

if __name__ == '__main__':
    # pip install ultralytics
    main()