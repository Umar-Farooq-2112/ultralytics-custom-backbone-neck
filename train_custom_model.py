"""
Train MobileNetV3-YOLO using standard YOLO API.

This script demonstrates how to train your custom MobileNetV3-YOLO model
using the exact same API as standard YOLO models (v8n, v8s, etc.).
"""

from ultralytics import YOLO


def main():
    """Train MobileNetV3-YOLO model using standard YOLO training pipeline."""
    
    # Method 1: Load from YAML config (RECOMMENDED)
    print("=" * 80)
    print("Loading MobileNetV3-YOLO from config...")
    print("=" * 80)
    model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')
    
    # Method 2: You can also load from checkpoint if you have trained weights
    # model = YOLO('path/to/mobilenetv3-yolo.pt')
    
    # Display model information
    print("\nModel loaded successfully!")
    print(f"Model: {model.model_name}")
    print(f"Task: {model.task}")
    
    # Train the model - EXACTLY like standard YOLO!
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    results = model.train(
        # Dataset configuration
        data='coco8.yaml',              # Path to dataset YAML (use coco8 for testing)
        
        # Training parameters
        epochs=100,                      # Number of training epochs
        imgsz=640,                       # Input image size
        batch=16,                        # Batch size (-1 for AutoBatch)
        
        # Model configuration
        device=0,                        # GPU device (0 for first GPU, 'cpu' for CPU)
        workers=8,                       # Number of dataloader workers
        
        # Optimization
        optimizer='AdamW',               # Optimizer (AdamW, SGD, Adam, etc.)
        lr0=0.001,                       # Initial learning rate
        lrf=0.01,                        # Final learning rate (lr0 * lrf)
        momentum=0.937,                  # SGD momentum/Adam beta1
        weight_decay=0.0005,             # Weight decay
        warmup_epochs=3.0,               # Warmup epochs
        warmup_momentum=0.8,             # Warmup initial momentum
        warmup_bias_lr=0.1,              # Warmup initial bias lr
        
        # Augmentation (data augmentation strength)
        hsv_h=0.015,                     # Hue augmentation
        hsv_s=0.7,                       # Saturation augmentation
        hsv_v=0.4,                       # Value augmentation
        degrees=0.0,                     # Rotation (+/- deg)
        translate=0.1,                   # Translation (+/- fraction)
        scale=0.5,                       # Scaling (+/- gain)
        shear=0.0,                       # Shear (+/- deg)
        perspective=0.0,                 # Perspective (+/- fraction)
        flipud=0.0,                      # Vertical flip probability
        fliplr=0.5,                      # Horizontal flip probability
        mosaic=1.0,                      # Mosaic augmentation probability
        mixup=0.0,                       # Mixup augmentation probability
        copy_paste=0.0,                  # Copy-paste augmentation probability
        
        # Training settings
        patience=50,                     # Early stopping patience (epochs)
        save=True,                       # Save checkpoints
        save_period=-1,                  # Save checkpoint every x epochs (-1 = disabled)
        cache=False,                     # Cache images (True/False/'ram'/'disk')
        
        # Validation
        val=True,                        # Validate during training
        plots=True,                      # Save plots
        
        # Output
        project='runs/train',            # Project directory
        name='mobilenetv3-yolo',         # Experiment name
        exist_ok=True,                   # Allow overwriting existing project
        
        # Advanced
        pretrained=False,                # Use pretrained weights (we have pretrained backbone)
        verbose=True,                    # Verbose output
        seed=0,                          # Random seed for reproducibility
        deterministic=True,              # Deterministic mode (slower but reproducible)
        single_cls=False,                # Train as single-class dataset
        rect=False,                      # Rectangular training
        cos_lr=False,                    # Cosine learning rate scheduler
        close_mosaic=10,                 # Disable mosaic augmentation for last N epochs
        amp=True,                        # Automatic Mixed Precision training
        fraction=1.0,                    # Dataset fraction to train on
        profile=False,                   # Profile ONNX and TensorRT speeds
        freeze=None,                     # Freeze layers (list of layer indices or None)
        
        # Multi-GPU training
        # device=[0, 1, 2, 3],           # Multi-GPU training
    )
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    
    # Results
    print("\nFinal metrics:")
    print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    
    # Save final model
    model.save('mobilenetv3-yolo-final.pt')
    print("\nModel saved to: mobilenetv3-yolo-final.pt")
    
    return model


def validate_model(model):
    """Validate the trained model."""
    print("\n" + "=" * 80)
    print("Running validation...")
    print("=" * 80)
    
    results = model.val(
        data='coco8.yaml',
        batch=16,
        imgsz=640,
        plots=True,
        save_json=True,
    )
    
    print("\nValidation results:")
    print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"  Precision: {results.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall: {results.results_dict.get('metrics/recall(B)', 0):.4f}")
    
    return results


def run_inference(model):
    """Run inference on test images."""
    print("\n" + "=" * 80)
    print("Running inference on test images...")
    print("=" * 80)
    
    # Predict on images
    results = model.predict(
        source='ultralytics/assets',     # Directory, image, video, URL
        imgsz=640,
        conf=0.25,                       # Confidence threshold
        iou=0.7,                         # NMS IoU threshold
        save=True,                       # Save results
        save_txt=True,                   # Save labels
        save_conf=True,                  # Save confidences
        show_labels=True,                # Show labels
        show_conf=True,                  # Show confidences
        line_width=2,                    # Bounding box line width
    )
    
    print(f"\nProcessed {len(results)} images")
    print("Results saved to: runs/detect/predict")
    
    return results


def export_model(model):
    """Export model to various formats."""
    print("\n" + "=" * 80)
    print("Exporting model...")
    print("=" * 80)
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    model.export(
        format='onnx',
        imgsz=640,
        dynamic=True,                    # Dynamic input shapes
        simplify=True,                   # Simplify ONNX model
    )
    print("✓ ONNX export complete")
    
    # Export to TorchScript
    print("\nExporting to TorchScript...")
    model.export(
        format='torchscript',
        imgsz=640,
    )
    print("✓ TorchScript export complete")
    
    # Other export formats available:
    # - format='engine' (TensorRT)
    # - format='coreml' (CoreML)
    # - format='saved_model' (TensorFlow SavedModel)
    # - format='pb' (TensorFlow GraphDef)
    # - format='tflite' (TensorFlow Lite)
    # - format='edgetpu' (TensorFlow Edge TPU)
    # - format='tfjs' (TensorFlow.js)
    # - format='paddle' (PaddlePaddle)
    # - format='ncnn' (NCNN)
    
    print("\nAll exports saved to model directory")


def benchmark_model(model):
    """Benchmark model performance."""
    print("\n" + "=" * 80)
    print("Benchmarking model...")
    print("=" * 80)
    
    results = model.benchmark(
        data='coco8.yaml',
        imgsz=640,
        half=False,                      # FP16 inference
        device=0,
    )
    
    print("\nBenchmark results:")
    print(results)
    
    return results


if __name__ == '__main__':
    # Train the model
    trained_model = main()
    
    # Optional: Run validation
    # validate_model(trained_model)
    
    # Optional: Run inference
    # run_inference(trained_model)
    
    # Optional: Export model
    # export_model(trained_model)
    
    # Optional: Benchmark
    # benchmark_model(trained_model)
    
    print("\n" + "=" * 80)
    print("All done! Your MobileNetV3-YOLO model is trained and ready to use!")
    print("=" * 80)
    print("\nQuick usage:")
    print("  from ultralytics import YOLO")
    print("  model = YOLO('mobilenetv3-yolo-final.pt')")
    print("  results = model.predict('image.jpg')")
