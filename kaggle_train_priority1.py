"""
Kaggle Training Script for Priority 1 Implementation
CSPResNet + P2 Detection + Multi-Scale Training

Usage:
    python kaggle_train_priority1.py --data your-data.yaml --epochs 300
"""

import argparse
from ultralytics import YOLO


def train_priority1(data_path, epochs=300, batch=16, device=0):
    """
    Train CSPResNet-YOLO-P2 model with Priority 1 improvements.
    
    Args:
        data_path (str): Path to data.yaml
        epochs (int): Number of training epochs
        batch (int): Batch size
        device (int): GPU device ID
    """
    
    print("="*80)
    print("PRIORITY 1 TRAINING - CSPResNet + P2 Detection + Multi-Scale")
    print("="*80)
    
    # Load model
    print("\n1. Loading model...")
    model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')
    print(f"   ✓ Model: {type(model.model).__name__}")
    print(f"   ✓ Backbone: {type(model.model.backbone).__name__}")
    print(f"   ✓ Neck: {type(model.model.neck).__name__}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"   ✓ Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Training configuration
    print("\n2. Training configuration...")
    print(f"   - Dataset: {data_path}")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch size: {batch}")
    print(f"   - Optimizer: AdamW")
    print(f"   - Learning rate: 0.001 → 0.00001 (cosine)")
    print(f"   - Multi-scale: Enabled (scale=0.9)")
    print(f"   - Device: cuda:{device}")
    
    # Start training
    print("\n3. Starting training...")
    print("-"*80)
    
    results = model.train(
        # Data
        data=data_path,
        
        # Training
        epochs=epochs,
        batch=batch,
        imgsz=640,
        
        # Optimizer
        optimizer='AdamW',
        lr0=0.001,          # Initial learning rate
        lrf=0.01,           # Final learning rate (lr0 * lrf = 0.00001)
        momentum=0.9,
        weight_decay=0.0005,
        
        # Learning rate schedule
        cos_lr=True,        # Cosine LR scheduler
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Multi-scale training (Priority 1 Feature #3)
        scale=0.9,          # Image scale (+/- gain)
        
        # Loss gains
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Augmentations (optimized for timber defects)
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        degrees=15.0,       # Rotation (wood can be rotated)
        translate=0.1,
        scale_aug=0.5,
        flipud=0.0,         # No vertical flip (wood grain has direction)
        fliplr=0.5,         # Horizontal flip OK
        mosaic=1.0,
        mixup=0.1,
        
        # Validation
        val=True,
        save_period=10,
        plots=True,
        patience=50,        # Early stopping
        
        # Inference
        conf=0.25,
        iou=0.7,
        max_det=300,
        
        # Device & workers
        device=device,
        workers=8,
        
        # Project
        project='runs/train',
        name='cspresnet-p2-priority1',
        exist_ok=False,
        verbose=True,
        
        # Advanced
        amp=True,           # Automatic Mixed Precision
        close_mosaic=10,    # Disable mosaic for final epochs
        seed=0,
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    
    # Print results
    print("\nFinal Metrics:")
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"   - mAP@0.5: {metrics.get('metrics/mAP50(B)', 'N/A')}")
        print(f"   - mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"   - Precision: {metrics.get('metrics/precision(B)', 'N/A')}")
        print(f"   - Recall: {metrics.get('metrics/recall(B)', 'N/A')}")
    
    # Compare with baseline
    print("\nExpected Improvement:")
    print("   - Baseline: 80% mAP")
    print("   - Target: 85% mAP")
    print("   - Check if achieved! ✓")
    
    print("\nModel saved to: runs/train/cspresnet-p2-priority1/weights/best.pt")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train Priority 1 Model')
    parser.add_argument('--data', type=str, required=True, 
                       help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=300,
                       help='Number of epochs (default: 300)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device ID (default: 0)')
    
    args = parser.parse_args()
    
    train_priority1(
        data_path=args.data,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device
    )


if __name__ == '__main__':
    main()
