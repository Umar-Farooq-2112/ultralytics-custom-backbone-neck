"""Test exact Kaggle training code."""
from ultralytics import YOLO

print("=" * 60)
print("TESTING EXACT KAGGLE CODE")
print("=" * 60)

try:
    print("\n✅ Step 1: Loading model (exact user code)...")
    model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')
    print("✅ Model loaded!")
    
    print(f"\n✅ Model: {model.model.__class__.__name__}")
    print(f"✅ Backbone: {model.model.backbone.__class__.__name__}")
    print(f"✅ Stride: {model.model.stride.tolist()}")
    print(f"✅ Parameters: {sum(p.numel() for p in model.model.parameters())/1e6:.2f}M")
    
    print("\n✅ Ready for training!")
    print("\nYou can now run on Kaggle:")
    print("""
model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')
model.train(
    data="defects-in-timber/data.yaml",
    imgsz=640,
    batch=16,
    epochs=150,
    optimizer="SGD",
    momentum=0.937,
    weight_decay=5e-4,
    lr0=0.01,
    lrf=0.01,
    warmup_epochs=5,
    patience=25,
    workers=4,
    device=0,
)
""")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
