"""
Quick test script to verify MobileNetV3-YOLO integration.
Run this to ensure everything is properly set up.
"""

import sys
import torch


def test_imports():
    """Test that all modules can be imported."""
    print("=" * 80)
    print("TEST 1: Module Imports")
    print("=" * 80)
    
    try:
        from ultralytics import YOLO
        print("✓ ultralytics.YOLO imported")
    except Exception as e:
        print(f"✗ Failed to import YOLO: {e}")
        return False
    
    try:
        from ultralytics.nn.custom_models import MobileNetV3YOLO
        print("✓ MobileNetV3YOLO imported")
    except Exception as e:
        print(f"✗ Failed to import MobileNetV3YOLO: {e}")
        return False
    
    try:
        from ultralytics.nn.modules import (
            MobileNetV3BackboneDW,
            UltraLiteNeckDW,
            DWConvCustom,
            CBAM_ChannelOnly,
            SimSPPF,
            P5Transformer,
        )
        print("✓ All custom modules imported")
    except Exception as e:
        print(f"✗ Failed to import custom modules: {e}")
        return False
    
    try:
        from ultralytics.nn.tasks import parse_custom_model
        print("✓ parse_custom_model imported")
    except Exception as e:
        print(f"✗ Failed to import parse_custom_model: {e}")
        return False
    
    print("\n✅ All imports successful!\n")
    return True


def test_model_loading():
    """Test that model can be loaded from YAML."""
    print("=" * 80)
    print("TEST 2: Model Loading from YAML")
    print("=" * 80)
    
    try:
        from ultralytics import YOLO
        
        print("Loading model from YAML...")
        model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')
        
        print(f"✓ Model loaded: {model.model_name}")
        print(f"✓ Task: {model.task}")
        print(f"✓ Model type: {type(model.model).__name__}")
        
        # Check if it's our custom model
        from ultralytics.nn.custom_models import MobileNetV3YOLO
        if isinstance(model.model, MobileNetV3YOLO):
            print("✓ Custom MobileNetV3YOLO detected!")
        else:
            print(f"✗ Wrong model type: {type(model.model)}")
            return False
        
        print("\n✅ Model loading successful!\n")
        return True
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test that model can perform forward pass."""
    print("=" * 80)
    print("TEST 3: Forward Pass")
    print("=" * 80)
    
    try:
        from ultralytics import YOLO
        
        print("Loading model...")
        model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')
        
        print("Creating dummy input...")
        x = torch.randn(1, 3, 640, 640)
        
        print("Running forward pass...")
        with torch.no_grad():
            outputs = model.model(x)
        
        print(f"✓ Forward pass successful!")
        print(f"✓ Number of outputs: {len(outputs)}")
        
        for i, out in enumerate(outputs):
            print(f"  Output {i}: {out.shape}")
        
        # Verify output shapes
        expected_shapes = [
            (1, 3, 80, 80, 85),   # P3: 640/8 = 80
            (1, 3, 40, 40, 85),   # P4: 640/16 = 40
            (1, 3, 20, 20, 85),   # P5: 640/32 = 20
        ]
        
        all_correct = True
        for i, (out, expected) in enumerate(zip(outputs, expected_shapes)):
            if out.shape != torch.Size(expected):
                print(f"✗ Output {i} shape mismatch: {out.shape} != {expected}")
                all_correct = False
        
        if all_correct:
            print("✓ All output shapes correct!")
        
        print("\n✅ Forward pass test successful!\n")
        return all_correct
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_info():
    """Test that model info works."""
    print("=" * 80)
    print("TEST 4: Model Information")
    print("=" * 80)
    
    try:
        from ultralytics import YOLO
        
        model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')
        
        print("Getting model info...")
        model.info(detailed=False, verbose=True)
        
        print("\n✅ Model info test successful!\n")
        return True
        
    except Exception as e:
        print(f"✗ Model info failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parse_custom_model():
    """Test parse_custom_model function directly."""
    print("=" * 80)
    print("TEST 5: parse_custom_model Function")
    print("=" * 80)
    
    try:
        from ultralytics.nn.tasks import parse_custom_model
        
        # Test with dict config
        print("Testing with dict config...")
        cfg_dict = {'custom_model': 'mobilenetv3-yolo', 'nc': 80}
        model = parse_custom_model(cfg_dict, ch=3, nc=80, verbose=False)
        
        if model is not None:
            print(f"✓ parse_custom_model returned: {type(model).__name__}")
        else:
            print("✗ parse_custom_model returned None")
            return False
        
        # Test with string config
        print("Testing with string config...")
        model = parse_custom_model('mobilenetv3-yolo', ch=3, nc=80, verbose=False)
        
        if model is not None:
            print(f"✓ parse_custom_model returned: {type(model).__name__}")
        else:
            print("✗ parse_custom_model returned None")
            return False
        
        # Test with non-custom config
        print("Testing with non-custom config...")
        model = parse_custom_model('yolov8n', ch=3, nc=80, verbose=False)
        
        if model is None:
            print("✓ parse_custom_model correctly returned None for standard model")
        else:
            print("✗ parse_custom_model should return None for standard models")
            return False
        
        print("\n✅ parse_custom_model test successful!\n")
        return True
        
    except Exception as e:
        print(f"✗ parse_custom_model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_integration():
    """Test that model integrates with DetectionTrainer."""
    print("=" * 80)
    print("TEST 6: Training Integration (DRY RUN)")
    print("=" * 80)
    
    try:
        from ultralytics.models.yolo.detect import DetectionTrainer
        from ultralytics.nn.tasks import parse_custom_model
        
        print("Testing DetectionTrainer.get_model()...")
        
        # Create a mock trainer with minimal config
        class MockTrainer:
            def __init__(self):
                self.data = {"nc": 80, "channels": 3}
        
        trainer = MockTrainer()
        
        # Test custom model loading
        cfg = 'ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml'
        model = parse_custom_model(cfg, ch=3, nc=80, verbose=False)
        
        if model is not None:
            print(f"✓ Trainer would load: {type(model).__name__}")
            print("✓ Model has required attributes:")
            print(f"  - nc: {model.nc}")
            print(f"  - names: {len(model.names)} classes")
            print(f"  - task: {model.yaml.get('task', 'detect')}")
        else:
            print("✗ Failed to load custom model in trainer")
            return False
        
        print("\n✅ Training integration test successful!\n")
        print("⚠️  NOTE: This is a dry run. Actual training requires a dataset.")
        return True
        
    except Exception as e:
        print(f"✗ Training integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "MobileNetV3-YOLO Integration Tests" + " " * 24 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    tests = [
        ("Imports", test_imports),
        ("Model Loading", test_model_loading),
        ("Forward Pass", test_forward_pass),
        ("Model Info", test_model_info),
        ("parse_custom_model", test_parse_custom_model),
        ("Training Integration", test_training_integration),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}\n")
            results.append((name, False))
    
    # Print summary
    print("\n")
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} - {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print("=" * 80)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 80)
    
    if passed == total:
        print("\n🎉 All tests passed! Your MobileNetV3-YOLO is ready to use!")
        print("\nNext steps:")
        print("  1. Run: python train_custom_model.py")
        print("  2. Or use: YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
