# Custom Model YAML Loading Fix

## Issue
When loading custom models via YAML (`mobilenetv3-yolo.yaml`), the code was failing with:
```
KeyError: 'backbone'
```

## Root Cause
The `DetectionModel.__init__` in `tasks.py` was:
1. Trying to access `self.yaml["backbone"][0][2]` without checking if the key exists or is in the expected format
2. Calling `parse_model(...)` which expects standard YAML with `backbone` and `head` keys
3. Not using the `parse_custom_model()` function that was already implemented

## Solution

### 1. Added Safety Check for YOLOv9 Silence Module (Line 409)
Changed from:
```python
if self.yaml["backbone"][0][2] == "Silence":
```

To:
```python
if "backbone" in self.yaml and isinstance(self.yaml["backbone"], list) and len(self.yaml["backbone"]) > 0:
    if isinstance(self.yaml["backbone"][0], list) and len(self.yaml["backbone"][0]) > 2:
        if self.yaml["backbone"][0][2] == "Silence":
```

This prevents `KeyError` when YAML doesn't have the expected structure.

### 2. Integrated Custom Model Parsing (Line 425-447)
Added check to use `parse_custom_model()` BEFORE calling `parse_model()`:

```python
# Check if this is a custom model
custom_model = parse_custom_model(self.yaml, ch=ch, nc=nc, verbose=verbose)
if custom_model is not None:
    # Use custom model architecture directly
    self.model = custom_model.model  # ModuleList with [backbone, neck, detect_head]
    self.save = []
    self.stride = custom_model.stride
    self.nc = custom_model.nc
    self._custom_model = custom_model
    self._is_custom_model = True
else:
    # Standard YAML-based model
    self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
    self._custom_model = None
    self._is_custom_model = False
```

### 3. Updated Forward Pass (Line 191)
Modified `_predict_once()` to use custom model's forward method:

```python
def _predict_once(self, x, profile=False, visualize=False, embed=None):
    # Use custom model's forward pass if available
    if hasattr(self, '_custom_model') and self._custom_model is not None:
        return self._custom_model(x)
    
    # Standard YAML model forward pass
    # ... (existing code)
```

### 4. Fixed Stride Initialization (Line 453-471)
Prevented stride overwriting for custom models:

```python
# Build strides (skip for custom models as they handle this themselves)
if not self._is_custom_model:
    m = self.model[-1]  # Detect()
    if isinstance(m, Detect):
        # ... initialize stride ...
    else:
        self.stride = torch.Tensor([32])  # default
# else: custom model already has stride set
```

## Files Modified
- `ultralytics/nn/tasks.py` (4 changes)

## Testing
All tests pass:
- ✅ Model loading via YOLO wrapper
- ✅ DetectionModel initialization
- ✅ Forward pass (inference)
- ✅ Training mode
- ✅ Backward pass (gradients)
- ✅ Stride initialization: `[4, 8, 16, 32]` ✓
- ✅ Parameter count: 5.17M ✓

## Usage
Now the custom model works seamlessly:

```python
from ultralytics import YOLO

# Load via YAML (now works!)
model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')

# Train
model.train(
    data="defects-in-timber/data.yaml",
    epochs=150,
    batch=16,
    device=0
)
```

## Result
✅ Custom models now load via YAML without errors  
✅ Compatible with standard YOLO training workflow  
✅ Ready for Kaggle deployment  
