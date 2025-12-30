"""Compatibility patch for torchvision.transforms.functional_tensor.

In torchvision >= 0.17, functional_tensor was renamed to _functional_tensor.
This module patches the old import path to work with basicsr.
"""

import sys
import importlib

def patch_torchvision():
    """Patch torchvision to restore functional_tensor module path."""
    try:
        # Check if the old module path already exists
        from torchvision.transforms import functional_tensor
        return  # Already works, no patch needed
    except ImportError:
        pass

    try:
        # Import the new path
        from torchvision.transforms import _functional_tensor

        # Create a module alias for the old path
        sys.modules['torchvision.transforms.functional_tensor'] = _functional_tensor
        print("Patched torchvision.transforms.functional_tensor for basicsr compatibility")
    except ImportError:
        print("Warning: Could not patch torchvision - _functional_tensor not found")


# Apply patch when module is imported
patch_torchvision()
