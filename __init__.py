from .nano_banana_v2 import NODE_CLASS_MAPPINGS as V2_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as V2_DISPLAY_MAPPINGS

# Try to import Tencent version (may fail if dependencies not installed)
try:
    from .nano_banana_tencent import NODE_CLASS_MAPPINGS as TENCENT_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as TENCENT_DISPLAY_MAPPINGS
    TENCENT_AVAILABLE = True
except ImportError as e:
    print(f"[Nano Banana] Tencent Cloud version not available: {e}")
    print("[Nano Banana] To use Tencent Cloud version, install: pip install -r requirements_tencent.txt")
    TENCENT_AVAILABLE = False

# Merge node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Add V2 (Gemini) nodes
NODE_CLASS_MAPPINGS.update(V2_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(V2_DISPLAY_MAPPINGS)

# Add Tencent nodes if available
if TENCENT_AVAILABLE:
    NODE_CLASS_MAPPINGS.update(TENCENT_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(TENCENT_DISPLAY_MAPPINGS)
    print("[Nano Banana] Registered: Gemini V2 + Tencent Cloud nodes")
else:
    print("[Nano Banana] Registered: Gemini V2 nodes only")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']