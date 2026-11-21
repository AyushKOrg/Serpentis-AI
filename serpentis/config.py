"""Configuration; no external inference services."""
import os
INTELLIGENCE_MODEL = os.getenv("SERPENTIS_INTELLIGENCE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
VISION_MODEL = os.getenv("SERPENTIS_VISION_MODEL", "HuggingFaceTB/SmolVLM-500M-Instruct")
LOAD_4BIT = os.getenv("SERPENTIS_4BIT", "1").lower() not in {"0", "false", "no"}
MAX_NEW_TOKENS = int(os.getenv("SERPENTIS_MAX_TOKENS", "512"))
DEVICE = os.getenv("SERPENTIS_DEVICE", "auto")
