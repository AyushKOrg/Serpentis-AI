"""Lazy local model loading; weights run locally."""
from threading import Lock
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText
from .config import INTELLIGENCE_MODEL, VISION_MODEL, LOAD_4BIT, DEVICE
_intel_lock=Lock(); _vision_lock=Lock(); _intel=_tok=None; _vision=_processor=None
def _device(): return DEVICE if DEVICE != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
def get_intelligence():
 global _intel,_tok
 with _intel_lock:
  if _intel is not None: return _intel,_tok
  dev=_device(); _tok=AutoTokenizer.from_pretrained(INTELLIGENCE_MODEL,use_fast=True)
  kw={"torch_dtype":torch.float16 if dev=="cuda" else torch.float32,"low_cpu_mem_usage":True}
  if dev=="cuda" and LOAD_4BIT:
   try:
    from transformers import BitsAndBytesConfig
    kw["quantization_config"]=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True); kw["device_map"]="auto"
   except Exception as e: print("4-bit unavailable, fallback:",e); kw["device_map"]={"":dev}
  else: kw["device_map"]={"":dev}
  _intel=AutoModelForCausalLM.from_pretrained(INTELLIGENCE_MODEL,**kw); _intel.eval(); return _intel,_tok
def get_vision():
 global _vision,_processor
 with _vision_lock:
  if _vision is not None: return _vision,_processor
  dev=_device(); _processor=AutoProcessor.from_pretrained(VISION_MODEL)
  _vision=AutoModelForImageTextToText.from_pretrained(VISION_MODEL,torch_dtype=torch.float16 if dev=="cuda" else torch.float32,low_cpu_mem_usage=True,device_map="auto" if dev=="cuda" else {"":dev}); _vision.eval(); return _vision,_processor
def unload_vision():
 global _vision,_processor
 with _vision_lock:
  _vision=_processor=None
  if torch.cuda.is_available(): torch.cuda.empty_cache()
