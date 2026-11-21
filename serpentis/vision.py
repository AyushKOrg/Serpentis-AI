"""Local image analysis and optional OCR."""
from PIL import Image
import torch
from .models import get_vision
def analyze_image(image,question="Describe this image and answer my question."):
 if image is None: return "Please upload an image first."
 if not isinstance(image,Image.Image): image=Image.open(image)
 image=image.convert("RGB"); model,processor=get_vision(); messages=[{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":question}]}]; prompt=processor.apply_chat_template(messages,add_generation_prompt=True); inputs=processor(text=prompt,images=[image],return_tensors="pt")
 try: inputs={k:v.to(next(model.parameters()).device) for k,v in inputs.items()}
 except Exception: pass
 with torch.inference_mode(): output=model.generate(**inputs,max_new_tokens=280,do_sample=False)
 return processor.batch_decode(output,skip_special_tokens=True)[0].strip()
def extract_ocr(image):
 try:
  import pytesseract
  return pytesseract.image_to_string(image).strip() or "No readable text detected."
 except Exception as exc: return f"OCR unavailable ({type(exc).__name__})."
