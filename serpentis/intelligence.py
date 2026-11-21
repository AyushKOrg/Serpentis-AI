"""Local text generation."""
import torch
from .models import get_intelligence
from .config import MAX_NEW_TOKENS
SYSTEM="You are SERPENTIS, a helpful, careful AI assistant. Be clear about uncertainty. Do not claim actions or access you do not have. For cybersecurity, support defensive work, authorized labs, and learning."
def chat(prompt,history=None,max_new_tokens=MAX_NEW_TOKENS):
 model,tokenizer=get_intelligence(); messages=[{"role":"system","content":SYSTEM}]
 for item in (history or [])[-8:]:
  if isinstance(item,dict) and item.get("role") in ("user","assistant") and isinstance(item.get("content"),str): messages.append({"role":item["role"],"content":item["content"]})
 messages.append({"role":"user","content":str(prompt)}); rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True); inputs=tokenizer(rendered,return_tensors="pt")
 try: inputs={k:v.to(next(model.parameters()).device) for k,v in inputs.items()}
 except Exception: pass
 with torch.inference_mode(): output=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=True,temperature=0.7,top_p=0.9,repetition_penalty=1.08,pad_token_id=tokenizer.eos_token_id,eos_token_id=tokenizer.eos_token_id)
 ans=tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:],skip_special_tokens=True).strip(); return ans or "I couldn't generate a response. Please try again."
