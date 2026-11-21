"""Narrow tool-assisted task runner."""
import json
from .intelligence import chat
from .tools import calculate
from .vision import analyze_image,extract_ocr
def run_agent(task,image=None):
 task=str(task or "").strip()
 if not task: return "Enter a task or question."
 if image is not None:
  visual=analyze_image(image,task); ocr=extract_ocr(image); context=f"Image observations (may be wrong):\\n{visual}\\nOCR (may contain errors):\\n{ocr}"
  return chat(f"User request: {task}\\n\\n{context}\\nAnswer carefully and flag uncertain observations.")
 plan=chat('For this task, decide if basic arithmetic tool is useful. If yes output JSON only: {"tool":"calculator","expression":"arithmetic"}. Otherwise output {"tool":"none"}. Task: '+task,max_new_tokens=100)
 try:
  a,b=plan.find("{"),plan.rfind("}"); data=json.loads(plan[a:b+1]) if a>=0 else {"tool":"none"}
 except Exception: data={"tool":"none"}
 if data.get("tool")=="calculator":
  try: val=calculate(data.get("expression","")); return chat(f"Task: {task}\\nCalculator result: {val}\\nExplain result and calculation briefly.")
  except Exception as e: return f"Calculator rejected expression: {e}\\n"+chat(task)
 return chat(task)
