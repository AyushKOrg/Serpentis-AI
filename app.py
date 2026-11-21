"""SERPENTIS local web demo; no third-party AI inference APIs."""
import os,gradio as gr
from serpentis.intelligence import chat
from serpentis.vision import analyze_image,extract_ocr
from serpentis.agent import run_agent
def vision_ui(image,question):
 if image is None: return "Upload an image first."
 return analyze_image(image,question or "Describe this image.")+"\\n\\n--- OCR (best-effort) ---\\n"+extract_ocr(image)
def agent_ui(task,image): return run_agent(task,image)
with gr.Blocks(title="SERPENTIS AI",theme=gr.themes.Soft(primary_hue="emerald",neutral_hue="slate")) as demo:
 gr.Markdown("# 🐍 SERPENTIS AI\\n**Local inference • No AI API calls • Models run on this machine/server**")
 gr.Markdown("Prototype: image answers and OCR can be wrong. Avoid uploading sensitive information to public demos.")
 with gr.Tab("Chat"):
  gr.ChatInterface(fn=chat,chatbot=gr.Chatbot(type="messages"),textbox=gr.Textbox(placeholder="Ask Serpentis…",container=False),examples=["Explain transformers simply","Write a Python function to sort a list","Solve: 12 / 3 * 5"],title="Intelligence")
 with gr.Tab("Vision + OCR"):
  im=gr.Image(type="pil",label="Image upload"); q=gr.Textbox(label="Question",placeholder="Describe the image or read its text"); btn=gr.Button("Analyze",variant="primary"); out=gr.Textbox(lines=10,label="Analysis"); btn.click(vision_ui,[im,q],out)
 with gr.Tab("Agent"):
  task=gr.Textbox(label="Task",placeholder="Calculate 15% of 240, or ask a question about an uploaded image…"); ai=gr.Image(type="pil",label="Optional image"); run=gr.Button("Run",variant="primary"); res=gr.Textbox(lines=10,label="Result"); run.click(agent_ui,[task,ai],res)
 with gr.Tab("Limits"):
  gr.Markdown("### Included\\n- Local Qwen2.5-7B text model (4-bit on compatible NVIDIA GPUs)\\n- Local SmolVLM image model, lazy-loaded\\n- Optional Tesseract OCR\\n- Restricted arithmetic tool\\n\\n**Weights:** pretrained open-weight downloads; code and orchestration live in this repository. This is not scratch-trained weights.\\n\\nNo shell execution, credential access, remote control, stealth, or unrestricted device automation.")
if __name__=="__main__": demo.launch(share=os.getenv("SERPENTIS_SHARE","0")=="1",server_name=os.getenv("SERPENTIS_HOST","127.0.0.1"),server_port=int(os.getenv("SERPENTIS_PORT","7860")))
