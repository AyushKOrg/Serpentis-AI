# SERPENTIS AI 🐍

Local-first multimodal AI prototype. **No external AI inference APIs.**

## Features
- Local text intelligence using Qwen2.5-7B-Instruct (4-bit on supported NVIDIA GPU)
- Local image understanding using SmolVLM-500M-Instruct, loaded on demand
- Optional Tesseract OCR
- Basic arithmetic tool with AST allowlisting (no Python eval)
- Gradio web interface: Chat, Vision/OCR, Agent

> **Weights vs code:** application/orchestration/tool code is in this repository. Model weights are pretrained open-weight downloads, not trained from scratch.

## Run
Python 3.10+ recommended; NVIDIA GPU recommended.
1. Install: pip install -r requirements.txt
2. Launch: python app.py
3. Open the local Gradio URL.

First run downloads weights and can take time and disk space. A T4 is expected to support 7B 4-bit text inference; vision is loaded separately and uses additional memory.

## Optional temporary public link
Set SERPENTIS_SHARE=1 before starting. The Gradio link is temporary and the host/session must remain active. Do not use it for sensitive uploads.

## Environment variables
SERPENTIS_INTELLIGENCE_MODEL (default Qwen/Qwen2.5-7B-Instruct)
SERPENTIS_VISION_MODEL (default HuggingFaceTB/SmolVLM-500M-Instruct)
SERPENTIS_4BIT=1 (default)
SERPENTIS_MAX_TOKENS=512
SERPENTIS_HOST=127.0.0.1
SERPENTIS_PORT=7860
SERPENTIS_SHARE=0 (default)

## Safety boundaries
Agent tools are deliberately restricted. No arbitrary shell, remote access, credential handling, stealth, or unrestricted device-control capability. Cybersecurity features should support defensive learning, authorized testing, and lab use only.
