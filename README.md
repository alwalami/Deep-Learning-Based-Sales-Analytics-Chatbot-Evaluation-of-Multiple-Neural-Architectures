# Deep-Learning-Based-Sales-Analytics-Chatbot-Evaluation-of-Multiple-Neural-Architectures

Offline Task-Oriented Dialogue Systems & Intent Classification
This repository contains a collection of Jupyter Notebooks and Python scripts evaluating different architectures for task-oriented dialogue systems. The projects range from specialized hybrid models (LSTM/GRU) for numerical sales queries to Generative AI implementations (Llama 2, Mistral, Phi-3) running fully offline for conversational ordering.

📂 File Overview
1. Sales Analytics Chatbot (Hybrid Architecture)

File: GRU_Intent_Model.ipynb

Description: A hybrid chatbot designed to answer specific analytical questions about sales data (e.g., "What is the total revenue?", "Best selling product?").

Architecture:

Intent Classification: Compares LSTM, Transformer, and GRU neural networks (built with TensorFlow/Keras) to classify user queries into specific intents.

Response Generation: Uses a deterministic, rule-based backend (Pandas) to query a CSV dataset (Coffee_Chain_Sales .csv) based on the predicted intent.

Key Dependencies: tensorflow, pandas, numpy, sklearn.

2. Comparative LLM OrderBot (Llama 2 vs. Mistral)

File: orderbot_project_(01).ipynb

Description: A comparative study of two 7-billion parameter LLMs acting as a Pizza OrderBot. It evaluates how well different models handle slot-filling (Name, Item, Size, Delivery Address).

Architecture:

Uses llama-cpp-python to load GGUF quantized models.

Models: Llama-2-7B-Chat vs. Mistral-7B-Instruct-v0.2.

Includes automated scenario testing and transcript scoring to benchmark model performance.

Key Dependencies: llama-cpp-python, huggingface_hub, pandas.

3. Optimized Offline OrderBot (Phi-3 Mini)

File: orderbot_offline_decoderonlyipynb.ipynb

Description: A lightweight implementation of the Pizza OrderBot using Microsoft's Small Language Model (SLM). Designed for high efficiency on consumer hardware.

Architecture:

Model: Microsoft Phi-3 Mini (4k-Instruct) in GGUF format (Q4 quantization).

Interface: Features a local graphical user interface (GUI) built with the Panel library for interactive browser-based chatting.

Workflow: Implements a strict "download once, run offline" pipeline.

Key Dependencies: llama-cpp-python, panel, jupyter_bokeh.

4. Source Code Reference

File: Python Code.docx

Description: A document containing raw Python code snippets.

Snippet 1: Source code for the Sales Chatbot (identical logic to GRU_Intent_Model.ipynb).

Snippet 2: A standalone implementation of an OrderBot using Qwen 2.5 (1.5B) and Panel.

🚀 Installation & Setup
To run these notebooks, you will need a Python environment with the following libraries installed:

Bash
# Core Machine Learning & Data
pip install tensorflow pandas numpy scikit-learn

# LLM Inference & Hugging Face
pip install llama-cpp-python huggingface_hub

# Interactive UI
pip install panel jupyter_bokeh
Note on Hardware Acceleration: For llama-cpp-python, installation commands may vary depending on your hardware (CUDA for NVIDIA GPUs, Metal for Apple Silicon) to enable faster inference.

🏃‍♂️ Usage Instructions
Running the Sales Analytics Bot (GRU_Intent_Model.ipynb)

Ensure you have a file named Coffee_Chain_Sales .csv in the root directory (or update the file path in the notebook).

Run the cells to train the LSTM, Transformer, and GRU models.

The notebook concludes with a simulation loop where you can ask questions like "Show me sales for Espresso."

Running the LLM OrderBots

First Run: Ensure you have an internet connection. The notebooks use huggingface_hub to automatically download the required GGUF model weights to a local models/ directory.

Subsequent Runs: The systems will detect the cached models and run fully offline.

Llama/Mistral Comparison: Run orderbot_project_(01).ipynb to see side-by-side transcript scoring.

Interactive UI: Run orderbot_offline_decoderonlyipynb.ipynb to launch the Panel chat interface in your browser.
