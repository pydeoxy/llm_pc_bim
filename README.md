# Chat with documents, IFC and Point cloud

## Project Overview

The **llm_pc_bim** project is a proof-of-concept aimed at developing a program that integrates open-source LLMs with function calling tools capabilities and neural networks for semantic segmentation of point clouds in the local environment. The system combines LLM pipelines, a chatbot-based user interface, function-calling tools for interacting with IFC models and point cloud data, and segmentation scripts to support a streamlined Scan-to-BIM workflow.

### Features
- The program running in fully local environment without share data to external parties.
- Document retrieval with retrieval-augmented generation (RAG) technique.
- Pipelines with function calling tools working with IFC and point clouds.
- Semantic segmentation tools for point cloud processing.
- Expandable architecture for future development.

### Main Python libraries used
- Haystack, for LLM-powered pipelines.
- Gradio, for user interface.
- PyG (PyTorch Geometric), for semantic segmentation.
- IfcOpenShell, for IFC related functions.
- Open3D, for point cloud processing and visualization.
- H5py, for point cloud datasets.

### Quick Start
```bash
pip install -r requirements.txt
python app.py
  ```
