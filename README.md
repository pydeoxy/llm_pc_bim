# llm_pc_bim

## Chatbot for AEC Documentation

### Features
- IFC model analysis
- Point cloud processing
- Document retrieval
- Expandable architecture

### Quick Start
```bash
pip install -r requirements.txt
python app.py
  ```

## **File Structure**

To ensure modularity, scalability, and maintainability, the project is organized as follows:
```
llm_pc_bim/
│   
├── chatcore/                   # Core functionality package
│   ├── __init__.py
│   ├── tools/                  # Modular tool implementations
│   │   ├── __init__.py
│   │   ├── ifc_tools.py        # IFC-related functions
│   │   ├── pc_tools.py         # Point Cloud-related functions
│   │   └── doc_processing.py
│   ├── pipelines/              # Haystack pipeline configurations
│   │   ├── __init__.py
│   │   ├── main_pipeline.py
│   │   └── doc_pipeline.py
│   ├── interfaces/             # Gradio UI components
│   │   ├── __init__.py
│   │   └── gradio_interface.py
│   └── utils/                  # Helper functions and utilities
│       ├── __init__.py
│       ├── file_handling.py
│       └── config_loader.py
├── examples/               # Example usage scripts
├── tests/                  # Unit/integration tests
├── docs/                   # Documentation
├── config/                 # Configuration files
│   └── settings.yaml
├── app.py                  # Main application entry point
├── requirements.txt
├── README.md
└── .gitignore
  ```
