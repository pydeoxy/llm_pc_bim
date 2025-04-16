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
│   │   └── doc_processing.py   # Documents processing
│   ├── pipelines/              # Haystack pipeline configurations
│   │   ├── __init__.py
│   │   ├── doc_pipeline.py     # RAG Pipeline for retrieving info from documents
│   │   ├── ifc_pipeline.py     # Pipeline incl. function calling with ifc models
│   │   ├── pc_pipeline.py      # Pipeline incl. function calling with point clouds
│   │   └── main_pipeline.py    # Main combined pipeline
│   ├── interfaces/             # Gradio UI components
│   │   ├── __init__.py
│   │   └── gradio_interface.py
│   └── utils/                  # Helper functions and utilities
│       ├── __init__.py
│       ├── config_loader.py    # Loading LLM and file configurations
│       ├── helpers.py          # Helper functions
│       └── prompts.py          # Prompts and prompt templates
├── pc_seg/                     # Scripts for semantic segmentation of point clouds
│   ├── checkpoints/            # ML checkpoints for semantic segmentation
│   ├── pc_dataset.py           # HDF5 dataset for point cloud
│   ├── pc_label_map.py         # Color and label map of points
│   ├── pyg_pointnet2.py        # PointNet++ model with pytorch_geometric
│   ├── ifc_sim_pc.ipynb        # Simulated point cloud from IFC model
│   ├── pc_to_h5.ipynb          # Convert point cloud to HDF5 dataset file
│   ├── s3dis_train.ipynb       # Training PointNet++ model from S3DIS dataset
│   ├── pc_seg_predict.ipynb    # Prediction of labels with pre-trained model
│   ├── sim_pc_train.ipynb      # Training PointNet++ model from simulated point cloud
│   ├── fine_tuning_train.ipynb # Fine-tuning training with simulated point cloud
│   ├── lora_train.ipynb        # LoRa training with simulated point cloud
│   └── lora_predict.ipynb      # Prediction of labels with pre-trained model and LoRa weights
├── tests/                  # Tests and temporary scripts
├── docs/                   # Documentation and model files
├── config/                 # Configuration files
│   ├── config.json         # Exchange file and folder locations for LLM pipelines
│   └── settings.yaml       # LLM settings
├── app.py                  # Main application entry point
├── requirements.txt        # Dependencies
├── README.md       
└── .gitignore
  ```
