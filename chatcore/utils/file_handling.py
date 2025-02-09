# In chatcore/utils/file_handling.py
FILE_HANDLERS = {
    '.ifc': handle_ifc,
    '.ply': handle_pointcloud,
    '.xyz': handle_pointcloud,  # New format
    '.pdf': handle_document
}