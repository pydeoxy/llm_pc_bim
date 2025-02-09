
# tests/test_ifc_tools.py
def test_ifc_info_retrieval():
    tool = IFCTool()
    result = tool.get_info("sample.ifc")
    assert "entities" in result["answers"][0]