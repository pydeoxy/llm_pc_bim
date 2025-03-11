from chatcore.interfaces.gradio_interface import create_interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(inbrowser = True)