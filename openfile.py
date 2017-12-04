import tkinter as tk
from tkinter import filedialog


def get_file():
    tk.Tk().withdraw()
    file_types = [('Image files', ('.png', '.jpg'))]
    filename = filedialog.askopenfilename(filetypes=file_types,
                                          title="Choose an Image File")
    return filename
