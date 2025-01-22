import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 导入主程序
from pdf_converter_gui import *

if __name__ == "__main__":
    root = tk.Tk()
    app = PDFConverterGUI(root)
    root.mainloop() 