import os
import sys

def setup_runtime_environment():
    # 获取程序运行路径
    if getattr(sys, 'frozen', False):
        application_path = sys._MEIPASS
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))
    
    # 设置必要的环境变量
    os.environ['TESSDATA_PREFIX'] = os.path.join(application_path, 'tessdata')
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 设置模型路径
    model_path = os.path.join(application_path, 'microsoft', 'trocr-base-handwritten')
    return model_path, application_path 