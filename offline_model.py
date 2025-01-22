import os
import sys
import shutil
from pathlib import Path
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def get_offline_model_path():
    """获取离线模型路径"""
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, 'offline_models')

def setup_offline_model():
    """设置离线模型"""
    try:
        model_path = get_offline_model_path()
        print(f"使用离线模型: {model_path}")
        
        # 加载处理器和模型
        processor = TrOCRProcessor.from_pretrained(model_path, local_files_only=True)
        model = VisionEncoderDecoderModel.from_pretrained(model_path, local_files_only=True)
        
        return processor, model
    except Exception as e:
        print(f"离线模型加载失败: {str(e)}")
        return None, None 