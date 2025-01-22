import os
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tqdm import tqdm
import shutil

def create_lite_model():
    """创建轻量级模型"""
    try:
        # 使用绝对路径
        model_path = os.path.join('D:', '20250122', 'offline_models')
        print(f"将模型保存到: {model_path}")
        
        # 如果目录已存在，先清理
        if os.path.exists(model_path):
            print("清理现有模型目录...")
            shutil.rmtree(model_path)
        
        # 创建目录
        os.makedirs(model_path, exist_ok=True)
        
        print("开始下载原始模型...")
        # 下载原始模型，使用进度条
        with tqdm(total=2, desc="下载进度") as pbar:
            processor = TrOCRProcessor.from_pretrained(
                'microsoft/trocr-base-handwritten',
                local_files_only=False,
                use_auth_token=None,
                trust_remote_code=True
            )
            pbar.update(1)
            
            model = VisionEncoderDecoderModel.from_pretrained(
                'microsoft/trocr-base-handwritten',
                local_files_only=False,
                use_auth_token=None,
                trust_remote_code=True
            )
            pbar.update(1)
        
        print("\n转换为轻量级模型...")
        # 转换为轻量级模型
        model.config.update({
            "hidden_size": 256,
            "intermediate_size": 1024,
            "num_attention_heads": 4,
            "num_hidden_layers": 4
        })
        
        print("保存轻量级模型...")
        # 保存轻量级模型
        processor.save_pretrained(model_path)
        model.save_pretrained(model_path)
        
        # 验证文件是否存在
        required_files = [
            'config.json',
            'preprocessor_config.json',
            'pytorch_model.bin',
            'special_tokens_map.json',
            'tokenizer_config.json'
        ]
        
        print("\n验证模型文件...")
        missing_files = []
        for file in required_files:
            file_path = os.path.join(model_path, file)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path) / (1024*1024)  # Convert to MB
                print(f"✓ {file}: {size:.2f} MB")
            else:
                missing_files.append(file)
                print(f"✗ {file}: 缺失")
        
        if missing_files:
            raise FileNotFoundError(f"缺少必要的模型文件: {', '.join(missing_files)}")
        
        print("\n验证模型功能...")
        # 验证模型
        test_processor = TrOCRProcessor.from_pretrained(model_path, local_files_only=True)
        test_model = VisionEncoderDecoderModel.from_pretrained(model_path, local_files_only=True)
        
        print("\n轻量级模型创建成功！")
        print(f"模型保存在: {model_path}")
        
        # 计算总大小
        total_size = sum(os.path.getsize(os.path.join(model_path, f)) 
                        for f in os.listdir(model_path) 
                        if os.path.isfile(os.path.join(model_path, f)))
        print(f"模型总大小: {total_size / (1024*1024):.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n模型创建失败: {str(e)}")
        # 清理不完整的下载
        if os.path.exists(model_path):
            try:
                shutil.rmtree(model_path)
                print(f"已清理不完整的下载: {model_path}")
            except Exception as cleanup_error:
                print(f"清理失败: {str(cleanup_error)}")
        return False

if __name__ == "__main__":
    print("PDF转Word工具 - 模型下载器")
    print("-" * 50)
    create_lite_model() 