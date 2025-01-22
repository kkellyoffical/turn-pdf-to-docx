# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_dynamic_libs

block_cipher = None

# 使用绝对路径
project_dir = 'D:\\20250122'
offline_model_path = os.path.join(project_dir, 'offline_models')

# 检查离线模型
if not os.path.exists(offline_model_path):
    raise FileNotFoundError(f"离线模型不存在: {offline_model_path}")

# 检查必要的模型文件
required_files = [
    'config.json',
    'preprocessor_config.json',
    'pytorch_model.bin',
    'special_tokens_map.json',
    'tokenizer_config.json'
]
for file in required_files:
    if not os.path.exists(os.path.join(offline_model_path, file)):
        raise FileNotFoundError(f"缺少模型文件: {file}")

# 收集所有需要的 DLL
binaries = []
binaries.extend(collect_dynamic_libs('torch'))
binaries.extend(collect_dynamic_libs('onnxruntime'))
binaries.extend(collect_dynamic_libs('cv2'))

# 添加 Visual C++ Runtime
binaries.extend([
    ('C:\\Windows\\System32\\msvcp140.dll', '.'),
    ('C:\\Windows\\System32\\vcruntime140.dll', '.'),
    ('C:\\Windows\\System32\\vcruntime140_1.dll', '.'),
])

a = Analysis(
    ['launcher.py'],
    pathex=[project_dir],
    binaries=binaries,
    datas=[
        # 添加离线模型
        (offline_model_path, 'offline_models'),
        # Tesseract 数据文件
        ('C:\\Program Files\\Tesseract-OCR\\tessdata', 'tessdata'),
    ],
    hiddenimports=[
        'torch',
        'transformers',
        'easyocr',
        'PIL',
        'cv2',
        'numpy',
        'fitz',
        'python-docx',
        'tkinter',
        'tkinter.ttk',
        'torch.jit',
        'pytorch_lightning',
        'torch.utils.data',
        'torch.utils.data._utils',
        'torch.utils.data.datapipes',
        'torch.utils.data.datapipes.iter',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='PDF转Word工具',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
) 