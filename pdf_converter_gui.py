import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pdf_converter import PDFConverter
import os
from runtime_config import setup_runtime_environment
from ttkbootstrap import Style  # 导入ttkbootstrap

model_path, app_path = setup_runtime_environment()

class PDFConverterGUI:
    def __init__(self, root):
        self.root = root
        self.style = Style()  # 初始化ttkbootstrap样式
        self.style.theme_use('flatly')  # 选择主题
        self.root.title("PDF转文档工具")
        self.root.geometry("600x400")
        self.converter = PDFConverter()
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 输入文件选择
        self.input_frame = ttk.LabelFrame(self.main_frame, text="选择PDF文件", padding="5")
        self.input_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.input_path = tk.StringVar()
        self.input_entry = ttk.Entry(self.input_frame, textvariable=self.input_path, width=50)
        self.input_entry.grid(row=0, column=0, padx=5)
        
        self.input_button = ttk.Button(self.input_frame, text="浏览", command=self.select_input_file)
        self.input_button.grid(row=0, column=1, padx=5)
        
        # 输出文件选择
        self.output_frame = ttk.LabelFrame(self.main_frame, text="保存位置", padding="5")
        self.output_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.output_path = tk.StringVar()
        self.output_entry = ttk.Entry(self.output_frame, textvariable=self.output_path, width=50)
        self.output_entry.grid(row=0, column=0, padx=5)
        
        self.output_button = ttk.Button(self.output_frame, text="浏览", command=self.select_output_file)
        self.output_button.grid(row=0, column=1, padx=5)
        
        # 转换按钮
        self.convert_button = ttk.Button(self.main_frame, text="开始转换", command=self.convert_file)
        self.convert_button.grid(row=2, column=0, columnspan=2, pady=20)
        
        # 进度条
        self.progress = ttk.Progressbar(self.main_frame, length=400, mode='determinate')
        self.progress.grid(row=3, column=0, columnspan=2, pady=5)
        
        # 状态标签
        self.status_var = tk.StringVar(value="准备就绪")
        self.status_label = ttk.Label(self.main_frame, textvariable=self.status_var)
        self.status_label.grid(row=4, column=0, columnspan=2, pady=5)

        # 设置窗口的样式
        self.style.configure('TButton', padding=6, relief='flat', background='#007BFF', foreground='white')
        self.style.map('TButton', background=[('active', '#0056b3')])
        self.style.configure('TLabel', font=('微软雅黑', 12))
        self.style.configure('TEntry', font=('微软雅黑', 12))

    def select_input_file(self):
        filename = filedialog.askopenfilename(
            title="选择PDF文件",
            filetypes=[("PDF文件", "*.pdf")]
        )
        if filename:
            print(f"选择的文件: {filename}")  # 打印选择的文件路径
            self.input_path.set(filename)
            # 自动设置输出文件名
            output_filename = os.path.splitext(filename)[0] + ".docx"
            self.output_path.set(output_filename)

    def select_output_file(self):
        filename = filedialog.asksaveasfilename(
            title="保存文件",
            filetypes=[("Word文档", "*.docx")],
            defaultextension=".docx"
        )
        if filename:
            self.output_path.set(filename)

    def convert_file(self):
        input_file = self.input_path.get()
        output_file = self.output_path.get()
        
        if not input_file or not output_file:
            messagebox.showerror("错误", "请选择输入和输出文件！")
            return
        
        self.progress['value'] = 0
        self.status_var.set("正在转换...")
        self.convert_button['state'] = 'disabled'
        self.root.update()
        
        try:
            print(f"输入文件: {input_file}")  # 打印输入文件路径
            print(f"输出文件: {output_file}")  # 打印输出文件路径
            success = self.converter.convert_pdf_to_docx(input_file, output_file)
            
            if success:
                self.status_var.set("转换完成！")
                messagebox.showinfo("成功", "文件转换成功！")
            else:
                self.status_var.set("转换失败！")
                messagebox.showerror("错误", "文件转换失败！")
        except Exception as e:
            self.status_var.set("转换错误！")
            messagebox.showerror("错误", f"转换过程中出现错误：{str(e)}")
        finally:
            self.convert_button['state'] = 'normal'

if __name__ == "__main__":
    root = tk.Tk()
    app = PDFConverterGUI(root)
    root.mainloop()