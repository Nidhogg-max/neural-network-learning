"""
gui.py - 图形化主界面，整合训练、预测、对比实验、摄像头识别
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import threading
import sys
import os
import time
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('TkAgg')

import camera

class MainGUI:
    def __init__(self, root):
        self.root = root
        root.title("手写数字识别系统")
        root.geometry("1100x750")
        root.resizable(True, True)

        self.current_process = None
        self.process_lock = threading.Lock()

        # ----- 笔记本 -----
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ---------- 控制台 ----------
        self.tab_main = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_main, text="控制台")

        title = tk.Label(self.tab_main, text="基于卷积神经网络的手写数字识别", font=("微软雅黑", 16, "bold"))
        title.pack(pady=10)

        btn_frame = tk.Frame(self.tab_main)
        btn_frame.pack(pady=10)

        self.train_btn = tk.Button(btn_frame, text="1. 训练模型", font=("微软雅黑", 12),
                                   width=15, height=2, command=self.run_train)
        self.train_btn.grid(row=0, column=0, padx=15, pady=5)

        self.predict_btn = tk.Button(btn_frame, text="2. 识别图片", font=("微软雅黑", 12),
                                     width=15, height=2, command=self.run_predict)
        self.predict_btn.grid(row=0, column=1, padx=15, pady=5)

        self.compare_btn = tk.Button(btn_frame, text="3. 对比实验", font=("微软雅黑", 12),
                                     width=15, height=2, command=self.run_compare)
        self.compare_btn.grid(row=0, column=2, padx=15, pady=5)

        self.stop_btn = tk.Button(btn_frame, text="终止任务", font=("微软雅黑", 12),
                                  width=15, height=2, command=self.stop_task, state=tk.DISABLED, bg="#f0ad4e")
        self.stop_btn.grid(row=0, column=3, padx=15, pady=5)

        # 日志
        log_frame = tk.LabelFrame(self.tab_main, text="运行日志", font=("微软雅黑", 10))
        log_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=("Consolas", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.status = tk.Label(self.tab_main, text="就绪", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # ---------- 结果展示 ----------
        self.tab_result = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_result, text="结果展示")

        result_canvas_frame = tk.Frame(self.tab_result)
        result_canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.result_canvas = tk.Canvas(result_canvas_frame)
        scrollbar = tk.Scrollbar(result_canvas_frame, orient=tk.VERTICAL, command=self.result_canvas.yview)
        self.result_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.result_frame = tk.Frame(self.result_canvas)
        self.result_canvas.create_window((0, 0), window=self.result_frame, anchor='nw')
        self.result_frame.bind("<Configure>", lambda e: self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all")))

        # ---------- 摄像头识别（双画面） ----------
        self.tab_camera = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_camera, text="摄像头识别")

        # 顶部的两个画面并排显示
        cam_display_frame = tk.Frame(self.tab_camera)
        cam_display_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # 原始画面
        left_frame = tk.Frame(cam_display_frame)
        left_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        tk.Label(left_frame, text="原始画面", font=("微软雅黑", 10)).pack()
        self.camera_label = tk.Label(left_frame, text="摄像头未开启", bg='black', fg='white')
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        # 二值化画面
        right_frame = tk.Frame(cam_display_frame)
        right_frame.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH, expand=True)
        tk.Label(right_frame, text="二值化预处理", font=("微软雅黑", 10)).pack()
        self.binary_label = tk.Label(right_frame, text="等待画面...", bg='black', fg='white')
        self.binary_label.pack(fill=tk.BOTH, expand=True)

        # 控制按钮和结果
        cam_btn_frame = tk.Frame(self.tab_camera)
        cam_btn_frame.pack(pady=5)

        self.cam_start_btn = tk.Button(cam_btn_frame, text="开启摄像头", command=self.start_camera)
        self.cam_start_btn.grid(row=0, column=0, padx=5)

        self.cam_stop_btn = tk.Button(cam_btn_frame, text="关闭摄像头", command=self.stop_camera, state=tk.DISABLED)
        self.cam_stop_btn.grid(row=0, column=1, padx=5)

        self.cam_result_label = tk.Label(self.tab_camera, text="识别结果：", font=("微软雅黑", 14))
        self.cam_result_label.pack(pady=5)

        # 摄像头相关变量
        self.cap = None
        self.camera_running = False
        self.camera_thread = None
        self.camera_recognizer = None

        # 重定向输出
        sys.stdout = self.TextRedirector(self.log_text)
        sys.stderr = self.TextRedirector(self.log_text)

    # ------------------- 输出重定向 -------------------
    class TextRedirector:
        def __init__(self, widget):
            self.widget = widget
        def write(self, text):
            self.widget.insert(tk.END, text)
            self.widget.see(tk.END)
            self.widget.update_idletasks()
        def flush(self):
            pass

    # ------------------- 终止任务（快速） -------------------
    def stop_task(self):
        with self.process_lock:
            if self.current_process is not None:
                try:
                    self.current_process.terminate()
                    try:
                        self.current_process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        self.current_process.kill()
                        self.current_process.wait()
                    self.log_text.insert(tk.END, "\n[系统] 任务已终止\n")
                    self.status.config(text="任务已终止")
                except Exception as e:
                    self.log_text.insert(tk.END, f"\n[错误] 终止任务失败: {e}\n")
                finally:
                    self.current_process = None
                    self.stop_btn.config(state=tk.DISABLED)
                    self.train_btn.config(state=tk.NORMAL)
                    self.compare_btn.config(state=tk.NORMAL)

    # ------------------- 训练 -------------------
    def run_train(self):
        if not os.path.exists("train.py"):
            messagebox.showerror("错误", "train.py 不存在")
            return
        self.status.config(text="正在训练模型...")
        self.log_text.insert(tk.END, "\n========== 开始训练 ==========\n")
        self.train_btn.config(state=tk.DISABLED)
        self.compare_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        threading.Thread(target=self._run_script, args=("train.py",), daemon=True).start()

    # ------------------- 预测 -------------------
    def run_predict(self):
        file_path = filedialog.askopenfilename(title="选择手写数字图片",
                                               filetypes=[("图片文件", "*.png *.jpg *.bmp")])
        if file_path:
            self.status.config(text=f"正在识别 {os.path.basename(file_path)}")
            self.log_text.insert(tk.END, f"\n========== 识别图片: {file_path} ==========\n")
            threading.Thread(target=self._run_script, args=("predict.py", file_path), daemon=True).start()
        else:
            self.log_text.insert(tk.END, "未选择文件\n")

    # ------------------- 对比实验（带自定义） -------------------
    def run_compare(self):
        self._show_compare_settings()

    def _show_compare_settings(self):
        settings_win = tk.Toplevel(self.root)
        settings_win.title("对比实验参数设置")
        settings_win.geometry("400x250")
        settings_win.transient(self.root)
        settings_win.grab_set()

        default_epochs = "5"
        default_samples = "1000,5000,10000,30000,60000"

        tk.Label(settings_win, text="每项训练轮次 (Epochs):", font=("微软雅黑", 10)).pack(pady=(15, 5))
        epochs_entry = tk.Entry(settings_win, width=20)
        epochs_entry.insert(0, default_epochs)
        epochs_entry.pack()

        tk.Label(settings_win, text="样本数列表 (用逗号分隔):", font=("微软雅黑", 10)).pack(pady=(10, 5))
        samples_entry = tk.Entry(settings_win, width=40)
        samples_entry.insert(0, default_samples)
        samples_entry.pack()

        tk.Label(settings_win, text="示例: 1000,5000,10000,30000,60000", font=("微软雅黑", 8), fg="gray").pack()

        btn_frame = tk.Frame(settings_win)
        btn_frame.pack(pady=20)

        def on_confirm():
            try:
                epochs = int(epochs_entry.get().strip())
                samples_str = samples_entry.get().strip()
                if not samples_str:
                    raise ValueError("样本列表不能为空")
                sample_list = [int(x.strip()) for x in samples_str.split(',') if x.strip()]
                if not sample_list:
                    raise ValueError("样本列表无效")
                settings_win.destroy()
                self._start_compare(epochs, samples_str)
            except ValueError as e:
                messagebox.showerror("输入错误", f"参数无效: {e}")

        def on_cancel():
            settings_win.destroy()

        tk.Button(btn_frame, text="开始实验", command=on_confirm, width=12).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="取消", command=on_cancel, width=8).pack(side=tk.LEFT)

    def _start_compare(self, epochs, samples_str):
        if not os.path.exists("compare.py"):
            messagebox.showerror("错误", "compare.py 不存在")
            return
        self.status.config(text=f"正在运行对比实验 (epochs={epochs})...")
        self.log_text.insert(tk.END, f"\n========== 开始对比实验 (epochs={epochs}, samples={samples_str}) ==========\n")
        self.compare_btn.config(state=tk.DISABLED)
        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        args = ["--epochs", str(epochs), "--samples", samples_str]
        threading.Thread(target=self._run_script, args=("compare.py", *args), daemon=True).start()

    # ------------------- 通用脚本运行 -------------------
    def _run_script(self, script, *args):
        try:
            python_exe = sys.executable
            cmd = [python_exe, "-u", script] + list(args)
            with self.process_lock:
                self.current_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                        text=True, bufsize=1, cwd=os.getcwd())
            process = self.current_process
            while True:
                if self.current_process is None:
                    break
                line = process.stdout.readline()
                if not line:
                    break
                self.log_text.insert(tk.END, line)
                self.log_text.see(tk.END)
                self.root.update_idletasks()
            process.wait()
            with self.process_lock:
                if self.current_process == process:
                    self.current_process = None
            returncode = process.returncode
            self.root.after(0, self._task_finished, script, returncode)
        except Exception as e:
            self.log_text.insert(tk.END, f"运行时异常: {e}\n")
            self.status.config(text="运行出错")
            self.root.after(0, self._reset_buttons)

    def _task_finished(self, script, returncode):
        self._reset_buttons()
        if returncode == 0:
            self.status.config(text="运行完成")
            if script == "train.py":
                self.show_training_results()
            elif script == "compare.py":
                self.show_compare_results()
            messagebox.showinfo("完成", f"{script} 执行成功！")
        else:
            self.status.config(text="运行出错，查看日志")
            messagebox.showerror("错误", f"{script} 执行失败，返回码 {returncode}")

    def _reset_buttons(self):
        self.stop_btn.config(state=tk.DISABLED)
        self.train_btn.config(state=tk.NORMAL)
        self.compare_btn.config(state=tk.NORMAL)

    # ------------------- 显示训练结果 -------------------
    def show_training_results(self):
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        image_files = [
            ("训练损失/准确率曲线", "training_curves.png"),
            ("混淆矩阵", "confusion_matrix.png"),
            ("归一化混淆矩阵", "confusion_matrix_norm.png"),
            ("各类别准确率", "class_accuracy.png"),
            ("错误样本", "error_samples.png"),
            ("学习率变化", "learning_rate.png"),
        ]

        row, col = 0, 0
        for title, fname in image_files:
            if os.path.exists(fname):
                try:
                    img = Image.open(fname)
                    img.thumbnail((300, 300))
                    photo = ImageTk.PhotoImage(img)
                    label = tk.Label(self.result_frame, image=photo, text=title, compound=tk.TOP, font=("微软雅黑", 10))
                    label.image = photo
                    label.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
                    col += 1
                    if col > 2:
                        col = 0
                        row += 1
                except Exception as e:
                    self.log_text.insert(tk.END, f"无法加载图片 {fname}: {e}\n")

        summary_file = "training_summary.txt"
        if os.path.exists(summary_file):
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = f.read()
            text_widget = tk.Text(self.result_frame, height=8, width=40, font=("Consolas", 10))
            text_widget.insert(tk.END, summary)
            text_widget.config(state=tk.DISABLED)
            text_widget.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

        self.notebook.select(self.tab_result)

    # ------------------- 显示对比结果 -------------------
    def show_compare_results(self):
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        fname = "sample_size_comparison.png"
        if os.path.exists(fname):
            img = Image.open(fname)
            img.thumbnail((600, 400))
            photo = ImageTk.PhotoImage(img)
            label = tk.Label(self.result_frame, image=photo, text="不同样本数对比", compound=tk.TOP, font=("微软雅黑", 12))
            label.image = photo
            label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        report_file = "comparison_report.txt"
        if os.path.exists(report_file):
            with open(report_file, 'r', encoding='utf-8') as f:
                report = f.read()
            text_widget = tk.Text(self.result_frame, height=15, width=80, font=("Consolas", 10))
            text_widget.insert(tk.END, report)
            text_widget.config(state=tk.DISABLED)
            text_widget.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.notebook.select(self.tab_result)

    # ------------------- 摄像头（双画面） -------------------
    def start_camera(self):
        if self.camera_running:
            return
        model_path = "cnn_mnist_model.pth"
        if not os.path.exists(model_path):
            messagebox.showerror("错误", "模型文件 cnn_mnist_model.pth 不存在，请先训练模型")
            return

        try:
            import cv2
        except ImportError:
            messagebox.showerror("错误", "未安装 opencv-python")
            return

        # 尝试使用USB摄像头（索引1），若失败则尝试索引0
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0) 
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头，请检查连接或索引")
            return

        try:
            self.camera_recognizer = camera.CameraRecognizer(model_path)
        except Exception as e:
            messagebox.showerror("错误", f"加载模型失败: {e}")
            self.cap.release()
            self.cap = None
            return

        self.camera_running = True
        self.cam_start_btn.config(state=tk.DISABLED)
        self.cam_stop_btn.config(state=tk.NORMAL)
        self.cam_result_label.config(text="识别结果：")

        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()

    def _camera_loop(self):
        import cv2
        from PIL import Image, ImageTk
        while self.camera_running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 新camera.py返回5个值：pred, conf, annotated, binary, processed
            pred, conf, annotated, binary, _ = self.camera_recognizer.predict_frame(frame, return_binary=True)

            # 显示原始画面
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(img.resize((480, 360)))
            self.camera_label.config(image=img_tk)
            self.camera_label.image = img_tk

            # 显示二值化图像（转为 RGB 以便显示）
            if binary is not None:
                binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
                bin_img = Image.fromarray(binary_rgb)
                bin_tk = ImageTk.PhotoImage(bin_img.resize((480, 360)))
                self.binary_label.config(image=bin_tk)
                self.binary_label.image = bin_tk
            else:
                self.binary_label.config(image='', text="未检测到数字")

            # 更新识别结果
            if pred is not None and conf >= 50:   # 置信度阈值
                self.cam_result_label.config(text=f"识别结果：数字 {pred}  置信度 {conf:.1f}%")
            else:
                self.cam_result_label.config(text="未检测到可靠数字")

            time.sleep(0.03)

        self.stop_camera()

    def stop_camera(self):
        self.camera_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.camera_label.config(image='', text="摄像头已关闭")
        self.binary_label.config(image='', text="等待画面...")
        self.cam_result_label.config(text="识别结果：")
        self.cam_start_btn.config(state=tk.NORMAL)
        self.cam_stop_btn.config(state=tk.DISABLED)

    def __del__(self):
        self.stop_camera()
        with self.process_lock:
            if self.current_process is not None:
                try:
                    self.current_process.terminate()
                except:
                    pass

if __name__ == '__main__':
    root = tk.Tk()
    app = MainGUI(root)
    root.mainloop()
