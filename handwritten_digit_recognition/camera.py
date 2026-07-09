"""
camera.py - 摄像头实时手写数字识别
多级阈值策略、预处理数字可视化、置信度校准
"""

import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np


class CNNHandwriting(nn.Module):
    def __init__(self):
        super(CNNHandwriting, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CameraRecognizer:
    def __init__(self, model_path='cnn_mnist_model.pth', device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNNHandwriting()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.debug_mode = True  # 默认开启调试，显示预处理数字
        print(f"[Camera] 模型加载成功，使用设备: {self.device}")

    def preprocess_frame(self, frame):
        """
        增强预处理：多级阈值策略，确保数字被正确提取
        返回: (数字PIL, 带框帧, 二值化图像, 预处理后的28x28数字)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # ----- 方法1：自适应阈值（尝试多个参数组合） -----
        best_thresh = None
        best_contour_area = 0
        best_params = None

        # 尝试多种参数组合
        param_combinations = [
            (11, 2), (11, 4), (11, 6),
            (15, 2), (15, 4), (15, 6),
            (19, 2), (19, 4), (19, 6),
        ]

        for block_size, c_val in param_combinations:
            block_size = block_size if block_size % 2 == 1 else block_size + 1
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, block_size, c_val)
            # 形态学去噪
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # 找到最大轮廓
                max_cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(max_cnt)
                if area > best_contour_area and 200 < area < w * h * 0.8:
                    best_contour_area = area
                    best_thresh = thresh
                    best_params = (block_size, c_val)

        # 如果自适应阈值效果不好，尝试Otsu阈值
        if best_thresh is None:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                best_thresh = thresh
            else:
                return None, frame, None, None

        if best_thresh is None:
            return None, frame, None, None

        thresh = best_thresh

        # ----- 轮廓检测与筛选 -----
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, frame, thresh, None

        # 筛选轮廓：面积在合理范围内
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100 or area > w * h * 0.7:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            # 宽高比合理范围
            aspect = cw / ch if ch > 0 else 0
            if aspect < 0.2 or aspect > 5.0:
                continue
            valid_contours.append(cnt)

        if not valid_contours:
            return None, frame, thresh, None

        # 选择面积最大的轮廓
        cnt = max(valid_contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(cnt)

        # ----- 扩展边距并裁切 -----
        margin = max(cw, ch) // 4  # 动态边距
        x = max(0, x - margin)
        y = max(0, y - margin)
        cw = min(w - x, cw + 2 * margin)
        ch = min(h - y, ch + 2 * margin)

        # 裁切数字
        digit = thresh[y:y+ch, x:x+cw]

        # 如果裁切区域太小，跳过
        if digit.shape[0] < 10 or digit.shape[1] < 10:
            return None, frame, thresh, None

        # ----- 将数字缩放到28x28（保持比例，居中） -----
        target_size = 28
        digit_pil = Image.fromarray(digit)
        # 缩放到28x28保持比例
        digit_pil.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        # 创建黑底画布
        canvas = Image.new('L', (target_size, target_size), 0)
        # 居中粘贴
        offset_x = (target_size - digit_pil.width) // 2
        offset_y = (target_size - digit_pil.height) // 2
        canvas.paste(digit_pil, (offset_x, offset_y))
        digit_array = np.array(canvas)
        digit_array = (digit_array > 128).astype(np.uint8) * 255
        digit_pil_final = Image.fromarray(digit_array)

        # ----- 绘制绿色框 -----
        annotated = frame.copy()
        cv2.rectangle(annotated, (x, y), (x + cw, y + ch), (0, 255, 0), 2)

        return digit_pil_final, annotated, thresh, digit_pil_final

    def predict_frame(self, frame, show_debug=False, return_binary=False):
        """
        预测一帧图像
        """
        if frame is None:
            return (None, 0, frame, None, None) if return_binary else (None, 0, frame)

        digit_pil, annotated, thresh, processed = self.preprocess_frame(frame)

        if digit_pil is None:
            cv2.putText(annotated, "No digit found", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if return_binary:
                return None, 0, annotated, thresh, None
            return None, 0, annotated

        # 模型预测
        tensor = self.transform(digit_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).item()
            confidence = torch.max(prob).item() * 100
            # 计算第二高概率（用于置信度校准）
            sorted_probs, _ = torch.sort(prob, dim=1, descending=True)
            second_conf = sorted_probs[0][1].item() * 100
            # 如果最高概率与第二高概率差距小于20%，认为不确定
            if confidence - second_conf < 20:
                confidence = max(confidence * 0.5, 30)  # 降低置信度

        # 在帧上绘制结果
        result_text = f"Pred: {pred}  Conf: {confidence:.1f}%"
        cv2.putText(annotated, result_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # 调试模式：显示预处理后的28x28数字
        if show_debug and processed is not None:
            # 将28x28数字放大显示在单独窗口
            debug_digit = np.array(processed)
            debug_digit = cv2.resize(debug_digit, (200, 200), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Debug - Processed Digit", debug_digit)
            cv2.imshow("Debug - Binary", thresh)
        else:
            try:
                cv2.destroyWindow("Debug - Processed Digit")
                cv2.destroyWindow("Debug - Binary")
            except:
                pass

        if return_binary:
            return pred, confidence, annotated, thresh, processed
        return pred, confidence, annotated


def run_camera(model_path='cnn_mnist_model.pth', camera_id=0):
    recognizer = CameraRecognizer(model_path)
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return

    print("[Camera] 按 'q' 退出，按 'd' 切换调试模式")
    print("[Camera] 调试模式下显示二值化图像和预处理后的28x28数字")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        pred, conf, annotated = recognizer.predict_frame(frame, show_debug=recognizer.debug_mode)
        cv2.imshow("Handwriting Recognition", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            recognizer.debug_mode = not recognizer.debug_mode
            print(f"[Camera] Debug mode: {recognizer.debug_mode}")
            if not recognizer.debug_mode:
                try:
                    cv2.destroyWindow("Debug - Processed Digit")
                    cv2.destroyWindow("Debug - Binary")
                except:
                    pass

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_camera()