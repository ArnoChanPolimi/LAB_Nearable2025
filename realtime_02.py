# # import sys
# # import cv2
# # import numpy as np
# # import time
# # import threading
# # from scipy.signal import butter, filtfilt, find_peaks
# # from PyQt5 import QtCore, QtGui, QtWidgets

# # # ----------------- 信号处理函数 -----------------
# # def bandpass_filter(signal, fs=30, lowcut=0.7, highcut=4.0):
# #     nyquist = 0.5 * fs
# #     low = lowcut / nyquist
# #     high = highcut / nyquist
# #     try:
# #         b, a = butter(4, [low, high], btype='band')
# #         if len(signal) > 3 * max(len(a), len(b)):
# #             filtered = filtfilt(b, a, signal)
# #             return filtered
# #         else:
# #             return signal
# #     except Exception as e:
# #         print(f"[ERROR] Filter failed: {e}")
# #         return signal

# # def estimate_hr_fft(signal, fs):
# #     n = len(signal)
# #     fft = np.fft.fft(signal - np.mean(signal))
# #     freqs = np.fft.fftfreq(n, d=1/fs)
# #     fft_magnitude = np.abs(fft)
# #     pos_mask = freqs > 0
# #     freqs = freqs[pos_mask]
# #     fft_magnitude = fft_magnitude[pos_mask]
# #     valid = (freqs >= 0.7) & (freqs <= 4.0)
# #     if not np.any(valid):
# #         return 0
# #     peak_freq = freqs[valid][np.argmax(fft_magnitude[valid])]
# #     hr_bpm = peak_freq * 60
# #     return hr_bpm

# # # ----------------- GUI 主窗口 -----------------
# # class RPPGWindow(QtWidgets.QWidget):
# #     def __init__(self):
# #         super().__init__()
# #         self.setWindowTitle("实时 rPPG 心率估计 + 血氧计对比")
# #         self.resize(900, 700)

# #         # 视频显示控件
# #         self.video_label = QtWidgets.QLabel()
# #         self.video_label.setFixedSize(640, 480)

# #         # 心率显示标签
# #         self.hr_label_rppg = QtWidgets.QLabel("rPPG HR: -- bpm")
# #         self.hr_label_spo2 = QtWidgets.QLabel("血氧计: Finger out")

# #         # 算法选择下拉菜单
# #         self.combo = QtWidgets.QComboBox()
# #         self.combo.addItems(["FFT"])
# #         self.combo.currentIndexChanged.connect(self.on_algo_change)

# #         # 布局
# #         vbox = QtWidgets.QVBoxLayout()
# #         vbox.addWidget(self.video_label)

# #         hbox = QtWidgets.QHBoxLayout()
# #         hbox.addWidget(QtWidgets.QLabel("选择心率估计算法:"))
# #         hbox.addWidget(self.combo)
# #         hbox.addStretch()
# #         vbox.addLayout(hbox)

# #         vbox.addWidget(self.hr_label_rppg)
# #         vbox.addWidget(self.hr_label_spo2)

# #         self.setLayout(vbox)

# #         # 摄像头初始化
# #         self.cap = cv2.VideoCapture(0)
# #         if not self.cap.isOpened():
# #             raise RuntimeError("无法打开摄像头")

# #         self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# #         self.signal = []
# #         self.times = []
# #         self.last_hr_update = 0
# #         self.estimation_interval = 1.0  # 秒
# #         self.current_algorithm = "FFT"

# #         # --- 血氧计数据相关变量 ---
# #         self.spo2_hr = 0.0
# #         self.finger_in = False
# #         self.lock = threading.Lock()  # 保护血氧计数据线程安全

# #         # 启动血氧计读取线程
# #         self.spo2_thread = threading.Thread(target=self.read_spo2_data, daemon=True)
# #         self.spo2_thread.start()

# #         # 启动定时器获取视频帧和更新界面
# #         self.timer = QtCore.QTimer()
# #         self.timer.timeout.connect(self.update_frame)
# #         self.timer.start(30)  # ~33fps

# #     def on_algo_change(self, index):
# #         self.current_algorithm = self.combo.currentText()

# #     # ====== 血氧计读取线程函数（修改点1） ======
# #     def read_spo2_data(self):
# #         # TODO: 替换下面伪代码为你实际血氧计读取代码
# #         # 这里模拟手指插入状态和心率数据，演示Finger out逻辑
# #         import random
# #         import time
# #         while True:
# #             try:
# #                 # 模拟90%几率检测到手指，心率60-80
# #                 finger = random.choices([True, False], weights=[9,1])[0]
# #                 hr = 60 + 20 * random.random() if finger else 0

# #                 with self.lock:
# #                     self.spo2_hr = hr
# #                     self.finger_in = finger

# #                 time.sleep(0.5)
# #             except Exception:
# #                 with self.lock:
# #                     self.spo2_hr = 0
# #                     self.finger_in = False
# #                 time.sleep(1)

# #     def update_frame(self):
# #         ret, frame = self.cap.read()
# #         if not ret:
# #             return

# #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #         faces = self.face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(100, 100))

# #         if len(faces) > 0:
# #             x, y, w, h = faces[0]

# #             # 额头ROI
# #             fh_x = x + int(0.25 * w)
# #             fh_y = y + int(0.05 * h)
# #             fh_w = int(0.5 * w)
# #             fh_h = int(0.15 * h)
# #             forehead_roi = frame[fh_y:fh_y+fh_h, fh_x:fh_x+fh_w]

# #             # 脸颊ROI（左脸）
# #             lc_x = x + int(0.2 * w)
# #             lc_y = y + int(0.4 * h)
# #             lc_w = int(0.15 * w)
# #             lc_h = int(0.36 * h)
# #             left_cheek_roi = frame[lc_y:lc_y+lc_h, lc_x:lc_x+lc_w]

# #             # 绿色通道均值信号采集（简单示例，实际可以融合多区域）
# #             green_mean = np.mean([
# #                 np.mean(forehead_roi[:, :, 1]),
# #                 np.mean(left_cheek_roi[:, :, 1])
# #             ])
# #             current_time = time.time()
# #             self.signal.append(green_mean)
# #             self.times.append(current_time)

# #             # 保留40秒数据
# #             cutoff = current_time - 40
# #             while self.times and self.times[0] < cutoff:
# #                 self.times.pop(0)
# #                 self.signal.pop(0)

# #             # 滤波和心率估计，1秒更新一次
# #             if (current_time - self.last_hr_update) > self.estimation_interval and len(self.signal) > 20:
# #                 fs = len(self.signal) / (self.times[-1] - self.times[0])
# #                 filtered = bandpass_filter(np.array(self.signal), fs)

# #                 hr = estimate_hr_fft(filtered, fs)

# #                 # 限制心率波动，防止突变（修改点2）
# #                 if not hasattr(self, 'last_hr'):
# #                     self.last_hr = hr
# #                 else:
# #                     if abs(hr - self.last_hr) > 15:
# #                         hr = (hr + self.last_hr) / 2
# #                     self.last_hr = hr

# #                 self.hr_label_rppg.setText(f"rPPG HR: {hr:.1f} bpm")
# #                 self.last_hr_update = current_time

# #             # 画出ROI框
# #             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
# #             cv2.rectangle(frame, (fh_x, fh_y), (fh_x+fh_w, fh_y+fh_h), (255, 0, 0), 2)
# #             cv2.rectangle(frame, (lc_x, lc_y), (lc_x+lc_w, lc_y+lc_h), (0, 255, 255), 2)

# #         else:
# #             self.hr_label_rppg.setText("rPPG HR: -- bpm")
# #             self.signal.clear()
# #             self.times.clear()

# #         # 显示血氧计心率或Finger out（修改点3）
# #         with self.lock:
# #             if self.finger_in and self.spo2_hr > 30:
# #                 spo2_text = f"血氧计 HR: {self.spo2_hr:.1f} bpm"
# #             else:
# #                 spo2_text = "血氧计: Finger out"
# #         self.hr_label_spo2.setText(spo2_text)

# #         # 显示摄像头画面
# #         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #         h, w, ch = rgb_frame.shape
# #         bytes_per_line = ch * w
# #         qt_img = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
# #         pixmap = QtGui.QPixmap.fromImage(qt_img)
# #         self.video_label.setPixmap(pixmap)

# #     def closeEvent(self, event):
# #         self.cap.release()
# #         event.accept()

# # if __name__ == "__main__":
# #     app = QtWidgets.QApplication(sys.argv)
# #     window = RPPGWindow()
# #     window.show()
# #     sys.exit(app.exec())



# import cv2
# import numpy as np
# from scipy.signal import butter, filtfilt
# import time

# # === 带通滤波函数 ===
# def bandpass_filter(signal, fs, lowcut=0.7, highcut=3.5, order=3):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     return filtfilt(b, a, signal)

# # === FFT频率估计心率 ===
# def estimate_hr_fft(signal, fs, min_hr=40, max_hr=180):
#     n = len(signal)
#     if n < 10:
#         return 0
#     fft = np.fft.rfft(signal - np.mean(signal))
#     freqs = np.fft.rfftfreq(n, d=1/fs)
#     fft = np.abs(fft)
#     mask = (freqs * 60 >= min_hr) & (freqs * 60 <= max_hr)
#     if not np.any(mask):
#         return 0
#     peak_freq = freqs[mask][np.argmax(fft[mask])]
#     return peak_freq * 60

# # === 简单心率平滑（限制跳变） ===
# class HeartRateSmoother:
#     def __init__(self, max_delta=10):
#         self.last_hr = None
#         self.max_delta = max_delta

#     def smooth(self, hr):
#         if hr < 40 or hr > 180:
#             return self.last_hr if self.last_hr else 0
#         if self.last_hr is None:
#             self.last_hr = hr
#             return hr
#         if abs(hr - self.last_hr) > self.max_delta:
#             hr = self.last_hr + np.sign(hr - self.last_hr) * self.max_delta
#         self.last_hr = hr
#         return hr

# def main():
#     # 加载 DNN 人脸检测模型（替换成你的模型路径）
#     model_dir = "models"
#     prototxt = model_dir + "/deploy.prototxt"
#     model = model_dir + "/res10_300x300_ssd_iter_140000.caffemodel"
#     net = cv2.dnn.readNetFromCaffe(prototxt, model)

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("无法打开摄像头")
#         return

#     fps = 20  # 假设采样频率
#     buffer_size = fps * 10  # 10秒缓冲区
#     green_buffer = []

#     hr_smoother = HeartRateSmoother(max_delta=10)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         h, w = frame.shape[:2]

#         # DNN 人脸检测
#         blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0,
#                                      (300,300), (104.0, 177.0, 123.0))
#         net.setInput(blob)
#         detections = net.forward()

#         face_box = None
#         confidence_threshold = 0.5
#         for i in range(detections.shape[2]):
#             conf = detections[0,0,i,2]
#             if conf > confidence_threshold:
#                 box = detections[0,0,i,3:7] * np.array([w,h,w,h])
#                 (x1, y1, x2, y2) = box.astype(int)
#                 face_box = (x1,y1,x2,y2)
#                 break  # 只取第一个检测到的人脸

#         if face_box is not None:
#             x1, y1, x2, y2 = face_box
#             # 画人脸框
#             cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

#             # 提取人脸ROI绿色通道均值作为rPPG信号点
#             face_roi = frame[y1:y2, x1:x2]
#             if face_roi.size > 0:
#                 green_mean = np.mean(face_roi[:,:,1])
#                 green_buffer.append(green_mean)
#                 if len(green_buffer) > buffer_size:
#                     green_buffer.pop(0)

#                 if len(green_buffer) == buffer_size:
#                     # 滤波
#                     filtered = bandpass_filter(np.array(green_buffer), fs=fps)

#                     # 估计心率
#                     hr = estimate_hr_fft(filtered, fs=fps)

#                     # 平滑心率
#                     hr = hr_smoother.smooth(hr)

#                     # 显示心率
#                     cv2.putText(frame, f"HR: {hr:.1f} bpm", (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
#         else:
#             green_buffer.clear()

#         cv2.imshow("rPPG Heart Rate Estimation", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

import sys
import time
import threading
import numpy as np
import cv2
from scipy.signal import butter, filtfilt
from PyQt5 import QtCore, QtGui, QtWidgets

# 假设你有 cms50d 库，下面是示范导入（请确保安装和正确）
# from cms50d import CMS50D

# ==== 带通滤波函数 ====
def bandpass_filter(signal, fs, lowcut=0.7, highcut=3.5, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# ==== FFT 频率估计心率 ====
def estimate_hr_fft(signal, fs, min_hr=40, max_hr=180):
    n = len(signal)
    if n < 10:
        return 0
    fft = np.fft.rfft(signal - np.mean(signal))
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft = np.abs(fft)
    mask = (freqs * 60 >= min_hr) & (freqs * 60 <= max_hr)
    if not np.any(mask):
        return 0
    peak_freq = freqs[mask][np.argmax(fft[mask])]
    return peak_freq * 60

# ==== 简单心率平滑（限制跳变） ====
class HeartRateSmoother:
    def __init__(self, max_delta=10):
        self.last_hr = None
        self.max_delta = max_delta

    def smooth(self, hr):
        if hr < 40 or hr > 180:
            return self.last_hr if self.last_hr else 0
        if self.last_hr is None:
            self.last_hr = hr
            return hr
        if abs(hr - self.last_hr) > self.max_delta:
            hr = self.last_hr + np.sign(hr - self.last_hr) * self.max_delta
        self.last_hr = hr
        return hr

# ==== CMS50D 血氧计串口读取线程 ====
class CMS50DReader(QtCore.QThread):
    new_hr = QtCore.pyqtSignal(float)
    finger_detected = QtCore.pyqtSignal(bool)

    def __init__(self, port='COM9'):
        super().__init__()
        self.port = port
        self.running = False
        self.monitor = None  # 设备对象

    def run(self):
        try:
            # 实例化设备，替换下面的伪代码为你自己的设备初始化代码
            # self.monitor = CMS50D(port=self.port)
            # self.monitor.connect()
            # self.monitor.start_live_acquisition()
            self.running = True
            while self.running:
                # 伪代码，替换为实际采集代码
                # data = self.monitor.get_latest_data()
                # if data and data['pulse_rate'] > 0:
                #     self.new_hr.emit(float(data['pulse_rate']))
                #     self.finger_detected.emit(True)
                # else:
                #     self.finger_detected.emit(False)
                # time.sleep(0.1)

                # 模拟测试，随机生成数据
                import random
                finger_in = random.choices([True, False], weights=[9,1])[0]
                if finger_in:
                    hr_val = 60 + 20 * random.random()
                    self.new_hr.emit(hr_val)
                self.finger_detected.emit(finger_in)
                time.sleep(0.5)
        except Exception as e:
            print(f"CMS50D error: {e}")

    def stop(self):
        self.running = False
        # if self.monitor:
        #     self.monitor.disconnect()

# ==== 主GUI窗口 ====
class RPPGWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("实时 rPPG 心率估计 + 血氧计对比")
        self.resize(900, 700)

        # 视频显示控件
        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(640, 480)

        # 心率显示标签
        self.hr_label_rppg = QtWidgets.QLabel("rPPG HR: -- bpm")
        self.hr_label_spo2 = QtWidgets.QLabel("血氧计: Finger out")

        # 布局
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.video_label)
        vbox.addWidget(self.hr_label_rppg)
        vbox.addWidget(self.hr_label_spo2)
        self.setLayout(vbox)

        # 摄像头初始化
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.signal = []
        self.times = []
        self.last_hr_update = 0
        self.estimation_interval = 1.0  # 秒
        self.current_hr = 0
        self.hr_smoother = HeartRateSmoother(max_delta=10)

        # 血氧计数据变量
        self.spo2_hr = 0.0
        self.finger_in = False

        # 启动血氧计读取线程
        self.spo2_reader = CMS50DReader(port='COM9')  # 修改成你的设备串口
        self.spo2_reader.new_hr.connect(self.update_spo2_hr)
        self.spo2_reader.finger_detected.connect(self.update_finger_status)
        self.spo2_reader.start()

        # 启动定时器获取视频帧和更新界面
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 约33fps

    def update_spo2_hr(self, hr):
        self.spo2_hr = hr
        self.update_labels()

    def update_finger_status(self, status):
        self.finger_in = status
        self.update_labels()

    def update_labels(self):
        if self.finger_in and self.spo2_hr > 30:
            spo2_text = f"血氧计 HR: {self.spo2_hr:.1f} bpm"
        else:
            spo2_text = "血氧计: Finger out"
        self.hr_label_spo2.setText(spo2_text)
        self.hr_label_rppg.setText(f"rPPG HR: {self.current_hr:.1f} bpm" if self.current_hr > 0 else "rPPG HR: -- bpm")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(100, 100))

        if len(faces) > 0:
            x, y, w, h = faces[0]

            # 额头ROI
            fh_x = x + int(0.25 * w)
            fh_y = y + int(0.05 * h)
            fh_w = int(0.5 * w)
            fh_h = int(0.15 * h)
            forehead_roi = frame[fh_y:fh_y+fh_h, fh_x:fh_x+fh_w]

            # 脸颊ROI（左脸）
            lc_x = x + int(0.2 * w)
            lc_y = y + int(0.4 * h)
            lc_w = int(0.15 * w)
            lc_h = int(0.36 * h)
            left_cheek_roi = frame[lc_y:lc_y+lc_h, lc_x:lc_x+lc_w]

            # 绿色通道均值信号采集
            green_mean = np.mean([
                np.mean(forehead_roi[:, :, 1]),
                np.mean(left_cheek_roi[:, :, 1])
            ])
            current_time = time.time()
            self.signal.append(green_mean)
            self.times.append(current_time)

            # 保留40秒数据
            cutoff = current_time - 40
            while self.times and self.times[0] < cutoff:
                self.times.pop(0)
                self.signal.pop(0)

            # 滤波和心率估计，1秒更新一次
            if (current_time - self.last_hr_update) > self.estimation_interval and len(self.signal) > 20:
                fs = len(self.signal) / (self.times[-1] - self.times[0])
                filtered = bandpass_filter(np.array(self.signal), fs)

                hr = estimate_hr_fft(filtered, fs)

                # 平滑心率，限制跳变
                hr = self.hr_smoother.smooth(hr)
                self.current_hr = hr
                self.last_hr_update = current_time
                self.update_labels()

            # 画ROI框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(frame, (fh_x, fh_y), (fh_x+fh_w, fh_y+fh_h), (255, 0, 0), 2)
            cv2.rectangle(frame, (lc_x, lc_y), (lc_x+lc_w, lc_y+lc_h), (0, 255, 255), 2)
        else:
            self.signal.clear()
            self.times.clear()
            self.current_hr = 0
            self.update_labels()

        # 显示摄像头画面
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_img = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.spo2_reader.stop()
        self.spo2_reader.wait()
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = RPPGWindow()
    window.show()
    sys.exit(app.exec())
