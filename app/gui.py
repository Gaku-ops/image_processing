import tkinter as tk
from tkinter import ttk
import json
import os

CONFIG_PATH = "../config/config.json"

class PatternConfigFrame(ttk.Frame):
    def __init__(self, parent, pattern_key, initial_data, save_callback):
        super().__init__(parent, padding=10)
        self.pattern_key = pattern_key
        self.save_callback = save_callback
        
        # Tkinter変数の初期化
        self.mode_var = tk.StringVar(value=initial_data.get("mode", "hsv"))
        self.send_image_var = tk.BooleanVar(value=initial_data.get("send_image", True))
        
        hsv_data = initial_data.get("hsv", {})
        self.h_min_var = tk.IntVar(value=hsv_data.get("h_min", 35))
        self.h_max_var = tk.IntVar(value=hsv_data.get("h_max", 85))
        self.s_min_var = tk.IntVar(value=hsv_data.get("s_min", 100))
        self.s_max_var = tk.IntVar(value=hsv_data.get("s_max", 255))
        self.v_min_var = tk.IntVar(value=hsv_data.get("v_min", 100))
        self.v_max_var = tk.IntVar(value=hsv_data.get("v_max", 255))
        
        self.exg_offset_var = tk.IntVar(value=initial_data.get("exg", {}).get("offset", 0))
        yolo_data = initial_data.get("yolo", {})
        self.yolo_conf_var = tk.DoubleVar(value=yolo_data.get("conf_thresh", 0.5))
        self.yolo_api_key_var = tk.StringVar(value=yolo_data.get("api_key", ""))
        model_ids = yolo_data.get("model_ids", ["yolov8n-640", "", ""])
        # Ensure the list has at least 3 elements
        while len(model_ids) < 3:
            model_ids.append("")
        self.yolo_model1_var = tk.StringVar(value=model_ids[0])
        self.yolo_model2_var = tk.StringVar(value=model_ids[1])
        self.yolo_model3_var = tk.StringVar(value=model_ids[2])
        self.create_widgets()

    def create_widgets(self):
        # 画像送信設定
        opt_frame = ttk.Frame(self)
        opt_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(opt_frame, text="MQTTでこのパターンの処理後画像を送信する", 
                        variable=self.send_image_var, command=self.save_callback).pack(anchor=tk.W)

        # モード選択
        mode_frame = ttk.LabelFrame(self, text="モード選択", padding=10)
        mode_frame.pack(fill=tk.X, pady=5)
        ttk.Radiobutton(mode_frame, text="HSV (緑色抽出)", variable=self.mode_var, value="hsv", command=self.save_callback).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="ExG + 大津の二値化", variable=self.mode_var, value="exg", command=self.save_callback).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="YOLO (AI物体検出)", variable=self.mode_var, value="yolo", command=self.save_callback).pack(anchor=tk.W)
        
        # HSV設定
        hsv_frame = ttk.LabelFrame(self, text="HSV設定", padding=10)
        hsv_frame.pack(fill=tk.X, pady=5)
        self.create_slider(hsv_frame, "H Min", self.h_min_var, 0, 179)
        self.create_slider(hsv_frame, "H Max", self.h_max_var, 0, 179)
        self.create_slider(hsv_frame, "S Min", self.s_min_var, 0, 255)
        self.create_slider(hsv_frame, "S Max", self.s_max_var, 0, 255)
        self.create_slider(hsv_frame, "V Min", self.v_min_var, 0, 255)
        self.create_slider(hsv_frame, "V Max", self.v_max_var, 0, 255)

        # ExG設定
        exg_frame = ttk.LabelFrame(self, text="ExG設定", padding=10)
        exg_frame.pack(fill=tk.X, pady=5)
        self.create_slider(exg_frame, "Threshold Offset", self.exg_offset_var, -100, 100)

        # YOLO設定
        yolo_frame = ttk.LabelFrame(self, text="YOLO (Roboflow) 設定", padding=10)
        yolo_frame.pack(fill=tk.X, pady=5)
        self.create_slider(yolo_frame, "Confidence", self.yolo_conf_var, 0.0, 1.0, is_float=True)
        
        # API Key
        self.create_entry(yolo_frame, "Roboflow API Key", self.yolo_api_key_var)
        
        # モデルID入力枠
        self.create_entry(yolo_frame, "Model ID 1", self.yolo_model1_var)
        self.create_entry(yolo_frame, "Model ID 2", self.yolo_model2_var)
        self.create_entry(yolo_frame, "Model ID 3", self.yolo_model3_var)

    def create_slider(self, parent, label_text, variable, from_, to, is_float=False):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=label_text, width=15).pack(side=tk.LEFT)
        scale = ttk.Scale(frame, from_=from_, to=to, variable=variable, orient=tk.HORIZONTAL, command=self.save_callback)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        val_label = ttk.Label(frame, width=5)
        val_label.pack(side=tk.LEFT)
        def update_label(*args):
            val_label.config(text=f"{variable.get():.2f}" if is_float else f"{variable.get()}")
        variable.trace_add("write", update_label)
        update_label()

    def create_entry(self, parent, label_text, variable):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=label_text, width=15).pack(side=tk.LEFT)
        entry = ttk.Entry(frame, textvariable=variable)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        variable.trace_add("write", self.save_callback)

    def get_data(self):
        return {
            "mode": self.mode_var.get(),
            "send_image": self.send_image_var.get(),
            "hsv": {
                "h_min": self.h_min_var.get(), "h_max": self.h_max_var.get(),
                "s_min": self.s_min_var.get(), "s_max": self.s_max_var.get(),
                "v_min": self.v_min_var.get(), "v_max": self.v_max_var.get()
            },
            "exg": { "offset": self.exg_offset_var.get() },
            "yolo": { 
                "conf_thresh": self.yolo_conf_var.get(),
                "api_key": self.yolo_api_key_var.get(),
                "model_ids": [
                    self.yolo_model1_var.get(),
                    self.yolo_model2_var.get(),
                    self.yolo_model3_var.get()
                ]
            }
        }

class ConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("画像処理パラメータ調整GUI (2パターン同時)")
        self.root.geometry("450x700")
        
        self.config = self.load_config()
        
        # Notebook (タブ) の作成
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # パターン1とパターン2のフレーム作成
        self.p1_frame = PatternConfigFrame(self.notebook, "pattern1", self.config.get("pattern1", {}), self.save_config)
        self.p2_frame = PatternConfigFrame(self.notebook, "pattern2", self.config.get("pattern2", {}), self.save_config)
        
        self.notebook.add(self.p1_frame, text="パターン 1")
        self.notebook.add(self.p2_frame, text="パターン 2")

        # 下部にキックボタンを追加
        kick_btn = ttk.Button(self.root, text="🚀 画像処理を実行（キック通知）", command=self.send_kick)
        kick_btn.pack(fill=tk.X, padx=10, pady=10)

    def send_kick(self):
        import paho.mqtt.client as mqtt
        import os
        # GUIがホストで動くかコンテナで動くかによってブローカーの向き先が変わるため、環境変数から取得しつつlocalhostをフォールバックに
        broker = os.environ.get("MQTT_BROKER", "localhost")
        port = int(os.environ.get("MQTT_PORT", 1883))
        try:
            client = mqtt.Client()
            client.connect(broker, port, 60)
            client.publish("tele/greenhouse/app_main/start", "kick_from_gui")
            client.disconnect()
            print("GUIからキック通知を送信しました。")
        except Exception as e:
            print(f"キック通知の送信に失敗しました: {e}")

    def load_config(self) -> dict:
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"ロードエラー: {e}")
        return {}

    def save_config(self, *args):
        new_config = {
            "pattern1": self.p1_frame.get_data(),
            "pattern2": self.p2_frame.get_data()
        }
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(new_config, f, indent=4)
            print("設定を保存しました。")
        except Exception as e:
            print(f"保存エラー: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ConfigGUI(root)
    root.mainloop()
