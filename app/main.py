import json
import time
import os
import cv2
import threading
import paho.mqtt.client as mqtt
from image_loader import load_images_from_folder
from image_processor import process_image
from result_sender import ResultSender

CONFIG_PATH = "../config/config.json"
DATA_DIR = "../data"
SHARED_DIR = "../shared_results"
KICK_TOPIC = "smart_agri/kick"

# 処理実行を要求するフラグ
process_trigger = False

def on_connect(client, userdata, flags, rc):
    print(f"MQTTブローカーに接続しました（キック待機用）。コード: {rc}")
    client.subscribe(KICK_TOPIC)
    print(f"トピック '{KICK_TOPIC}' をサブスクライブしました。")

def on_message(client, userdata, msg):
    global process_trigger
    print(f"キック通知を受信しました: {msg.topic}")
    process_trigger = True

def load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"設定ファイルの読み込みエラー: {e}")
        return {}

def process_and_send(pattern_name, img, config, sender, filename):
    mode = config.get("mode", "hsv")
    send_image = config.get("send_image", False)
    
    results = []
    
    # YOLOの場合はマルチモデル対応
    if mode == "yolo":
        yolo_config = config.get("yolo", {})
        model_ids = yolo_config.get("model_ids", [])
        api_key = yolo_config.get("api_key", "")
        
        # モデルが設定されていない場合のフォールバック
        if not model_ids:
            model_ids = [""]
            
        for idx, model_id in enumerate(model_ids):
            if not model_id:
                continue
            
            temp_config = config.copy()
            # yolo設定をコピーしてmodel_idとapi_keyを付与
            temp_yolo_config = yolo_config.copy()
            temp_yolo_config["model_id"] = model_id
            temp_yolo_config["api_key"] = api_key
            temp_config["yolo"] = temp_yolo_config
            
            res = process_image(img.copy(), temp_config)
            sub_pattern_name = f"{pattern_name}_yolo_{idx+1}"
            results.append((sub_pattern_name, res))
    else:
        # 古典的手法など
        res = process_image(img.copy(), config)
        results.append((pattern_name, res))

    for name, res in results:
        res_img = res.get("processed_image")
        result_value = res.get("result_value", 0)
        
        payload = {
            "mode": res.get("mode"),
            "result_value": result_value
        }
        
        if send_image:
            payload["image_b64"] = sender.encode_image(res_img)
            
        # 独立した結果としてMQTT送信
        sender.send_single_result(filename, name, payload)
        
        # 検出結果がある場合のみ共有フォルダに保存
        if result_value > 0:
            os.makedirs(SHARED_DIR, exist_ok=True)
            save_path = os.path.join(SHARED_DIR, f"{name}_{filename}")
            cv2.imwrite(save_path, res_img)
            
        # プレビュー表示
        cv2.imshow(f"Preview - {name}", res_img)

def main():
    global process_trigger
    print("メイン統括プログラム（司令塔）を開始します...")
    broker = os.environ.get("MQTT_BROKER", "mqtt-broker")
    port = int(os.environ.get("MQTT_PORT", 1883))
    
    # キック動作用のMQTTクライアント設定
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(broker, port, 60)
        client.loop_start()  # バックグラウンドで受信ループを開始
    except Exception as e:
        print(f"MQTTブローカーへの接続に失敗しました: {e}")
        
    sender = ResultSender(broker=broker, port=port)
    
    last_modified = 0
    current_config = {}

    while True:
        try:
            # GUIのためにcv2のイベントループを回しておく
            if cv2.waitKey(100) & 0xFF == 27:
                break
                
            if os.path.exists(CONFIG_PATH):
                mtime = os.path.getmtime(CONFIG_PATH)
                if mtime != last_modified:
                    current_config = load_config()
                    last_modified = mtime
                    print(f"設定が更新されました。")
                    
            # フラグが立っていなければ待機
            if not process_trigger:
                time.sleep(0.5)
                continue

            print("処理を開始します...")
            images = load_images_from_folder(DATA_DIR)
            if not images:
                print(f"{DATA_DIR} に画像が見つかりません。")
                process_trigger = False
                continue

            # 2パターンの設定を取得
            p1_config = current_config.get("pattern1", {})
            p2_config = current_config.get("pattern2", {})

            for i, (file_path, img) in enumerate(images):
                filename = os.path.basename(file_path)
                
                # ====== パターン1 と パターン2 の処理 ======
                process_and_send("pattern1", img, p1_config, sender, filename)
                process_and_send("pattern2", img, p2_config, sender, filename)
                
                # 処理が終わった元の画像を削除する
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"画像削除エラー: {e}")
            
            print("処理が完了しました。次のキックを待機します...")
            # 全ての画像を処理したらフラグを戻す
            process_trigger = False
            
        except KeyboardInterrupt:
            print("処理を終了します。")
            break
        except Exception as e:
            print(f"エラー: {e}")
            process_trigger = False
            time.sleep(2)
            
    client.loop_stop()

if __name__ == "__main__":
    main()
