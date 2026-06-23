import json
import time
import logging
import os
import cv2
import threading
import paho.mqtt.client as mqtt
# from image_loader import load_images_from_folder  # ← 不要になったのでコメントアウト
from image_processor import process_image
from result_sender import ResultSender
from pathlib import Path

CONFIG_PATH = "/app/config/config.json"
DATA_DIR = Path("/app/captured_images")
SHARED_DIR = "/app/shared_results"

KICK_TOPIC = "tele/greenhouse/app_main/start"
STOP_TOPIC = "tele/greenhouse/app_main/stop"

# フラグ管理・設定データのグローバル化
process_trigger = False
stop_trigger = False
requested_filename = "latest.jpg"
current_config = {}  # 毎サイクル最新の設定を参照できるようにグローバル化


def on_connect(client, userdata, flags, rc):
    print(f"MQTTブローカーに接続しました。コード: {rc}")
    client.subscribe(KICK_TOPIC)
    client.subscribe(STOP_TOPIC)
    print(f"トピック '{KICK_TOPIC}', '{STOP_TOPIC}' をサブスクライブしました。")


def on_message(client, userdata, msg):
    global process_trigger, stop_trigger, requested_filename, current_config
    print(f"MQTT通知を受信しました: {msg.topic}")
    
    if msg.topic == KICK_TOPIC:
        payload_text = msg.payload.decode("utf-8", errors="ignore")
        filename = "latest.jpg"
        try:
            payload = json.loads(payload_text)
            filename = payload.get("filename", filename)
        except json.JSONDecodeError:
            if payload_text.strip():
                filename = payload_text.strip()

        requested_filename = filename
        process_trigger = True
        print(f"処理対象ファイルを設定しました: {requested_filename}")
    elif msg.topic == STOP_TOPIC:
        print("停止信号を受信しました！")
        stop_trigger = True


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
    
    if mode == "yolo":
        yolo_config = config.get("yolo", {})
        model_ids = yolo_config.get("model_ids", [])
        api_key = os.environ.get("ROBOFLOW_API_KEY", yolo_config.get("api_key", ""))
        
        if not model_ids:
            model_ids = [""]
            
        for idx, model_id in enumerate(model_ids):
            if not model_id:
                continue
            
            temp_config = config.copy()
            temp_yolo_config = yolo_config.copy()
            temp_yolo_config["model_id"] = model_id
            temp_yolo_config["api_key"] = api_key
            temp_config["yolo"] = temp_yolo_config
            
            # image_processor で画像処理（YOLO推論）を実行
            res = process_image(img.copy(), temp_config)
            sub_pattern_name = f"{pattern_name}_yolo_{idx+1}"
            results.append((sub_pattern_name, res))
    else:
        res = process_image(img.copy(), config)
        results.append((pattern_name, res))

    # ─── 📊 検出サイズ（w, h）の算出とMQTT送信処理 ───
    for name, res in results:
        res_img = res.get("processed_image")
        result_value = res.get("result_value", 0) # 検出数
        
        # 基本のペイロードを作成
        payload = {
            "mode": res.get("mode"),
            "result_value": result_value,
            "detected_objects": []  # 各オブジェクトのサイズを入れる配列
        }
        
        # image_processor側から生の検出リスト（predictions等）が返ってきている場合、サイズを抽出
        # ※お使いのimage_processorの戻り値の構造（辞書のキー）に合わせて調整してください
        predictions = res.get("predictions", [])
        for pred in predictions:
            # 枠のサイズ（ピクセル）を取得
            box_w = pred.get("width", 0)
            box_h = pred.get("height", 0)
            class_name = pred.get("class", "unknown")
            confidence = pred.get("confidence", 0.0)
            
            payload["detected_objects"].append({
                "class": class_name,
                "confidence": confidence,
                "width_px": int(box_w),
                "height_px": int(box_h)
            })
        
        if send_image:
            payload["image_b64"] = sender.encode_image(res_img)
            
        # Node-RED等に向けてMQTT送信
        sender.send_single_result(filename, name, payload)
        
        os.makedirs(SHARED_DIR, exist_ok=True)
        save_path = os.path.join(SHARED_DIR, f"{name}_{filename}")
        cv2.imwrite(save_path, res_img)


def main():
    global process_trigger, stop_trigger, current_config
    print("メイン統括プログラム（司令塔）を開始します...")
    broker = os.environ.get("MQTT_BROKER", "mosquitto")
    port = int(os.environ.get("MQTT_PORT", 1883))
    
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(broker, port, 60)
        client.loop_start()  # バックグラウンドで待ち受け開始
    except Exception as e:
        print(f"MQTTブローカーへの接続に失敗しました: {e}")
        return
        
    sender = ResultSender(client=client)
    
    # 起動時の初期値として config.json をロード
    current_config = load_config()

    print("【待機状態】Node-REDからの信号を待っています...")

    # ターゲットとなる最新画像のフルパスを定義
    LATEST_IMAGE_PATH = None

    while True:
        try:
            # 停止フラグが立っていたら、ループを抜けてコンテナを終了する
            if stop_trigger:
                print("停止フラグを検知したため、プログラムを安全に終了します。")
                break

            # キック信号が来るまで待機
            if not process_trigger:
                time.sleep(0.5)
                continue

            print(f"キック信号を検知しました。処理対象ファイル: {requested_filename} をチェックします...")
            
            LATEST_IMAGE_PATH = os.path.join(DATA_DIR, requested_filename)

            # ─── 🛠️ 修正：指定されたファイルが存在するか直接チェック ───
            if not os.path.exists(LATEST_IMAGE_PATH):
                print(f"❌ エラー: {LATEST_IMAGE_PATH} が見つかりません。指定ファイルが存在するか確認してください。")
                process_trigger = False
                continue

            # OpenCVで指定ファイルを読み込む
            img = cv2.imread(LATEST_IMAGE_PATH)
            if img is None:
                print("❌ エラー: latest.jpg の読み込みに失敗しました（書き込み中によるファイルの破損など）。")
                process_trigger = False
                continue

            # 設定値のパース準備
            p1_config = current_config.get("pattern1", {}).copy()
            if "mode" in p1_config and p1_config["mode"] in p1_config:
                p1_config.update(p1_config.get(p1_config["mode"], {}))

            p2_config = current_config.get("pattern2", {}).copy()
            if "mode" in p2_config and p2_config["mode"] in p2_config:
                p2_config.update(p2_config.get(p2_config["mode"], {}))

            # ─── 🛠️ 修正：ループ処理を撤廃し、要求されたファイルに対して1回だけ処理を実行 ───
            process_and_send("pattern1", img, p1_config, sender, requested_filename)
            # process_and_send("pattern2", img, p2_config, sender, requested_filename)
            
            print("💾 [KEEP] latest.jpg の画像処理とMQTT送信が完了しました。ファイルを保持します。")
            process_trigger = False  
            
        except KeyboardInterrupt:
            print("ユーザーにより終了します。")
            break
        except Exception as e:
            error_msg = f"画像処理中に致命的エラーが発生しました: {str(e)}"
            print(error_msg)
            sender.send_error(filename="latest.jpg", error_message=str(e))
            process_trigger = False
            time.sleep(2)
            
    # ループを抜けたらMQTT接続を綺麗に閉じて終了
    client.loop_stop()
    client.disconnect()
    print("プログラムが正常に終了しました。")


if __name__ == "__main__":
    main()