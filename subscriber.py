import paho.mqtt.client as mqtt
import json
import base64
import cv2
import numpy as np

# MQTTブローカーの設定（Windowsのローカルホストを経由してコンテナに接続）
BROKER_ADDRESS = "localhost"
PORT = 1883
TOPIC = "smart_agri/result"

def on_connect(client, userdata, flags, rc):
    print(f"MQTTブローカーに接続しました！ (コード: {rc})")
    print(f"トピック '{TOPIC}' をサブスクライブ（購読）開始します...")
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    try:
        # 受け取ったメッセージ（JSON）を辞書に変換
        payload = json.loads(msg.payload.decode('utf-8'))
        
        filename = payload.get("filename", "unknown")
        pattern_name = payload.get("pattern_name", "unknown")
        data = payload.get("data", {})
        
        print(f"\n--- メッセージを受信しました: {filename} ---")
        
        mode = data.get("mode", "unknown")
        val = data.get("result_value", 0)
        print(f"[{pattern_name}] モード: {mode}, 結果: {val}")
        
        # Base64画像が含まれていれば復元して表示
        image_b64 = data.get("image_b64", "")
        if image_b64:
            img_data = base64.b64decode(image_b64)
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if img is not None:
                # ウィンドウ名を受信側専用にする
                window_name = f"Received: {filename} - {pattern_name}"
                cv2.imshow(window_name, img)
        
        # 画面を更新（1ミリ秒待つ）
        cv2.waitKey(1)

    except Exception as e:
        print(f"メッセージ処理エラー: {e}")

# クライアントの作成と設定
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print("受信プログラムを起動します。終了するには Ctrl + C を押してください。")
try:
    # 接続して受信ループを永遠に回す
    client.connect(BROKER_ADDRESS, PORT, 60)
    client.loop_forever()
except KeyboardInterrupt:
    print("受信プログラムを終了します。")
    cv2.destroyAllWindows()
