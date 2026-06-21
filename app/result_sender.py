import json
import base64
import cv2
import paho.mqtt.client as mqtt
import numpy as np

class ResultSender:
    # ✨ 修正：新しく接続を作るのではなく、メインのクライアント(client)を外から受け取るようにします
    def __init__(self, client: mqtt.Client, topic: str = "tale/greenhouse/image_processor/result"):
        self.client = client
        self.topic = topic
        print("ResultSender がメインのMQTT接続を継承しました。")
            
    def encode_image(self, image: np.ndarray) -> str:
        """画像をBase64形式にエンコードして返す"""
        if image is None or image.size == 0:
            return ""
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buffer).decode('utf-8')
            
    def send_dual_result(self, filename: str, p1_data: dict, p2_data: dict) -> bool:
        try:
            payload = {
                "filename": filename,
                "pattern1": p1_data,
                "pattern2": p2_data
            }
            json_payload = json.dumps(payload)
            self.client.publish(self.topic, json_payload)
            print(f"[{filename}] 2パターンの結果を送信しました。")
            return True
        except Exception as e:
            print(f"MQTT送信エラー: {e}")
            return False

    def send_single_result(self, filename: str, pattern_name: str, data: dict) -> bool:
        try:
            payload = {
                "filename": filename,
                "pattern_name": pattern_name,
                "data": data
            }
            json_payload = json.dumps(payload)
            self.client.publish(self.topic, json_payload)
            print(f"[{filename}] {pattern_name} の結果を送信しました。")
            return True
        except Exception as e:
            print(f"MQTT送信エラー: {e}")
            return False

    def send_error(self, filename: str, error_message: str) -> bool:
        try:
            if self.topic.endswith("/result"):
                error_topic = self.topic[:-7] + "/error"
            else:
                error_topic = self.topic.rsplit('/', 1)[0] + "/error"

            payload = {
                "status": "error",
                "filename": filename,
                "message": error_message
            }
            json_payload = json.dumps(payload)
            
            self.client.publish(error_topic, json_payload, qos=1)
            print(f"[{filename}] 🚨 エラー通知をトピック '{error_topic}' に送信しました。")
            return True
        except Exception as e:
            print(f"MQTTエラー送信失敗: {e}")
            return False