import json
import base64
import cv2
import paho.mqtt.client as mqtt
import numpy as np

class ResultSender:
    def __init__(self, broker: str = "mqtt-broker", port: int = 1883, topic: str = "smart_agri/result"):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client = mqtt.Client()
        
        try:
            self.client.connect(self.broker, self.port, 60)
            print(f"MQTTブローカー ({self.broker}:{self.port}) に接続しました。")
        except Exception as e:
            print(f"MQTTブローカー接続エラー: {e}")
            
    def encode_image(self, image: np.ndarray) -> str:
        """画像をBase64形式にエンコードして返す"""
        if image is None or image.size == 0:
            return ""
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buffer).decode('utf-8')
            
    def send_dual_result(self, filename: str, p1_data: dict, p2_data: dict) -> bool:
        """
        2パターンの処理結果を1つのJSONにまとめて送信する。
        p1_data, p2_data には "mode", "result_value", 必要に応じて "image_b64" を含める。
        """
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
        """
        1つの処理結果をJSONにして送信する。独立した結果として扱う。
        """
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
