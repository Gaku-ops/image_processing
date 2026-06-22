import os
import cv2
import numpy as np

# ✨【超安全化】ultralytics が無くても、エラーを出さずに通常処理（HSV/ExG）を生かす
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from inference import get_model
    import supervision as sv
except ImportError:
    get_model = None
    sv = None

# モデルのキャッシュ用辞書
local_model_cache = {}
model_cache = {}


def get_roboflow_model(model_id: str, api_key: str):
    if not get_model:
        print("inference または supervision がインストールされていません。")
        return None
    if not model_id or not api_key:
        return None
        
    cache_key = f"{model_id}_{api_key}"
    if cache_key in model_cache:
        return model_cache[cache_key]
        
    try:
        model = get_model(model_id=model_id, api_key=api_key)
        model_cache[cache_key] = model
        return model
    except Exception as e:
        print(f"Roboflowモデル({model_id})の初期化エラー: {e}")
        return None


def process_image(img: np.ndarray, config: dict) -> dict:
    """
    画像データと設定パラメータを受け取り、指定されたモードで処理を行うメイン関数。
    """
    mode = config.get("mode", "hsv")
    
    # 元画像がない場合は空で返す
    if img is None or img.size == 0:
        return {"mode": mode, "processed_image": img, "result_value": 0}
        
    if mode == "hsv":
        return _process_hsv(img, config.get("hsv", {}))
    elif mode == "exg":
        return _process_exg(img, config.get("exg", {}))
    elif mode == "yolo":
        return _process_yolo(img, config.get("yolo", {}))
    else:
        # デフォルトはそのまま返す
        return {"mode": "unknown", "processed_image": img.copy(), "result_value": 0}


def _process_hsv(img: np.ndarray, params: dict) -> dict:
    """モード1: HSV色空間での緑色抽出"""
    h_min = params.get("h_min", 35)
    h_max = params.get("h_max", 85)
    s_min = params.get("s_min", 100)
    s_max = params.get("s_max", 255)
    v_min = params.get("v_min", 100)
    v_max = params.get("v_max", 255)
    
    # ✨ OpenCVの上限値180に安全にクリップ
    h_min = int(np.clip(h_min, 0, 180))
    h_max = int(np.clip(h_max, 0, 180))
    
    # HSV空間へ変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 閾値で二値化
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # 抽出部分を可視化するため、元画像にマスクをかける
    result_img = cv2.bitwise_and(img, img, mask=mask)
    
    # 面積の割合（%）を計算
    total_pixels = img.shape[0] * img.shape[1]
    green_pixels = cv2.countNonZero(mask)
    percentage = (green_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0
    
    return {
        "mode": "hsv",
        "processed_image": result_img,
        "result_value": round(percentage, 2)
    }


def _process_exg(img: np.ndarray, params: dict) -> dict:
    """モード2: 拡張緑色指標(ExG) + 大津の二値化"""
    offset = params.get("offset", 0)
    
    # OpenCVはBGRなので分割
    b, g, r = cv2.split(img.astype(np.float32))
    
    # ExG = 2G - R - B
    exg = 2.0 * g - r - b
    
    # 負の値を0にし、255に正規化してuint8に変換
    exg[exg < 0] = 0
    exg = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 大津の二値化
    thresh_val, mask = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 手動オフセットを適用して再計算
    if offset != 0:
        new_thresh = np.clip(thresh_val + offset, 0, 255)
        _, mask = cv2.threshold(exg, new_thresh, 255, cv2.THRESH_BINARY)
        
    # 結果画像（マスク適用）
    result_img = cv2.bitwise_and(img, img, mask=mask)
    
    # 面積の割合（%）
    total_pixels = img.shape[0] * img.shape[1]
    plant_pixels = cv2.countNonZero(mask)
    percentage = (plant_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0
    
    return {
        "mode": "exg",
        "processed_image": result_img,
        "result_value": round(percentage, 2)
    }


def _process_yolo(img: np.ndarray, params: dict) -> dict:
    """モード3: YOLOによる物体検出（ローカル.pt / Roboflow ハイブリッド）"""
    conf_thresh = params.get("conf_thresh", 0.5)
    model_id = params.get("model_id", "")
    api_key = params.get("api_key", "")
    
    # ─── ✨ もし入力された名前が「.pt」で終わる場合（ローカル実行） ───
    if model_id.endswith(".pt"):
        if YOLO is None:
            return {
                "mode": "yolo", "processed_image": img.copy(), "result_value": -1,
                "status": "error", "error_message": "ultralytics がインストールされていません。"
            }
        
        # パス指定がない場合はデフォルトで /app/config 内を探す設定
        if os.path.isabs(model_id):
            model_path = model_id
        else:
            model_path = os.path.join("/app/config", model_id)

        if not os.path.exists(model_path):
            return {
                "mode": "yolo", "processed_image": img.copy(), "result_value": -1,
                "status": "error", "error_message": f"モデルファイルが見つかりません: {model_path}"
            }
            
        try:
            if model_path not in local_model_cache:
                local_model_cache[model_path] = YOLO(model_path)
            model = local_model_cache[model_path]
            
            results = model(img, conf=conf_thresh)[0]
            annotated_image = results.plot()
            detected_count = len(results.boxes)
            
            # ─── 🛠️ 修正: メインプログラムにサイズ情報を引き渡すためのリスト作成 ───
            predictions = []
            for box in results.boxes:
                # 座標(x1, y1, x2, y2)から幅と高さを計算
                xyxy = box.xyxy[0].tolist()
                w = xyxy[2] - xyxy[0]
                h = xyxy[3] - xyxy[1]
                
                # クラスIDから名前（potted plantなど）を取得
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                predictions.append({
                    "class": class_name,
                    "confidence": confidence,
                    "width": w,
                    "height": h
                })
            
            return {
                "mode": "yolo",
                "processed_image": annotated_image,
                "predictions": predictions,  # メイン側に横流し
                "result_value": detected_count
            }
        except Exception as e:
            return {
                "mode": "yolo", "processed_image": img.copy(), "result_value": -1,
                "status": "error", "error_message": f"ローカルYOLO推論エラー: {str(e)}"
            }

    # ─── 💡 以降は、既存のRoboflow用ロジック ───
    roboflow_model = get_roboflow_model(model_id, api_key)
    if roboflow_model is None or sv is None:
        return {
            "mode": "yolo", "processed_image": img.copy(), "result_value": -1,
            "status": "error", "error_message": "Roboflow model initialization failed"
        }
        
    try:
        results = roboflow_model.infer(img)
    except Exception as e:
        print(f"Roboflow推論エラー: {e}")
        return {
            "mode": "yolo", "processed_image": img.copy(), "result_value": -1,
            "status": "error", "error_message": f"Inference failed: {str(e)}"
        }

    if isinstance(results, list): 
        result = results[0]
    else: 
        result = results
        
    detections = sv.Detections.from_inference(result)
    detections = detections[detections.confidence >= conf_thresh]
    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    annotated_image = img.copy()
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    
    # Roboflow側のpredictions構造にも合わせるための処理
    predictions = []
    for i in range(len(detections)):
        xyxy = detections.xyxy[i]
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        c_name = detections.data['class_name'][i] if 'class_name' in detections.data else str(detections.class_id[i])
        predictions.append({
            "class": c_name,
            "confidence": float(detections.confidence[i]),
            "width": w,
            "height": h
        })

    labels = [
        f"{detections.data['class_name'][i]} {detections.confidence[i]:.2f}"
        for i in range(len(detections))
    ] if "class_name" in detections.data else [
        f"{detections.class_id[i]} {detections.confidence[i]:.2f}"
        for i in range(len(detections))
    ]
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    
    return {
        "mode": "yolo", 
        "processed_image": annotated_image, 
        "predictions": predictions, 
        "result_value": len(detections)
    }