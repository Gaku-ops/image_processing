import cv2
import numpy as np
try:
    from inference import get_model
    import supervision as sv
except ImportError:
    get_model = None
    sv = None

# Roboflowモデルのキャッシュ用辞書
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
    画像データと設定パラメータを受け取り、指定されたモードで処理を行う関数。
    外部ファイルや通信に依存しない純粋な関数として設計。
    
    Args:
        img (np.ndarray): 処理対象の画像 (BGRフォーマット)
        config (dict): パラメータを格納した辞書
            例: {
                "mode": "hsv" | "exg" | "yolo",
                "hsv": {"h_min":0, "h_max":180, "s_min":0, ...},
                "exg": {"offset": 0},
                "yolo": {"conf_thresh": 0.5}
            }
            
    Returns:
        dict: {
            "mode": 実行されたモード(str),
            "processed_image": 処理後の画像(np.ndarray),
            "result_value": 抽出面積(%) や 検出数などの数値結果(float|int)
        }
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
    
    # OpenCVはBGRなので、分割
    b, g, r = cv2.split(img.astype(np.float32))
    
    # ExG = 2G - R - B
    exg = 2.0 * g - r - b
    
    # 負の値を0にし、255に正規化してuint8に変換
    exg[exg < 0] = 0
    exg = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 大津の二値化
    # 戻り値の thresh は自動計算された閾値
    thresh_val, mask = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 手動オフセットを適用して再計算（もしオフセットが指定されている場合）
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
    """モード3: YOLO(Roboflow)による物体検出"""
    conf_thresh = params.get("conf_thresh", 0.5)
    model_id = params.get("model_id", "")
    api_key = params.get("api_key", "")
    
    roboflow_model = get_roboflow_model(model_id, api_key)
    
    if roboflow_model is None or sv is None:
        # モデルがない場合はエラーを返す代わりに元画像をそのまま返す
        return {"mode": "yolo", "processed_image": img.copy(), "result_value": 0}
        
    # 推論実行 (Inference SDK)
    results = roboflow_model.infer(img)
    if isinstance(results, list):
        result = results[0]
    else:
        result = results
        
    # Supervisionで結果をパース
    detections = sv.Detections.from_inference(result)
    
    # 信頼度の閾値でフィルタリング
    detections = detections[detections.confidence >= conf_thresh]
    
    # 描画
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    annotated_image = img.copy()
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    
    labels = [
        f"{detections.data['class_name'][i]} {detections.confidence[i]:.2f}"
        for i in range(len(detections))
    ] if "class_name" in detections.data else [
        f"{detections.class_id[i]} {detections.confidence[i]:.2f}"
        for i in range(len(detections))
    ]
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    
    # 検出されたオブジェクト数
    detected_count = len(detections)
    
    return {
        "mode": "yolo",
        "processed_image": annotated_image,
        "result_value": detected_count
    }
