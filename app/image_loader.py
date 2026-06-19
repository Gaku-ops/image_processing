import cv2
import os
import glob
import time
import numpy as np

def load_images_from_folder(folder_path: str, extensions: tuple = ('.jpg', '.jpeg', '.png'), delay_seconds: int = 3) -> list:
    """
    指定されたフォルダから画像を読み込みます。
    
    Args:
        folder_path (str): 読み込む画像が配置されているフォルダパス
        extensions (tuple): 対象とする画像の拡張子
        
    Returns:
        list: 読み込まれた画像(OpenCVのNumPy配列)のリスト。ファイルがない場合は空リスト。
    """
    if not os.path.exists(folder_path):
        print(f"エラー: フォルダが存在しません - {folder_path}")
        return []

    images = []
    current_time = time.time()
    
    # 拡張子ごとに検索
    for ext in extensions:
        search_path = os.path.join(folder_path, f"*{ext}")
        for file_path in glob.glob(search_path):
            # 書き込み中の読み込み事故を防ぐため、最終更新から指定秒数（3秒）経過していないファイルはスキップ
            mtime = os.path.getmtime(file_path)
            if current_time - mtime < delay_seconds:
                print(f"書き込み待機中 (あと約 {int(delay_seconds - (current_time - mtime))}秒): {file_path}")
                continue
                
            # 日本語パス対応などのために cv2.imdecode を使用する方が安全ですが、今回は通常のimreadを使用
            img = cv2.imread(file_path)
            if img is not None:
                images.append((file_path, img))
                print(f"読み込み成功: {file_path}")
            else:
                print(f"読み込み失敗: {file_path}")
                
    # 大文字の拡張子も考慮
    for ext in extensions:
        search_path = os.path.join(folder_path, f"*{ext.upper()}")
        for file_path in glob.glob(search_path):
            mtime = os.path.getmtime(file_path)
            if current_time - mtime < delay_seconds:
                continue
                
            img = cv2.imread(file_path)
            if img is not None:
                images.append((file_path, img))
                print(f"読み込み成功: {file_path}")

    return images

def load_single_image(file_path: str) -> np.ndarray:
    """
    単一の画像を読み込みます。
    
    Args:
        file_path (str): 画像ファイルのパス
        
    Returns:
        np.ndarray or None: 読み込まれた画像データ
    """
    if not os.path.exists(file_path):
        print(f"エラー: ファイルが存在しません - {file_path}")
        return None
        
    return cv2.imread(file_path)
