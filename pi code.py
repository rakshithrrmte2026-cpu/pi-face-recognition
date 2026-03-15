import cv2 import time import numpy as np import pandas as pd from insightface.app import
FaceAnalysis import faiss import os from threading import Thread from collections import deque
from picamera2 import Picamera2 import onnxruntime as ort
================================
0. Async Frame Capture using Pi Camera
================================
class PiVideoStream: def init(self, width=640, height=480): self.picam2 = Picamera2()
self.preview_config = self.picam2.create_preview_configuration( main={"size": (width, height),
"format": "RGB888"} ) self.picam2.configure(self.preview_config)
self.picam2.set_controls({"AwbEnable": True}) self.picam2.start()
self.stopped = False
self.frame = None
self.width = width
self.height = height
def start(self):
Thread(target=self.update, daemon=True).start()
return self
def update(self):
while not self.stopped:
self.frame = self.picam2.capture_array()
def read(self):
return self.frame
def stop(self):
self.stopped = True
self.picam2.stop()
================================
1. Load face database
================================
DB_CSV = "/home/rakshith/face_embeddings22.csv" if not os.path.exists(DB_CSV): raise
FileNotFoundError(f"{DB_CSV} not found!")
df = pd.read_csv(DB_CSV) name_col = "label" embed_cols = [str(i) for i in range(512)]
database = pd.DataFrame() database["name"] = df[name_col] database["embedding"] =
df[embed_cols].apply(lambda row: row.values.astype("float32"), axis=1)
database["embedding"] =database["embedding"].apply(lambda e: e / np.linalg.norm(e))
dim = 512 faiss_index = faiss.IndexFlatIP(dim) emb_matrix =
np.vstack(database["embedding"].values).astype("float32") faiss_index.add(emb_matrix)
print(f"􊵸􊵹 Database loaded with {len(database)} faces. FAISS index ready.")
================================
2. Initialize InsightFace
================================
so = ort.SessionOptions() so.graph_optimization_level =
ort.GraphOptimizationLevel.ORT_ENABLE_ALL so.intra_op_num_threads = 4
recognizer = FaceAnalysis( name="buffalo_l", providers=['CPUExecutionProvider'], sess_opts=so )
recognizer.prepare(ctx_id=0, det_size=(320, 320)) print("􋱅􋱆􋱇􋱈􋱉􋱊􋱋􋱌 InsightFace initialized with buffalo_l.")
================================
3. Video stream settings
================================
cap = PiVideoStream(width=640, height=480).start() print("􈙐􈙑􈙔􈙒􈙓 Press 'q' to quit.") start_time =
time.time()
frame_count = 0 skip_interval = 2 target_min_fps, target_max_fps = 10, 15 fps_smooth = 0 last_time
= time.time()
target_fps = 25 frame_time = 1.0 / target_fps
detect_res = (320, 240) face_queue = deque() max_faces_per_frame = 3
================================
Face recognition cache (IoU)
================================
face_cache = {} # key: bbox tuple -> {'name': str, 'last_seen': timestamp} CACHE_MAX_AGE = 1.0
# seconds IOU_THRESHOLD = 0.5
def iou(boxA, boxB): xA = max(boxA[0], boxB[0]) yA = max(boxA[1], boxB[1]) xB = min(boxA[2],
boxB[2]) yB = min(boxA[3], boxB[3]) interArea = max(0, xB - xA) * max(0, yB - yA) if interArea
== 0: return 0.0 boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1]) boxBArea = (boxB[2]-
boxB[0]) * (boxB[3]-boxB[1]) return interArea / float(boxAArea + boxBArea - interArea)
def get_cached_face_name_iou(face, cache, max_age=CACHE_MAX_AGE,iou_threshold=IOU_THRESHOLD): x1, y1, x2, y2 = face.bbox.astype(int) now = time.time() for
cached_box, info in cache.items(): if now - info['last_seen'] > max_age: continue if iou((x1, y1, x2,
y2), cached_box) > iou_threshold: info['last_seen'] = now return info['name'], cached_box return
None, None
================================
4. CPU temperature helper
================================
def get_cpu_temp(): try: with open("/sys/class/thermal/thermal_zone0/temp") as f: return int(f.read())
/ 1000 except: return 0
import os
import pandas as pd
import numpy as np
# =========================
# CONFIG
# =========================
ROOT_FOLDER = "./dynamic_csvs"
SIMILARITY_THRESHOLD = 0.3
# -------------------------
# Ground truth mapping
# -------------------------
SUBJECT_MAP = {
"S1_run": "Adarsh",
"S2_run": "Akash",
"S3_run": "Manjunath"
}
SCENARIO_SUBJECTS = {
"D1-01": ["Adarsh","Akash","Manjunath"],
"D1-02": ["Adarsh","Akash","Manjunath"],
"D1-03": ["Adarsh","Akash","Manjunath"],
"D1-04": ["Adarsh","Akash","Manjunath"],
"D2-01": ["Adarsh"],
"D2-02": ["Adarsh"],
"D2-03": ["Akash"],
"D2-04": ["Akash"],
"D2-05": ["Manjunath"],
"D2-06": ["Manjunath"],
"D3-01": ["Adarsh"],
"D3-02": ["Adarsh"],
"D3-03": ["Akash"],
"D3-04": ["Akash"],
"D3-05": ["Manjunath"],
"D3-06": ["Manjunath"],
"D4-01": ["Adarsh","Akash","Manjunath"],
"D4-02": ["Adarsh","Akash","Manjunath"],
"D4-03": ["Adarsh","Akash","Manjunath"],
"D4-04": ["Adarsh","Akash","Manjunath"]
}
# =========================
# METRIC FUNCTIONS
# =========================
def compute_metrics(df, gt_names):
# Accuracy
correct = ((df['predicted_name'].isin(gt_names)) &
(df['similarity'] >= SIMILARITY_THRESHOLD))
accuracy = correct.sum() / len(df) * 100
# Missed detection
missed = (df['predicted_name'] == "Unknown").sum() / len(df) * 100
# False recognition
far = ((~df['predicted_name'].isin(gt_names)) &
(df['predicted_name'] != "Unknown")).sum() / len(df) * 100
# ID switch (frame-normalized)
df_sorted = df.sort_values(['frame_id','face_idx'])
last_id = {}
id_switches = 0
for _, row in df_sorted.iterrows():
idx = row['face_idx']
name = row['predicted_name']
if idx in last_id and last_id[idx] != name and name != "Unknown":
id_switches += 1
last_id[idx] = name
id_switch_rate = id_switches / df['frame_id'].nunique()
# FPS & latency
df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'], unit='s')
diffs = df_sorted['timestamp'].diff().dt.total_seconds().dropna()
latency_mean = diffs.mean() * 1000
latency_max = diffs.max() * 1000
fps_mean = 1 / diffs.mean()
fps_std = diffs.std()
fps_std = (1 / fps_std) if fps_std > 0 else 0
return {
"accuracy": accuracy,
"far": far,
"missed": missed,
"id_switch": id_switch_rate,
"fps_mean": fps_mean,
"fps_std": fps_std,
"latency_mean": latency_mean,
"latency_max": latency_max
}
# =========================
# AGGREGATION PER SCENARIO
# =========================
def process_scenario(folder_path):
scenario_id = os.path.basename(folder_path).split("_")[0]
metrics = []
for file in os.listdir(folder_path):
if not file.endswith(".csv"):
continue
df = pd.read_csv(os.path.join(folder_path, file))
file_key = file.replace(".csv","")
if file_key in SUBJECT_MAP:
gt = [SUBJECT_MAP[file_key]]
else:
gt = SCENARIO_SUBJECTS[scenario_id]
metrics.append(compute_metrics(df, gt))
return pd.DataFrame(metrics)
# =========================
# PRINT TABLE
# =========================
def print_table(title, table_folder):
print(f"\n{title}")
base = os.path.join(ROOT_FOLDER, table_folder)
for scenario in sorted(os.listdir(base)):
df = process_scenario(os.path.join(base, scenario))
acc = f"{df.accuracy.mean():.1f} ± {df.accuracy.std():.1f}"
far = f"{df.far.mean():.1f} ± {df.far.std():.1f}"
miss = f"{df.missed.mean():.1f} ± {df.missed.std():.1f}"
ids = f"{df.id_switch.mean():.1f} ± {df.id_switch.std():.1f}"
fps = f"{df.fps_mean.mean():.1f} ± {df.fps_std.mean():.1f}"
lat = f"{df.latency_mean.mean():.0f} ± {df.latency_max.mean():.0f}"
print(f"{scenario} | {acc} | {far} | {miss} | {ids} | {fps} | {lat}")
# =========================
# RUN
# =========================
print_table("Table B — Dynamic Single-Person", "B_Dynamic_Single_Core")
print_table("Table C — Speed Sensitivity", "C_Dynamic_Speed_Sensitivity")
print_table("Table D — Distance Under Motion", "D_Dynamic_Distance_Motion")
print_table("Table E — Multi-Person (3 Faces)", "E_Dynamic_MultiPerson")
