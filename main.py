import os
import cv2
import time
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
# ==========================
# CONFIG
# ==========================

FACE_DB = "dataset/train"
TEST_DB = "dataset/test"
THRESHOLD = 0.55
DET_SIZE = (640, 640)
USE_GPU = False

# ==========================
# MODEL INITIALIZATION
# ==========================

ctx_id = 0 if USE_GPU else -1
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=ctx_id, det_size=DET_SIZE)

# ==========================
# UTILS
# ==========================

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ==========================
# BUILD DATABASE
# ==========================

def build_face_db():
    print("\nEncoding Face Database...")
    db = {}

    for person in tqdm(os.listdir(FACE_DB)):
        person_path = os.path.join(FACE_DB, person)
        if not os.path.isdir(person_path):
            continue

        embeddings = []

        for img_name in os.listdir(person_path):
            img = cv2.imread(os.path.join(person_path, img_name))
            faces = app.get(img)

            if len(faces) == 0:
                continue

            embeddings.append(faces[0].embedding)

        if embeddings:
            db[person] = np.mean(embeddings, axis=0)

    np.save("face_db_embeddings.npy", db)
    print(f"Database built with {len(db)} identities\n")


# ==========================
# LOAD DATABASE
# ==========================

def load_face_db():
    if not os.path.exists("face_db_embeddings.npy"):
        build_face_db()
    return np.load("face_db_embeddings.npy", allow_pickle=True).item()


# ==========================
# REALTIME RECOGNITION
# ==========================

def realtime_recognition(db):

    cap = cv2.VideoCapture(0)
    frame_fps = []
    prev_time = time.perf_counter()

    print("\nStarting Real-Time Recognition (Press ESC to exit)...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.perf_counter()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        frame_fps.append(fps)

        faces = app.get(frame)

        for face in faces:
            emb = face.embedding

            best_id = "Unknown"
            best_score = 0

            for name, db_emb in db.items():
                score = cosine_similarity(emb, db_emb)
                if score > best_score:
                    best_score = score
                    best_id = name

            if best_score < THRESHOLD:
                best_id = "Unknown"

            box = face.bbox.astype(int)
            color = (0,255,0) if best_id!="Unknown" else (0,0,255)

            cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),color,2)
            cv2.putText(frame,f"{best_id} {best_score:.2f}",
                        (box[0],box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

        cv2.putText(frame,f"FPS: {fps:.2f}",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

        cv2.imshow("FaceTrack - RetinaFace + ArcFace", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    frame_fps = np.array(frame_fps)
    latencies = 1 / frame_fps

    print("\n========== FPS REPORT ==========")
    print(f"Max FPS : {frame_fps.max():.2f}")
    print(f"Avg FPS : {frame_fps.mean():.2f}")
    print(f"Min FPS : {frame_fps.min():.2f}")
    print(f"Std FPS : {frame_fps.std():.2f}")
    print(f"Throughput : {len(frame_fps)/latencies.sum():.2f} fps")
    print(f"P50 Latency: {np.percentile(latencies,50):.4f}s")
    print(f"P95 Latency: {np.percentile(latencies,95):.4f}s")
    print(f"P99 Latency: {np.percentile(latencies,99):.4f}s")
    print("================================\n")


# ==========================
# EVALUATION METRICS
# ==========================

def evaluate_metrics(db):

    print("\nEvaluating Offline Metrics...\n")

    y_true = []
    y_score = []
    rank1_correct = 0
    total_samples = 0

    det_times, emb_times, match_times = [], [], []

    for person in tqdm(os.listdir(TEST_DB)):
        person_path = os.path.join(TEST_DB, person)
        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            img = cv2.imread(os.path.join(person_path, img_name))

            t0 = time.perf_counter()
            faces = app.get(img)
            det_times.append(time.perf_counter() - t0)

            if len(faces) == 0:
                continue

            t1 = time.perf_counter()
            emb = faces[0].embedding
            emb_times.append(time.perf_counter() - t1)

            scores = {}
            t2 = time.perf_counter()
            for name, db_emb in db.items():
                scores[name] = cosine_similarity(emb, db_emb)
            match_times.append(time.perf_counter() - t2)

            best_id = max(scores, key=scores.get)
            best_score = scores[best_id]

            total_samples += 1
            if best_id == person:
                rank1_correct += 1

            y_score.append(best_score)
            y_true.append(1 if best_id == person else 0)

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    preds = (y_score >= THRESHOLD).astype(int)

    cm = confusion_matrix(y_true, preds, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)

    far = fp / (fp + tn + 1e-6)
    frr = fn / (fn + tp + 1e-6)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    rank1 = rank1_correct / total_samples

    print("\n========== PERFORMANCE REPORT ==========")
    print(f"Accuracy     : {accuracy*100:.2f}%")
    print(f"Precision    : {precision*100:.2f}%")
    print(f"Recall       : {recall*100:.2f}%")
    print(f"F1 Score     : {f1*100:.2f}%")
    print(f"FAR          : {far*100:.2f}%")
    print(f"FRR          : {frr*100:.2f}%")
    print(f"AUC          : {roc_auc:.4f}")
    print(f"EER          : {eer:.4f}")
    print(f"Rank-1 Acc   : {rank1*100:.2f}%")
    print("----------------------------------------")
    print(f"Avg Embedding Time: {np.mean(emb_times)*1000:.3f} ms")
    print(f"Avg Matching Time : {np.mean(match_times)*1000:.3f} ms")
    print(f"Avg Detection Time: {np.mean(det_times)*1000:.3f} ms")
    print(f"Total Pipeline    : {(np.mean(det_times)+np.mean(emb_times)+np.mean(match_times))*1000:.3f} ms")
    print("========================================\n")

    # ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – RetinaFace + ArcFace")
    plt.legend()
    plt.show()

    # Confusion Matrix Heatmap
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Negative", "Pred Positive"],
                yticklabels=["Actual Negative", "Actual Positive"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

    # FAR vs FRR Curve
    plt.figure()
    plt.plot(thresholds, fpr, label="FAR")
    plt.plot(thresholds, fnr, label="FRR")
    plt.xlabel("Threshold")
    plt.ylabel("Error Rate")
    plt.title("FAR vs FRR Curve")
    plt.legend()
    plt.show()


# ==========================
# MAIN
# ==========================

if __name__ == "__main__":

    print("\n========== FACETRACK POC ==========")

    db = load_face_db()
    realtime_recognition(db)
    evaluate_metrics(db)