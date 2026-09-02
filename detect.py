#!/usr/bin/env python3
"""
detect.py - Recunoastere de obiecte in imagini si video
Proiect examen: Prelucrarea Imaginilor

Foloseste YOLOv8 (Ultralytics) - o retea neurala convolutionala de tip
"single-stage detector" - pentru a detecta si localiza obiecte in:
  - o imagine statica
  - un fisier video
  - fluxul video de la webcam (live)

Pipeline (pe scurt, pentru prezentare):
  1. Preprocesare: imaginea/frame-ul e redimensionat la 640x640 si
     normalizat (valori pixeli in [0,1]).
  2. Extractie de caracteristici: un backbone CNN (CSPDarknet) extrage
     harti de caracteristici la mai multe scari (pentru obiecte mici,
     medii, mari).
  3. Predictie: pentru fiecare celula dintr-o grila, reteaua prezice
     direct cutii de incadrare (bounding boxes), un scor de incredere
     (confidence) si o distributie de probabilitate pe clase (varianta
     "anchor-free" folosita de YOLOv8).
  4. Post-procesare: se elimina cutiile cu incredere sub prag
     (--conf) si se aplica Non-Maximum Suppression (NMS) pentru a
     elimina detectiile duplicate ale aceluiasi obiect (cutii care se
     suprapun mult - IoU mare).
  5. Se deseneaza cutiile ramase peste imagine/frame, cu eticheta
     clasei si scorul de incredere.

Utilizare:
  python detect.py --source poza.jpg
  python detect.py --source clip.mp4
  python detect.py --source 0                 # webcam (index 0)
  python detect.py --source poza.jpg --conf 0.5 --no-show
"""

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "weights" / "yolov8n.pt"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def load_model(model_path: Path):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "[EROARE] Pachetul 'ultralytics' nu este instalat.\n"
            "Ruleaza: pip install -r requirements.txt"
        )

    if not model_path.exists():
        print(
            f"[INFO] Nu am gasit modelul local la {model_path}.\n"
            "       Incerc sa descarc 'yolov8n.pt' automat (necesita internet)."
        )
        model_path = "yolov8n.pt"  # ultralytics il descarca automat

    print(f"[INFO] Incarc modelul: {model_path}")
    model = YOLO(str(model_path))
    return model


def classify_source(source: str):
    if source.isdigit():
        return "webcam", int(source)
    p = Path(source)
    ext = p.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image", p
    if ext in VIDEO_EXTS:
        return "video", p
    sys.exit(f"[EROARE] Extensie necunoscuta pentru sursa: {source}")


def summarize(results, model) -> str:
    """Construieste un rezumat text: cate obiecte din fiecare clasa au fost gasite."""
    names = model.names
    counts = Counter()
    for box in results.boxes:
        cls_id = int(box.cls[0])
        counts[names[cls_id]] += 1
    if not counts:
        return "Niciun obiect detectat peste pragul de incredere ales."
    parts = [f"{cls}: {n}" for cls, n in sorted(counts.items(), key=lambda x: -x[1])]
    return "Detectii -> " + ", ".join(parts)


def run_on_image(model, image_path: Path, conf: float, save: bool, show: bool):
    frame = cv2.imread(str(image_path))
    if frame is None:
        sys.exit(f"[EROARE] Nu pot citi imaginea: {image_path}")

    results = model.predict(frame, conf=conf, verbose=False)[0]
    annotated = results.plot()  # deseneaza cutiile + etichetele

    print(summarize(results, model))

    if save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        out_path = OUTPUT_DIR / f"detectat_{image_path.stem}.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"[INFO] Rezultat salvat in: {out_path}")

    if show:
        _safe_show("Detectie obiecte - imagine (apasa orice tasta pentru a inchide)", annotated, wait=0)


def run_on_video(model, video_path, conf: float, save: bool, show: bool, is_webcam: bool):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[EROARE] Nu pot deschide sursa video: {video_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save and not is_webcam:
        OUTPUT_DIR.mkdir(exist_ok=True)
        out_path = OUTPUT_DIR / f"detectat_{Path(str(video_path)).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (width, height))
        print(f"[INFO] Salvez videoclipul adnotat in: {out_path}")

    total_counts = Counter()
    prev_time = time.time()
    frame_idx = 0

    print("[INFO] Ruleaza detectia... apasa 'q' in fereastra video pentru a opri.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            results = model.predict(frame, conf=conf, verbose=False)[0]
            annotated = results.plot()

            for box in results.boxes:
                total_counts[model.names[int(box.cls[0])]] += 1

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(
                annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )

            if writer is not None:
                writer.write(annotated)

            if show:
                stop = _safe_show("Detectie obiecte - video (q = iesire)", annotated, wait=1)
                if stop:
                    break
    except KeyboardInterrupt:
        print("\n[INFO] Oprit de utilizator (Ctrl+C).")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # mediu fara GUI (ex. server fara display) - ignoram

    print(f"[INFO] Total cadre procesate: {frame_idx}")
    if total_counts:
        parts = [f"{cls}: {n}" for cls, n in sorted(total_counts.items(), key=lambda x: -x[1])]
        print("Total detectii pe tot clipul -> " + ", ".join(parts))


def _safe_show(window_name: str, frame, wait: int) -> bool:
    """Afiseaza o fereastra cu imaginea. Returneaza True daca userul a apasat 'q'.
    Daca nu exista mediu grafic disponibil (ex. server fara display), nu pica programul."""
    try:
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(wait) & 0xFF
        return key == ord("q")
    except cv2.error as e:
        print(f"[AVERTISMENT] Nu pot afisa fereastra grafica ({e}). "
              f"Continui doar cu salvarea rezultatelor.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Recunoastere de obiecte in imagini/video (YOLOv8)")
    parser.add_argument("--source", required=True,
                         help="Cale catre imagine, cale catre video, sau index webcam (ex: 0)")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Cale catre fisierul de model .pt")
    parser.add_argument("--conf", type=float, default=0.4, help="Prag de incredere (0-1)")
    parser.add_argument("--no-save", action="store_true", help="Nu salva rezultatul pe disc")
    parser.add_argument("--no-show", action="store_true", help="Nu deschide fereastra grafica")
    args = parser.parse_args()

    model = load_model(Path(args.model))
    kind, source = classify_source(args.source)

    save = not args.no_save
    show = not args.no_show

    if kind == "image":
        run_on_image(model, source, args.conf, save, show)
    elif kind == "video":
        run_on_video(model, str(source), args.conf, save, show, is_webcam=False)
    else:  # webcam
        run_on_video(model, source, args.conf, save=False, show=show, is_webcam=True)


if __name__ == "__main__":
    main()
