#!/usr/bin/env python3
"""
clasic_haar.py - Detectie de obiecte prin metoda CLASICA (Haar Cascade)

Acest script e un BONUS pentru proiect: arata ca intelegi si o metoda
"clasica" de prelucrare a imaginilor (dinainte de deep learning), utila
daca profesorul intreaba "ce alte metode exista in afara de retele
neuronale?" sau daca YOLO nu poate rula (fara internet / fara GPU /
problema de instalare de ultima ora).

Cum functioneaza Haar Cascade (Viola-Jones, 2001) - de explicat la examen:
  1. Imaginea e convertita in tonuri de gri (grayscale) - metoda foloseste
     doar contrast de lumina/intensitate, nu culoare.
  2. Se calculeaza o "imagine integrala" (integral image) care permite
     calculul foarte rapid al sumei pixelilor din orice regiune dreptunghiulara.
  3. Un set de "trasaturi Haar" (dreptunghiuri alb/negru, ex: ochi mai
     inchisi la culoare decat obrajii) sunt aplicate ca niste filtre peste
     ferestre glisante (sliding window) de diverse dimensiuni.
  4. Un clasificator in cascada (mai multe etaje, fiecare tot mai strict)
     elimina rapid ferestrele care clar NU contin obiectul cautat, pastrand
     doar candidatii promitatori - de-aia e foarte rapid (ruleaza si pe CPU
     slab, in timp real).
  5. Ferestrele care trec toate etajele cascadei sunt marcate ca detectii.

Diferenta fata de YOLO (detect.py):
  - Haar Cascade: metoda clasica, bazata pe trasaturi desenate manual
    (hand-crafted features), foarte rapida, dar mai putin precisa si
    limitata la un singur tip de obiect per model (fete, ochi, etc.)
  - YOLO: retea neuronala convolutionala, invata singura trasaturile din
    date de antrenament, detecteaza zeci/sute de clase de obiecte simultan,
    mai precisa dar are nevoie de mai multa putere de calcul.

Utilizare:
  python clasic_haar.py --source poza.jpg
  python clasic_haar.py --source 0        # webcam
"""

import argparse
import sys
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# Haar cascades vin incluse cu opencv-python, nu necesita descarcare separata.
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"


def detect_faces(frame, face_cascade, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # imbunatateste contrastul -> detectii mai bune

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "fata", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=8)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    return frame, len(faces)


def main():
    parser = argparse.ArgumentParser(description="Detectie clasica (Haar Cascade) - fata + ochi")
    parser.add_argument("--source", required=True, help="Cale imagine/video sau index webcam (ex: 0)")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
    if face_cascade.empty():
        sys.exit("[EROARE] Nu am putut incarca haarcascade_frontalface_default.xml")

    source = args.source
    is_webcam = source.isdigit()
    src = int(source) if is_webcam else source

    p = Path(source) if not is_webcam else None
    is_image = (p is not None) and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            sys.exit(f"[EROARE] Nu pot citi imaginea: {source}")
        frame, n = detect_faces(frame, face_cascade, eye_cascade)
        print(f"[INFO] Fete detectate: {n}")
        OUTPUT_DIR.mkdir(exist_ok=True)
        out_path = OUTPUT_DIR / f"haar_{p.stem}.jpg"
        cv2.imwrite(str(out_path), frame)
        print(f"[INFO] Salvat: {out_path}")
        if not args.no_show:
            try:
                cv2.imshow("Detectie clasica (Haar) - apasa o tasta", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except cv2.error:
                pass
    else:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            sys.exit(f"[EROARE] Nu pot deschide sursa: {source}")
        print("[INFO] Apasa 'q' pentru a opri.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame, n = detect_faces(frame, face_cascade, eye_cascade)
            if not args.no_show:
                try:
                    cv2.imshow("Detectie clasica (Haar) - q = iesire", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    break
        cap.release()
        if not args.no_show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    main()
