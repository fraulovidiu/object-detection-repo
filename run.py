#!/usr/bin/env python3
"""
run.py - Meniu simplu pentru prezentare la examen.

Nu trebuie sa iti amintesti comenzi CLI: ruleaza "python run.py" si
alege optiunea potrivita din meniu.
"""

import sys
from pathlib import Path

from detect import load_model, run_on_image, run_on_video, DEFAULT_MODEL

SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = SCRIPT_DIR / "sample_media"


def find_sample(exts):
    if not SAMPLE_DIR.exists():
        return None
    for f in sorted(SAMPLE_DIR.iterdir()):
        if f.suffix.lower() in exts:
            return f
    return None


def menu():
    print("=" * 60)
    print(" Recunoastere de obiecte in imagini/video - Demo examen")
    print("=" * 60)
    print("1. Detectie pe o imagine din sample_media/")
    print("2. Detectie pe un video din sample_media/")
    print("3. Detectie live pe webcam")
    print("4. Detectie pe un fisier ales de tine (introduci calea)")
    print("0. Iesire")
    print("-" * 60)
    return input("Alege o optiune: ").strip()


def main():
    model = load_model(Path(DEFAULT_MODEL))

    while True:
        choice = menu()

        if choice == "0":
            print("La revedere!")
            sys.exit(0)

        elif choice == "1":
            img = find_sample({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
            if img is None:
                print("[!] Nu am gasit nicio imagine in sample_media/. "
                      "Pune o poza acolo sau foloseste optiunea 4.")
                continue
            print(f"[INFO] Folosesc imaginea: {img}")
            run_on_image(model, img, conf=0.4, save=True, show=True)

        elif choice == "2":
            vid = find_sample({".mp4", ".avi", ".mov", ".mkv", ".m4v"})
            if vid is None:
                print("[!] Nu am gasit niciun video in sample_media/. "
                      "Pune un clip acolo sau foloseste optiunea 4.")
                continue
            print(f"[INFO] Folosesc videoclipul: {vid}")
            run_on_video(model, str(vid), conf=0.4, save=True, show=True, is_webcam=False)

        elif choice == "3":
            print("[INFO] Pornesc webcam-ul... apasa 'q' in fereastra pentru a opri.")
            run_on_video(model, 0, conf=0.4, save=False, show=True, is_webcam=True)

        elif choice == "4":
            path = input("Introdu calea catre imagine/video: ").strip()
            p = Path(path)
            if not p.exists():
                print("[!] Fisierul nu exista.")
                continue
            ext = p.suffix.lower()
            if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                run_on_image(model, p, conf=0.4, save=True, show=True)
            else:
                run_on_video(model, str(p), conf=0.4, save=True, show=True, is_webcam=False)

        else:
            print("[!] Optiune invalida.")


if __name__ == "__main__":
    main()
