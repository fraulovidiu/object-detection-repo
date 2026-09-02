# Recunoaștere de obiecte în imagini și video

Detecție de obiecte cu YOLOv8 (Ultralytics), pe imagine, fișier video
sau webcam live. Include și un script bonus cu detecție clasică
(Haar Cascade, față + ochi).

## Cerințe

- Python 3.9+
- pip

## Instalare

```bash
git clone https://github.com/fraulovidiu/object-detection-repo.git
cd object-detection-repo

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Modelul (`weights/yolov8n.pt`) este inclus în repo — nu necesită
descărcare separată.

## Utilizare

Meniu interactiv:

```bash
python run.py
```

Sau direct din linia de comandă:

```bash
python detect.py --source cale/catre/imagine.jpg
python detect.py --source cale/catre/video.mp4
python detect.py --source 0              # webcam
```

Opțiuni: `--conf <0-1>` (prag de încredere, implicit 0.4),
`--no-save`, `--no-show`.

Detecție clasică (Haar Cascade):

```bash
python clasic_haar.py --source cale/catre/imagine.jpg
```

Rezultatele se salvează în `outputs/`.

## Structură

```
detect.py          # detectie YOLOv8 - imagine/video/webcam
run.py              # meniu interactiv
clasic_haar.py       # detectie clasica (Haar Cascade)
requirements.txt
weights/yolov8n.pt   # model pre-antrenat
sample_media/        # imagini/video de test
exemple/              # rezultate deja generate
outputs/              # rezultatele noilor rulari
```

## Licență

MIT
