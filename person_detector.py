"""
person_detector.py

Classe que usa Ultralytics YOLOv8 para detectar pessoas via webcam,
mostrando caixas, centro da pessoa e log de direção para mover a câmera.

Instalação:
pip install ultralytics opencv-python
"""

import cv2
from ultralytics import YOLO
import time


class PersonDetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf_thresh: float = 0.35, cam_index: int = 0, device: str = None):
        """
        model_name: nome do modelo (ex: "yolov8n.pt", "yolov8m.pt", caminho local para .pt)
        conf_thresh: limiar mínimo de confiança para considerar uma detecção
        cam_index: índice da webcam (0 normalmente é a webcam interna)
        device: 'cpu' ou 'cuda' (None = automático)
        """
        self.model_name = model_name
        self.conf_thresh = conf_thresh
        self.cam_index = cam_index

        # inicializa o modelo YOLOv8
        print(f"Carregando modelo {self.model_name}...")
        self.model = YOLO(model_name)  # cria o modelo
        if device:
            self.model.to(device)  # move para CPU/GPU

        # nomes de classes
        try:
            self.names = self.model.names
        except Exception:
            self.names = {0: "person"}

        print("Modelo carregado com sucesso ✅")

    def start(self):
        """Abre a webcam e inicia a detecção em tempo real. Pressione 'q' para sair."""
        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            raise RuntimeError(f"❌ Não foi possível abrir a câmera index={self.cam_index}")

        # tolerância em pixels para considerar a pessoa centralizada
        tolerance = 30

        # pega dimensões da imagem
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2

        print("🟢 Iniciando detecção (pressione 'q' para encerrar)...")
        fps_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Falha ao capturar frame.")
                break

            # inferência com YOLOv8
            results = self.model(frame, conf=self.conf_thresh, verbose=False)

            for res in results:
                boxes = getattr(res, "boxes", None)
                if boxes is None:
                    continue

                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = self.names.get(cls, str(cls))

                    # desenha apenas se for pessoa
                    if label.lower() != "person":
                        continue

                    x1, y1, x2, y2 = map(int, xyxy)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # --- log para mover câmera ---
                    move_x = ""
                    move_y = ""

                    if cx < frame_center_x - tolerance:
                        move_x = "esquerda"
                    elif cx > frame_center_x + tolerance:
                        move_x = "direita"
                    else:
                        move_x = "centralizado"

                    if cy < frame_center_y - tolerance:
                        move_y = "cima"
                    elif cy > frame_center_y + tolerance:
                        move_y = "baixo"
                    else:
                        move_y = "centralizado"

                    print(f"[LOG] Pessoa detectada → mover câmera: X: {move_x}, Y: {move_y}, Confiança: {conf:.2f}")

                    # desenha caixa e centro da pessoa
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # calcula FPS
            fps = 1.0 / (time.time() - fps_time + 1e-8)
            fps_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # mostra frame
            cv2.imshow("YOLOv8 - Person Detector (Pressione 'q' para sair)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("🛑 Detecção encerrada.")


if __name__ == "__main__":
    # exemplo de uso
    detector = PersonDetector(model_name="yolov8n.pt", conf_thresh=0.4, cam_index=0, device="cpu")
    detector.start()
