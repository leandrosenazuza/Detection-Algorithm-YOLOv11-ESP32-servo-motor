import cv2
from ultralytics import YOLO
import time
import requests

class PersonDetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf_thresh: float = 0.35, cam_index: int = 0, device: str = None):
        self.model_name = model_name
        self.conf_thresh = conf_thresh
        self.cam_index = cam_index

        print(f"Carregando modelo {self.model_name}...")
        self.model = YOLO(model_name)
        if device:
            self.model.to(device)

        try:
            self.names = self.model.names
        except Exception:
            self.names = {0: "person"}

        print("Modelo carregado com sucesso ✅")

    def enviar_comando(self, direcao: str):
        url = f"http://localhost:8080/servidor-esp/servo-motor/{direcao}"
        try:
            requests.get(url, timeout=0.5)
            print(f"[HTTP] Comando enviado: {direcao}")
        except requests.RequestException as e:
            print(f"[ERRO HTTP] Não foi possível enviar comando: {direcao} ({e})")

    def start(self):
        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            raise RuntimeError(f"❌ Não foi possível abrir a câmera index={self.cam_index}")

        tolerance = 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2

        last_command_x = "centralizado"
        last_command_y = "centralizado"

        print("🟢 Iniciando detecção (pressione 'q' para encerrar)...")
        fps_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, conf=self.conf_thresh, verbose=False)

            for res in results:
                boxes = getattr(res, "boxes", None)
                if boxes is None:
                    continue

                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    cls = int(box.cls[0])
                    label = self.names.get(cls, str(cls))

                    if label.lower() != "person":
                        continue

                    x1, y1, x2, y2 = map(int, xyxy)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # calcula direção
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

                    # envia comando HTTP somente se mudou
                    if move_x != last_command_x:
                        self.enviar_comando(move_x)
                        last_command_x = move_x

                    if move_y != last_command_y:
                        self.enviar_comando(move_y)
                        last_command_y = move_y

                    # desenha caixa e centro
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"{label}", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            fps = 1.0 / (time.time() - fps_time + 1e-8)
            fps_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("YOLOv8 - Person Detector (Pressione 'q' para sair)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("🛑 Detecção encerrada.")


if __name__ == "__main__":
    detector = PersonDetector(model_name="yolov8n.pt", conf_thresh=0.4, cam_index=0, device="cpu")
    detector.start()
