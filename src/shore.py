import cv2
import numpy as np
import time
import threading
import pandas as pd
import bluetooth
import random
import csv
import RPi.GPIO as GPIO
from PIL import Image
from tflite_runtime.interpreter import Interpreter, load_delegate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from RPLCD.i2c import CharLCD

# =================================================================
# 1. 설정 및 초기화
# =================================================================
PORT = 1
SERVO1_PIN = 22 
SERVO2_PIN = 27
VIB_PIN = 17
LOG_FILE = "land_log.csv"

shared_data = {
    "animal": 0.0,       # 동물 감지
    "quake": 0.0,        # 지진 규모
    "is_tsunami": 0,     # AI 예측 결과
    "wave": 0.0,         # 수신 파고
    "auto_mode": False,  # 자동 모드
    "running": True
}

# LCD
try:
    lcd = CharLCD('PCF8574', 0x27)
    lcd.clear(); lcd.write_string("System Init...")
except: lcd = None

# GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO1_PIN, GPIO.OUT); GPIO.setup(SERVO2_PIN, GPIO.OUT)
GPIO.setup(VIB_PIN, GPIO.IN)
servo1 = GPIO.PWM(SERVO1_PIN, 50); servo2 = GPIO.PWM(SERVO2_PIN, 50)
servo1.start(0); servo2.start(0)

def set_angle(servo, angle):
    duty = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty)
    time.sleep(0.02)
    servo.ChangeDutyCycle(0)

set_angle(servo1, 0); set_angle(servo2, 180)

# 로그
try:
    with open(LOG_FILE, 'x', newline='') as f:
        csv.writer(f).writerow(['Time', 'Vibration', 'Animal', 'Quake', 'AI_Pred', 'Wave_Recv', 'Action'])
except: pass

def save_log_land(vib, animal, quake, ai_pred, wave, action):
    now = datetime.now().strftime("%H:%M:%S")
    with open(LOG_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([now, vib, animal, f"{quake:.1f}", ai_pred, f"{wave:.1f}", action])

def get_evac_msg(wave_val):
    if wave_val < 0.5: return "Stay off Beach"
    elif wave_val < 2.0: return "Evac: 2F+"
    elif wave_val < 5.0: return "Evac: 4F+"
    elif wave_val < 10.0: return "Evac: 7F+"
    else: return "!! RUN TO MT !!"

# =================================================================
# 2. AI 모델 학습 및 로드
# =================================================================
# [A] Fish Detection
MODEL_PATH = "fish_overfit.tflite"
try:
    interpreter = Interpreter(model_path=MODEL_PATH, experimental_delegates=[load_delegate("libedgetpu.so.1")])
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape'][1:3]
    LABELS = {0:"Fish", 1:"No Fish"}
except: exit()

# [B] Tsunami AI (RandomForest)
print(">>> [System] 지진 모델 학습 중...")
tsunami_model = None
try:
    df = pd.read_csv("earthquake_data_tsunami.csv")
    X = df[['magnitude', 'depth', 'latitude', 'longitude']]
    y = df['tsunami']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tsunami_model = RandomForestClassifier(n_estimators=100, random_state=42)
    tsunami_model.fit(X_train, y_train)
    print(">>> [System] 학습 완료!")
except: print("[오류] CSV 파일 없음")

def predict_tsunami_ai(mag):
    if tsunami_model is None: return 0
    # 임의의 깊이/위치값 사용 (시뮬레이션)
    return tsunami_model.predict(pd.DataFrame([[mag, 20, 37.5, 131.0]], columns=['magnitude', 'depth', 'latitude', 'longitude']))[0]

# =================================================================
# 3. 통신 스레드
# =================================================================
def bluetooth_thread_func():
    global shared_data
    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_sock.bind(("", PORT))
    server_sock.listen(1)
    print("📡 [Server] 통신 대기 중...")

    while shared_data["running"]:
        try:
            client_sock, client_info = server_sock.accept()
            print(f"✅ 연결됨: {client_info}")
            if lcd: lcd.clear(); lcd.write_string("Connected!")

            while shared_data["running"]:
                try:
                    # 1. 송신 (Ping)
                    anim = shared_data["animal"]
                    quake = shared_data["quake"]
                    is_tsu = shared_data["is_tsunami"]
                    
                    # [핵심 수정] 경보 발송 조건 완화 (Fail-safe)
                    # AI가 0이라도, 지진이 4.5 이상이면 무조건 보냄!
                    if anim > 0 or quake >= 4.5 or is_tsu == 1:
                        msg = f"WARNING,{anim},{quake}"
                        status_txt = "WARNING"
                    else:
                        msg = "SAFE,0,0"
                        status_txt = "SAFE"
                    
                    client_sock.send(msg.encode())
                    print(f"[TX] {msg} (AI판단:{is_tsu})")

                    # 2. 수신 (Pong)
                    raw = client_sock.recv(1024).decode().strip()
                    evac_msg = "-"
                    
                    if raw:
                        print(f"[RX] {raw}")
                        if ":" in raw:
                            header, value = raw.split(":", 1)
                            
                            if header == "WAVE":
                                wave_val = float(value)
                                shared_data["wave"] = wave_val
                                
                                evac_msg = get_evac_msg(wave_val)
                                if lcd:
                                    lcd.clear(); lcd.write_string(f"Wave: {wave_val}m")
                                    lcd.cursor_pos=(1,0); lcd.write_string(evac_msg)

                                # 방파벽 가동 (3m 이상)
                                if wave_val >= 3.0:
                                    set_angle(servo1, 180); set_angle(servo2, 0)
                                else:
                                    set_angle(servo1, 0); set_angle(servo2, 180)

                            elif header == "RESCUE":
                                if lcd:
                                    lcd.clear(); lcd.write_string("!! RESCUE !!")
                                    lcd.cursor_pos=(1,0)
                                    try:
                                        lat, lon = value.split(",")
                                        lcd.write_string(f"{float(lat):.2f}, {float(lon):.2f}")
                                    except: lcd.write_string(value[:16])

                    save_log_land(0, anim, quake, is_tsu, shared_data["wave"], status_txt)
                    time.sleep(1.0)

                except (OSError, bluetooth.BluetoothError):
                    print("끊김. 재접속 대기..."); break
            client_sock.close()
        except: time.sleep(1)
    server_sock.close()

# =================================================================
# 4. 메인 루프
# =================================================================
def main():
    t = threading.Thread(target=bluetooth_thread_func)
    t.daemon = True; t.start()
    cap = cv2.VideoCapture(0)
    
    last_servo_state = 0
    print("🎥 [Main] 가동 ('a':자동, 'q':종료)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: continue

            # [A] 지진 시뮬레이션 & AI
            is_vib = GPIO.input(VIB_PIN)
            if shared_data["auto_mode"] or is_vib == 1:
                mag = round(random.uniform(3.5, 9.0), 1)
                shared_data["quake"] = mag
                shared_data["is_tsunami"] = predict_tsunami_ai(mag)
            else:
                shared_data["quake"] = 0.0
                shared_data["is_tsunami"] = 0

            # [B] 동물 감지
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img).resize(input_details[0]['shape'][1:3])
            input_data = np.expand_dims(np.array(img_pil)/255.0, axis=0).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]['index'])[0][0]
            
            class_id = 1 if out > 0.5 else 0
            label = LABELS[class_id]
            shared_data["animal"] = 1.0 if label == "Fish" else 0.0

            # [C] 서보 제어 (즉각 반응)
            target_state = 0
            if shared_data["wave"] >= 3.0: target_state = 1
            elif shared_data["animal"] > 0: target_state = 1
            
            if target_state != last_servo_state:
                if target_state == 1: 
                    for a in range(0, 181, 20): set_angle(servo1, a); set_angle(servo2, 180-a)
                else:
                    for a in range(180, -1, -20): set_angle(servo1, a); set_angle(servo2, 180-a)
                last_servo_state = target_state

            # [D] 화면
            color = (0,0,255) if shared_data["quake"] > 0 else (0,255,0)
            cv2.putText(frame, f"Fish: {label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Quake: {shared_data['quake']} (AI:{shared_data['is_tsunami']})", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            evac_txt = get_evac_msg(shared_data["wave"])
            cv2.putText(frame, f"{evac_txt} ({shared_data['wave']}m)", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            cv2.imshow("Land Node", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('a'): shared_data["auto_mode"] = not shared_data["auto_mode"]

    except KeyboardInterrupt: pass
    finally:
        shared_data["running"] = False
        cap.release(); cv2.destroyAllWindows()
        servo1.stop(); servo2.stop(); GPIO.cleanup()

if __name__ == "__main__":
    main()


