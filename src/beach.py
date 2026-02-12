import sys
import time
import serial
import socket
import smbus2
import pynmea2
import numpy as np
import RPi.GPIO as GPIO
import tflite_runtime.interpreter as tflite
import csv
from datetime import datetime
from ctypes import c_short

# --- 설정 ---
LAND_MAC = "2C:CF:67:8C:25:C1"
SCALER_MEAN = [1015.20, 15.53, 75.28, 5.90, 0.0, 0.0]
SCALER_SCALE = [8.02, 8.49, 14.04, 3.14, 1.0, 1.0]
I2C_BUS = 1
ADDR_BME = 0x76; ADDR_MPU = 0x68; PIN_COLLISION = 17
MODEL_PATH = "tsunami_model.tflite"
LOG_FILE = "sea_log.csv"

try: bus = smbus2.SMBus(I2C_BUS)
except: sys.exit(1)
GPIO.setmode(GPIO.BCM); GPIO.setup(PIN_COLLISION, GPIO.IN, pull_up_down=GPIO.PUD_UP)
try: ser = serial.Serial("/dev/serial0", 9600, timeout=0.5)
except: ser = None
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except: sys.exit(1)

try:
    with open(LOG_FILE, 'x', newline='') as f:
        csv.writer(f).writerow(['Time', 'Temp', 'Press', 'Shake', 'Lat', 'Lon', 'Quake_In', 'Wave_Out', 'Msg_Type'])
except: pass

# --- 함수 ---
def read_bme280_safe():
    try: import my_bme280; return my_bme280.readBME280All(ADDR_BME)
    except: return 20.0, 1013.0, 50.0

def read_mpu6050_shake():
    try:
        data = bus.read_i2c_block_data(ADDR_MPU, 0x3B, 6)
        x = c_short((data[0]<<8)+data[1]).value; y = c_short((data[2]<<8)+data[3]).value; z = c_short((data[4]<<8)+data[5]).value
        return abs(((x**2+y**2+z**2)**0.5/16384.0)-1.0)*10.0
    except: return 0.0

def predict(sensor_vals, land_vals):
    raw = np.array([sensor_vals['press'], sensor_vals['temp'], sensor_vals['hum'], sensor_vals['shake'], land_vals['quake'], land_vals['animal']])
    norm = (raw - np.array(SCALER_MEAN)) / np.array(SCALER_SCALE)
    input_tensor = np.array([norm], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    base = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
    
    # 하이브리드 보정
    q = land_vals['quake']
    corr = 0.0
    if q >= 3.0: corr = (q - 3.0) * 2.0
    if q >= 7.0: corr *= 1.2
    return max(0.0, round(base + corr, 2))

def save_log(s_val, l_val, wave, msg_type):
    now = datetime.now().strftime("%H:%M:%S")
    with open(LOG_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([now, f"{s_val['temp']:.1f}", f"{s_val['press']:.0f}", f"{s_val['shake']:.2f}",
                                f"{s_val['lat']:.4f}", f"{s_val['lon']:.4f}", l_val['quake'], wave, msg_type])

# --- 메인 ---
def main():
    try: bus.write_byte_data(ADDR_MPU, 0x6B, 0)
    except: pass
    print(f"\n>>> 바다 노드 (효율화 모드) <<<")
    
    sensor_vals = {'temp':0, 'press':0, 'hum':0, 'shake':0, 'lat':0.0, 'lon':0.0}
    land_vals = {'quake':0.0, 'animal':0.0}
    btn_press_start = None; is_rescue_mode = False

    while True:
        client_sock = None
        try:
            print("[BT] 육지 노드 연결 시도...")
            client_sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
            client_sock.connect((LAND_MAC, 1))
            print("[BT] 연결 성공!")

            while True:
                try:
                    # 1. 수신
                    raw = client_sock.recv(1024).decode().strip()
                    if not raw: break
                    print(f"[수신] {raw}")
                    
                    if raw.startswith("WARNING"):
                        parts = raw.split(',')
                        land_vals['animal'] = float(parts[1])
                        land_vals['quake'] = float(parts[2])
                        tsunami_alert = True
                    else: # SAFE
                        land_vals['animal'] = 0.0
                        land_vals['quake'] = 0.0
                        tsunami_alert = False

                    # 2. 센싱 (항상 수행 - 데이터 축적용)
                    t, p, h = read_bme280_safe()
                    sensor_vals.update({'temp':t, 'press':p, 'hum':h, 'shake':read_mpu6050_shake()})
                    if ser:
                        cnt=0
                        while ser.in_waiting > 0 and cnt<10:
                            try:
                                l = ser.readline().decode('utf-8', errors='ignore').strip()
                                if l.startswith('$GNGGA') or l.startswith('$GPGGA'):
                                    m = pynmea2.parse(l)
                                    if m.latitude > 0: sensor_vals['lat']=m.latitude; sensor_vals['lon']=m.longitude
                            except: pass
                            cnt+=1

                    # 3. 버튼 확인
                    if GPIO.input(PIN_COLLISION) == 0:
                        if btn_press_start is None: btn_press_start = time.time()
                        elif (time.time() - btn_press_start) > 1.0: is_rescue_mode = True
                    else: btn_press_start = None

                    # 4. 송신 (조건부)
                    if is_rescue_mode:
                        # [1순위] 구조
                        s_lat = sensor_vals['lat'] if sensor_vals['lat']>0 else 37.24
                        s_lon = sensor_vals['lon'] if sensor_vals['lon']>0 else 131.86
                        msg = f"RESCUE:{s_lat:.4f},{s_lon:.4f}"
                        print(f"🚨 [긴급] 구조신호 전송")
                        save_log(sensor_vals, land_vals, 0, "RESCUE")
                        
                    elif tsunami_alert:
                        # [2순위] 쓰나미 경보 시에만 파고 계산해서 보냄
                        wave = predict(sensor_vals, land_vals)
                        msg = f"WAVE:{wave:.2f}"
                        print(f"🌊 [위험] 파고 {wave:.2f}m 전송")
                        save_log(sensor_vals, land_vals, wave, "WAVE_ALERT")
                        
                    else:
                        # [3순위] 평시 - 파고 안 보냄 (절약)
                        msg = "STATUS:SAFE"
                        print(f"✅ [안전] 상태 양호 (전송 생략)")
                        save_log(sensor_vals, land_vals, 0, "STATUS_OK")

                    client_sock.send(msg.encode())

                except (BrokenPipeError, OSError): break
        except: time.sleep(3)
        finally: 
            if client_sock: client_sock.close()

if __name__ == "__main__":
    main()

