from machine import I2C, Pin, freq
import time
import gc
import network
import struct

# 1. Optimización Extrema del Sistema
network.WLAN(network.STA_IF).active(False) # Apagar WiFi para maximizar estabilidad de RAM
gc.collect()
freq(160000000) # CPU a 160MHz para garantizar el timing de 1ms

# Configuración I2C: ESP8266 (SCL=GPIO0, SDA=GPIO2)
i2c = I2C(scl=Pin(0), sda=Pin(2), freq=400000)
ADS_ADDR = 0x48

# CONFIGURACIÓN TÉCNICA PARA JUECES / ARTÍCULO:
# Registro 0x01 (Config): b'\x42\x83'
# - MUX: 100 (Diferencial AINP=AIN0, AINN=AIN1) -> MODO DIFERENCIAL ACTIVO
# - PGA: 001 (+/- 4.096V) -> Resolución: 2.0 mV/bit (12-bit)
# - MODE: 0 (Modo Continuo)
# - DR: 100 (1600 SPS internos para evitar latencia en muestreo de 1000Hz)
i2c.writeto_mem(ADS_ADDR, 0x01, b'\x42\x83')

# 2. Configuración del Experimento
DURACION_SEG = 80
SPS_TARGET = 1000
TAMANO_BUFFER = 100 # Bloques de 100ms para escritura eficiente en Flash
TOTAL_MUESTRAS = DURACION_SEG * SPS_TARGET

filename = "fist_gesture_raw.bin"

# Pre-asignación de memoria para evitar pausas del Garbage Collector
buffer = bytearray(TAMANO_BUFFER * 2) 
buf_temp = bytearray(2)
read_into = i2c.readfrom_mem_into
us = time.ticks_us

print(f"--- ADS1015: CAPTURA DIFERENCIAL SINCRONIZADA v4 ---")
print(f"Modo: Diferencial A0-A1 | Frecuencia: {SPS_TARGET} Hz")
print(f"Protocolo: 80s (10 repeticiones 5s ON / 3s OFF)")
time.sleep(2)

try:
    with open(filename, "wb") as f:
        print("Adquiriendo datos... Mantenga el ritmo del protocolo.")
        start_time = time.ticks_ms()
        timestamp_proxima_muestra = us()
        
        # Bucle principal por bloques
        for i in range(TOTAL_MUESTRAS // TAMANO_BUFFER):
            for j in range(TAMANO_BUFFER):
                # Sincronización precisa de 1000 microsegundos
                while time.ticks_diff(timestamp_proxima_muestra, us()) > 0:
                    pass
                
                # Captura de datos I2C
                read_into(ADS_ADDR, 0x00, buf_temp)
                
                # Almacenamiento rápido en buffer de RAM
                ptr = j * 2
                buffer[ptr] = buf_temp[0]
                buffer[ptr+1] = buf_temp[1]
                
                # Programar siguiente muestra
                timestamp_proxima_muestra = time.ticks_add(timestamp_proxima_muestra, 1000)
            
            # Volcado de bloque a memoria Flash
            f.write(buffer)
            
            # Feedback de progreso
            if i % 50 == 0:
                print(f"Progreso: {int((i * TAMANO_BUFFER / TOTAL_MUESTRAS) * 100)}%")

        end_time = time.ticks_ms()

    # --- REPORTE TÉCNICO FINAL ---
    
    # Cálculo del promedio de las ÚLTIMAS 20 LECTURAS (2 ciclos @ 100Hz)
    suma_ultimas_20 = 0
    # Extraemos del último buffer guardado en RAM
    for k in range(20):
        # Localizar la muestra k empezando desde el final del buffer
        p = (TAMANO_BUFFER - 1 - k) * 2
        v_raw = (buffer[p] << 8) | buffer[p+1]
        
        # Manejo de signo (entero de 16 bits)
        if v_raw > 32767: v_raw -= 65536
        
        # Ajuste ADS1015: El dato está en los bits 15-4
        v_12bit = v_raw >> 4
        # Multiplicar por LSB (2.0 mV para rango de 4.096V)
        suma_ultimas_20 += (v_12bit * 2.0)

    promedio_final_mv = suma_ultimas_20 / 20
    dur_real = (end_time - start_time) / 1000

    print("-" * 50)
    print("EXPERIMENTO FINALIZADO CON ÉXITO")
    print("-" * 50)
    print(f"Frecuencia real (SPS): {TOTAL_MUESTRAS / dur_real:.2f} Hz")
    print(f"Duración real: {dur_real:.2f} segundos")
    print(f"Muestras promediadas (Reporte): 20 (2 periodos @ 100Hz)")
    print(f"Valor DC de reposo (Offset): {promedio_final_mv:.2f} mV")
    print(f"Archivo guardado: {filename}")
    print("-" * 50)

except Exception as e:
    print(f"\n[ERROR]: {e}")