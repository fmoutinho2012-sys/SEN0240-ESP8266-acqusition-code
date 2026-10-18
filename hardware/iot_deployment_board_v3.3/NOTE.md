# IoT Deployment Board v3.3 (author's design)

Schematic and Gerber files of the microcontroller board used for the recordings. From the schematic: ESP-01 module, AMS1117 3.3 V regulator, battery connector, power jack (4.5–12 V), reset and flash buttons, serial header, and a sensor connector with supply, GPIO0 (I2C clock), GPIO2 (I2C data) and ground. The design and its bench tests are described in the author's book *ESP8266 + MicroPython* (see the paper's reference [13]).

Version 3.0 is described in the book; the schematic file is model 3.3. Model 3.3 is the board used for the recordings (confirmed by the author).

In the recordings the ADS1015 module took its 3.3 V from the board's regulator (fed from the 5.0 V bench supply or the 4.8 V Ni-Cd pack), and the sensor's 3.3 V was taken from the supply pins of the ADS1015 module. The 5 V pin of the sensor connector (CN7, the unregulated input rail in the schematic) was not used. The sensor ground goes directly to the board ground and meets the converter ground at a single point (star connection). The schematic has a single ground net.
