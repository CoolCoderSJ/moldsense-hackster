import time, os
bag = input("Enter bag number: ")
expired = input("Is this pre- or post- mold innoculation? [y/N]")
if expired == "y": expired = 1
else: expired = 0

import bme680
sensor = bme680.BME680(bme680.I2C_ADDR_SECONDARY)
sensor.set_humidity_oversample(bme680.OS_2X)
sensor.set_pressure_oversample(bme680.OS_4X)
sensor.set_temperature_oversample(bme680.OS_8X)
sensor.set_filter(bme680.FILTER_SIZE_3)
sensor.set_gas_status(bme680.ENABLE_GAS_MEAS)
sensor.set_gas_heater_temperature(320)
sensor.set_gas_heater_duration(150)
sensor.select_gas_heater_profile(0)


print('Calibration data:')
for name in dir(sensor.calibration_data):

    if not name.startswith('_'):
        value = getattr(sensor.calibration_data, name)

        if isinstance(value, int):
            print('{}: {}'.format(name, value))

print('\n\nInitial reading:')
for name in dir(sensor.data):
    value = getattr(sensor.data, name)

    if not name.startswith('_'):
        print('{}: {}'.format(name, value))


import board
import adafruit_bitbangio as bitbangio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

i2c = bitbangio.I2C(board.D27, board.D17)
ads = ADS.ADS1115(i2c)

channel = AnalogIn(ads, ADS.P0)

def read_voltage(channel):
    buf = list()
    for i in range(10): # Take 10 samples
        buf.append(channel.voltage)
    buf.sort() # Sort samples and discard highest and lowest
    buf = buf[2:-2]
    avg = (sum(map(float,buf))/6) # Get average value from remaining 6
    return round(avg,2)


def collect():
    print("bag,expired,temperature,gas,humidity,pressure,voltage", file=open("testing.csv", "w"))

    for i in range(16):
        voltage = read_voltage(channel)
        value = voltage
        if sensor.get_sensor_data():
            print(f"{bag},{expired},{sensor.data.temperature},{sensor.data.gas_resistance},{sensor.data.humidity-5},{sensor.data.pressure},{value}", file=open("testing.csv", "a"))
            print(f"{i} Temperature: {sensor.data.temperature}, Gas Resistance: {sensor.data.gas_resistance} ohms, Humidity: {sensor.data.humidity}%, Pressure: {sensor.data.pressure}hPa, pH: {value}")
        time.sleep(2)

    print("All values recorded")