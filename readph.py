import board
import adafruit_bitbangio as bitbangio
import time
import sys
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# Setup
i2c = bitbangio.I2C(board.D27, board.D17)
ads = ADS.ADS1115(i2c)

def read_voltage(channel):
    buf = list()
    for i in range(10): # Take 10 samples
        buf.append(channel.voltage)
    buf.sort() # Sort samples and discard highest and lowest
    buf = buf[2:-2]
    avg = (sum(map(float,buf))/6) # Get average value from remaining 6
    return round(avg,2)

if __name__ == '__main__':
    channel = AnalogIn(ads, ADS.P0)
    print(read_voltage(channel) / 3.3 * 14)