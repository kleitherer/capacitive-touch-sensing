import time
import sys
import ADS1256
import spidev
import RPi.GPIO as GPIO
import matplotlib.pyplot as plt

"""
Author: Yuchi Hsu

Step 0.
Connect the P29 (GPIO21) on Raspberry Pi to the AD7 A pin on the ADC board.

Step 1. 
Run the code and check the frequency (sample rate). It should be only around a few hundred Hz (too slow).

Step 2. 
Set SETUP_SPI to True. Run the code again and check the frequency. It should be around 3500 Hz (still too slow).

Step 3. 
Set SET_CHANNEL to True. Run the code again and check the frequency. 
It should be around 20000 Hz. You could take a look at the implementation of the ADS1256_GetChannalValue and see 
why it's slower than the ADC.ADS1256_Read_ADC_Data.

Take a look at the `adc_value_vs_time.png` file.

Step 4.
Try to improve the GPIO signal by increasing the GPIO_FREQ_FACTOR (lower the frequency).

"""


SETUP_SPI = False 
SET_CHANNEL = False
FREQUENCY = 20000
GPIO_FREQ_FACTOR = 1 # The frequency of the GPIO signal is GPIO_FREQ_FACTOR times less than the frequency of the ADC signal


class FrequencyDetector:
    def __init__(self):
        self.current_time = time.time()
        self.last_time = self.current_time
        self.start_time = self.current_time
        self.count = 0
        
    def update(self):
        self.count += 1
        self.current_time = time.time()
        
    def realtime_update(self):
        self.last_time = self.current_time
        self.current_time = time.time()
        self.count += 1
       
        dt = self.current_time - self.last_time
        if dt > 0:
            freq = 1/dt
            sys.stdout.write("\r")
            sys.stdout.write(f"{freq:.4f} Hz       ")
            sys.stdout.flush()
    def get_freq(self):
        return self.count / (self.current_time - self.start_time)
    
class FrequencyRegulator:
    def __init__(self, frequency):
        self.frequency = frequency
        self.start_time = time.time()
        self.dt = 1/self.frequency
        self.next_pulse_time = self.start_time + self.dt
    
    def is_next_pulse_ok(self):
        if time.time() > self.next_pulse_time:
            self.next_pulse_time += self.dt
            return True
        return False
        

def main():
    
    # GPIO setup
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(21, GPIO.OUT)
    
    
    # ADC setup
    ADC = ADS1256.ADS1256()
    ADC.ADS1256_init()
    # ADC.ADS1256_ConfigADC(2, 0xF0)
    # ADC.ADS1256_WriteReg(0, 0x01) 
    
    # SPI setup
    if SETUP_SPI:
        SPI = spidev.SpiDev(0, 0)
        SPI.mode = 0b01
        SPI.max_speed_hz = 3000000
    
    freq_detector = FrequencyDetector()
    freq_regulator = FrequencyRegulator(FREQUENCY)
    
    gpio_freq_factor = GPIO_FREQ_FACTOR
    
    if SET_CHANNEL:
        ADC.ADS1256_SetChannal(7)
        ADC.ADS1256_WriteCmd(0xFC) #sync
        ADC.ADS1256_WriteCmd(0x00) #wakeup  
        
    adc_data = []
    time_data = []
    
    gpio_state = 0
    gpio_count = 0
    
    start_time = time.time()
    
    while(len(adc_data) < 500):
        if not freq_regulator.is_next_pulse_ok():
            continue
        
        freq_detector.update()
        
        if gpio_count == gpio_freq_factor:
            gpio_count = 0
            if gpio_state == 0:
                GPIO.output(21, GPIO.HIGH)
                gpio_state = 1
            else:
                GPIO.output(21, GPIO.LOW)
                gpio_state = 0
        gpio_count += 1
        
        if SET_CHANNEL:
            adc_value = ADC.ADS1256_Read_ADC_Data()*5.0/0x7fffff
        else:
            adc_value = ADC.ADS1256_GetChannalValue(7)*5.0/0x7fffff
            
        adc_data.append(adc_value)
        time_data.append(time.time() - start_time)
    print(f"Frequency: {freq_detector.get_freq():.4f} Hz")
    # plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(time_data[400:500], adc_data[400:500])
    plt.ylim(-1,6)
    plt.xlabel("Time (s)")
    plt.ylabel("ADC Value")
    plt.title("ADC Value vs Time")
    plt.grid(True)
    plt.savefig("adc_value_vs_time.png")
    

if __name__ == "__main__":
    main()