# -*- coding: utf-8 -*-
import smbus
import time
import numpy as np  
import keyboard
import matplotlib.pyplot as plt

ACCL_ADDR = 0x19
GYRO_ADDR = 0x69

bus = smbus.SMBus(1)

# 加速度センサの設定
# Select PMU_Range register, 0x0F(15)
#       0x03(03)    Range = +/- 2g
bus.write_byte_data(ACCL_ADDR, 0x0F, 0x03)
# Select PMU_BW register, 0x10(16)
#       0x08(08)    Bandwidth = 7.81 Hz
bus.write_byte_data(ACCL_ADDR, 0x10, 0x08)
# Select PMU_LPW register, 0x11(17)
#       0x00(00)    Normal mode, Sleep duration = 0.5ms
bus.write_byte_data(ACCL_ADDR, 0x11, 0x00)

time.sleep(0.5)

# ジャイロセンサの設定
# Select Range register, 0x0F(15)
#       0x04(04)    Full scale = +/- 125 degree/s
bus.write_byte_data(GYRO_ADDR, 0x0F, 0x04)
# Select Bandwidth register, 0x10(16)
#       0x07(07)    ODR = 100 Hz
bus.write_byte_data(GYRO_ADDR, 0x10, 0x07)
# Select LPM1 register, 0x11(17)
#       0x00(00)    Normal mode, Sleep duration = 2ms
bus.write_byte_data(GYRO_ADDR, 0x11, 0x00)
time.sleep(0.5)

def accl():
    xA = yA = zA = 0

    try:
        data = bus.read_i2c_block_data(0x19, 0x02, 6)
        # Convert the data to 12-bits
        xA = ((data[1] * 256) + (data[0] & 0xF0)) / 16
        if xA > 2047:
            xA -= 4096
        yA = ((data[3] * 256) + (data[2] & 0xF0)) / 16
        if yA > 2047:
            yA -= 4096
        zA = ((data[5] * 256) + (data[4] & 0xF0)) / 16
        if zA > 2047:
            zA -= 4096
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))

    return xA, yA, zA

def gyro():
    xG = yG = zG = 0

    try:
        data = bus.read_i2c_block_data(GYRO_ADDR, 0x02, 6)
        # Convert the data
        xG = (data[1] * 256) + data[0]
        if xG > 32767:
            xG -= 65536

        yG = (data[3] * 256) + data[2]
        if yG > 32767:
            yG -= 65536

        zG = (data[5] * 256) + data[4]
        if zG > 32767:
            zG -= 65536

    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))

    return xG, yG, zG

def main():
    data = np.empty((0, 6))
    Datacollect = False
    previous_shift= False
    print("Press 'SHIFT' to start/stop data collection and 'Ctrl+C' to exit.")
    
    plt.ion()  # Turn on interactive mode
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    x_data, y_data_accl, y_data_gyro = [], [], []

    try:
        while True:
            xAccl, yAccl, zAccl = accl()
            xGyro, yGyro, zGyro = gyro()
            if Datacollect:
                feature = np.array([xGyro, yGyro, zGyro,xAccl,yAccl,zAccl], dtype=float)
                data = np.vstack((data, feature))
                # Update data for plotting
                x_data.append(len(x_data))
                y_data_accl.append([xAccl, yAccl, zAccl])
                y_data_gyro.append([xGyro, yGyro, zGyro])

                # Plotting
                if len(x_data) > 100:  # Keep the plot window to the latest 100 points
                    x_data = x_data[-100:]
                    y_data_accl = y_data_accl[-100:]
                    y_data_gyro = y_data_gyro[-100:]

                axs[0].cla()
                axs[1].cla()
                axs[0].plot(x_data, [y[0] for y in y_data_accl], label='X Accl')
                axs[0].plot(x_data, [y[1] for y in y_data_accl], label='Y Accl')
                axs[0].plot(x_data, [y[2] for y in y_data_accl], label='Z Accl')
                axs[0].legend(loc='upper left')
                axs[0].set_title('Acceleration Data')
                axs[1].plot(x_data, [y[0] for y in y_data_gyro], label='X Gyro')
                axs[1].plot(x_data, [y[1] for y in y_data_gyro], label='Y Gyro')
                axs[1].plot(x_data, [y[2] for y in y_data_gyro], label='Z Gyro')
                axs[1].legend(loc='upper left')
                axs[1].set_title('Gyroscope Data')
                plt.pause(0.01)

            current_shift = keyboard.is_pressed('shift')
            if current_shift and not previous_shift:
                Datacollect = not Datacollect
                print("Data collection: {}".format(Datacollect))
                if not Datacollect:
                    print(data.shape)
                    filename = input("Enter the filename to save the data: ")
                    np.savetxt(filename + ".csv", data, delimiter=",")
                    print(f"Data saved to {filename}.csv")
            previous_shift=current_shift
    except KeyboardInterrupt:
        print("Exiting")
        plt.ioff()  # Turn off interactive mode
        plt.show()

if __name__ == "__main__":
    main()