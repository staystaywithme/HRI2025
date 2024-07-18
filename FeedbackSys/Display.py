import torch
import torch.nn as nn
from lstm_cnn_net import lstm_cnn
from dataprocess import reduce0, reducenoice, data301

import numpy as np

import smbus
import time
import struct
from gpiozero import MCP3008
import numpy as np

##########
# クラス
##########

class BNO055:
    ##########
    # クラス変数
    ##########
    BNO055_ADDRESS_A                = 0x28
    BNO055_ADDRESS_B                = 0x29
    BNO055_ID                       = 0xA0
    
    # Power mode settings
    POWER_MODE_NORMAL               = 0X00
    POWER_MODE_LOWPOWER             = 0X01
    POWER_MODE_SUSPEND              = 0X02
    
    # Operation mode settings
    OPERATION_MODE_CONFIG           = 0X00
    OPERATION_MODE_ACCONLY          = 0X01
    OPERATION_MODE_MAGONLY          = 0X02
    OPERATION_MODE_GYRONLY          = 0X03
    OPERATION_MODE_ACCMAG            = 0X04
    OPERATION_MODE_ACCGYRO           = 0X05
    OPERATION_MODE_MAGGYRO           = 0X06
    OPERATION_MODE_AMG              = 0X07
    OPERATION_MODE_IMUPLUS           = 0X08
    OPERATION_MODE_COMPASS           = 0X09
    OPERATION_MODE_M4G              = 0X0A
    OPERATION_MODE_NDOF_FMC_OFF      = 0X0B
    OPERATION_MODE_NDOF              = 0X0C
    
    # Output vector type
    VECTOR_ACCELEROMETER            = 0x08
    VECTOR_MAGNETOMETER              = 0x0E
    VECTOR_GYROSCOPE                = 0x14
    VECTOR_EULER                    = 0x1A
    VECTOR_LINEARACCEL              = 0x28
    VECTOR_GRAVITY                  = 0x2E
    
    # REGISTER DEFINITION START
    BNO055_PAGE_ID_ADDR             = 0X07
    
    BNO055_CHIP_ID_ADDR             = 0x00
    BNO055_ACCEL_REV_ID_ADDR         = 0x01
    BNO055_MAG_REV_ID_ADDR           = 0x02
    BNO055_GYRO_REV_ID_ADDR          = 0x03
    BNO055_SW_REV_ID_LSB_ADDR        = 0x04
    BNO055_SW_REV_ID_MSB_ADDR        = 0x05
    BNO055_BL_REV_ID_ADDR            = 0X06
    
    # Accel data register 
    BNO055_ACCEL_DATA_X_LSB_ADDR      = 0X08
    BNO055_ACCEL_DATA_X_MSB_ADDR      = 0X09
    BNO055_ACCEL_DATA_Y_LSB_ADDR      = 0X0A
    BNO055_ACCEL_DATA_Y_MSB_ADDR      = 0X0B
    BNO055_ACCEL_DATA_Z_LSB_ADDR      = 0X0C
    BNO055_ACCEL_DATA_Z_MSB_ADDR      = 0X0D
    
    # Mag data register 
    BNO055_MAG_DATA_X_LSB_ADDR       = 0X0E
    BNO055_MAG_DATA_X_MSB_ADDR       = 0X0F
    BNO055_MAG_DATA_Y_LSB_ADDR       = 0X10
    BNO055_MAG_DATA_Y_MSB_ADDR       = 0X11
    BNO055_MAG_DATA_Z_LSB_ADDR       = 0X12
    BNO055_MAG_DATA_Z_MSB_ADDR       = 0X13
    
    # Gyro data registers 
    BNO055_GYRO_DATA_X_LSB_ADDR      = 0X14
    BNO055_GYRO_DATA_X_MSB_ADDR      = 0X15
    BNO055_GYRO_DATA_Y_LSB_ADDR      = 0X16
    BNO055_GYRO_DATA_Y_MSB_ADDR      = 0X17
    BNO055_GYRO_DATA_Z_LSB_ADDR      = 0X18
    BNO055_GYRO_DATA_Z_MSB_ADDR      = 0X19
    
    # Euler data registers 
    BNO055_EULER_H_LSB_ADDR          = 0X1A
    BNO055_EULER_H_MSB_ADDR          = 0X1B
    BNO055_EULER_R_LSB_ADDR          = 0X1C
    BNO055_EULER_R_MSB_ADDR          = 0X1D
    BNO055_EULER_P_LSB_ADDR          = 0X1E
    BNO055_EULER_P_MSB_ADDR          = 0X1F
    
    # Quaternion data registers 
    BNO055_QUATERNION_DATA_W_LSB_ADDR        = 0X20
    BNO055_QUATERNION_DATA_W_MSB_ADDR        = 0X21
    BNO055_QUATERNION_DATA_X_LSB_ADDR        = 0X22
    BNO055_QUATERNION_DATA_X_MSB_ADDR        = 0X23
    BNO055_QUATERNION_DATA_Y_LSB_ADDR        = 0X24
    BNO055_QUATERNION_DATA_Y_MSB_ADDR        = 0X25
    BNO055_QUATERNION_DATA_Z_LSB_ADDR        = 0X26
    BNO055_QUATERNION_DATA_Z_MSB_ADDR        = 0X27
    
    # Linear acceleration data registers 
    BNO055_LINEAR_ACCEL_DATA_X_LSB_ADDR      = 0X28
    BNO055_LINEAR_ACCEL_DATA_X_MSB_ADDR      = 0X29
    BNO055_LINEAR_ACCEL_DATA_Y_LSB_ADDR      = 0X2A
    BNO055_LINEAR_ACCEL_DATA_Y_MSB_ADDR      = 0X2B
    BNO055_LINEAR_ACCEL_DATA_Z_LSB_ADDR      = 0X2C
    BNO055_LINEAR_ACCEL_DATA_Z_MSB_ADDR      = 0X2D
    
    # Gravity data registers 
    BNO055_GRAVITY_DATA_X_LSB_ADDR           = 0X2E
    BNO055_GRAVITY_DATA_X_MSB_ADDR           = 0X2F
    BNO055_GRAVITY_DATA_Y_LSB_ADDR           = 0X30
    BNO055_GRAVITY_DATA_Y_MSB_ADDR           = 0X31
    BNO055_GRAVITY_DATA_Z_LSB_ADDR           = 0X32
    BNO055_GRAVITY_DATA_Z_MSB_ADDR           = 0X33
    
    # Temperature data register 
    BNO055_TEMP_ADDR                 = 0X34
    
    # Status registers 
    BNO055_CALIB_STAT_ADDR           = 0X35
    BNO055_SELFTEST_RESULT_ADDR      = 0X36
    BNO055_INTR_STAT_ADDR            = 0X37
    
    BNO055_SYS_CLK_STAT_ADDR         = 0X38
    BNO055_SYS_STAT_ADDR             = 0X39
    BNO055_SYS_ERR_ADDR              = 0X3A
    
    # Unit selection register 
    BNO055_UNIT_SEL_ADDR             = 0X3B
    BNO055_DATA_SELECT_ADDR          = 0X3C
    
    # Mode registers 
    BNO055_OPR_MODE_ADDR             = 0X3D
    BNO055_PWR_MODE_ADDR             = 0X3E
    
    BNO055_SYS_TRIGGER_ADDR          = 0X3F
    BNO055_TEMP_SOURCE_ADDR          = 0X40
    
    # Axis remap registers 
    BNO055_AXIS_MAP_CONFIG_ADDR      = 0X41
    BNO055_AXIS_MAP_SIGN_ADDR        = 0X42
    
    # SIC registers 
    BNO055_SIC_MATRIX_0_LSB_ADDR     = 0X43
    BNO055_SIC_MATRIX_0_MSB_ADDR     = 0X44
    BNO055_SIC_MATRIX_1_LSB_ADDR     = 0X45
    BNO055_SIC_MATRIX_1_MSB_ADDR     = 0X46
    BNO055_SIC_MATRIX_2_LSB_ADDR     = 0X47
    BNO055_SIC_MATRIX_2_MSB_ADDR     = 0X48
    BNO055_SIC_MATRIX_3_LSB_ADDR     = 0X49
    BNO055_SIC_MATRIX_3_MSB_ADDR     = 0X4A
    BNO055_SIC_MATRIX_4_LSB_ADDR     = 0X4B
    BNO055_SIC_MATRIX_4_MSB_ADDR     = 0X4C
    BNO055_SIC_MATRIX_5_LSB_ADDR     = 0X4D
    BNO055_SIC_MATRIX_5_MSB_ADDR     = 0X4E
    BNO055_SIC_MATRIX_6_LSB_ADDR     = 0X4F
    BNO055_SIC_MATRIX_6_MSB_ADDR     = 0X50
    BNO055_SIC_MATRIX_7_LSB_ADDR     = 0X51
    BNO055_SIC_MATRIX_7_MSB_ADDR     = 0X52
    BNO055_SIC_MATRIX_8_LSB_ADDR     = 0X53
    BNO055_SIC_MATRIX_8_MSB_ADDR     = 0X54
    
    # Accelerometer Offset registers 
    ACCEL_OFFSET_X_LSB_ADDR          = 0X55
    ACCEL_OFFSET_X_MSB_ADDR          = 0X56
    ACCEL_OFFSET_Y_LSB_ADDR          = 0X57
    ACCEL_OFFSET_Y_MSB_ADDR          = 0X58
    ACCEL_OFFSET_Z_LSB_ADDR          = 0X59
    ACCEL_OFFSET_Z_MSB_ADDR          = 0X5A
    
    # Magnetometer Offset registers 
    MAG_OFFSET_X_LSB_ADDR            = 0X5B
    MAG_OFFSET_X_MSB_ADDR            = 0X5C
    MAG_OFFSET_Y_LSB_ADDR            = 0X5D
    MAG_OFFSET_Y_MSB_ADDR            = 0X5E
    MAG_OFFSET_Z_LSB_ADDR            = 0X5F
    MAG_OFFSET_Z_MSB_ADDR            = 0X60
    
    # Gyroscope Offset register s
    GYRO_OFFSET_X_LSB_ADDR           = 0X61
    GYRO_OFFSET_X_MSB_ADDR           = 0X62
    GYRO_OFFSET_Y_LSB_ADDR           = 0X63
    GYRO_OFFSET_Y_MSB_ADDR           = 0X64
    GYRO_OFFSET_Z_LSB_ADDR           = 0X65
    GYRO_OFFSET_Z_MSB_ADDR           = 0X66
    
    # Radius registers 
    ACCEL_RADIUS_LSB_ADDR            = 0X67
    ACCEL_RADIUS_MSB_ADDR            = 0X68
    MAG_RADIUS_LSB_ADDR              = 0X69
    MAG_RADIUS_MSB_ADDR              = 0X6A
    
    # REGISTER DEFINITION END
    
    def __init__(self, bus, address=BNO055_ADDRESS_A, mode=OPERATION_MODE_NDOF):
        self._bus = bus
        self._address = address
        self._mode = mode
        self._write_byte(BNO055.BNO055_OPR_MODE_ADDR, BNO055.OPERATION_MODE_CONFIG)
        time.sleep(0.025)
        self._write_byte(BNO055.BNO055_PWR_MODE_ADDR, BNO055.POWER_MODE_NORMAL)
        self._write_byte(BNO055.BNO055_PAGE_ID_ADDR, 0)
        self._write_byte(BNO055.BNO055_SYS_TRIGGER_ADDR, 0x00)
        self._write_byte(BNO055.BNO055_OPR_MODE_ADDR, self._mode)
        time.sleep(0.025)

    def _write_byte(self, register, value):
        self._bus.write_byte_data(self._address, register, value)

    def _read_vector(self, vector_type, count=3):
        data = self._bus.read_i2c_block_data(self._address, vector_type, count*2)
        result = []
        for i in range(count):
            result.append(struct.unpack_from('<h', bytearray(data), i*2)[0] / 16.0)
        return result

    def read_euler(self):
        return self._read_vector(BNO055.VECTOR_EULER)

    def read_gyroscope(self):
        return self._read_vector(BNO055.VECTOR_GYROSCOPE)

    def read_linear_acceleration(self):
        return self._read_vector(BNO055.VECTOR_LINEARACCEL)

##########
# メイン
##########
def main():
    pot= MCP3008(0)
    I2C_BUS = 1
    sensor1_address = BNO055.BNO055_ADDRESS_A  # 0x28
    sensor2_address = BNO055.BNO055_ADDRESS_B  # 0x29
    bus = smbus.SMBus(I2C_BUS)
    sensor1 = BNO055(bus, sensor1_address)
    sensor2 = BNO055(bus, sensor2_address)

    data = np.empty((0, 13))

   
    while True:
        #euler1 = sensor1.read_euler()
        gyroscope1 = sensor1.read_gyroscope()
        linear_accel1 = sensor1.read_linear_acceleration()
        #euler2 = sensor2.read_euler()
        gyroscope2 = sensor2.read_gyroscope()
        linear_accel2 = sensor2.read_linear_acceleration()
        #euler1 = ','.join(map(str,euler1))
        gyroscope1 = ','.join(map(str,gyroscope1))
        linear_accel1 = ','.join(map(str,linear_accel1))
        #euler2 = ','.join(map(str,euler2))
        gyroscope2 = ','.join(map(str,gyroscope2))
        linear_accel2 = ','.join(map(str,linear_accel2))
        pot_value = f"{pot.value:.3f}"
        time.sleep(0.01)
        
        feature= np.array([ *gyroscope1.split(','), *linear_accel1.split(','),
                            *gyroscope2.split(','), *linear_accel2.split(','), pot_value],dtype=float)
        timing = feature[13]
        print("{}, {}, {}".format(gyroscope1, gyroscope2, pot_value))
        while timing != 0:
            data= np.vstack((data, feature))
            print(data.shape)
        
        data= reduce0(data)
        data= reducenoice(data)
        data= data301(data)
        print(data.shape)

        #predict
        model = lstm_cnn()
        model.load_state_dict(torch.load('model.pth'))
        model.eval()
        data = torch.tensor(data, dtype=torch.float32)
        data = data.unsqueeze(0)
        output = model(data)
        print(output)
        

if __name__ == "__main__":
    main()

