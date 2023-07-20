import numpy as np

def mpu9250_conv(acc_xyz, gyro_xyz, accel_sens, gyro_sens, acc_max_value, gyr_max_value):
    # convert to acceleration in g and gyro dps

    a_xyz = np.divide(acc_xyz, acc_max_value) * accel_sens
    w_xyz = np.divide(gyro_xyz, gyr_max_value) * gyro_sens

    return a_xyz, w_xyz








