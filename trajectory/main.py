# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 20:26:42 2023

@author: NOA
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from quaternions import quaternRotate
from convert_raw_data import mpu9250_conv
from AHRS import AHRS


# Global system variables
#a_x, a_y, a_z = 0.0, 0.0, 0.0  # accelerometer measurements
#w_x, w_y, w_z = 0.0, 0.0, 0.0  # gyroscope measurements in rad/s


#recopilació de dades, cambiar el nom de l'arxiu pickle per a tractar amb els altres tests
with open('C:/Users/NOA/Documents/MATCAD/TFG/df_RGC.pkl', 'rb') as archivo:
    dataset = pickle.load(archivo)

dataset = dataset.drop(dataset[dataset['Indice_VEST'] == '-'].index)

#Quedar-se amb les columnes d'interés
dataframe = dataset[['acelerometro_x', 'acelerometro_y', 'acelerometro_z', 'giroscopo_x', 'giroscopo_y', 'giroscopo_z']]

#Calcular el màxim de cada columna per a normalitzar les dades
diccionario = {nombre_columna: dataframe[nombre_columna].max() for nombre_columna in dataframe.columns}

acc_max = [diccionario['acelerometro_x'], diccionario['acelerometro_y'], diccionario['acelerometro_z']]
gyr_max = [diccionario['giroscopo_x'], diccionario['giroscopo_y'], diccionario['giroscopo_z']]

#Obtenir el màxim per cada característica per separat
acc_max_value = np.max(acc_max)
gyr_max_value = np.max(gyr_max)

gyro_sens, acc_sens = 16, 2000 #MPU6050_start()

#dataset = dataframe.iloc[[0]] #agafar nomes la mostra d'1 pacient
acceleration_values = dataset[['acelerometro_x', 'acelerometro_y', 'acelerometro_z']].values
gyroscope_values = dataset[['giroscopo_x', 'giroscopo_y', 'giroscopo_z']].values

acceleration_values = np.array([[np.array(value) for value in row] for row in acceleration_values])
gyroscope_values = np.array([[np.array(value) for value in row] for row in gyroscope_values])

#Escalar els valors de l'acceleròmetre i el giròscop
acc_values_scaled, gyr_values_scaled = mpu9250_conv(acceleration_values,gyroscope_values, acc_sens, gyro_sens, acc_max_value, gyr_max_value)

#Aproximació de l'error de mesura del giròscop
#El calculem a partir de les dades del ROA que són gairebé estàtiques

gyro_data = dataset[['giroscopo_x', 'giroscopo_y', 'giroscopo_z']]
gyro_data = gyro_data.reset_index(drop=True)

#Recorrem totes les dades per a convertir la l'array en una matriu 3D
gyro_data_3d = []
for i in range(gyro_data.shape[0]):
    row_list = []
    for j in range(gyro_data.shape[1]):
        row_list.append(np.array(gyro_data.iloc[i, j]))
    gyro_data_3d.append(row_list)

# Passem a array per obtenir la forma: (n_filas, n_columnes, longitud_llista)
gyro_data_3d = np.array(gyro_data_3d)

#eliminem una dimensio per a calcular la desviacio estandard per totes les dades de cada eix
gyro_data_reshaped = gyro_data_3d.reshape(gyro_data_3d.shape[0]*gyro_data_3d.shape[2], gyro_data_3d.shape[1])
 
mean_per_col = np.mean(gyro_data_reshaped, axis=0)
data_escaled = gyro_data_reshaped - mean_per_col
std_per_col = np.std(data_escaled, axis=0)
std_per_col = np.divide(std_per_col, gyr_max_value) * gyro_sens
gyro_error = (np.sum(std_per_col)) / 3

##################START OF THE BUCLE per each sample

# System constants
deltat = 1/40 #40 Hz
gyroMeasError = gyro_error # gyroscope measurement error in rad/s (shown as 5 deg/s)
beta = math.sqrt(3.0 / 4.0) * gyroMeasError  # compute beta

#Matriu per emmagatzemar les trajectòries
n_trajectories = acc_values_scaled.shape[0]
n_points = len(acc_values_scaled[0][0])
n_dimensions = 3  # Coordenades X, Y, Z

trajectories_matrix = np.zeros((n_trajectories,n_dimensions, n_points))

for sample in range(acc_values_scaled.shape[0]):
    acc_xyz = acc_values_scaled[sample]
    gyr_xyz = gyr_values_scaled[sample]
    
    # Rotar el vector de referencia (eix X) utilitzant els quaternionss estimats
    rotated_vector = np.array([1.0, 0.0, 0.0])
    
    trajectory = []
    
    #Inicialització filtre AHRS: Attitude and Heading Reference System
    AHRSalgorithm = AHRS(SamplePeriod=1/40, Kp=1,Ki=1, KpInit=1)
    
    total_samples = acc_xyz.shape[1]  # 30 segons de dades, 1200 valors aprox.
    time = np.linspace(0, 30, total_samples)
    
    # Iterar a traves de les dades i estimar l'orientació en cada pas de temps
    for i in range(total_samples):
        AHRSalgorithm.Kp = 0

        #Actualització orientació
        AHRSalgorithm.UpdateIMU(gyr_xyz[:,i].flatten(), acc_xyz[:,i].flatten())
    
        quaternions = AHRSalgorithm.Quaternion
        rotated_vector = quaternRotate(rotated_vector, quaternions)
        
        #Afegir la posició actual de la IMU a la trajectòria
        if i == 0:
            trajectory.append(rotated_vector)
        else:
            trajectory.append(trajectory[-1] + rotated_vector * deltat)
    
    trajectory = np.array(trajectory)
    
    #emmagatzemar la trajetoria
    trajectories_matrix[sample, :, :] = trajectory.T
    
    
    # Plotejar la trajectòria en 3D   
    
    #obtenció ID's
    ids = dataset['id']
    ids = ids.reset_index(drop=True)
    
    #Obtenció índex VEST
    Índex_VEST = dataset['Indice_VEST'].astype(float)/100
    label = Índex_VEST.astype(np.float32)
    label = label.reset_index(drop=True)
    label = label*100
    
    #càlcul rang plots
    min_val_x = np.min(trajectories_matrix[:, 0, :])
    max_val_x = np.max(trajectories_matrix[:, 0, :])
    min_val_y = np.min(trajectories_matrix[:, 1, :])
    max_val_y = np.max(trajectories_matrix[:, 1, :])
    min_val_z = np.min(trajectories_matrix[:, 2, :])
    max_val_z = np.max(trajectories_matrix[:, 2, :])

    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='{}'.format(ids[sample]))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim(min_val_x, max_val_x)
    ax.set_ylim(min_val_y, max_val_y)
    ax.set_zlim(min_val_z, max_val_z)
    titol = 'Trajectòria durant el test de Romberg - Índex VEST: {}'.format(label[sample])
    plt.title(titol)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Define la ruta de la carpeta donde guardar los gráficos
    ruta_carpeta = "C:/Users/NOA/Documents/MATCAD/TFG/trajectories_IMU/RGC"
        
    nombre_archivo = "traj_{}.png".format(sample)
    
    if label[sample]<0.5:
        
      ruta_archivo = ruta_carpeta + '/positives/'
      
      # Crea la carpeta si no existeix
      if not os.path.exists(ruta_archivo):
          os.makedirs(ruta_archivo)
          
      ruta_archivo += nombre_archivo
      
    else:
        
      ruta_archivo = ruta_carpeta + '/negatives/'
      
      # Crea la carpeta si no existeix
      if not os.path.exists(ruta_archivo):
          os.makedirs(ruta_archivo)

      ruta_archivo += nombre_archivo
    
    plt.savefig(ruta_archivo)


with open('C:/Users/NOA/Documents/MATCAD/TFG/trajectories_IMU/df_RGC.pkl', 'wb') as file:
    pickle.dump(trajectories_matrix, file)
    
with open('C:/Users/NOA/Documents/MATCAD/TFG/trajectories_IMU/labels_RGC.pkl', 'wb') as file:
    pickle.dump(label, file)


















