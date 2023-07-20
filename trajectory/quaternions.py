# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 20:46:08 2023

@author: NOA
"""
import numpy as np

def quaternRotate(v, q):

    """
    Rotar un vector v utilizando un cuaternión q.

    Parameters:
        v (numpy array): Vector tridimensional a rotar.
        q (numpy array): Cuaternión que representa la rotación.

    Returns:
        numpy array: Vector v rotado.
    """
    def quaternProd(a, b):
        """
        Multiplicar dos cuaterniones a y b.

        Parameters:
            a (numpy array): Cuaternión a.
            b (numpy array): Cuaternión b.

        Returns:
            numpy array: Resultado de la multiplicación de los cuaterniones.
        """
        ab = np.zeros(4)
        ab[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
        ab[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
        ab[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
        ab[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
        return ab

    def quaternConj(q):
        """
        Calcular el conjugado de un cuaternión q.

        Parameters:
            q (numpy array): Cuaternión.

        Returns:
            numpy array: Conjugado del cuaternión q.
        """
        # Implementación del conjugado de un cuaternión
        qConj = np.zeros(4)
        qConj[0] = q[0]
        qConj[1:] = -q[1:]
        return qConj
    
    
    v = np.array(v)
    q = np.array(q)

    # Convertir el vector v en un cuaternión con parte escalar igual a cero
    v_quat = np.zeros(4)
    v_quat[1:] = v

    # Multiplicar el cuaternión de rotación q con el cuaternión v_quat
    rotated_v_quat = quaternProd(quaternProd(q, v_quat), quaternConj(q))

    # Extraer las componentes imaginarias del resultado y devolver el vector rotado
    rotated_v = rotated_v_quat[1:]

    return rotated_v