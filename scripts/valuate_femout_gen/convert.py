#!/usr/bin/env python3

from numpy.polynomial import Polynomial as Poly
import numpy as np

# props given by Polynomial fitting: C, S, K
temp = np.array([300, 500, 800])
props_h = np.array([[200, 285, 330], 
                    [170, 210, 200], 
                    [  2, 1.2, 0.6]])
props_c = np.array([[700, 330, 285], 
                    [260, 240, 160], 
                    [0.7, 0.8, 1.2]])

props_h = [Poly.fit(temp, prop, deg=2, domain=[-100,100]) for prop in props_h]
props_c = [Poly.fit(temp, prop, deg=2, domain=[-100,100]) for prop in props_c]

T = np.linspace(300, 800, 101)
props_h = np.array([item(T) for item in props_h])
props_c = np.array([item(T) for item in props_c])

# save datas to files
np.savetxt('gen_input_h.txt', np.c_[T, props_h.T], fmt='%.4f', header='T, C, S, K')
np.savetxt('gen_input_c.txt', np.c_[T, props_c.T], fmt='%.4f', header='T, C, S, K')
