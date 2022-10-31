#!/usr/bin/env python3

from TEoutput.femout import BulkSegment, LayerSegment, GenElement
from TEoutput.utils import get_root_logger
from numpy.polynomial import Polynomial as Poly
import numpy as np
import os


############# Default Setting #############
T = np.array([300, 500, 800])
props_h = np.array([[200, 285, 330],    # unit: S/cm
                    [170, 210, 200],    # unit: uV/K
                    [  2, 1.2, 0.6]])   # unit: W/(m.K)
props_c = np.array([[700, 330, 285], 
                    [260, 240, 160], 
                    [0.7, 0.8, 1.2]])
mode = 'YitaMax'
Th, Tc = 800, 300
Lh, Lc =   2,   1               # unit: mm
Rc_h, Rc_mid, Rc_c = 10, 0, 10  # unit: uOhm.cm^2
Kc_h, Kc_mid, Kc_c =  1, 2,  0  # unit: cm^2.K/W
###########################################

# config logger
LEVELNUM = 20
logger = get_root_logger(level=LEVELNUM)

# header
sname = '{} - {}'.format(logger.name, os.path.basename(__file__))
logger.info('Calculate TE element performance of generator: %s', sname)

# contact (Rc, Kc) in uOhm.cm^2, Kc in cm^2.K/W
contact_Th  = LayerSegment(Rc=Rc_h,   Kc=Kc_h)
contact_Mid = LayerSegment(Rc=Rc_mid, Kc=Kc_mid)
contact_Tc  = LayerSegment(Rc=Rc_c,   Kc=Kc_c)

# props given by Polynomial fitting: C, S, K
ZT_h = 1E-10 * props_h[1]*props_h[1]*props_h[0]/props_h[2]*T
ZT_c = 1E-10 * props_c[1]*props_c[1]*props_c[0]/props_c[2]*T

logger.info('Thermoelectirc figure-of-merit:')
logger.info('     T: %s', '  '.join(['{:8.4f}'.format(item) for item in T]))
logger.info('  ZT_h: %s', '  '.join(['{:8.4f}'.format(item) for item in ZT_h]))
logger.info('  ZT_c: %s', '  '.join(['{:8.4f}'.format(item) for item in ZT_c]))

props_h = [Poly.fit(T, prop, deg=2, domain=[-100,100]) for prop in props_h]
props_c = [Poly.fit(T, prop, deg=2, domain=[-100,100]) for prop in props_c]
TEM_h = BulkSegment(props_h, isPoly=True, Length=Lh)
TEM_c = BulkSegment(props_c, isPoly=True, Length=Lc)

segments = [contact_Th, TEM_h, contact_Mid, TEM_c, contact_Tc]

results = GenElement.valuate(Th, Tc, segments, mode)
logger.info('Results:')
for key, value in results.items():
    logger.info('{:>8s}: {}'.format(key, value))
logger.info('(DONE)')
