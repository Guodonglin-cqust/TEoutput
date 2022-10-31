#!/usr/bin/env python3

from TEoutput.femout import BulkSegment, LayerSegment, GenElement
from TEoutput.utils import get_root_logger
from numpy.polynomial import Polynomial as Poly
import numpy as np
import os


############# Default Setting #############     
fileinput_h = 'gen_input_h.txt'
fileinput_c = 'gen_input_c.txt'
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

# props read from files: T, C, S, K
fileinput_h = 'gen_input_h.txt'
fileinput_c = 'gen_input_c.txt'
props_h = np.loadtxt(fileinput_h, unpack=True, ndmin=2)
props_c = np.loadtxt(fileinput_c, unpack=True, ndmin=2)
ZT_h = 1E-10 * props_h[2]*props_h[2]*props_h[1]/props_h[3]*props_h[0]
ZT_c = 1E-10 * props_c[2]*props_c[2]*props_c[1]/props_c[3]*props_c[0]

nstep = int(len(ZT_h)/2)
logger.info('Thermoelectirc figure-of-merit:')
logger.info('     T: %s', '  '.join(['{:8.4f}'.format(item) for item in props_h[0,::nstep]]))
logger.info('  ZT_h: %s', '  '.join(['{:8.4f}'.format(item) for item in ZT_h[::nstep]]))
logger.info('  ZT_c: %s', '  '.join(['{:8.4f}'.format(item) for item in ZT_c[::nstep]]))

TEM_h = BulkSegment(props_h, Length=Lh)
TEM_c = BulkSegment(props_c, Length=Lc)

segments = [contact_Th, TEM_h, contact_Mid, TEM_c, contact_Tc]

results = GenElement.valuate(Th, Tc, segments, mode)
logger.info('Results:')
for key, value in results.items():
    logger.info('{:>8s}: {}'.format(key, value))
logger.info('(DONE)')
