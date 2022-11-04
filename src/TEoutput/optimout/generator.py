import numpy as np
import logging

from .core import Element
from ..utils import AttrDict


logger = logging.getLogger(__name__)

class GenElement(Element):
    def __init__(self, Th=None, Tc=None, segments=None, CSA=100):
        super().__init__(Ta=Th, Tb=Tc,
                         segments=segments,
                         CSA=CSA)
    @property
    def Th(self):
        return self.endtemp[0]
    
    @Th.setter
    def Th(self, value):
        self.endtemp[0] = value
    
    @property
    def Tc(self):
        return self.endtemp[1]
    
    @Tc.setter
    def Tc(self, value):
        self.endtemp[1] = value
    
    def build(self, CSA=None, dT_x=None, T_x=None):
        logger.info('Begin building process ...')
        Th, Tc = self.endtemp
        if Th < Tc:
            raise RuntimeError('Th should be higher than Tc')
        output = self._build(CSA, dT_x, T_x)
        logger.info('Finish building process')
        return output

    def simulate(self, I=0, CSA=None, 
                 maxiter=30, miniter=1, 
                 mixing=0.9, tol=1E-4):
        if isinstance(I, str):
            # YitaMax, PowerMax, OpenCircuit, ShortCircuit
            I_str = I[0].lower()
            if I_str == 'o':
                mode = 'open-circuit'
            elif I_str == 's':
                mode = 'short-circuit'
            elif I_str == 'p':
                mode = 'Max.Pout'
            elif I_str == 'y':
                mode = 'Max.Yita'
            else:
                raise ValueError('Invalid value of I')
        else:
            # Work under assigned current
            mode = 'I={} A'.format(I)
        logger.info('Begin simulating where %s ...', mode)
            
        args = [I, CSA, maxiter, miniter, mixing, tol]
        Jphi_Th, Jphi_df, Vo_df, Ix, succeed, rsts = self._simulate(*args)
        if not succeed:
            results = None
            logger.info('Stop simulating for reaching maxiter')
        else:
            results = AttrDict()
            Vout = (-1)*Vo_df       # mV
            Pout = 1E-3 * Ix*Vout   # W
            results['I'] = Ix
            results['Vout'] = Vout
            results['Qhot'] = Jphi_Th
            results['Pout'] = Pout
            results['Yita'] = 100 * (Pout/Jphi_Th)
            logger.info('Finish simulating process')
        return results
    
    def _parse_current(self, I):
        # parse Ix from input I
        # value, YitaMax, PowerMax, OpenCircuit, ShortCircuie
        if isinstance(I, str):
            # YitaMax, PowerMax, OpenCircuit, ShortCircuie
            I_str = I[0].lower()
            if I_str == 'o':
                Ix = 0
            else:
                prfs = self.get_prfs()
                Isc = prfs['Isc']
                mopt = prfs['mopt']
                if I_str == 's':
                    Ix = Isc
                elif I_str == 'p':
                    Ix = Isc/2
                elif I_str == 'y':
                    Ix = Isc/(1+mopt)
                else:
                    raise ValueError('Invalid value of I')
        else:
            Ix = I
        return Ix
    
    def get_prfs(self):
        Ta, Tb = self.endtemp
        deltaT = Ta-Tb
        
        props = ['R', 'S', 'K', 'Rx', 'Sx', 'ST']
        cums = self._get_cums(props)
        
        Ltot = cums['Ltot']
        R_cums = cums['R']
        S_cums = cums['S']*(-1)
        K_cums = cums['K']*(-1)
        Rx_cums = cums['Rx']
        Sx_cums = cums['Sx']*(-1)
        ST_cums = cums['ST']
        
        prfs = AttrDict()
        prfs['Voc'] = S_cums            # mV
        prfs['Rin'] = R_cums            # mOhm
        prfs['Isc'] = S_cums/R_cums     # A
        p1 = ST_cums/Ltot/S_cums        # 1, => Tp/DT from Seb. coef.
        p2 = 1-Sx_cums/Ltot/S_cums      # 1, => 1/2
        p3 = 1-Rx_cums/Ltot/R_cums      # 1, => 1/2
        Tpr = p1+p2-p3                  # 1, => Tp/DT
        prfs['Tp'] = Tpr*deltaT         # K, Tp
        ZTeng = 1E-3*S_cums*S_cums/(R_cums*K_cums/Ltot)  # 1, => Zeng*DT
        ZTp = ZTeng*Tpr                 # 1, => Zeng*Tp
        prfs['mopt'] = np.sqrt(1+ZTp)
        return prfs     # Voc, Rin, Isc, Tp, mopt
    
    @classmethod
    def valuate(cls, Th, Tc, segments, mode='YitaMax'):
        gen = cls(Th, Tc, segments)
        gen.build()
        results = gen.simulate(I=mode)
        return results