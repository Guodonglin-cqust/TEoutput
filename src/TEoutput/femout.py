from abc import ABC, abstractmethod
from scipy.integrate import cumtrapz, trapz
import numpy as np
import logging

from .utils import AttrDict, Metric

logger = logging.getLogger(__name__)

class Segment(ABC):
    Length = 1          # mm
    Area = 100          # mm^2
    Perimeter = 40      # mm
    Ngrid = 101         # 1
    _xi = []            # mm
    _dT_x = []          # K/mm
    _T_x = []           # K
    
    @property
    def endtemp(self):
        Ta, *_, Tb = self._T_x
        return Ta, Tb
    
    @property
    def gradtemp(self):
        dTa, *_, dTb = self._dT_x
        return dTa, dTb
    
    def __str__(self):
        clsname = self.__class__.__name__
        dsp = '{}(Length={}, Area={}, Perimeter={})'
        fmt = dsp.format(clsname, self.Length, self.Area, self.Perimeter)
        return fmt
    
    @property
    @abstractmethod
    def profile(self):
        R_cum = ...         # integral(Rho, dL)/A           , mOhm
        S_cum = ...         # integral(S*(-dT/dx), dL)      , mV
        K_cum = ...         # integral(1/Rth*(-dT/dx), dL)*A, W*mm
        Rx_cum = ...        # integral(x*Rho, dL/A)         , mOhm.mm
        Sx_cum = ...        # integral(x*S*(-dT/dx), dL)    , mV.mm
        ST_cum = ...        # integral(T*S, dL)             , mV.mm
        return R_cum, S_cum, K_cum, Rx_cum, Sx_cum, ST_cum
    
    @abstractmethod
    def heatflow(self, I):
        qa = ...        # W
        qb = ...        # W
        return qa, qb
    
    @abstractmethod
    def phix(self, I):
        phi_r = ...         # W/A = V
        ux = ...            # K/mm/A, same as dT_x/I
        vx = ...            # K/W/mm, same as dT_x/Jphi_r
        ux_cum = ...        # K/A, same as T_x/I
        vx_cum = ...        # K/W, same as T_x/Jphi_r
        return phi_r, ux, vx, ux_cum, vx_cum
    
class BulkSegment(Segment):
    def __init__(self, props, isPoly=False, 
                 Length=1, Area=100, Perimeter=40,
                 Ngrid=101):
        if isPoly: 
            Rho, S, Rth = props
            self._Rho = Rho
            self._S = S
            self._Rth = Rth
            self._datas = props
        else:
            T, C, S, K = props
            Rho = 1E4/C
            Rth = 1/K
            self._Rho = lambda Ti: np.interp(Ti, T, Rho)
            self._S   = lambda Ti: np.interp(Ti, T, S)
            self._Rth = lambda Ti: np.interp(Ti, T, Rth)
        self.Length = Length
        self.Area = Area
        self.Perimeter = Perimeter
        self.Ngrid = Ngrid
        self._xi = np.linspace(0, Length, Ngrid)    # mm
        self._T_x = np.ones(Ngrid)                  # K
        self._dT_x = np.zeros(Ngrid)                # K/mm
    
    def Rho_S_Rth(self, x):
        Tx = np.interp(x, self._ix, self._T_x)
        return self._Rho(Tx), self._S(Tx), self._Rth(Tx)
    
    def Rho(self, x):
        return self.Rho_S_Rth(x)[0]
    
    def S(self, x):
        return self.Rho_S_Rth(x)[1]
    
    def Rth(self, x):
        return self.Rho_S_Rth(x)[2]
    
    def C(self, x):
        return 1E4 / self.Rho_S_Rth(x)[0]
    
    def K(self, x):
        return 1/self.Rho_S_Rth(x)[2]
    
    @property
    def profile(self):
        Area = self.Area                # mm^2
        xi = self._xi                   # mm
        T_x = self._T_x                 # K
        dT_x = (-1)*self._dT_x          # K/mm
        
        Rho_x = self._Rho(T_x)          # uOhm.m <--> mOhm.mm
        R_cum = trapz(Rho_x, xi)/Area   # mOhm
        Rx_cum = trapz(xi*Rho_x, xi)/Area   # mOhm.mm
        
        S_x = 1E-3 * self._S(T_x)       # uV/K --> mV/K
        S_cum = trapz(S_x*dT_x, xi)     # mV
        Sx_cum = trapz(xi*S_x*dT_x, xi) # mV.mm
        ST_cum = trapz(S_x*T_x, xi)     # mV.mm
        
        K_x = 1E-3 / self._Rth(T_x)     # W/(m.K) --> W/(mm.K)
        K_cum = trapz(K_x*dT_x, xi)*Area    # W*mm
        
        # R_cum = ...         # integral(Rho, dL)/A           , mOhm
        # S_cum = ...         # integral(S*(-dT/dx), dL)      , mV
        # K_cum = ...         # integral(1/Rth*(-dT/dx), dL)*A, W*mm
        # Rx_cum = ...        # integral(x*Rho, dL/A)         , mOhm.mm
        # Sx_cum = ...        # integral(x*S*(-dT/dx), dL)    , mV.mm
        # ST_cum = ...        # integral(T*S, dL)             , mV.mm
        return R_cum, S_cum, K_cum, Rx_cum, Sx_cum, ST_cum
    
    def heatflow(self, I):
        Area = self.Area                # mm^2
        Ta, Tb = self.endtemp           # K
        dTa, dTb = self.gradtemp        # K/mm
        Rth_a = 1E3 * self._Rth(Ta)     # m.K/W --> mm.K/W
        Rth_b = 1E3 * self._Rth(Tb)
        S_a = 1E-6 * self._S(Ta)        # uV/K  --> V/K
        S_b = 1E-6 * self._S(Tb)
        qa = S_a*Ta*I-dTa/Rth_a*Area    # W
        qb = S_b*Tb*I-dTb/Rth_b*Area    
        return qa, qb
    
    def phix(self, I):
        Area = self.Area                # mm^2
        xi = self._xi                   # mm
        T_x = self._T_x                 # K
        dT_x = self._dT_x               # K/mm
        S_x = 1E-6 * self._S(T_x)       # uV/K   --> V/K
        Rho_x = 1E-3 * self._Rho(T_x)   # uOhm.m --> Ohm.mm
        Rth_x = 1E3 * self._Rth(T_x)    # m.K/W  --> mm.K/W
        S_cum = cumtrapz(S_x*dT_x, xi, initial=0)   # V
        Rho_cum = cumtrapz(Rho_x, xi, initial=0)    # Ohm.mm^2
        phi_r = (S_cum+I*Rho_cum/Area)          # W/A = V
        ux = (S_x*T_x-phi_r)*Rth_x/Area         # K/mm/A
        vx = Rth_x/Area                         # K/W/mm 
        ux_cum = cumtrapz(ux, xi, initial=0)    # K/A
        vx_cum = cumtrapz(vx, xi, initial=0)    # K/W
        return phi_r, ux, vx, ux_cum, vx_cum

class LayerSegment(Segment):
    Rc = None   # uOhm.cm^2
    Kc = None   # cm^2.K/W
    def __init__(self, Rc=None, Kc=None, S=None, 
                 Area=100, Perimeter=40):
        self.isPoly = False
        self._Rc = Rc       # uOhm.cm^2
        self._Kc = Kc       # cm^2.K/W
        self._S = S         # uV/K
        self.Length = 0
        self.Area = Area
        self.Perimeter = Perimeter
        self.Ngrid = 2
        self._xi = np.zeros(2)      # mm
        self._T_x = np.ones(2)      # K
        self._dT_x = np.zeros(2)    # W, same as AKdT_x or Jphi_r
    
    @property
    def Rc(self):
        value = self._Rc
        if (value is None) or (value <= 0):
            return 0
        else:
            return value
    
    @property
    def Kc(self):
        value = self._Kc
        if (value is None) or (value <= 0):
            return 0
        else:
            return value    
    
    @property
    def S(self):
        value = self._S
        if (value is None) or (value <= 0):
            return 0
        else:
            return value
    
    @property
    def profile(self):
        Area = self.Area / 100          # mm^2 --> cm^2
        R_cum = 1E-3 * self.Rc/Area     # uOhm.cm^2/cm^2 --> mOhm
        Rx_cum = 0                      # mOhm.mm
        
        Ta, Tb = self.endtemp           # K
        S = 1E-3 * self.S               # uV/K --> mV/K
        S_cum = S*(Ta-Tb)               # mV
        Sx_cum = 0                      # mV.mm
        ST_cum = 0                      # mV.mm

        K_cum = 0                       # K = q*L/dT => 0 (L=0)
        
        # R_cum = ...         # integral(Rho, dL)/A           , mOhm
        # S_cum = ...         # integral(S*(-dT/dx), dL)      , mV
        # K_cum = ...         # integral(1/Rth*(-dT/dx), dL)*A, W/mm
        # Rx_cum = ...        # integral(x*Rho, dL/A)         , mOhm.mm
        # Sx_cum = ...        # integral(x*S*(-dT/dx), dL)    , mV.mm
        # ST_cum = ...        # integral(T*S, dL)             , mV.mm
        return R_cum, S_cum, K_cum, Rx_cum, Sx_cum, ST_cum
    
    def heatflow(self, I):
        Area = self.Area                # mm^2
        Ta, Tb = self.endtemp           # K
        AKdTa, AKdTb = self.gradtemp    # W
        S = 1E-6 * self.S               # uV/K      --> V/K
        Rc = 1E-4 * self.Rc             # uOhm.cm^2 --> Ohm.mm^2
        qa = I*(S*Ta-1/2*I*Rc/Area)-AKdTa   # W
        qb = I*(S*Tb+1/2*I*Rc/Area)-AKdTb   # W
        return qa, qb
        
    def phix(self, I):
        Area = self.Area
        Ta, Tb = self.endtemp
        T_x = self._T_x
        S = 1E-6 * self.S   # uV/K      --> V/K
        Rc = 1E-4 * self.Rc # uOhm.cm^2 --> Ohm.mm^2
        Kc = 1E2 * self.Kc  # cm^2.K/W  --> mm^2.K/W
        Ngrid = self.Ngrid
        phi_r = np.zeros(Ngrid)
        ux_cum = np.zeros(Ngrid)
        vx_cum = np.zeros(Ngrid)
        
        phi_r[-1] = (S*(Tb-Ta)+I*Rc/Area)   # W/A = V
        ux = S*T_x - phi_r                  # W/A = V
        vx = np.ones_like(phi_r)            # 1, different with super Segment()
        ux_cum[-1] = (S*Ta-1/2*I*Rc/Area)*Kc/Area   # K/A
        vx_cum[-1] = Kc/Area                        # K/W, same with the super()
        return phi_r, ux, vx, ux_cum, vx_cum

class Element(ABC):
    def __init__(self, Ta=None, Tb=None, segments=None, CSA=100):
        self.endtemp = [Ta, Tb]
        self.CSA = CSA
        if segments is None:
            self._segments = []
        else:
            self._segments = segments
        self._Areas = []        # ratio of seg.Area/CSA
        self._Perimeter = []    # reserved attr.
        self._Ltot = 0
        
    @property
    def segments(self):
        # read-only
        return self._segments
    
    def add_segment(self, segment):
        self._segments.append(segment)
    
    def build(self, CSA=None, dT_x=None, T_x=None):
        clsname = self.__class__.__name__
        # check Ta, Tb
        Ta, Tb = self.endtemp
        if (Ta is None) or (Tb is None):
            dsp = 'End temperature are required to build {}'
            raise ValueError(dsp.format(clsname))
        else:
            logger.info('End temperatures are {} K, {} K'.format(Ta, Tb))
        
        if CSA is not None:
            self.CSA = CSA
            logger.info('Cross-sectional area (CSA) is set to {} mm^2'.format(CSA))
        else:
            logger.info('Cross-sectional area (CSA) is {} mm^2'.format(self.CSA))
        
        # check segments, get _Areas
        segs = self.segments
        if (segs is None) or (len(segs) == 0):
            dsp = 'Segments are required to build {}'
            raise ValueError(dsp.format(clsname))
        elif not all(map(lambda seg: isinstance(seg, Segment), segs)):
            dsp = 'A Segment() object is required to build {}'
            raise ValueError(dsp.format(clsname))
        else:
            rAreas, Ltot = [], 0
            logger.info('Segments list:')
            for seg in segs:
                rAreas.append(seg.Area/self.CSA)
                Ltot += seg.Length
                logger.info('    %s', str(seg))
            self._Areas = rAreas
            self._Ltot = Ltot
            logger.info('Total length is {} mm'.format(Ltot))
            
        # initialize dT_x, T_x
        if (dT_x is None) or (T_x is None):
            intercept = Ta
            slope = (Tb-Ta)/Ltot
            coor = 0
            dT_x, T_x = [], []
            for seg in segs:
                dT_x.append(slope * np.ones_like(seg._xi))
                T_x.append(slope*(coor+seg._xi)+intercept)
                coor = coor + seg.Length
            dsp = 'Initialize temperature in linear distribution'
            logger.info(dsp)
        else:
            dsp = 'Initialize temperature using customized value'
            logger.info(dsp)
        self._update_temp(dT_x, T_x)
    
    def simulate(self, I, CSA=None,
                  maxiter=30, miniter=1,
                  mixing=1.0, tol=1E-4):
        if CSA is not None:
            self.CSA = CSA
            logger.info('Value of CSA is set to {} mm^2'.format(CSA))
        
        dT_x, T_x = [], []
        for seg, area in zip(self.segments, self._Areas):
            seg.Area = self.CSA*area
            dT_x.append(seg._dT_x)
            T_x.append(seg._T_x)
        logger.info('Update areas of segments')
        
        dsp = 'Using %s._get_metric() to evaluate itol'
        logger.debug(dsp, self.__class__.__name__)
        
        rst = [dT_x, T_x,]
        rsts = []
        succeed = True
        for epoch in range(maxiter+1):
            Ix = self._parse_current(I)
            Jphi_Ta, Jphi_Tb, Vdiff, dT_x2, T_x2 = self._get_phi_t2(Ix)
            rstx = [Jphi_Ta, Jphi_Tb, Vdiff, Ix]
            rst.extend(rstx)
            if len(rsts) == 0:
                itol = [np.inf,]
                header = '{:>6s}{:>10s}{:>10s}{:>10s}{:>10s}{:>12s}'
                args= ('epoch', 'Jphi_Ta', 'Jphi_Tb', 'Vdiff', 'Ix', 'itol')
                logger.info(header.format(*args))
            else:
                itol = self._get_metric(rsts[-1][:-1], rst)
            dsp = '{:6d}'+'{:10.4f}'*len(rstx)+'{:12.4E}'*len(itol)
            logger.info(dsp.format(epoch, *rstx, *itol))
            rst.append(itol)
            rsts.append(rst)

            if (itol[0] > tol) or (epoch < miniter):
                rst = [dT_x2, T_x2]
                self._update_temp(dT_x2, T_x2, mixing)
            else:
                break
        else:
            succeed = False

        return Jphi_Ta, Jphi_Tb, Vdiff, succeed, rsts
        
    def _get_phi_t2(self, I):
        phixs = [seg.phix(I) for seg in self.segments]
            
        ux_cums, vx_cums, phi_cums, phi_vx_cums = [], [], [], []
        phi_cum_v = 0
        for phix in phixs:
            phi_r, _, _, ux_cum, vx_cum = phix
            ux_cums.append(ux_cum[-1])
            vx_cums.append(vx_cum[-1])
            phi_cums.append(phi_cum_v)
            phi_vx_cums.append(phi_cum_v*vx_cum[-1])
            phi_cum_v += phi_r[-1]
        
        Ta, Tb = self.endtemp
        Jphi_Ta = ((Ta-Tb)+I*(sum(ux_cums)-sum(phi_vx_cums)))/sum(vx_cums)
        Jphi_Tb = Jphi_Ta + I*phi_cum_v
        Vdiff = -1E3*phi_cum_v    # phi_Ta - phi_Tb in mV
        
        T_cum_v = Ta
        dT_x2 = []
        T_x2 = []
        for phix, phi_cum_i in zip(phixs, phi_cums):
            _, ux, vx, ux_cum, vx_cum = phix
            Jphi_Ta_i = Jphi_Ta+I*phi_cum_i
            dT_x_i = I*ux - Jphi_Ta_i*vx
            T_x_i = T_cum_v + I*ux_cum - Jphi_Ta_i*vx_cum
            T_cum_v = T_x_i[-1]
            dT_x2.append(dT_x_i)
            T_x2.append(T_x_i)
        return Jphi_Ta, Jphi_Tb, Vdiff, dT_x2, T_x2
    
    def _get_metric(self, rst, rst2):
        # rst: ['dT_x', 'T_x', 'Jphi_Ta', 'Jphi_Tb', 'Vdiff', 'Ix']
        Jphi = rst[2:4]          # [Jphi_Ta, Jphi_Tb]
        Jphi2 = rst2[2:4]
        itol = Metric.RMSE(Jphi, Jphi2)
        return [itol,]             # W
    
    def _update_temp(self, dT_x, T_x, mixing=None):
        for seg, dT_x_i, T_x_i in zip(self.segments, dT_x, T_x):
            if mixing is None:
                seg._dT_x = dT_x_i
                seg._T_x = T_x_i
            else:
                seg._dT_x += mixing*(dT_x_i-seg._dT_x)
                seg._T_x  += mixing*(T_x_i -seg._T_x)
    
    @abstractmethod
    def _parse_current(self, I):
        # value, YitaMax, PowerMax, OpenCircuit, ShortCircuie
        Ix = ...                # parse Ix from input I
        return Ix
    
    @abstractmethod
    def get_prfs(self):
        # profiles, such Voc, Rin, Isc, Tp, mopt for generator
        prfs = AttrDict()
        return prfs
    
    @classmethod
    @abstractmethod
    def valuate(cls):
        pass

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
    
    def build(self, dT_x=None, T_x=None):
        logger.info('Begin building process ...')
        Th, Tc = self.endtemp
        if Th < Tc:
            raise RuntimeError('Th should be higher than Tc')
        output = super().build(dT_x, T_x)
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
        Jphi_Th, Jphi_Tc, Vdiff, succeed, rsts = super().simulate(*args)
        if not succeed:
            results = None
            logger.info('Stop simulating for reaching maxiter')
        else:
            results = AttrDict()
            results['Vout'] = Vdiff
            results['Qhot'] = Jphi_Th
            results['Pout'] = Jphi_Th - Jphi_Tc
            results['Yita'] = 100 * (1-Jphi_Tc/Jphi_Th)
            logger.info('Finish simulating process')
        return results
    
    def _get_metric(self, rst, rst2):
        A0 = self.CSA/100    # mm^2 to cm^2
        itol = np.array(super()._get_metric(rst, rst2))
        return itol/A0       # W/cm^2
    
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
        Ltot = self._Ltot
        L_cum_v = 0
        R_cums, S_cums, K_cums, Rx_cums, Sx_cums, ST_cums = 0, 0, 0, 0, 0, 0
        for seg in self.segments:
            R_cum, S_cum, K_cum, Rx_cum, Sx_cum, ST_cum = seg.profile
            R_cums += R_cum                             # mOhm
            S_cums += S_cum                             # mV
            K_cums += K_cum                             # W*mm
            Rx_cums += (L_cum_v*R_cum + Rx_cum)         # mOhm.mm
            Sx_cums += L_cum_v*S_cum + Sx_cum           # mV.mm
            ST_cums += ST_cum                           # mV.mm
            L_cum_v += seg.Length                       # mm
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
    def valuate(cls, Th, Tc, segments, mode='MaxYita'):
        gen = cls(Th, Tc, segments)
        gen.build()
        results = gen.simulate(I=mode)
        return results

class Couple(ABC):
    # share the same (absoluate) current 
    def __init__(self, elements=None):
        raise NotImplementedError

class Cascade():
    def __init__(self, modules=None):
        raise NotImplementedError
