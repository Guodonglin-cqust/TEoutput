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
    
    @abstractmethod
    def get_cums(self):
        R_cum = ...         # integral(1/C, dL)/A           , mOhm
        S_cum = ...         # integral(S*(dT/dx), dL)       , mV
        K_cum = ...         # integral(K*(dT/dx), dL)*A     , W*mm
        Rx_cum = ...        # integral(x/C, dL/A)           , mOhm.mm
        Sx_cum = ...        # integral(x*S*(dT/dx), dL)     , mV.mm
        ST_cum = ...        # integral(T*S, dL)             , mV.mm
        return R_cum, S_cum, K_cum, Rx_cum, Sx_cum, ST_cum
    
    @abstractmethod
    def heatflow(self, I):
        qa = ...        # W
        qb = ...        # W
        return qa, qb
    
    @abstractmethod
    def phix(self, I):
        Jphi_r = ...        # W
        ux = ...            # K/mm  , same as dT_x
        vx = ...            # K/W/mm, same as dT_x/Jphi_r
        ux_cum = ...        # K  , same as T_x
        vx_cum = ...        # K/W, same as T_x/Jphi_r
        vdf = ...           # mV
        return Jphi_r, ux, vx, ux_cum, vx_cum, vdf
    
class BulkSegment(Segment):
    def __init__(self, props, isPoly=False, 
                 Length=1, Area=100, Perimeter=40,
                 Ngrid=101):
        if isPoly: 
            C, S, K = props
            self._C = C
            self._S = S
            self._K = K
            self._datas = props
        else:
            T, C, S, K = props
            self._C = lambda Ti: np.interp(Ti, T, C)
            self._S = lambda Ti: np.interp(Ti, T, S)
            self._K = lambda Ti: np.interp(Ti, T, K)
        self.Length = Length
        self.Area = Area
        self.Perimeter = Perimeter
        self.Ngrid = Ngrid
        self._xi = np.linspace(0, Length, Ngrid)    # mm
        self._T_x = np.ones(Ngrid)                  # K
        self._dT_x = np.zeros(Ngrid)                # K/mm
    
    def props_x(self, x):
        Tx = np.interp(x, self._ix, self._T_x)
        return self._C(Tx), self._S(Tx), self._K(Tx)
    
    def C(self, x):
        return self.props_x(x)[0]
    
    def S(self, x):
        return self.props_x(x)[1]
    
    def K(self, x):
        return self.props_x(x)[2]
    
    def Rho(self, x):
        return 1E4 / self.props_x(x)[0]
    
    def Rth(self, x):
        return 1/self.props_x(x)[2]
    
    def get_cums(self):
        Area = self.Area                # mm^2
        xi = self._xi                   # mm
        T_x = self._T_x                 # K
        dT_x = self._dT_x               # K/mm
        
        Rho_x = 1E4 / self._C(T_x)      # S/cm --> uOhm.m <--> mOhm.mm
        R_cum = trapz(Rho_x, xi)/Area   # mOhm
        Rx_cum = trapz(xi*Rho_x, xi)/Area   # mOhm.mm
        
        S_x = 1E-3 * self._S(T_x)       # uV/K --> mV/K
        S_cum = trapz(S_x*dT_x, xi)     # mV
        Sx_cum = trapz(xi*S_x*dT_x, xi) # mV.mm
        ST_cum = trapz(S_x*T_x, xi)     # mV.mm
        
        K_x = 1E-3 * self._K(T_x)       # W/(m.K) --> W/(mm.K)
        K_cum = trapz(K_x*dT_x, xi)*Area    # W*mm
        
        # R_cum = ...         # integral(1/C, dL)/A           , mOhm
        # S_cum = ...         # integral(S*(dT/dx), dL)       , mV
        # K_cum = ...         # integral(K*(dT/dx), dL)*A     , W*mm
        # Rx_cum = ...        # integral(x/C, dL/A)           , mOhm.mm
        # Sx_cum = ...        # integral(x*S*(dT/dx), dL)     , mV.mm
        # ST_cum = ...        # integral(T*S, dL)             , mV.mm
        return R_cum, S_cum, K_cum, Rx_cum, Sx_cum, ST_cum
    
    def heatflow(self, I):
        Area = self.Area                # mm^2
        Ta, Tb = self.endtemp           # K
        dTa, dTb = self.gradtemp        # K/mm
        K_a = 1E3 * self._K(Ta)         # m.K/W --> mm.K/W
        K_b = 1E3 * self._K(Tb)
        S_a = 1E-6 * self._S(Ta)        # uV/K  --> V/K
        S_b = 1E-6 * self._S(Tb)
        qa = S_a*Ta*I-K_a*dTa*Area      # W
        qb = S_b*Tb*I-K_b*dTb*Area    
        return qa, qb
    
    def phix(self, I):
        Area = self.Area                # mm^2
        xi = self._xi                   # mm
        T_x = self._T_x                 # K
        dT_x = self._dT_x               # K/mm
        S_x = 1E-6 * self._S(T_x)       # uV/K   --> V/K
        Rho_x = 1E1 / self._C(T_x)      # S/cm   --> Ohm.cm  --> Ohm.mm
        K_x = 1E-3 * self._K(T_x)       # W/(m.K) --> W/(mm.K)
        S_cum = cumtrapz(S_x*dT_x, xi, initial=0)   # V
        Rho_cum = cumtrapz(Rho_x, xi, initial=0)    # Ohm.mm^2
        phi_r = (S_cum+I*Rho_cum/Area)          # W/A = V
        Jphi_r = I*phi_r                        # W
        ux = (I*S_x*T_x-Jphi_r)/(Area*K_x)      # K/mm
        vx = 1/(Area*K_x)                       # K/W/mm 
        ux_cum = cumtrapz(ux, xi, initial=0)    # K
        vx_cum = cumtrapz(vx, xi, initial=0)    # K/W
        vdf = 1E3 * phi_r[-1]                   # mV
    
        # Jphi_r = ...        # W
        # ux = ...            # K/mm  , same as dT_x
        # vx = ...            # K/W/mm, same as dT_x/Jphi_r
        # ux_cum = ...        # K  , same as T_x
        # vx_cum = ...        # K/W, same as T_x/Jphi_r
        # vdf = ...           # mV
        return Jphi_r, ux, vx, ux_cum, vx_cum, vdf

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
    
    def get_cums(self):
        Area = self.Area / 100          # mm^2 --> cm^2
        R_cum = 1E-3 * self.Rc/Area     # uOhm.cm^2/cm^2 --> mOhm
        Rx_cum = 0                      # mOhm.mm
        
        Ta, Tb = self.endtemp           # K
        S = 1E-3 * self.S               # uV/K --> mV/K
        S_cum = S*(Tb-Ta)               # mV
        Sx_cum = 0                      # mV.mm
        ST_cum = 0                      # mV.mm

        K_cum = 0                       # K = q*L/dT => 0 (L=0)
        
        # R_cum = ...         # integral(1/C, dL)/A           , mOhm
        # S_cum = ...         # integral(S*(dT/dx), dL)       , mV
        # K_cum = ...         # integral(K*(dT/dx), dL)*A     , W*mm
        # Rx_cum = ...        # integral(x/C, dL/A)           , mOhm.mm
        # Sx_cum = ...        # integral(x*S*(dT/dx), dL)     , mV.mm
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
        Jphi_r = I*phi_r                    # W
        ux = I*S*T_x - Jphi_r               # W
        vx = np.ones_like(Jphi_r)           # 1, different with super Segment()
        ux_cum[-1] = I*(S*Ta-1/2*I*Rc/Area)*Kc/Area # K
        vx_cum[-1] = Kc/Area                # K/W, same with the super()
        vdf = 1E3 * phi_r[-1]               # mV
        
        # Jphi_r = ...        # W
        # ux = ...            # K/mm  , same as dT_x
        # vx = ...            # K/W/mm, same as dT_x/Jphi_r
        # ux_cum = ...        # K  , same as T_x
        # vx_cum = ...        # K/W, same as T_x/Jphi_r
        # vdf = ...           # mV
        return Jphi_r, ux, vx, ux_cum, vx_cum, vdf

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

        return Jphi_Ta, Jphi_Tb, Vdiff, Ix, succeed, rsts
        
    def _get_phi_t2(self, I):
        phixs = [seg.phix(I) for seg in self.segments]
            
        ux_cums, vx_cums, Jphi_cums, Jphi_vx_cums = [], [], [], []
        Jphi_cum_v, Vdiff = 0, 0
        for phix in phixs:
            Jphi_r, _, _, ux_cum, vx_cum, vdf = phix
            ux_cums.append(ux_cum[-1])
            vx_cums.append(vx_cum[-1])
            Jphi_cums.append(Jphi_cum_v)
            Jphi_vx_cums.append(Jphi_cum_v*vx_cum[-1])
            Jphi_cum_v += Jphi_r[-1]
            Vdiff += vdf
        
        Ta, Tb = self.endtemp
        Jphi_Ta = ((Ta-Tb)+(sum(ux_cums)-sum(Jphi_vx_cums)))/sum(vx_cums)
        Jphi_Tb = Jphi_Ta + Jphi_cum_v
        
        T_cum_v = Ta
        dT_x2 = []
        T_x2 = []
        for phix, Jphi_cum_i in zip(phixs, Jphi_cums):
            _, ux, vx, ux_cum, vx_cum, _ = phix
            Jphi_Ta_i = Jphi_Ta+Jphi_cum_i
            dT_x_i = ux - Jphi_Ta_i*vx
            T_x_i = T_cum_v + ux_cum - Jphi_Ta_i*vx_cum
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
    
    def _get_cums(self):
        L_cum_v = 0
        R_cums, S_cums, K_cums, Rx_cums, Sx_cums, ST_cums = 0, 0, 0, 0, 0, 0
        for seg in self.segments:
            R_cum, S_cum, K_cum, Rx_cum, Sx_cum, ST_cum = seg.get_cums()
            R_cums += R_cum                             # mOhm
            S_cums += S_cum                             # mV
            K_cums += K_cum                             # W*mm
            Rx_cums += L_cum_v*R_cum + Rx_cum           # mOhm.mm
            Sx_cums += L_cum_v*S_cum + Sx_cum           # mV.mm
            ST_cums += ST_cum                           # mV.mm
            L_cum_v += seg.Length                       # mm
        return R_cums, S_cums, K_cums, Rx_cums, Sx_cums, ST_cums
    
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
        Jphi_Th, Jphi_Tc, Vdiff, Ix, succeed, rsts = super().simulate(*args)
        if not succeed:
            results = None
            logger.info('Stop simulating for reaching maxiter')
        else:
            results = AttrDict()
            results['I'] = Ix
            results['Vout'] = (-1)*Vdiff
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
        R_cums, S_cums, K_cums, Rx_cums, Sx_cums, ST_cums = self._get_cums()
        S_cums, Sx_cums, K_cums = (-1)*np.array([S_cums, Sx_cums, K_cums])
        
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

class Couple(ABC):
    # share the same (absoluate) current 
    def __init__(self, elements=None):
        raise NotImplementedError

class Cascade():
    def __init__(self, modules=None):
        raise NotImplementedError
