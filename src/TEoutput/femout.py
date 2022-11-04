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
    def get_cums(self, x0=0):
        # cums = {
        #     'R': ... ,  # integral(1/C, dL)/A           , mOhm
        #     'S': ... ,  # integral(S*(dT/dx), dL)       , mV
        #     'K': ... ,  # integral(K*(dT/dx), dL)*A     , W*mm
        #     'W': ... , 
        #     'Rx':... ,  # integral(x/C, dL/A)           , mOhm.mm
        #     'Sx':... ,  # integral(x*S*(dT/dx), dL)     , mV.mm
        #     'ST':... ,  # integral(T*S, dL)             , mV.mm
        #     'Wx':... ,
        # }
        # return AttrDict(cums)
        pass
    
    @abstractmethod
    def heatflow(self, I):
        qa = ...        # W
        qb = ...        # W
        return qa, qb
    
    @abstractmethod
    def phix(self, I, Jphi0=0):
        # rst = {
        #     'Jphi_r': ... ,         # np.vstack([I*S_cum, I*IR_cum,])   , W
        #     'Jphi_x': ... ,         # Jphi_r.sum(axis=0)                , W
        #     'ux': ... ,             # (Pi*I-(Jphi0+Jphi_x))/K_x         , K/mm
        #     'vx': ... ,             # 1/K_x                             , K/W/mm
        #     'ux_cum': ... ,         # cumtrapz(ux, xi, initial=0)       , K
        #     'vx_cum': ... ,         # cumtrapz(vx, xi, initial=0)       , K/W
        #     'ux_df': ... ,          # ux_cum[-1]                        , K
        #     'vx_df': ... ,          # vx_cum[-1]                        , K/W
        #     'Vo_df': ... ,          # 1E3 * (S_cum[-1]+IR_cum[-1])      , mV
        #     'Jphi_df': ... ,        # Jphi_x[-1]                        , W
        # }
        # return AttrDict(rst)
        pass
    
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
    
    def get_cums(self, x0=0):
        Area = self.Area                # mm^2
        xi = x0 + self._xi              # mm
        T_x = self._T_x                 # K
        dT_x = self._dT_x               # K/mm
        
        Rho_x = 1E4 / self._C(T_x)      # S/cm --> uOhm.m <--> mOhm.mm
        S_x = 1E-3 * self._S(T_x)       # uV/K --> mV/K
        K_x = 1E-3 * self._K(T_x)       # W/(m.K) --> W/(mm.K)
        W_x = 0                         # other heat flow
        
        cums = {
            'R': trapz(Rho_x, xi)/Area,     # integral(1/C, dL)/A           , mOhm
            'S': trapz(S_x*dT_x, xi),       # integral(S*(dT/dx), dL)       , mV
            'K': trapz(K_x*dT_x, xi)*Area,  # integral(K*(dT/dx), dL)*A     , W*mm
            'W': 0,                         
            'Rx': trapz(xi*Rho_x, xi)/Area, # integral(x/C, dL/A)           , mOhm.mm
            'Sx': trapz(xi*S_x*dT_x, xi),   # integral(x*S*(dT/dx), dL)     , mV.mm
            'ST': trapz(S_x*T_x, xi),       # integral(T*S, dL)             , mV.mm
            'Wx': 0,
        }
        return AttrDict(cums)
    
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
    
    def phix(self, I, Jphi0=0):
        Area = self.Area                # mm^2
        xi = self._xi                   # mm
        T_x = self._T_x                 # K
        dT_x = self._dT_x               # K/mm
        S_x = 1E-6 * self._S(T_x)       # uV/K   --> V/K
        R_x = 1E1  / self._C(T_x)/Area  # cm/S/mm^2 --> Ohm.mm/mm^2 = Ohm/mm
        K_x = 1E-3 * self._K(T_x)*Area  # W/(m.K)*mm^2 --> W/(mm.K)*mm^2 = W.mm/K
        S_cum = cumtrapz(S_x*dT_x, xi, initial=0)   # V
        IR_cum = I * cumtrapz(R_x, xi, initial=0)   # A*Ohm = V
        
        Jphi_r = np.vstack([I*S_cum, I*IR_cum,])    # W
        Jphi_x = Jphi_r.sum(axis=0)                 # W
        ux = (I*S_x*T_x - Jphi0 - Jphi_x)/K_x       # K/mm
        vx = 1/K_x                                  # K/W/mm
        ux_cum = cumtrapz(ux, xi, initial=0)        # K
        vx_cum = cumtrapz(vx, xi, initial=0)        # K/W
        
        rst = {
            'Jphi_r': Jphi_r,
            'Jphi_x': Jphi_x,
            'ux': ux,
            'vx': vx,
            'ux_cum': ux_cum,
            'vx_cum': vx_cum,
            'ux_df': ux_cum[-1],
            'vx_df': vx_cum[-1],
            'Vo_df': 1E3 * (S_cum[-1]+IR_cum[-1]),  # mV
            'Jphi_df': Jphi_x[-1]                   # W
        }
        return AttrDict(rst)

class LayerSegment(Segment):
    Rc = None   # uOhm.cm^2
    Kc = None   # cm^2.K/W
    def __init__(self, Rc=None, Kc=None, S=None, 
                 Area=100, Perimeter=40):
        self.isPoly = False
        self.Length = 0
        self.Rc = 0 if Rc is None else Rc       # uOhm.cm^2
        self.Kc = 0 if Kc is None else Kc       # cm^2.K/W
        self.S  = 0 if S  is None else S        # uV/K
        self.Area = Area
        self.Perimeter = Perimeter
        self.Ngrid = 2
        self._xi = np.zeros(2)      # mm
        self._T_x = np.ones(2)      # K
        self._dT_x = np.zeros(2)    # W, same as AKdT_x or Jphi_r
    
    def get_cums(self, x0=0):
        Area = self.Area            # mm^2
        Ta, Tb = self.endtemp       # K
        Rc = 1E-1 * self.Rc/Area    # uOhm.cm^2/mm^2 --> mOhm
        S = 1E-3 * self.S           # uV/K --> mV/K
        W_x = 0                     # other heat flow
        
        cums = {
            'R': Rc,                # integral(1/C, dL)/A           , mOhm
            'S': S*(Tb-Ta),         # integral(S*(dT/dx), dL)       , mV                    
            'K': 0,                 # integral(K*(dT/dx), dL)*A     , W*mm, K_cum = L/Kc*DT
            'W': 0,
            'Rx': x0*Rc,            # integral(x/C, dL/A)           , mOhm.mm
            'Sx': x0*S*(Tb-Ta),     # integral(x*S*(dT/dx), dL)     , mV.mm
            'ST': 0,                # integral(T*S, dL)             , mV.mm
            'Wx': 0,
        }
        return AttrDict(cums)
    
    def heatflow(self, I):
        Area = self.Area                    # mm^2
        Ta, Tb = self.endtemp               # K
        AKdTa, AKdTb = self.gradtemp        # W
        S = 1E-6 * self.S                   # uV/K      --> V/K
        Rc = 1E-4 * self.Rc                 # uOhm.cm^2 --> Ohm.mm^2
        qa = I*(S*Ta-1/2*I*Rc/Area)-AKdTa   # W
        qb = I*(S*Tb+1/2*I*Rc/Area)-AKdTb   # W
        return qa, qb
        
    def phix(self, I, Jphi0=0):
        Area = self.Area
        Ta, Tb = self.endtemp
        T_x = self._T_x
        xr = np.linspace(0, 1, self.Ngrid)
        S = 1E-6 * self.S           # uV/K --> V/K
        Rc = 1E-4 * self.Rc/Area    # uOhm.cm^2/mm^2 --> Ohm
        Kc = 1E2  * self.Kc/Area    # cm^2.K/W/mm^2  --> K/W
        S_cum = S * (T_x-Ta)        # V
        IR_cum = I * Rc * xr        # V
        
        Jphi_r = np.vstack([I*S_cum, I*IR_cum,])    # W
        Jphi_x = Jphi_r.sum(axis=0)                 # W. I*S*(T_x-Ta)+I*I*Rc*xr
        ux = I*S*T_x - Jphi0 - Jphi_x               # W
        vx = np.ones_like(xr)       # 1. ux, vx are over-writing
        ux_cum = cumtrapz(ux, xr, initial=0)*Kc     # K
        vx_cum = Kc*xr          # K/W. ux_cum,vx_cum are inheriting
        
        rst = {
            'Jphi_r': Jphi_r,
            'Jphi_x': Jphi_x,
            'ux': ux,
            'vx': vx,
            'ux_cum': ux_cum,
            'vx_cum': vx_cum,
            'ux_df': ux_cum[-1],
            'vx_df': vx_cum[-1],
            'Vo_df': 1E3 * (S_cum[-1]+IR_cum[-1]),  # mV
            'Jphi_df': Jphi_x[-1],          # W. I*S*(Tb-Ta)+I*I*Rc
        }
        return AttrDict(rst)

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
        
    @property
    def segments(self):
        # read-only
        return self._segments
    
    def add_segment(self, segment):
        self._segments.append(segment)
    
    @abstractmethod  
    def build(self, CSA=None, dT_x=None, T_x=None):
        logger.info('Begin building process ...')
        output = self._build(CSA, dT_x, T_x)
        logger.info('Finish building process')
        return output
    
    @abstractmethod
    def simulate(self, I, CSA=None,
                maxiter=30, miniter=1,
                mixing=1.0, tol=1E-4):
        logger.info('Begin simulating ...')
        args = [I, CSA, maxiter, miniter, mixing, tol]
        results = self._simulate(*args)
        logger.info('Finish simulating process')
        return results
    
    def _build(self, CSA=None, dT_x=None, T_x=None):
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
    
    def _simulate(self, I, CSA=None,
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
            Jphi_Ta, Jphi_df, Vo_df, dT_x2, T_x2 = self._get_phi_t2(Ix)
            rstx = [Jphi_Ta, Jphi_df, Vo_df, Ix]
            rst.extend(rstx)
            if len(rsts) == 0:
                itol = [np.inf,]
                logger.info('Slove by iterative temperature distribution ...')
                header = '{:>6s}{:>10s}{:>10s}{:>10s}{:>10s}{:>12s}'
                args= ('epoch', 'Jphi_Ta', 'Jphi_df', 'Vo_df', 'Ix', 'itol')
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

        return Jphi_Ta, Jphi_df, Vo_df, Ix, succeed, rsts
        
    def _get_phi_t2(self, I):
        ux_cums, vx_cums, Vo_df, Jphi_df = 0, 0, 0, 0
        phixs = []
        for seg in self.segments:
            phix = seg.phix(I, Jphi0=Jphi_df)
            ux_cums += phix['ux_df']
            vx_cums += phix['vx_df']
            Vo_df   += phix['Vo_df']
            Jphi_df += phix['Jphi_df']
            phixs.append(phix)
        
        Ta, Tb = self.endtemp
        Jphi_Ta = ((Ta-Tb)+ux_cums)/vx_cums
        
        T_cum_v = Ta
        dT_x2 = []
        T_x2 = []
        for phix in phixs:
            dT_x_i = phix['ux'] - Jphi_Ta*phix['vx']
            T_x_i = T_cum_v + phix['ux_cum'] - Jphi_Ta*phix['vx_cum']
            T_cum_v = T_x_i[-1]
            dT_x2.append(dT_x_i)
            T_x2.append(T_x_i)
        return Jphi_Ta, Jphi_df, Vo_df, dT_x2, T_x2
    
    def _get_metric(self, rst, rst2):
        # rst: ['dT_x', 'T_x', 'Jphi_Ta', 'Jphi_df', 'Vo_df', 'Ix']
        Jphi = rst[2:4]         # [Jphi_Ta, Jphi_df]
        Jphi2 = rst2[2:4]
        itol = Metric.RMSE(Jphi, Jphi2)
        Area = self.CSA/100         # cm^2
        return [itol/Area,]         # W/cm^2
    
    def _update_temp(self, dT_x, T_x, mixing=None):
        for seg, dT_x_i, T_x_i in zip(self.segments, dT_x, T_x):
            if mixing is None:
                seg._dT_x = dT_x_i
                seg._T_x = T_x_i
            else:
                seg._dT_x += mixing*(dT_x_i-seg._dT_x)
                seg._T_x  += mixing*(T_x_i -seg._T_x)
        Ta = self.segments[0]._T_x[0]
        Tb = self.segments[-1]._T_x[-1]
        self.endtemp = [Ta, Tb]
        
    def _get_cums(self, props=None):
        xp = 0
        cums = []
        for seg in self.segments:
            cums.append(seg.get_cums(x0=xp))
            xp += seg.Length
        cums = AttrDict.sum(cums, keys=props)
        cums['Ltot'] = xp
        return cums
    
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

class Module(ABC):
    # share the same (absoluate) current 
    def __init__(self, elements=None):
        raise NotImplementedError

class Cascade():
    def __init__(self, modules=None):
        raise NotImplementedError
