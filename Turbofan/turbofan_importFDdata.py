import pandas as pd
   
    
def turbofan_importFDdata(file):       
    colname = ['Unit', 'Time', 'Setting1', 'Setting2', 'Setting3', 'FanInletTemp',
    'LPCOutletTemp', 'HPCOutletTemp', 'LPTOutletTemp', 'FanInletPres',
    'BypassDuctPres', 'TotalHPCOutletPres', 'PhysFanSpeed', 'PhysCoreSpeed',
    'EnginePresRatio', 'StaticHPCOutletPres', 'FuelFlowRatio', 'CorrFanSpeed',
    'CorrCoreSpeed', 'BypassRatio', 'BurnerFuelAirRatio', 'BleedEnthalpy',
    'DemandFanSpeed', 'DemandCorrFanSpeed', 'HPTCoolantBleed', 'LPTCoolantBleed'];
    
    data = pd.read_csv(file, header=None, delimiter=' ').dropna(axis=1)
    data.columns = colname
    return data