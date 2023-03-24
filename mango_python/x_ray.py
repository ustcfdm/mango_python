import numpy as np
import pandas as pd
import os
import random
import tqdm
from scipy import signal

from . import files as fs


def load_mu_data(material: str, cross_section: str = 'Total') -> tuple[np.ndarray, np.ndarray] | pd.DataFrame:
    # TODO: add comment
    # get linear attenuation coefficient [cm^-1] of given material
    # cross_section: 'PhotoElectric', 'Compton', 'Rayleigh', 'Total', 'All'
    # If 'All', return a DataFrame
    
    filename = os.path.join(os.path.dirname(__file__), 'mu_data', f'{material}.txt')
    
    df = pd.read_csv(filename, sep=' ', comment='#', names=['Energy', 'Rayleigh', 'Compton', 'PhotoElectric', 'PairProduction(NuclearField)', 'PairProduction(ElectronField)', 'Total', 'Total-Rayleigh'], index_col=False)
    
    df['Energy'] /= 1000    # convert unit to keV
    
    if cross_section == 'PhotoElectric':
        return np.array(df['Energy']), np.array(df['PhotoElectric'])
    elif cross_section == 'Compton':
        return np.array(df['Energy']), np.array(df['Compton'])
    elif cross_section == 'Rayleigh':
        return np.array(df['Energy']), np.array(df['Rayleigh'])
    elif cross_section == 'PairProduction':
        return np.array(df['Energy']), np.array(df['PairProduction(NuclearField)']) + np.array(df['PairProduction(ElectronField)'])
    elif cross_section == 'Total':
        return np.array(df['Energy']), np.array(df['Total'])
    else:
        return df
    

def list_mu_material() -> list[str]:
    """List available materials having linear attenuation coefficient data

    Returns
    -------
    list[str]
        List of materials
    """
    
    folder = os.path.join(os.path.dirname(__file__), 'mu_data')
    files = fs.dir_reg(folder, '.*txt')
    
    return [os.path.basename(file)[0:-4] for file in files]


def get_material_density(material: str) -> float:
    """Get mass density (g/cm3) of given material

    Parameters
    ----------
    material : str
        Material name

    Returns
    -------
    float
        Material mass density (g/cm3)
    """
    material_density_dic = {
        # compound
        'adipose': 0.95,
        'air': 1.205e-3,
        'blood': 1.06,
        'bone': 1.92,
        'brain': 1.04,
        'CaC2O4': 2.12,
        'CaCl2': 2.15,
        'CeCl3': 3.97,
        'CdTe': 5.85,
        'CsI': 4.51,
        'gland': 1.02,
        'GOS': 7.32,
        'hydroxyapatite': 3.1,
        'kidney': 1.05,
        'liver': 1.06,
        'muscle': 1.05,
        'NaI': 3.67,
        'PMMA': 1.18,
        'spleen': 1.06,
        'teflon': 2.2,
        'tissue': 1.06,
        'UricAcid': 1.87,
        'water': 1,
        # element
        'Ag': 10.5,
        'Al': 2.7,
        'Au': 19.3,
        'C': 2.26,
        'Ca': 1.55,
        'Ce': 6.78,
        'Cl': 3.17,
        'Cu': 8.96,
        'Fe': 7.86,
        'Gd': 7.89,
        'H': 0.0899,
        'I': 4.92,
        'K': 0.86,
        'Mg': 1.74,
        'N': 1.251,
        'Na': 0.97,
        'O': 1.429,
        'P': 1.82,
        'Rh': 12.4,
        'S': 2.07,
        'Si': 2.33,
        'Sn': 7.3,
        'Tm': 9.33,
        'W': 19.3,
    }
    
    if material in material_density_dic:
        return material_density_dic[material]
    else:
        raise ValueError(f'No density data of material "{material}"!')

    
def sample_photon_energy_after_compton(incident_energy: float) -> float:
    # TODO: add comment
    # incident_energy (keV)
    # calculate photon energy afeter Compton scattering using Monte Carlo with Kahn Method
    
    # # planck constant x speed of light [eV nm]
    # hc = 1239.841984
    # electron's static energy [keV]
    mec2 = 511.0
    
    # # incident photon wave length [nm]
    # lamda = hc / (incident_energy * 1e6)
    # # normalize the unit of wave length
    # lamda /= hc / mec2
    
    # convert energy [keV] to unit wave length (ratio to electron wave length)
    lamda = mec2 / incident_energy
    
    
    while True:
        r1, r2, r3 = random.random(), random.random(), random.random()
        
        if r1 < (lamda+2) / (9*lamda+2):
            x = 2 * r2 / lamda + 1
            if r3 <= 4 * (1/x - 1/(x*x)):
                break
            else:
                continue
        else:
            x = (lamda+2) / (lamda+2*(1-r2))
            if r3 <= 0.5 * (1/x + (1-lamda*x + lamda)**2):
                break
            else:
                continue
            
    # lamda_new = x * lamda * hc / mec2
    # energy_after_scatter = hc / lamda * 1e-6
    
    # lamda_new = x * lamda
    
    # convert unit wave length to energy [keV]
    energy_after_scatter = mec2 / (x * lamda)
    
    return energy_after_scatter


def sample_compton_scatter_angle(incident_energy: float, energy_after_scatter: float, angle_unit: str = 'degree'):
    # TODO: add comment
    # get photon angle after Compton scattering
    # photon energy unit: keV
    
    cos_theta = 1 - (1/energy_after_scatter - 1/incident_energy) * 511.0
    
    theta = np.arccos(cos_theta)
    
    if angle_unit == 'degree':
        theta *= 180 / np.pi
        
    return theta


def _sample_interaction_type(number_of_photons: int, *args):
    # TODO: add comment
    
    # number of types
    n = len(args)
    
    # prepare result array
    inter_type = np.zeros(number_of_photons, dtype='int')
    
    # threshold for each cross-section type
    th_cs = np.cumsum(args) / np.sum(args)
    
    # get random number
    r = np.random.random(number_of_photons)
    
    # assign inter_type
    for n, th in enumerate(th_cs[:-1]):
        inter_type[r >= th] = n + 1
        
    return inter_type

    
def sample_deposited_energy(photon_energy: float, material: str, number_of_photons: int = 1e7):
    # TODO: add comment
    # photon_energy: keV
    # material: 'Si', 'CdTe', etc.
    
    number_of_photons = int(number_of_photons)

    deposit_energy = np.zeros(number_of_photons)
    
    # load cross-section data
    cs = load_mu_data(material, 'All')
    
    # get cross-section of Compton, Rayleigh, and PhotoElectric ratio
    cs_compton = np.interp(photon_energy, cs['Energy'], cs['Compton'])
    cs_photoelectric = np.interp(photon_energy, cs['Energy'], cs['PhotoElectric'])
    cs_rayleigh = np.interp(photon_energy, cs['Energy'], cs['Rayleigh'])
    
    # sample interaction type
    inter_type = _sample_interaction_type(number_of_photons, cs_photoelectric, cs_compton, cs_rayleigh)
    
    # PhotoElectric: deposit energy is equal to input photon energy
    deposit_energy[inter_type == 0] = photon_energy
    # Rayleigh: deposit energy is zero
    deposit_energy[inter_type == 2] = 0
    
    # For Compton scattering, we need to sample deposit energy
    idx_compton, *_ = np.where(inter_type == 1)
    
    for idx in idx_compton:
        deposit_energy[idx] = photon_energy - sample_photon_energy_after_compton(photon_energy)
        # print(deposit_energy[idx], 'keV')
        
    return deposit_energy


def calc_deposit_energy_table(material: str, max_energy: int = 150) -> np.ndarray:
    # TODO: add comment
    # calculate deposit energy for different input photon energy
    
    energy = np.arange(1, max_energy+1)
    
    R = np.zeros((energy.size, energy.size))
    
    bins = np.arange(0.5, max_energy+1, 1)
    
    number_of_photons = 1e7
    
    for en in tqdm.tqdm(energy, 'Calculating deposited energy'):
        depo_en = sample_deposited_energy(en, material, number_of_photons)
        
        # make histogram data
        h = np.histogram(depo_en, bins)
        
        R[:, en-1] = h[0]
        
    # normalize R
    R /= number_of_photons
    
    # save to file in 'mu_data' folder
    filename = os.path.join(os.path.dirname(__file__), 'mu_data', material+'_energy_response_no_blur.npy')
    
    np.save(filename, R)
    
    return R
    

def load_energy_response_matrix(material: str, sigma: float) -> np.ndarray:
    # TODO: add comment
    # load energy response matrix and filter with a gaussian kernel
    
    # load energy response matrix file
    filename = os.path.join(os.path.dirname(__file__), 'mu_data', material+'_energy_response_no_blur.npy')
    R = np.load(filename)
    
    # prepare Gaussian filter
    if sigma == None:
        return R
    
    x = np.linspace(-20, 20, 41)
    g = 1 / (np.sqrt(2*np.pi) * sigma) * np.exp(- x**2 / (2* sigma**2))
    
    # do the convolution
    g = np.reshape(g, (-1, 1))
    # R2 = np.convolve(R, g, 'same')
    R2 = signal.convolve2d(R, g, mode='same', boundary='symm')
    
    return R2



def _test():
    
    E = 80
    
    en = sample_deposited_energy(E, 'Si')
    
    sigma = 4
    x = np.linspace(-20, 20, 41)
    
    g = 1 / (np.sqrt(2*np.pi) * sigma) * np.exp(- x**2 / (2* sigma**2))
    
    # g = np.ones(15) / 15
    
    bins = np.arange(0.5, 100+1, 1)
    x = np.arange(1, 100+1, step=1)
    h = np.histogram(en, bins=bins)
    
    h = np.convolve(h[0], g, 'same')
    
    import matplotlib.pyplot as plt
    # plt.hist(en, bins=80)
    
    plt.show()
    
    
    # en, pe = load_mu_data('Si', 'PhotoElectric')

    # df = load_mu_data('Si', 'All')
    
    # print(df)
    
    
def _test2():
    # calc_deposit_energy_table('Si')
    load_energy_response_matrix('Si', 4)
    
if __name__ == '__main__':
    R = _test2()





