import numpy as np
import warnings


def norm_pmatrix(pmatrix: np.ndarray, du: float = None) -> np.ndarray:
    """Normalize p-matrix with given detector element size 'du'

    Parameters
    ----------
    pmatrix : np.ndarray
        P-matrix (3 x 4)
    du : float, optional
        Detector element size. If None, nothing will change. By default None

    Returns
    -------
    np.ndarray
        Normalized p-matrix

    Raises
    ------
    ValueError
        If input p-matrix is not in shape of (3, 4)
    """
    
    if du == None:
        return pmatrix
    
    nrows, ncols = pmatrix.shape
    if (nrows != 3) or (ncols != 4):
        raise ValueError(f'The shape of pmatrix is not (3, 4)!')
    
    # Matrix Q
    qmatrix = np.linalg.inv(pmatrix[:, 0:3])
    
    lamda = du / np.linalg.norm(qmatrix[:,0])
    
    return pmatrix / lamda


def parse_pmatrix(pmatrix: np.ndarray, du: float = None) -> tuple:

    if du is not None:
        pmatrix = norm_pmatrix(pmatrix)

    # Matrix Q
    qmatrix = np.linalg.inv(pmatrix[:, 0:3])

    # source-origin distance
    r_s = -qmatrix @ pmatrix[:,3]
    sod = np.linalg.norm(r_s)

    # source-detector distance
    e_u = qmatrix[:,0]
    e_v = qmatrix[:,1]
    r_sd = qmatrix[:,2]

    n = np.cross(e_u, e_v)
    sdd = np.abs(np.dot(r_sd, n)) / np.linalg.norm(n)

    return sod, sdd



def fit_pmatrix(points_xyz: np.ndarray, points_uv: np.ndarray, du: float = None) -> np.ndarray:
    """Calculate p-matrix with given points of (x,y,z) and (u,v) using pseudo-inverse method.

    Parameters
    ----------
    points_xyz : np.ndarray
        Points coordinate (x,y,z), should be in shape (n, 3)
    points_uv : np.ndarray
        Points detector projection coordinate (u,v), should be in shape (n, 2)
    du : float, optional
        Detector element size, by default None

    Returns
    -------
    np.ndarray
        P-matrix in shape (3, 4)
    """
    
    nrows_xyz, ncols_xyz = points_xyz.shape
    nrows_uv, ncols_uv = points_uv.shape
    
    if ncols_xyz != 3:
        raise ValueError(f'points_xyz should have three columns!')
    if ncols_uv != 2:
        raise ValueError(f'points_uv should have two columns!')
    if nrows_xyz != nrows_uv:
        raise ValueError(f'points_xyz and points_uv should have same number of rows')
    if nrows_xyz < 6:
        warnings.warn('Number of points is less than 6. Results may be not accurate.', stacklevel=2)
    
    # Build matrix of X 
    X = np.zeros((2*nrows_xyz, 11))
    for n in range(nrows_xyz):
        x = points_xyz[n, 0]
        y = points_xyz[n, 1]
        z = points_xyz[n, 2]
        u = points_uv[n, 0]
        v = points_uv[n, 1]
        
        X[n*2] = [x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z]
        X[n*2+1] = [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z]
        
    # Build matrix of U
    U = np.reshape(points_uv, (-1, 1))

    # Solve p-matrix
    # pmatrix_v = np.linalg.inv(np.transpose(X) @ X) @ (np.transpose(X) @ U)
    pmatrix_v, *_ = np.linalg.lstsq(X, U)
    
    # Add p_34 and reshape
    pmatrix = np.reshape(np.append(pmatrix_v, 1), (3, 4))
    
    # Normalize p-matrix with detector element size
    pmatrix = norm_pmatrix(pmatrix, du)
    
    return pmatrix


def calc_pmatrix_analytical(sod: float, sdd: float, du: float, u_shift: float = 0, v_shift: float = 0, alpha: float = 0, beta: float = 0, gamma: float = 0, theta: float = 0, phi: float = 0) -> np.ndarray:
    # TODO: add comment
    # calculate p-matrix analytically based on given geometry description
    # sod: source-origin distance [mm]
    # sdd: source-detector distance [mm]
    # du: detector element size [mm]
    # u_shift: detector shift distance along u direction [mm]
    # v_shift: detector shift distance along v direction [mm]
    # alpha, beta, gamma: rotation angle of detector along x, y and z axis, respectively [radius]
    # theta, phi: angle of source described in a custom spherical coordinate system (phi starts from -y direction) [radius]

    # Instead of calculating Q=Rz·Ry·Rx·Q0 followed by calculating Q_inv, we directly calculate Q_inv with equation:
    #    Q_inv = Q0_inv · Rx_inv · Ry_inv · Rz_inv

    # Matrix of Q0 (just a demonstration)
    # Q0 = np.array([[du, 0,  u_shift],
    #                [0,  0,  sdd],
    #                [0,  du, v_shift]])

    # Matrix of Q0 inverse
    Q0_inv = np.array([[1/du, -u_shift/(du*sdd),    0],
                       [   0, -v_shift/(du*sdd), 1/du],
                       [   0,             1/sdd,    0]])
    
    # Rotation matrix Rx, Ry, Rz inverse
    Rx_inv = np.array([[            1,              0,               0],
                       [            0,  np.cos(alpha),   np.sin(alpha)],
                       [            0, -np.sin(alpha),   np.cos(alpha)]])

    Ry_inv = np.array([[ np.cos(beta),              0,   -np.sin(beta)],
                       [            0,              1,               0],
                       [ np.sin(beta),              0,    np.cos(beta)]])

    Rz_inv = np.array([[ np.cos(gamma),  np.sin(gamma),               0],
                       [-np.sin(gamma),  np.cos(gamma),               0],
                       [             0,              0,               1]])

    # Calculate Q_inv = Q0_inv · Rx_inv · Ry_inv · Rz_inv
    Q_inv = Q0_inv @ Rx_inv @ Ry_inv @ Rz_inv

    # source position
    xs =  sod * np.cos(theta) * np.sin(phi)
    ys = -sod * np.cos(theta) * np.cos(phi)
    zs =  sod * np.sin(theta)

    # final p-matrix
    pmatrix = Q_inv @ np.array([[1, 0, 0, -xs],
                                [0, 1, 0, -ys],
                                [0, 0, 1, -zs]])

    return pmatrix


def calc_uv_from_xyz_pmatrix(points_xyz: np.ndarray, pmatrix: np.ndarray) -> np.ndarray:
    """P-matrix formward projection. Calculate (u,v) from (x,y,z) using p-matrix.

    Parameters
    ----------
    points_xyz : np.ndarray
        Points cooridnate (x,y,z), should be in shape of (n,3)
    pmatrix : np.ndarray
        P-matrix, should be in shape of (3,4)

    Returns
    -------
    np.ndarray
        Points (u,v), in shape of (n,2)

    Raises
    ------
    ValueError
        Input points (x,y,z) is not in shape of (n,3)
    ValueError
        Input p-matrix is not int shape of (3,4)
    """
    
    n, ncols = points_xyz.shape
    if ncols != 3:
        raise ValueError('points_xyz is not in (n, 3) shape')
    
    nrows, ncols = pmatrix.shape
    if (nrows != 3) or (ncols != 4):
        raise ValueError('pmatrix is not in (3, 4) shape')
    
    # Build matrix on the right-hand side (x,y,z,1)
    X = np.concatenate((points_xyz, np.ones((n,1))), axis=1).transpose()
    
    # Calculate matrix U (alpha*u, alpha*v, alpha)
    U = pmatrix @ X
    
    points_uv = U[0:2, :] / U[2, :]
    
    return points_uv.transpose()
    

# ====================================================================================

def test():
    n = 6
    xyz = np.random.random((n,3))
    uv = np.random.random((n,2))
    
    # uv = np.array([[1,2], [3,4], [5,6], [7,8]])
    
    p = fit_pmatrix(xyz, uv, 1.01)
    print(p)
    
    
if __name__ == '__main__':
    test()