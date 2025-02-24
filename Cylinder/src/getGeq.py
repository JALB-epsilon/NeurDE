import numpy as np
from .multinv import multinv

def levermore_Geq(ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay):
    """Calculates the Levermore equilibrium distribution function (optimized)."""
    ux[np.abs(ux) < 1e-5] = 0
    uy[np.abs(uy) < 1e-5] = 0
    T[np.abs(T) < 1e-5] = 0
    rho[np.abs(rho) < 1e-5] = 0

    Y, X = ux.shape
    Qn = int(Qn)
    ex = ex.squeeze()
    ey = ey.squeeze()

    uu = ux**2 + uy**2
    E = T * Cv + 0.5 * uu
    H = E + T

    w = np.zeros((Qn, Y, X))  
    w[:4, :, :] = (1 - T) * T * 0.5
    w[4:8, :, :] = T**2 * 0.25
    w[8, :, :] = (1 - T)**2

    f = np.zeros((Qn, Y, X))
    F = np.zeros((3, Y, X))
    J = np.zeros((3, 3, Y, X))

    for _ in range(20):
        khi[np.abs(khi) < 1e-5] = 0
        zetax[np.abs(zetax) < 1e-5] = 0
        zetay[np.abs(zetay) < 1e-5] = 0

        f = w * np.exp(khi[None, :, :] + zetax[None, :, :] * ex[:, None, None] + zetay[None, :, :] * ey[:, None, None])

        F[0, :, :] = f.sum(axis=0) - 2 * E
        F[1, :, :] = (ex[:, None, None] * f).sum(axis=0) - 2 * ux * H
        F[2, :, :] = (ey[:, None, None] * f).sum(axis=0) - 2 * uy * H


        J[0, 0, :, :] = f.sum(axis=0)
        J[0, 1, :, :] = np.einsum("q,qyx->yx", ex, f)  
        J[0, 2, :, :] = np.einsum("q,qyx->yx", ey, f)
        J[1, 0, :, :] = J[0, 1, :, :]
        J[1, 1, :, :] = np.einsum("q,qyx->yx", ex**2, f)
        J[1, 2, :, :] = np.einsum("q,qyx->yx", ex * ey, f)
        J[2, 0, :, :] = J[0, 2, :, :]
        J[2, 1, :, :] = J[1, 2, :, :]
        J[2, 2, :, :] = np.einsum("q,qyx->yx", ey**2, f)
        
        IJ = multinv(J) 
        IJ = IJ.reshape((3,3,Y,X), order="F")

        khi1 = khi.copy()
        zetax1 = zetax.copy()
        zetay1 = zetay.copy()

        khi -= (IJ[0, 0] * F[0] + IJ[0, 1] * F[1] + IJ[0, 2] * F[2])
        zetax -= (IJ[1, 0] * F[0] + IJ[1, 1] * F[1] + IJ[1, 2] * F[2])
        zetay -= (IJ[2, 0] * F[0] + IJ[2, 1] * F[1] + IJ[2, 2] * F[2])

        dkhi = np.abs(khi - khi1)
        dzetax = np.abs(zetax - zetax1)
        dzetay = np.abs(zetay - zetay1)

        mx = np.max(np.array([np.max(dkhi), np.max(dzetax), np.max(dzetay)]))  
        if mx < 1e-6:
            break

    Feq = w * rho[None, :, :] * np.exp(khi[None, :, :] + zetax[None, :, :] * ex[:, None, None] + zetay[None, :, :] * ey[:, None, None])

    return Feq, khi, zetax, zetay


def levermore_Geq_BCs(ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, row, col):
    """Calculates Levermore equilibrium for boundary conditions (optimized)."""

    # Ensure numerical stability
    ux[np.abs(ux) < 1e-5] = 0
    uy[np.abs(uy) < 1e-5] = 0
    T[np.abs(T) < 1e-5] = 0
    rho[np.abs(rho) < 1e-5] = 0
    eps = 1e-12

    Y, X = ux.shape
    Qn = int(Qn)
    ex = ex.squeeze()
    ey = ey.squeeze()

    # Precompute common terms
    uu = ux**2 + uy**2
    E = T * Cv + 0.5 * uu
    H = E + T

    # Precompute weights
    w = np.zeros((Qn, Y, X))
    w[:4, row, col] = (1 - T[row, col]) * T[row, col] / 2
    w[4:8, row, col] = T[row, col]**2 / 4
    w[8, row, col] = (1 - T[row, col])**2

    R = len(row)

    # Newton-Raphson iteration
    for _ in range(20):
        khi[np.abs(khi) < 1e-5] = 0
        zetax[np.abs(zetax) < 1e-5] = 0
        zetay[np.abs(zetay) < 1e-5] = 0

        # Calculate exponential term
        exponent = khi[None, row, col] + zetax[None, row, col] * ex[:, None] + zetay[None, row, col] * ey[:, None]
        f = w[:, row, col] * np.exp(exponent)

        # Calculate F and J using vectorized operations
        F = np.zeros((3, R))
        F[0] = np.sum(f, axis=0) - 2 * E[row, col]
        #precompute ex and ey dot f 
        ex_dot_f = np.dot(ex, f)
        ey_dot_f = np.dot(ey, f)

        F[1] = ex_dot_f - 2 * ux[row, col] * H[row, col]
        F[2] = ey_dot_f - 2 * uy[row, col] * H[row, col]

        J = np.zeros((3, 3, R))
        J[0, 0] = np.sum(f, axis=0)
        J[0, 1] = ex_dot_f
        J[0, 2] = ey_dot_f
        J[1, 0] = ex_dot_f
        J[1, 1] = np.dot(ex**2, f)
        J[1, 2] = np.dot(ex * ey, f)
        J[2, 0] = ey_dot_f
        J[2, 1] = np.dot(ey * ex, f)
        J[2, 2] = np.dot(ey**2, f)


        # Calculate inverse Jacobian and update parameters
        IJ = np.linalg.inv(J.transpose(2, 0, 1))  # Transpose to get correct shape for batched inverse
        d_params = -np.matmul(IJ, F).transpose(0, 2, 1)[:, 0, :]


        khi_old = khi[row, col].copy()
        zetax_old = zetax[row, col].copy()
        zetay_old = zetay[row, col].copy()

        khi[row, col] += d_params[:, 0]
        zetax[row, col] += d_params[:, 1]
        zetay[row, col] += d_params[:, 2]

        # Calculate convergence criteria
        dkhi = np.abs((khi[row, col] - khi_old) / (khi_old + eps))
        dzetax = np.abs((zetax[row, col] - zetax_old) / (zetax_old + eps))
        dzetay = np.abs((zetay[row, col] - zetay_old) / (zetay_old + eps))

        max_diff = max(np.max(dkhi), np.max(dzetax), np.max(dzetay))

        if max_diff < 1e-6:
            break

    # Calculate final equilibrium distribution
    exponent = khi[None, row, col] + zetax[None, row, col] * ex[:, None] + zetay[None, row, col] * ey[:, None]
    Feq = w[:, row, col] * rho[row, col] * np.exp(exponent)

    return Feq, khi, zetax, zetay

'''def levermore_Geq_BCs(ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, row, col):
    """Calculates Levermore equilibrium for boundary conditions (similar to the well-tested version)."""
    ux[np.abs(ux) < 1e-5] = 0
    uy[np.abs(uy) < 1e-5] = 0
    T[np.abs(T) < 1e-5] = 0
    rho[np.abs(rho) < 1e-5] = 0
    eps=1e-12
    Y, X = ux.shape
    Qn = int(Qn)
    ONE9 = np.ones((1,Qn))
    ex = ex.squeeze()
    ey = ey.squeeze()

    uu = ux**2 + uy**2
    E = T * Cv + 0.5 * uu  # Corrected: 0.5 * uu
    H = E + T

    w = np.zeros((Qn, Y,X), order='F')
    for i in range(4):
        w[i,row,col] = (1-T[row,col])*T[row,col]/2
    for i in range(4, 8):
        w[i,row,col] = T[row,col]**2/4
    w[8,row,col] = (1-T[row,col])**2

    R = len(row)
    C = 1

    f = np.zeros((Qn,R), order='F')
    F = np.zeros((3,Y,X), order='F')
    J = np.zeros((3,3,Y,X), order='F')

    dkhi = np.zeros_like(khi, order='F')
    dzetax = np.zeros_like(zetax, order='F')
    dzetay = np.zeros_like(zetay, order='F')

    for j in range(20):
        khi[np.abs(khi)<1e-5] = 0
        zetax[np.abs(zetax)<1e-5] = 0
        zetay[np.abs(zetay)<1e-5] = 0

        f = w[:, row, col] * np.exp(khi[None,row,col] + zetax[None,row,col]*ex[:,None] + zetay[None,row,col]*ey[:,None])
        #for i in range(Qn):
        #    f[i,row] = (w[i,row,col] * np.exp(khi[row,col] + zetax[row,col]*ex[i] + zetay[row,col]*ey[i]))

        F = np.zeros((3, len(row)), order= 'F')
        F[0,row] = ONE9.dot(f.reshape(Qn,R*C, order='F')) - 2*E[row,col]
        F[1,row] = ex.dot(f.reshape(Qn,R*C, order='F')) - 2*ux[row,col]*H[row,col]
        F[2,row] = ey.dot(f.reshape(Qn,R*C, order='F')) - 2*uy[row,col]*H[row,col]
        J = np.zeros((3, 3, Y), order='F')

        J[0,0,row] = f.sum(axis=0)
        J[0,1,row] = ex.dot(f.reshape(Qn,R*C, order='F'))
        J[0,2,row] = ey.dot(f.reshape(Qn,R*C, order='F'))
        J[1,0,row] = ex.dot(f.reshape(Qn,R*C, order='F'))
        J[1,1,row] = (ex*ex).dot(f.reshape(Qn,R*C, order='F'))
        J[1,2,row] = (ex*ey).dot(f.reshape(Qn,R*C, order='F'))
        J[2,0,row] = ey.dot(f.reshape(Qn,R*C, order='F'))
        J[2,1,row] = (ey*ex).dot(f.reshape(Qn,R*C, order='F'))
        J[2,2,row] = (ey*ey).dot(f.reshape(Qn,R*C, order='F'))
        IJ = multinv(J)
        khi1 = khi.copy()
        zetax1 = zetax.copy() # very important to use copy, in python array is stored by reference 
        zetay1 = zetay.copy()

        # newton Method to find root
        khi[row,col] = khi[row,col] -(IJ[0, 0] * F[0] + IJ[0, 1] * F[1] + IJ[0, 2] * F[2])        
        zetax[row,col] =zetax[row,col]- (IJ[1, 0] * F[0] + IJ[1, 1] * F[1] + IJ[1, 2] * F[2])
        zetay[row,col] =zetay[row,col]- (IJ[2, 0] * F[0] + IJ[2, 1] * F[1] + IJ[2, 2] * F[2]) 
        
        dkhi[row,col] = np.abs((khi[row,col] - khi1[row,col]) / (khi1[row,col]+eps))
        dzetax[row,col] = np.abs((zetax[row,col] - zetax1[row,col]) / (zetax1[row,col]+eps))
        dzetay[row,col] = np.abs((zetay[row,col] - zetay1[row,col]) / (zetay1[row,col]+eps))

        mkhi = np.max(np.abs(dkhi[row,col]))
        mzetax = np.max(np.abs(dzetax[row,col]))
        mzetay = np.max(np.abs(dzetay[row,col]))

        mx = max([mkhi, mzetax, mzetay])

        if mx < 1e-6:
            break

    Feq = w[:, row, col] * rho[row, col] * np.exp(khi[None, row, col] + zetax[None, row, col] * ex[:, None] + zetay[None, row, col] * ey[:, None])
    #Feq = np.zeros((Qn,Y))
    #for i in range(Qn):
    #    Feq[i,:] = w[i,row,col]*(rho[row,col]*np.exp(khi[row,col]+zetax[row,col]*ex[i]+zetay[row,col]*ey[i]))
    
    return Feq,khi,zetax,zetay'''

def levermore_Geq_Obs(ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, Obs):
    """Calculates Levermore equilibrium for observed points (optimized)."""
    ux[np.abs(ux) < 1e-5] = 0
    uy[np.abs(uy) < 1e-5] = 0
    T[np.abs(T) < 1e-5] = 0
    rho[np.abs(rho) < 1e-5] = 0

    ex = ex.squeeze()
    ey = ey.squeeze()
    eps = 1e-12

    uu = ux[Obs]**2 + uy[Obs]**2
    E = T[Obs] * Cv + uu / 2
    H = E + T[Obs]
    L = np.arange(len(uu))

    w = np.zeros((Qn, len(L)))
    one_minus_T = 1 - T[Obs]
    w[:4, L] = one_minus_T * T[Obs] * 0.5
    w[4:8, L] = T[Obs]**2 * 0.25
    w[8, L] = one_minus_T**2

    dkhi = np.zeros_like(khi, order='F')
    dzetax = np.zeros_like(zetax, order='F')
    dzetay = np.zeros_like(zetay, order='F')

    for _ in range(20):
        khi[np.abs(khi) < 1e-5] = 0
        zetax[np.abs(zetax) < 1e-5] = 0
        zetay[np.abs(zetay) < 1e-5] = 0

        f = w * np.exp(khi[Obs] + zetax[None, Obs] * ex[:, None] + zetay[None, Obs] * ey[:, None])

        #precompute ex and ey dot f 
        ex_dot_f = np.dot(ex, f)
        ey_dot_f = np.dot(ey, f)

        F = np.zeros((3, len(L)))
        F[0, L] = np.sum(f, axis=0) - 2 * E
        F[1, L] = ex_dot_f - 2 * ux[Obs] * H
        F[2, L] = ey_dot_f - 2 * uy[Obs] * H
        

        J = np.zeros((3, 3, len(L)))
        J[0, 0, L] = np.sum(f, axis=0)
        J[0, 1, L] = ex_dot_f
        J[0, 2, L] = ey_dot_f
        J[1, 0, L] = ex_dot_f
        J[1, 1, L] = np.dot(ex**2, f)
        J[1, 2, L] = np.dot(ex * ey, f)
        J[2, 0, L] = ey_dot_f
        J[2, 1, L] = np.dot(ey * ex, f)
        J[2, 2, L] = np.dot(ey**2, f)


        IJ = np.linalg.inv(J.transpose(2, 0, 1))

        d_params = -np.matmul(IJ, F).transpose(0, 2, 1)[:, 0, :]

        khi_old = khi[Obs].copy()
        zetax_old = zetax[Obs].copy()
        zetay_old = zetay[Obs].copy()

        khi[Obs] += d_params[:, 0] 
        zetax[Obs] += d_params[:, 1]
        zetay[Obs] += d_params[:, 2] 

        dkhi[Obs] = np.abs((khi[Obs] - khi_old) / (khi_old + eps))
        dzetax[Obs] = np.abs((zetax[Obs] - zetax_old) / (zetax_old + eps))
        dzetay[Obs] = np.abs((zetay[Obs] - zetay_old) / (zetay_old + eps))

        max_diff = np.max(np.stack((dkhi[Obs], dzetax[Obs], dzetay[Obs])), axis=0).max()

        if max_diff < 1e-6:
            break

    Feq = w * rho[None, Obs] * np.exp(khi[None, Obs] + zetax[None, Obs] * ex[:, None] + zetay[None, Obs] * ey[:, None])

    return Feq, khi, zetax, zetay

'''def levermore_Geq_Obs(ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, Obs):
    """Calculates Levermore equilibrium for observed points (similar to the well-tested version)."""
    ux[np.abs(ux) < 1e-5] = 0
    uy[np.abs(uy) < 1e-5] = 0
    T[np.abs(T) < 1e-5] = 0
    rho[np.abs(rho) < 1e-5] = 0
    eps = 1e-12
    ex = ex.squeeze()
    ey = ey.squeeze()
    Qn = int(Qn)

    uu = ux[Obs]**2 + uy[Obs]**2
    E = T[Obs] * Cv + 0.5 * uu  # Corrected: 0.5 * uu
    H = E + T[Obs]
    L = np.arange(len(uu))

    w = np.zeros((Qn,len(L)))  # Qn rows, number of observed points columns
    w[:4, :] = (1 - T[Obs]) * T[Obs] * 0.5  # Vectorized
    w[4:8, :] = T[Obs]**2 * 0.25  # Vectorized
    w[8, :] = (1 - T[Obs])**2  # Vectorized

    dkhi = np.zeros_like(khi, order='F')
    dzetax = np.zeros_like(zetax, order='F')
    dzetay = np.zeros_like(zetay, order='F')

    for _ in range(20):
        khi[np.abs(khi)<1e-5] = 0
        zetax[np.abs(zetax)<1e-5] = 0
        zetay[np.abs(zetay)<1e-5] = 0

        #f = np.zeros((Qn, len(L)), order='F')
        f = w * np.exp(khi[None, Obs] + zetax[None, Obs]*ex[:, None] + zetay[None, Obs]*ey[:, None])
        #for i in range(Qn):
        #    f[i,L] = w[i,L] * np.exp(khi[Obs] + zetax[Obs]*ex[i] + zetay[Obs]*ey[i])

        F = np.zeros((3, len(L)), order='F')
        F[0,L] = np.sum(f[:,L], axis=0) - 2*E
        F[1,L] = ex.dot(f[:,L]) - 2*ux[Obs]*H
        F[2,L] = ey.dot(f[:,L]) - 2*uy[Obs]*H

        J = np.zeros((3, 3, len(L)), order='F')
        J[0,0,L] = np.sum(f[:,L], axis=0)
        J[0,1,L] = ex.dot(f[:,L])
        J[0,2,L] = ey.dot(f[:,L])
        J[1,0,L] = ex.dot(f[:,L])
        J[1,1,L] = (ex**2).dot(f[:,L])
        J[1,2,L] = (ex*ey).dot(f[:,L])
        J[2,0,L] = ey.dot(f[:,L])
        J[2,1,L] = (ex*ey).dot(f[:,L])
        J[2,2,L] = (ey**2).dot(f[:,L])

        IJ = multinv(J)

        khi1 = khi[Obs].copy()
        zetax1 = zetax[Obs].copy()
        zetay1 = zetay[Obs].copy()

        sz = F[0,L].shape
        khi[Obs] = khi[Obs] - (IJ[0,0,L].reshape(sz, order='F')*F[0,L] + IJ[0,1,L].reshape(sz, order='F')*F[1,L] + IJ[0,2,L].reshape(sz, order='F')*F[2,L])
        zetax[Obs] = zetax[Obs] - (IJ[1,0,L].reshape(sz, order='F')*F[0,L] + IJ[1,1,L].reshape(sz, order='F')*F[1,L] + IJ[1,2,L].reshape(sz, order='F')*F[2,L])
        zetay[Obs] = zetay[Obs] - (IJ[2,0,L].reshape(sz, order='F')*F[0,L] + IJ[2,1,L].reshape(sz, order='F')*F[1,L] + IJ[2,2,L].reshape(sz, order='F')*F[2,L])
        dkhi[Obs] = np.abs((khi[Obs] - khi1) / (khi1+eps))
        dzetax[Obs] = np.abs((zetax[Obs] - zetax1) / (zetax1+eps))
        dzetay[Obs] = np.abs((zetay[Obs] - zetay1) / (zetay1+eps))

        mkhi = np.min(dkhi[Obs])
        mzetax = np.min(dzetax[Obs])
        mzetay = np.min(dzetay[Obs])

        mx = min(mkhi, mzetax, mzetay)

        if mx < 1e-6:
            break

    Feq = w * rho[None, Obs] * np.exp(khi[None, Obs] + zetax[None, Obs]*ex[:, None] + zetay[None, Obs]*ey[:, None])
    #Feq = np.zeros((Qn, len(L)), order='F')
    #for i in range(Qn):
    #    Feq[i,L] = w[i,L] * rho[Obs] * np.exp(khi[Obs] + zetax[Obs]*ex[i] + zetay[Obs]*ey[i])
    return Feq, khi, zetax, zetay'''

'''
def levermore_Geq_Obs(ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, Obs):
    ux[np.abs(ux)<1e-5] = 0
    uy[np.abs(uy)<1e-5] = 0
    T[np.abs(T)<1e-5] = 0
    rho[np.abs(rho)<1e-5] = 0

    ex = ex.squeeze()
    ey = ey.squeeze()
    eps=1e-12

    uu = ux[Obs]**2 + uy[Obs]**2
    E = T[Obs]*Cv + uu/2
    H = E + T[Obs]
    L = np.arange(len(uu))

    w = np.zeros((Qn, len(L)), order='F')
    one_minus_T = 1-T[Obs]
    w[:4,L] = one_minus_T*T[Obs]*0.5
    w[4:8,L] = T[Obs]**2*0.25
    w[8,L] = one_minus_T**2


    dkhi = np.zeros_like(khi, order='F')
    dzetax = np.zeros_like(zetax, order='F')
    dzetay = np.zeros_like(zetay, order='F')

    for _ in range(20):
        khi[np.abs(khi)<1e-5] = 0
        zetax[np.abs(zetax)<1e-5] = 0
        zetay[np.abs(zetay)<1e-5] = 0

        f = w * np.exp(khi[Obs] + zetax[None, Obs]*ex[:,None] + zetay[None, Obs]*ey[:,None])

        F = np.zeros((3, len(L)), order='F')
        F[0,L] = np.sum(f[:,L], axis=0) - 2*E
        F[1,L] = ex.dot(f[:,L]) - 2*ux[Obs]*H
        F[2,L] = ey.dot(f[:,L]) - 2*uy[Obs]*H

        J = np.zeros((3, 3, len(L)), order='F')
        J[0,0,L] = np.sum(f[:,L], axis=0)
        J[0,1,L] = ex.dot(f[:,L])
        J[0,2,L] = ey.dot(f[:,L])
        J[1,0,L] = ex.dot(f[:,L])
        J[1,1,L] = (ex**2).dot(f[:,L])
        J[1,2,L] = (ex*ey).dot(f[:,L])
        J[2,0,L] = ey.dot(f[:,L])
        J[2,1,L] = (ex*ey).dot(f[:,L])
        J[2,2,L] = (ey**2).dot(f[:,L])

        IJ = multinv(J)

        khi1 = khi[Obs].copy()
        zetax1 = zetax[Obs].copy()
        zetay1 = zetay[Obs].copy()

        sz = F[0,L].shape
        khi[Obs] = khi[Obs] - (IJ[0,0,L].reshape(sz, order='F')*F[0,L] + IJ[0,1,L].reshape(sz, order='F')*F[1,L] + IJ[0,2,L].reshape(sz, order='F')*F[2,L])
        zetax[Obs] = zetax[Obs] - (IJ[1,0,L].reshape(sz, order='F')*F[0,L] + IJ[1,1,L].reshape(sz, order='F')*F[1,L] + IJ[1,2,L].reshape(sz, order='F')*F[2,L])
        zetay[Obs] = zetay[Obs] - (IJ[2,0,L].reshape(sz, order='F')*F[0,L] + IJ[2,1,L].reshape(sz, order='F')*F[1,L] + IJ[2,2,L].reshape(sz, order='F')*F[2,L])
        dkhi[Obs] = np.abs((khi[Obs] - khi1) / (khi1+eps))
        dzetax[Obs] = np.abs((zetax[Obs] - zetax1) / (zetax1+eps))
        dzetay[Obs] = np.abs((zetay[Obs] - zetay1) / (zetay1+eps))

        mkhi = np.min(dkhi[Obs])
        mzetax = np.min(dzetax[Obs])
        mzetay = np.min(dzetay[Obs])

        mx = min(mkhi, mzetax, mzetay)

        if mx < 1e-5:
            break

    Feq = w * rho[None, Obs] * np.exp(khi[None, Obs] + zetax[None, Obs]*ex[:,None] + zetay[None, Obs]*ey[:,None])

    return Feq, khi, zetax, zetay


def Geq_BCs(ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, row, col):
    ux[np.abs(ux)<1e-5] = 0
    uy[np.abs(uy)<1e-5] = 0
    T[np.abs(T)<1e-5] = 0
    rho[np.abs(rho)<1e-5] = 0
    eps=1e-12
    Y,X = ux.shape
    Qn = int(Qn)
    ONE9 = np.ones((1,Qn))
    ex = ex.squeeze()
    ey = ey.squeeze()

    uu = ux**2 + uy**2
    E = T*Cv + uu/2
    H = E + T

    w = np.zeros((Qn, Y,X), order='F')
    one_minus_T = 1-T[row,col]
    w[:4,row,col] = (one_minus_T)*T[row,col]*0.5
    w[4:8,row,col] = T[row,col]**2*0.25
    w[8,row,col] = (one_minus_T)**2
    for i in range(4):
        w[i,row,col] = (1-T[row,col])*T[row,col]/2
    for i in range(4, 8):
        w[i,row,col] = T[row,col]**2/4
    w[8,row,col] = (1-T[row,col])**2

    R = len(row)
    C = 1

    f = np.zeros((Qn,R), order='F')
    F = np.zeros((3,Y,X), order='F')
    J = np.zeros((3,3,Y,X), order='F')

    dkhi = np.zeros_like(khi, order='F')
    dzetax = np.zeros_like(zetax, order='F')
    dzetay = np.zeros_like(zetay, order='F')

    for j in range(20):
        khi[np.abs(khi)<1e-5] = 0
        zetax[np.abs(zetax)<1e-5] = 0
        zetay[np.abs(zetay)<1e-5] = 0

        f[i,row] = (w[i,row,col] * np.exp(khi[None,row,col] + zetax[row,col]*ex[i] + zetay[row,col]*ey[i]))

        for i in range(Qn):
            f[i,row] = (w[i,row,col] * np.exp(khi[row,col] + zetax[row,col]*ex[i] + zetay[row,col]*ey[i]))

        F = np.zeros((3, len(row)), order= 'F')
        F[0,row] = ONE9.dot(f.reshape(Qn,R*C, order='F')) - 2*E[row,col]
        F[1,row] = ex.dot(f.reshape(Qn,R*C, order='F')) - 2*ux[row,col]*H[row,col]
        F[2,row] = ey.dot(f.reshape(Qn,R*C, order='F')) - 2*uy[row,col]*H[row,col]
        J = np.zeros((3, 3, Y), order='F')

        J[0,0,row] = f.sum(axis=0)
        J[0,1,row] = ex.dot(f.reshape(Qn,R*C, order='F'))
        J[0,2,row] = ey.dot(f.reshape(Qn,R*C, order='F'))
        J[1,0,row] = ex.dot(f.reshape(Qn,R*C, order='F'))
        J[1,1,row] = (ex*ex).dot(f.reshape(Qn,R*C, order='F'))
        J[1,2,row] = (ex*ey).dot(f.reshape(Qn,R*C, order='F'))
        J[2,0,row] = ey.dot(f.reshape(Qn,R*C, order='F'))
        J[2,1,row] = (ey*ex).dot(f.reshape(Qn,R*C, order='F'))
        J[2,2,row] = (ey*ey).dot(f.reshape(Qn,R*C, order='F'))
        IJ = multinv(J)
        khi1 = khi.copy()
        zetax1 = zetax.copy() # very important to use copy, in python array is stored by reference 
        zetay1 = zetay.copy()

        # newton Method to find root
        khi[row,col] = khi[row,col] -(IJ[0, 0] * F[0] + IJ[0, 1] * F[1] + IJ[0, 2] * F[2])         # remove reshape in Tran's code becasue of automatic broadcasting in numpy
        zetax[row,col] =zetax[row,col]- (IJ[1, 0] * F[0] + IJ[1, 1] * F[1] + IJ[1, 2] * F[2])
        zetay[row,col] =zetay[row,col]- (IJ[2, 0] * F[0] + IJ[2, 1] * F[1] + IJ[2, 2] * F[2]) 
        
        dkhi[row,col] = np.abs((khi[row,col] - khi1[row,col]) / (khi1[row,col]+eps))
        dzetax[row,col] = np.abs((zetax[row,col] - zetax1[row,col]) / (zetax1[row,col]+eps))
        dzetay[row,col] = np.abs((zetay[row,col] - zetay1[row,col]) / (zetay1[row,col]+eps))

        mkhi = np.max(np.abs(dkhi[row,col]))
        mzetax = np.max(np.abs(dzetax[row,col]))
        mzetay = np.max(np.abs(dzetay[row,col]))

        mx = max([mkhi, mzetax, mzetay])

        if mx < 1e-5:
            break
    Feq = np.zeros((Qn,Y))
    for i in range(Qn):
        Feq[i,:] = w[i,row,col]*(rho[row,col]*np.exp(khi[row,col]+zetax[row,col]*ex[i]+zetay[row,col]*ey[i]))
    
    return Feq,khi,zetax,zetay'''