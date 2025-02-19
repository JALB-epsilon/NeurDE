import numpy as np
from .multinv import multinv

def levermore_Geq(ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay):
    """Calculates the Levermore equilibrium distribution function (optimized)."""
    ux[np.abs(ux) < 1e-6] = 0
    uy[np.abs(uy) < 1e-6] = 0
    T[np.abs(T) < 1e-6] = 0
    rho[np.abs(rho) < 1e-6] = 0

    Y, X = ux.shape
    Qn = int(Qn)
    ONE9 = np.ones((1, Qn))
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
        khi[np.abs(khi) < 1e-6] = 0
        zetax[np.abs(zetax) < 1e-6] = 0
        zetay[np.abs(zetay) < 1e-6] = 0

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
    ONE9 = np.ones(Qn, order='F')

    uu = ux**2 + uy**2
    E = T*Cv + uu/2
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

        if mx < 1e-6:
            break
    Feq = np.zeros((Qn,Y))
    for i in range(Qn):
        Feq[i,:] = w[i,row,col]*(rho[row,col]*np.exp(khi[row,col]+zetax[row,col]*ex[i]+zetay[row,col]*ey[i]))
    
    return Feq,khi,zetax,zetay

def levermore_Geq_Obs (ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, Obs):
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

    w = np.zeros((9, len(L)), order='F')
    w[0,L] = (1-T[Obs])*T[Obs]/2
    w[1,L] = (1-T[Obs])*T[Obs]/2
    w[2,L] = (1-T[Obs])*T[Obs]/2
    w[3,L] = (1-T[Obs])*T[Obs]/2
    w[4,L] = T[Obs]**2/4
    w[5,L] = T[Obs]**2/4
    w[6,L] = T[Obs]**2/4
    w[7,L] = T[Obs]**2/4
    w[8,L] = (1-T[Obs])**2

    dkhi = np.zeros_like(khi, order='F')
    dzetax = np.zeros_like(zetax, order='F')
    dzetay = np.zeros_like(zetay, order='F')

    for j in range(20):
        khi[np.abs(khi)<1e-5] = 0
        zetax[np.abs(zetax)<1e-5] = 0
        zetay[np.abs(zetay)<1e-5] = 0

        f = np.zeros((Qn, len(L)), order='F')
        for i in range(Qn):
            f[i,L] = w[i,L] * np.exp(khi[Obs] + zetax[Obs]*ex[i] + zetay[Obs]*ey[i])

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

    Feq = np.zeros((Qn, len(L)), order='F')
    for i in range(Qn):
        Feq[i,L] = w[i,L] * rho[Obs] * np.exp(khi[Obs] + zetax[Obs]*ex[i] + zetay[Obs]*ey[i])
    return Feq, khi, zetax, zetay

'''
def levermore_Geq_Obs(ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, Obs):
    ux[np.abs(ux)<1e-6] = 0
    uy[np.abs(uy)<1e-6] = 0
    T[np.abs(T)<1e-6] = 0
    rho[np.abs(rho)<1e-6] = 0

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
        khi[np.abs(khi)<1e-6] = 0
        zetax[np.abs(zetax)<1e-6] = 0
        zetay[np.abs(zetay)<1e-6] = 0

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

        if mx < 1e-6:
            break

    Feq = w * rho[None, Obs] * np.exp(khi[None, Obs] + zetax[None, Obs]*ex[:,None] + zetay[None, Obs]*ey[:,None])

    return Feq, khi, zetax, zetay


def Geq_BCs(ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, row, col):
    ux[np.abs(ux)<1e-6] = 0
    uy[np.abs(uy)<1e-6] = 0
    T[np.abs(T)<1e-6] = 0
    rho[np.abs(rho)<1e-6] = 0
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
        khi[np.abs(khi)<1e-6] = 0
        zetax[np.abs(zetax)<1e-6] = 0
        zetay[np.abs(zetay)<1e-6] = 0

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

        if mx < 1e-6:
            break
    Feq = np.zeros((Qn,Y))
    for i in range(Qn):
        Feq[i,:] = w[i,row,col]*(rho[row,col]*np.exp(khi[row,col]+zetax[row,col]*ex[i]+zetay[row,col]*ey[i]))
    
    return Feq,khi,zetax,zetay'''