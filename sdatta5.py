import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

def Fhalf(x):
    xx = np.linspace(0, abs(x) + 10, 251)
    dx = xx[1] - xx[0]
    fx = (2 * dx / np.sqrt(np.pi)) * np.sqrt(xx) / (1 + np.exp(xx - x))
    y = np.sum(fx)
    return y

def fig53():
    # Constants (all MKS, except energy which is in eV)
    hbar = 1.06e-34
    q = 1.6e-19
    epsil = 10 * 8.85e-12
    kT = 0.025
    m = 0.25 * 9.1e-31
    n0 = 2 * m * kT * q / (2 * np.pi * (hbar**2))
    a = 3e-10
    t = (hbar**2) / (2 * m * (a**2) * q)
    beta = q * a * a / epsil
    Ns = 15
    Nc = 70
    Np = Ns + Nc + Ns
    XX = a * 1e9 * np.arange(1, Np + 1)
    mu = 0.318
    Nd = 2 * ((n0/2)**1.5) * Fhalf(mu/kT)
    Nd = Nd * np.concatenate([np.ones(Ns), 0.5*np.ones(Nc), np.ones(Ns)])

    # d2/dx2 matrix for Poisson solution
    D2 = -2 * np.eye(Np) + np.diag(np.ones(Np-1), 1) + np.diag(np.ones(Np-1), -1)
    D2[0, 0] = -1
    D2[-1, -1] = -1  # zero field condition

    # Hamiltonian matrix
    T = 2 * t * np.eye(Np) - t * np.diag(np.ones(Np-1), 1) - t * np.diag(np.ones(Np-1), -1)
    Jop = (q * t / ((Np-1) * hbar)) * 1j * (np.diag(np.ones(Np-1), -1) - np.diag(np.ones(Np-1), 1))

    # Energy grid
    NE = 301
    E = np.linspace(-0.25, 0.5, NE)
    dE = E[1] - E[0]
    print(f"dE = {dE}")
    zplus = 1j * 1e-12
    f0 = n0 * np.log(1 + np.exp((mu - E) / kT))

    # Initial guess for U
    U = np.concatenate([np.zeros(Ns), 0.2*np.ones(Nc), np.zeros(Ns)])

    # Voltage bias steps
    NV = 5
    VV = np.linspace(0, 0.25, NV)
    dV = VV[1] - VV[0]
    Fn = mu * np.ones(Np)
    UU = np.zeros((Np, NV))
    J = np.zeros((Np, NV))

    for kV in range(NV):
        V = VV[kV]
        print(f"V = {V}")
        f1 = n0 * np.log(1 + np.exp((mu - E) / kT))
        f2 = n0 * np.log(1 + np.exp((mu - V - E) / kT))

        in_val = 10
        while in_val > 0.01:
            sig1 = np.zeros((Np, Np), dtype=complex)
            sig2 = np.zeros((Np, Np), dtype=complex)
            rho = np.zeros((Np, Np), dtype=complex)

            sigs = -1j * 0.0125 * np.concatenate([np.zeros(Ns), np.ones(Nc), np.zeros(Ns)])
            sigs = np.diag(sigs)
            gams = 1j * (sigs - sigs.conj().T)
            gams = np.diag(np.diag(gams))

            for k in range(NE):
                fs = n0 * np.log(1 + np.exp((Fn - E[k]) / kT))
                sigin = fs * gams
                sigin = np.diag(np.diag(sigin))

                ck = 1 - ((E[k] + zplus - U[0]) / (2 * t))
                ka = np.arccos(ck)
                sig1[0, 0] = -t * np.exp(1j * ka)
                gam1 = 1j * (sig1 - sig1.conj().T)

                ck = 1 - ((E[k] + zplus - U[-1]) / (2 * t))
                ka = np.arccos(ck)
                sig2[-1, -1] = -t * np.exp(1j * ka)
                gam2 = 1j * (sig2 - sig2.conj().T)

                G = linalg.inv((E[k] + zplus) * np.eye(Np) - T - np.diag(U) - sig1 - sig2 - sigs)
                A1 = G.conj().T @ gam1 @ G
                A2 = G.conj().T @ gam2 @ G

                rho += (dE * (f1[k] * A1 + f2[k] * A2 + G.conj().T @ sigin @ G) / (2 * np.pi))

            n = (1 / a) * np.real(np.diag(rho))
            JJ = (-0.5 * q) * np.real(np.diag((Jop @ rho) + (rho @ Jop)))
            dJ = np.diff(JJ)
            dJ = np.concatenate(([0, 0], dJ[1:-1], [0]))
            dFn = 10 * V * Np * dJ / np.sum(np.abs(JJ))
            Fn = Fn - dFn

            # Correction dU from Poisson
            D = np.zeros(Np)
            for k in range(Np):
                z = (Fn[k] - U[k]) / kT
                D[k] = 2 * ((n0/2)**1.5) * ((Fhalf(z + 0.1) - Fhalf(z)) / 0.1) / kT

            dN = n - Nd + (1 / beta) * D2 @ U
            dU = -beta * linalg.inv(D2 - beta * np.diag(D)) @ dN
            U += dU

            # Check for convergence
            inj = Np * (np.max(JJ[1:-1]) - np.min(JJ[1:-1])) / np.sum(np.abs(JJ))
            ind = np.max(np.abs(dN)) / np.max(Nd)
            in_val = ind + inj
            if kV == 0:
                in_val = ind

            print(f"inj = {inj}, ind = {ind}, in_val = {in_val}")

        UU[:, kV] = U
        J[:, kV] = -0.5 * q * np.diag((rho @ Jop) + (Jop @ rho))
        Fn += np.concatenate([np.zeros(Ns), np.linspace(0, -dV, Nc), -dV * np.ones(Ns)])

    II = np.sum(J, axis=0)
    Fn -= np.concatenate([np.zeros(Ns), np.linspace(0, -dV, Nc), -dV * np.ones(Ns)])

    # Plotting
    plt.figure()
    plt.plot(VV, II)
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.title('I-V Characteristic')
    plt.grid(True)
    plt.show()

    return VV, II, XX, J, Fn, UU

# Run the function
VV, II, XX, J, Fn, UU = fig53()