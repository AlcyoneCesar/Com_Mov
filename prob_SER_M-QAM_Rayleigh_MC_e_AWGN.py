import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt 


# ============================================================
#  SER em canal AWGN para M-QAM (teórico)
# ============================================================

def ser_mqam_AWGN(M, gama_s_dB):
    """
    Probabilidade de erro de símbolo (SER) em canal AWGN
    para modulação M-QAM (aproximação clássica).
    """
    gama_s = 10**(gama_s_dB / 10)  # SNR média por símbolo (linear)
    k = np.sqrt(M)
    Pe = 4 * (1 - 1/k) * norm.sf(np.sqrt(3 * gama_s / (M - 1)))
    return Pe


# ============================================================
#  SER em canal Rayleigh para M-QAM (teórico)
# ============================================================

def ser_mqam_Rayleigh_teorico(M, gama_s_dB):
    """
    Probabilidade de erro de símbolo (SER) em canal Rayleigh
    para modulação M-QAM, usando a expressão em função de C_M.
    """
    gama_s = 10**(gama_s_dB / 10)  # SNR média por símbolo (linear)

    # C_M(γ̄_s)
    C_M = np.sqrt((1.5 * gama_s) / (M - 1 + 1.5 * gama_s))

    # Fator geométrico da constelação M-QAM
    a = (np.sqrt(M) - 1) / np.sqrt(M)

    Pe = 2 * a * (1 - C_M) - (a**2) * (1 - (4/np.pi) * C_M * np.arctan(1/C_M))
    return Pe


# ============================================================
#  Geração de constelação M-QAM quadrada normalizada
# ============================================================

def gerar_constelacao_mqam(M):
    """
    Gera constelação quadrada M-QAM (M = 4, 16, 64, ...)
    com energia média de símbolo normalizada para 1.
    Retorna:
      - array complex com os pontos de constelação,
      - vetor de níveis reais (e imaginários) em cada eixo.
    """
    L = int(np.sqrt(M))
    assert L * L == M, "M deve ser quadrado perfeito (4, 16, 64, ...)."

    # Níveis em cada eixo: - (L-1), ..., (L-1) com passo 2 (ex: -3, -1, 1, 3)
    niveis = np.arange(-L + 1, L, 2)

    I, Q = np.meshgrid(niveis, niveis)
    const = I + 1j * Q
    const = const.flatten()

    # Energia média da constelação não normalizada: Es = 2/3 * (M - 1)
    Es = 2/3 * (M - 1)

    # Normalizar para Es_médio = 1
    const = const / np.sqrt(Es)
    niveis_norm = niveis / np.sqrt(Es)

    return const, niveis_norm


# ============================================================
#  Simulação Monte Carlo em canal Rayleigh para M-QAM
# ============================================================

def simular_ser_mqam_Rayleigh_MC(M, gama_s_dB, N_sym=10**5, seed=1234):
    """
    Simulação Monte Carlo da SER em canal Rayleigh para M-QAM.
    Gera N_sym símbolos, canal Rayleigh e ruído AWGN para cada SNR.
    Retorna um vetor Pe_MC com mesmo tamanho de gama_s_dB.
    """
    rng = np.random.default_rng(seed)

    # Constelação e níveis normalizados
    const, niveis_norm = gerar_constelacao_mqam(M)

    # Sorteio de símbolos transmitidos (índices na constelação)
    idx_tx = rng.integers(0, M, size=N_sym)
    s = const[idx_tx]  # símbolos transmitidos (energia média 1)

    # Canal Rayleigh: h ~ CN(0,1)
    h_real = rng.normal(0.0, 1/np.sqrt(2), size=N_sym)
    h_imag = rng.normal(0.0, 1/np.sqrt(2), size=N_sym)
    h = h_real + 1j * h_imag

    Pe_MC = np.zeros_like(gama_s_dB, dtype=float)

    # Loop sobre os valores de SNR média por símbolo
    for i, g_dB in enumerate(gama_s_dB):
        gama_s = 10**(g_dB / 10)  # SNR média por símbolo (linear)

        # Es = 1 (constelação normalizada), então N0 = Es / γ̄s = 1 / γ̄s
        # Variância de cada componente do ruído: sigma2 = N0 / 2
        if gama_s > 0:
            sigma2 = 1 / (2 * gama_s)
        else:
            sigma2 = 1e3  # ruído muito forte para SNR média bem negativa

        n_real = rng.normal(0.0, np.sqrt(sigma2), size=N_sym)
        n_imag = rng.normal(0.0, np.sqrt(sigma2), size=N_sym)
        n = n_real + 1j * n_imag

        # Sinal recebido: r = h * s + n
        r = h * s + n

        # Equalização com CSI perfeita: r_eq = r / h
        r_eq = r / h

        # Detecção por mínimos quadrados: decidir nível mais próximo por eixo
        rI = r_eq.real
        rQ = r_eq.imag

        dist_I = np.abs(rI[:, None] - niveis_norm[None, :])
        dist_Q = np.abs(rQ[:, None] - niveis_norm[None, :])

        idx_I_hat = np.argmin(dist_I, axis=1)
        idx_Q_hat = np.argmin(dist_Q, axis=1)

        s_hat = niveis_norm[idx_I_hat] + 1j * niveis_norm[idx_Q_hat]

        # Contar erros de símbolo
        erros = np.count_nonzero(s_hat != s)
        Pe_MC[i] = erros / N_sym

    return Pe_MC


# ============================================================
#  Bloco principal: SER M-QAM em Rayleigh (teo + MC) e AWGN (teo)
# ============================================================

if __name__ == "__main__":

    M_values = [4, 16, 64]                   # ordens de modulação M-QAM
   # gama_s_dB = np.linspace(-30, -10, 21)    # SNR média por símbolo em dB
    gama_s_dB = np.linspace(-30, 30, 21)    # SNR média por símbolo em dB
    plt.figure(figsize=(10, 6))
    cores = ["tab:blue", "tab:orange", "tab:green"]
    marcadores = ["o", "s", "^"]

    for idx, M in enumerate(M_values):
        # Curva teórica AWGN
        Pe_AWGN_teo = ser_mqam_AWGN(M, gama_s_dB)

        # Curva teórica Rayleigh
        Pe_Rayleigh_teo = ser_mqam_Rayleigh_teorico(M, gama_s_dB)

        # Simulação Monte Carlo em Rayleigh
        Pe_Rayleigh_MC = simular_ser_mqam_Rayleigh_MC(M, gama_s_dB,
                                                      N_sym=10**5,
                                                      seed=1234+idx)

        # Plot AWGN teórico (linha pontilhada)
        plt.semilogy(gama_s_dB, Pe_AWGN_teo,
                     color=cores[idx],
                     linestyle=":",
                     label=f"M-QAM AWGN teórico (M={M})")

        # Plot Rayleigh teórico (linha contínua)
        plt.semilogy(gama_s_dB, Pe_Rayleigh_teo,
                     color=cores[idx],
                     linestyle="-",
                     label=f"M-QAM Rayleigh teórico (M={M})")

        # Pontos da simulação Rayleigh (marcadores)
        plt.semilogy(gama_s_dB, Pe_Rayleigh_MC,
                     color=cores[idx],
                     linestyle="",
                     marker=marcadores[idx],
                     markersize=6,
                     label=f"M-QAM Rayleigh MC (M={M})")

    plt.xlabel('SNR média por símbolo, $\\bar{\\gamma}_s$ (dB)')
    plt.ylabel('Probabilidade de erro de símbolo (SER)')
    plt.title('SER para M-QAM em canais AWGN e Rayleigh\n'
              '(curvas teóricas e simulação Monte Carlo)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.legend()
    plt.ylim(1e-4, 1)
    # plt.xlim(-30, -10)
    plt.xlim(-30, 30)
    plt.tight_layout()
    plt.savefig("ser_mqam_awgn_rayleigh_teo_MC.png", dpi=150)
    plt.show(block=False)
