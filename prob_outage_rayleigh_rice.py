import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv  # Bessel modificada de 1ª espécie


# ============================================================
#  Função Marcum-Q de ordem 1 (implementação manual)
# ============================================================

def marcum_q1(a, b, terms=50):
    """
    Implementação manual da função Marcum Q de ordem 1.

    Q1(a, b) = exp(-(a^2 + b^2)/2) * sum_{k=0}^{∞} (a/b)^k * I_k(a b)

    Parâmetros:
      a, b : floats ou arrays numpy (broadcasting suportado)
      terms : número de termos do somatório (50 é seguro para boa precisão)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    expo = np.exp(-(a*a + b*b) / 2.0)
    soma = np.zeros_like(b, dtype=float)

    ab = a * b
    ratio = a / b

    for k in range(terms):
        soma += (ratio**k) * iv(k, ab)

    return expo * soma


# ============================================================
#  Funções principais de canal e outage
# ============================================================

def gera_rayleigh(N):
    """
    Gera canal complexo Rayleigh com E[|h|^2] = 1
    e retorna a envoltória beta = |h|.
    """
    # Parte real e imaginária ~ N(0, 1/sqrt(2)) -> potência total = 1
    h_real = np.random.normal(0.0, 1/np.sqrt(2), N)
    h_imag = np.random.normal(0.0, 1/np.sqrt(2), N)
    h = h_real + 1j * h_imag
    beta = np.abs(h)
    return beta, h


def gera_rice(N, K):
    """
    Gera canal complexo Rice com fator K (ESCALA LINEAR),
    normalizado para E[|h|^2] = 1.
    Retorna a envoltória beta = |h|.
    """
    # Normalização: E[|h|^2] = A^2 + 2*sigma^2 = 1
    sigma2 = 1.0 / (2.0 * (K + 1.0))
    sigma = np.sqrt(sigma2)
    A = np.sqrt(K / (K + 1.0))  # componente LOS (amplitude real, fase = 0)

    X = np.random.normal(0.0, 1.0, N)
    Y = np.random.normal(0.0, 1.0, N)
    h = A + sigma * (X + 1j * Y)

    beta = np.abs(h)
    return beta, h


def calc_outage(gamma_s, gamma_th_lin):
    """
    Calcula a probabilidade de outage de forma explícita, por contagem:
      P_out(γ_th) ≈ Ns / N,
    onde Ns é o número de amostras com γ_s < γ_th.

    gamma_s: vetor de SNR instantânea (linear)
    gamma_th_lin: vetor de limiares (linear)
    """
    N = len(gamma_s)
    p_out = np.zeros_like(gamma_th_lin)

    for i, th in enumerate(gamma_th_lin):
        Ns = np.count_nonzero(gamma_s < th)  # contagem discreta
        p_out[i] = Ns / N

    return p_out


def p_out_rayleigh_analitico(gamma_th_lin, gamma_med_lin):
    """
    Outage Rayleigh analítico:
      P_out(γ_th) = 1 - exp(-γ_th / γ̄_s)
    """
    return 1.0 - np.exp(-gamma_th_lin / gamma_med_lin)


def p_out_rice_analitico(gamma_th_lin, gamma_med_lin, K):
    """
    Outage Rice analítico usando a função Marcum Q de ordem 1:

      P_out(γ_th) = 1 - Q1( sqrt(2 K), sqrt( 2 * (K+1)/γ̄_s * γ_th ) )

    Aqui:
      - gamma_th_lin: vetor de limiar (linear)
      - gamma_med_lin: SNR média (linear)
      - K: fator de Rice em ESCALA LINEAR (0.1, 1, 10, etc.)
    """
    a = np.sqrt(2.0 * K)
    b = np.sqrt(2.0 * (K + 1.0) * gamma_th_lin / gamma_med_lin)
    return 1.0 - marcum_q1(a, b)


# ============================================================
#  Script principal
# ============================================================

def main():
    # -------------------------
    # Parâmetros da simulação
    # -------------------------
    N = 10**5                          # número de amostras
    gama_s_med_dB = np.array([-20.0, 0.0, 20.0])  # SNR média em dB
    gama_th_dB = np.linspace(-30.0, 30.0, 601)    # limiar de SNR em dB
    gama_th_lin = 10**(gama_th_dB / 10.0)         # limiar em escala linear

    # Fatores de Rice (ESCALA LINEAR)
    K_list = np.array([0.1, 1.0, 10.0])

    # -----------------------------------
    # 1) Canal Rayleigh: analítico x empírico
    # -----------------------------------
    beta_rayleigh, h_rayleigh = gera_rayleigh(N)
    print("E[β^2] Rayleigh ≈", np.mean(beta_rayleigh**2))

    p_out_analit_rayleigh = []
    p_out_emp_rayleigh = []

    for g_med_dB in gama_s_med_dB:
        g_med_lin = 10**(g_med_dB / 10.0)

        # SNR instantânea: γ_s = γ̄_s * β^2
        gama_s = g_med_lin * (beta_rayleigh**2)

        # Outage analítico Rayleigh
        p_analit = p_out_rayleigh_analitico(gama_th_lin, g_med_lin)

        # Outage empírico por contagem
        p_emp = calc_outage(gama_s, gama_th_lin)

        p_out_analit_rayleigh.append(p_analit)
        p_out_emp_rayleigh.append(p_emp)

    p_out_analit_rayleigh = np.array(p_out_analit_rayleigh)
    p_out_emp_rayleigh = np.array(p_out_emp_rayleigh)

    # -----------------------------------
    # 2) Canal Rice: analítico x empírico
    #    para K_R ∈ {0.1, 1, 10}
    # -----------------------------------
    resultados_rice_analit = {}  # dict[(K, g_med_dB)] -> curva analítica
    resultados_rice_emp = {}     # dict[(K, g_med_dB)] -> curva empírica

    for K in K_list:
        beta_rice, h_rice = gera_rice(N, K)
        print(f"E[β^2] Rice (K={K}) ≈", np.mean(beta_rice**2))

        for g_med_dB in gama_s_med_dB:
            g_med_lin = 10**(g_med_dB / 10.0)
            gama_s_r = g_med_lin * (beta_rice**2)

            # Analítico Rice
            p_rice_analit = p_out_rice_analitico(gama_th_lin, g_med_lin, K)
            # Empírico Rice
            p_rice_emp = calc_outage(gama_s_r, gama_th_lin)

            resultados_rice_analit[(K, g_med_dB)] = p_rice_analit
            resultados_rice_emp[(K, g_med_dB)] = p_rice_emp

    # ============================================================
    # 3) Gráficos
    # ============================================================

    cores = ["tab:blue", "tab:orange", "tab:green"]

    # ---------------- Rayleigh ----------------
    plt.figure(figsize=(10, 6))

    for i, g_med_dB in enumerate(gama_s_med_dB):
        # Analítico
        plt.semilogy(
            gama_th_dB,
            p_out_analit_rayleigh[i, :],
            color=cores[i],
            linestyle="-",
            label=f"Rayleigh analítico, γ̄s = {g_med_dB:.0f} dB"
        )
        # Empírico
        plt.semilogy(
            gama_th_dB,
            p_out_emp_rayleigh[i, :],
            color=cores[i],
            linestyle="--",
            marker="o",
            markevery=40,
            label=f"Rayleigh empírico, γ̄s = {g_med_dB:.0f} dB"
        )

    plt.xlabel("Limiar de SNR, γ_th (dB)")
    plt.ylabel("Probabilidade de Outage")
    plt.title("Probabilidade de interrupção em função do limiar γ_th\n"
              "para diferentes valores de SNR média, canal Rayleigh")
    plt.grid(True, which="both", linestyle="--", linewidth=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("prob_outage_Rayleigh.png", dpi=150)
    plt.show(block=False)

    # ---------------- Rice: um gráfico por K ----------------
    for K in K_list:
        plt.figure(figsize=(10, 6))
        K_dB = 10 * np.log10(K)

        for i, g_med_dB in enumerate(gama_s_med_dB):
            p_analit = resultados_rice_analit[(K, g_med_dB)]
            p_emp = resultados_rice_emp[(K, g_med_dB)]

            # Analítico
            plt.semilogy(
                gama_th_dB,
                p_analit,
                color=cores[i],
                linestyle="-",
                label=f"Rice analítico, γ̄s = {g_med_dB:.0f} dB"
            )
            # Empírico
            plt.semilogy(
                gama_th_dB,
                p_emp,
                color=cores[i],
                linestyle="--",
                marker="s",
                markevery=40,
                label=f"Rice empírico, γ̄s = {g_med_dB:.0f} dB"
            )

        plt.xlabel("Limiar de SNR, γ_th (dB)")
        plt.ylabel("Probabilidade de Outage")
        plt.title(f"Probabilidade de interrupção em função do limiar γ_th\n"
                  f"canal Rice, K_R = {K:.1f} (≈ {K_dB:.1f} dB)")
        plt.grid(True, which="both", linestyle="--", linewidth=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"prob_outage_Rice_K_{K:.1f}.png", dpi=150)
        plt.show(block=False)


if __name__ == "__main__":
    main()
