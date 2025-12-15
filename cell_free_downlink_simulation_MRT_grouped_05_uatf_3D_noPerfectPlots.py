
import matplotlib
matplotlib.use("Agg")  # backend não interativo, não cria janelas

import numpy as np
import matplotlib.pyplot as plt

from sharc.campaigns.base_campaign import BaseCampaign


# -------------------------------------------------------------------------
# Função utilitária: CDF empírica (ECDF)
# -------------------------------------------------------------------------
def empirical_cdf(x):
    """
    Retorna (x_sorted, F) para CDF empírica de um vetor 1D x.
    """
    x = np.asarray(x).ravel()
    x_sorted = np.sort(x)
    N = len(x_sorted)
    if N == 0:
        return x_sorted, np.array([])
    F = np.arange(1, N + 1) / N
    return x_sorted, F


class CellFreeDownlinkSimulationCampaign(BaseCampaign):
    """
    Cell-Free Massive MIMO downlink simulation campaign (MRT) com 3 regimes:

        1) perfect       -> upper bound (CSI perfeito para formação de feixe)
        2) statistical   -> baseline simples (apenas estatística de larga-escala)
        3) uatf_mmse     -> modelo principal (MMSE no AP + prelog + SINR UatF fechada)

    Notas importantes (alinhadas com a hipótese típica de CF downlink):
      - Estimação de canal ocorre SOMENTE nos APs via pilotos em uplink
      - UEs NÃO estimam o canal em downlink (sem DL pilots no modelo)
      - Para o caso uatf_mmse, a SINR é calculada via "use-and-then-forget"
        (UE conhece apenas estatísticas efetivas, não realizações instantâneas).
    """

    def __init__(self, parameters: dict):
        super().__init__(parameters)

        # --- Parâmetros básicos ---
        self.fc_GHz = parameters.get("fc_GHz", 3.0)          # frequência de portadora [GHz]
        self.c = 3e8                                         # velocidade da luz [m/s]
        self.lambda_c = self.c / (self.fc_GHz * 1e9)         # comprimento de onda [m]

        self.d0 = parameters.get("d0", 1.0)                  # distância de referência [m]

        self.Bw_MHz = parameters.get("Bw_MHz", 20.0)         # largura de banda [MHz]
        self.noise_figure_dB = parameters.get("noise_figure_dB", 9.0)  # fator de ruído [dB]

        self.h_AP = parameters.get("h_AP", 15.0)             # altura APs [m]
        self.h_UE = parameters.get("h_UE", 1.65)             # altura UEs [m]

        self.T = parameters.get("T", 296.15)                 # temperatura [K]
        self.k_B = 1.38064852e-23                            # constante de Boltzmann [J/K]

        self.Lx = parameters.get("Lx", 1000.0)               # comprimento da área [m]
        self.Ly = parameters.get("Ly", 1000.0)               # largura da área [m]

        # Flag (herdada, não usada aqui)
        self.interactive_plots = parameters.get("interactive_plots", False)

        # Valores iniciais de M e K (podem ser sobrescritos nas varreduras)
        self.num_APs = parameters.get("num_APs", 100)
        self.num_UEs = parameters.get("num_UEs", 20)

        # Expoente de path-loss
        self.n_PL = parameters.get("n_PL", 2.8)

        # Shadow fading [dB]
        self.sigma_sf_dB = parameters.get("sigma_sf_dB", 8.0)

        # --- Parâmetros de UL pilots / DL ---
        self.Pp_mW = parameters.get("Pp_mW", 200.0)          # potência de piloto UL [mW]
        self.Pp_W = self.Pp_mW / 1000.0                      # [W]

        self.Pdl_mW = parameters.get("Pdl_mW", 200.0)        # potência DL [mW] (por AP)
        self.Pdl_W = self.Pdl_mW / 1000.0                    # [W]

        # Comprimento dos pilotos (tau_p = tau_cf)
        self.Tau_p = int(parameters.get("Tau_p", 50))        # símbolos de piloto
        self.tau_p = max(self.Tau_p, 1)

        # Bloco de coerência: tau_c = Tc * Bc (em amostras)
        self.Tc_ms = float(parameters.get("Tc_ms", 1.0))     # [ms]
        self.Bc_kHz = float(parameters.get("Bc_kHz", 200.0)) # [kHz]
        self.tau_c = int(np.floor((self.Tc_ms * 1e-3) * (self.Bc_kHz * 1e3)))
        self.tau_c = max(self.tau_c, 1)

        if self.tau_p >= self.tau_c:
            # garante ao menos 1 símbolo útil (evita divisão por zero)
            self.tau_p = max(self.tau_c - 1, 1)

        self.tau_dl = max(self.tau_c - self.tau_p, 1)
        self.prelog = self.tau_dl / self.tau_c  # fator (1 - tau_p/tau_c)

        # Número de blocos de coerência por rede
        self.Nbc = int(parameters.get("Nbc", 100))

        # Número de redes (topologias) avaliadas
        self.Ncf = int(parameters.get("Ncf", 300))

        # Normalização de potência: "per_AP" ou "network"
        self.power_normalization_mode = parameters.get("power_normalization_mode", "per_AP")

        # Precoder: MRT
        modo_param = parameters.get("precoder_mode", "MRT").lower()
        if modo_param not in ["mrt", "mrt_only"]:
            print("Aviso: este script suporta apenas MRT. Ignorando precoder_mode passado e usando MRT.")
        self.precoder_mode = "MRT"

        # Pilotos: estratégia de alocação
        # - "orthogonal_if_possible": se tau_p >= K -> ortogonal; senão, reuse (k mod tau_p)
        self.pilot_assignment = parameters.get("pilot_assignment", "orthogonal_if_possible")

        # Validação por Monte Carlo do caso MMSE (opcional)
        self.validate_mmse_mc = bool(parameters.get("validate_mmse_mc", False))
        self.Nbc_validation = int(parameters.get("Nbc_validation", min(self.Nbc, 50)))

        # Flags de plot / saída
        self.do_plots = parameters.get("do_plots", True)
        self.save_figs = parameters.get("save_figs", True)
        self.fig_prefix = parameters.get("fig_prefix", "cf_dl")
        self.verbose = parameters.get("verbose", True)

        # Quais curvas de CSI devem aparecer nos gráficos (por padrão: sem "perfect")
        self.plot_csi_types = parameters.get("plot_csi_types", ["statistical", "uatf_mmse"])
       
        # RNG (opcionalmente reproduzível)
        self.seed = parameters.get("seed", None)
        self.rng = np.random.default_rng(self.seed)

    # -------------------------------------------------------------------------
    # Helpers básicos
    # -------------------------------------------------------------------------
    @property
    def Bw_Hz(self):
        return self.Bw_MHz * 1e6

    @property
    def noise_figure_linear(self):
        return 10 ** (self.noise_figure_dB / 10.0)

    # -------------------------------------------------------------------------
    # Cálculos físicos
    # -------------------------------------------------------------------------
    def calculate_noise_power(self):
        """
        Potência de ruído térmico na recepção do UE em Watts:
        Pn = k_B * T * Bw * F
        """
        return self.k_B * self.T * self.Bw_Hz * self.noise_figure_linear

    def distribute_nodes(self):
        """
        Distribui APs e UEs uniformemente em (x,y) na área [0,Lx]x[0,Ly],
        e adiciona a coordenada de altura (z) fixa para cada tipo.
        Retorna:
            AP_positions: (M, 3) -> (x, y, h_AP)
            UE_positions: (K, 3) -> (x, y, h_UE)
        """
        AP_xy = self.rng.random((self.num_APs, 2)) * np.array([self.Lx, self.Ly])
        UE_xy = self.rng.random((self.num_UEs, 2)) * np.array([self.Lx, self.Ly])
        AP_z = np.full((self.num_APs, 1), float(self.h_AP))
        UE_z = np.full((self.num_UEs, 1), float(self.h_UE))
        AP_positions = np.hstack([AP_xy, AP_z])
        UE_positions = np.hstack([UE_xy, UE_z])
        return AP_positions, UE_positions

    @staticmethod
    def calculate_distances(P_AP, P_UE):
        """Calcula distâncias Euclidianas 3D entre APs e UEs.

        P_AP: (M, 3) -> (x, y, z)
        P_UE: (K, 3) -> (x, y, z)
        Retorna: distances (M, K)
        """
        M = P_AP.shape[0]
        K = P_UE.shape[0]
        distances = np.zeros((M, K))
        for m in range(M):
            for k in range(K):
                distances[m, k] = np.linalg.norm(P_AP[m] - P_UE[k])
        distances[distances < 1e-3] = 1e-3
        return distances

    def path_loss_CI_dB(self, distances):
        fc_Hz = self.fc_GHz * 1e9
        PL_d0_dB = 20 * np.log10(4 * np.pi * self.d0 * fc_Hz / self.c)
        X_sf_dB = self.rng.normal(0.0, self.sigma_sf_dB, size=distances.shape)
        PL_dB = PL_d0_dB + 10 * self.n_PL * np.log10(distances / self.d0) + X_sf_dB
        return PL_dB

    @staticmethod
    def path_loss_linear(PL_dB):
        return 10 ** (-PL_dB / 10.0)

    def small_scale_fading(self, num_APs, num_UEs):
        hI = self.rng.normal(0, 1 / np.sqrt(2), (num_APs, num_UEs))
        hQ = self.rng.normal(0, 1 / np.sqrt(2), (num_APs, num_UEs))
        return hI + 1j * hQ

    @staticmethod
    def overall_channel_coefficients(h_mk, beta_mk):
        return h_mk * np.sqrt(beta_mk)

    # -------------------------------------------------------------------------
    # Pilotos + MMSE: parâmetros fechados (gamma) e correlações
    # -------------------------------------------------------------------------
    def assign_pilots(self, K):
        """
        Retorna pilot_index[k] em {0,...,tau_p-1}.

        - Se tau_p >= K: pilotos ortogonais (sem contaminação)
        - Se tau_p < K: reuse simples (k mod tau_p)
        """
        if self.pilot_assignment == "orthogonal_if_possible":
            if self.tau_p >= K:
                return np.arange(K)  # "índice" único por UE (não há reuso)
            return np.mod(np.arange(K), self.tau_p)

        # fallback: sempre reuse
        return np.mod(np.arange(K), self.tau_p)

    def compute_mmse_gamma_and_corr(self, beta, pilot_index, noise_power_W):
        """
        Modelo: g_mk ~ CN(0, beta_mk) e estimação MMSE nos APs via UL pilots.

        Retorna:
          gamma[m,k] = E{|ĝ_mk|^2} (qualidade da estimação)
          corr[m,k,j] = E{ g_mk^* ĝ_mj }  (necessário para interferência coerente com pilot contamination)

        Observação:
          corr != 0 apenas quando UE k e UE j usam o mesmo piloto.
        """
        M, K = beta.shape
        gamma = np.zeros((M, K), dtype=np.float64)

        # Pré-computar "psi" por grupo de pilotos
        # psi_{m,p} = tau_p * Pp * sum_{i:pilot_i=p} beta_{m,i} + noise
        pilots = np.unique(pilot_index)
        psi = np.zeros((M, len(pilots)), dtype=np.float64)
        pilot_to_col = {p: idx for idx, p in enumerate(pilots)}

        for p in pilots:
            users = np.where(pilot_index == p)[0]
            beta_sum = np.sum(beta[:, users], axis=1)  # (M,)
            psi[:, pilot_to_col[p]] = self.tau_p * self.Pp_W * beta_sum + noise_power_W

        for k in range(K):
            p = pilot_index[k]
            psi_m = psi[:, pilot_to_col[p]]
            gamma[:, k] = (self.tau_p * self.Pp_W * (beta[:, k] ** 2)) / psi_m

        # corr[m,k,j] apenas quando mesmo piloto:
        # E{ g_mk^* ĝ_mj } = tau_p * Pp * beta_mk * beta_mj / psi_{m,pilot(j)}
        # equivalente a: beta_mk * (gamma_mj / beta_mj)   (quando beta_mj>0)
        corr = np.zeros((M, K, K), dtype=np.float64)

        for j in range(K):
            p = pilot_index[j]
            psi_m = psi[:, pilot_to_col[p]]
            # fator comum por m: tau_p*Pp*beta_mj/psi
            fac = (self.tau_p * self.Pp_W * beta[:, j]) / psi_m  # (M,)
            same = np.where(pilot_index == p)[0]
            # para todos k que compartilham o piloto de j:
            # corr[m,k,j] = beta[m,k] * fac[m]
            corr[:, same, j] = beta[:, same] * fac[:, None]

        return gamma, corr

    # -------------------------------------------------------------------------
    # Precoding (MRT) + normalização de potência (igual ao seu estilo)
    # -------------------------------------------------------------------------
    def compute_precoder_matrix(self, H_eff, precoder_type):
        """
        MRT (alinhado com y = H^H x): V = H_eff
        """
        if precoder_type.upper() == "MRT":
            return H_eff.astype(np.complex128)
        raise ValueError(f"Tipo de precoder desconhecido (esperado MRT): {precoder_type}")

    def normalize_precoder(self, V):
        """
        Normalização instantânea, como no seu script original:
        - per_AP: cada linha tem potência = Pdl_W
        - network: potência total = M * Pdl_W
        """
        M, _ = V.shape
        if self.power_normalization_mode == "per_AP":
            W = np.zeros_like(V, dtype=np.complex128)
            for m in range(M):
                row = V[m, :]
                row_power = np.sum(np.abs(row) ** 2)
                if row_power > 0:
                    alpha_m = np.sqrt(self.Pdl_W / row_power)
                    W[m, :] = alpha_m * row
        elif self.power_normalization_mode == "network":
            total_power_target = M * self.Pdl_W
            fro2 = np.sum(np.abs(V) ** 2)
            if fro2 > 0:
                alpha = np.sqrt(total_power_target / fro2)
                W = alpha * V
            else:
                W = np.zeros_like(V, dtype=np.complex128)
        else:
            raise ValueError(f"Modo de normalização de potência desconhecido: {self.power_normalization_mode}")
        return W

    # -------------------------------------------------------------------------
    # SINR instantânea (Monte Carlo) + taxa efetiva (com prelog)
    # -------------------------------------------------------------------------
    def compute_dl_sinr_and_rate(self, H_actual, W, noise_power_W):
        Heff = H_actual.conj().T @ W  # K x K

        desired_power = np.abs(np.diag(Heff)) ** 2
        total_power = np.sum(np.abs(Heff) ** 2, axis=1)
        interference_power = total_power - desired_power

        sinr_linear = desired_power / (interference_power + noise_power_W)
        sinr_linear = np.maximum(sinr_linear, 1e-15)

        # taxa efetiva: inclui overhead de pilotos (prelog)
        rate_bps = self.prelog * self.Bw_Hz * np.log2(1.0 + sinr_linear)
        sinr_dB = 10 * np.log10(sinr_linear)
        return sinr_linear, sinr_dB, rate_bps

    # -------------------------------------------------------------------------
    # SINR UatF fechada (analítica) para MR + MMSE (modelo principal)
    # -------------------------------------------------------------------------
    def compute_uatf_sinr_mr_mmse(self, beta, gamma, corr, noise_power_W):
        """
        Calcula SINR UatF por UE (vetor K, determinístico para uma topologia),
        assumindo precoding MR baseado em ĝ e normalização por potência média.

        Convenção:
          - y_k = h_k^H x + n
          - w_{m,j} = alpha_m * ĝ_{m,j}
          - Restrição de potência média por AP:
              E[ sum_j |w_{m,j}|^2 ] = alpha_m^2 * sum_j gamma_{m,j} = Pdl_W

        Termos (para UE k):
          A_k = E{ h_k^H w_k } = sum_m alpha_m * gamma_{m,k}
          E{|h_k^H w_j|^2} = sum_m alpha_m^2 * beta_{m,k} * gamma_{m,j} + |sum_m alpha_m * corr_{m,k,j}|^2

        Denominador UatF:
          sum_j E{|h_k^H w_j|^2} - |A_k|^2 + noise
        """
        M, K = beta.shape

        # alpha_m por potência média
        if self.power_normalization_mode == "per_AP":
            sum_gamma_m = np.sum(gamma, axis=1)  # (M,)
            alpha_m = np.zeros(M, dtype=np.float64)
            nonzero = sum_gamma_m > 0
            alpha_m[nonzero] = np.sqrt(self.Pdl_W / sum_gamma_m[nonzero])
        else:
            sum_gamma = np.sum(gamma)
            if sum_gamma > 0:
                alpha = np.sqrt((M * self.Pdl_W) / sum_gamma)
            else:
                alpha = 0.0
            alpha_m = np.full(M, alpha, dtype=np.float64)

        # A_k (K,)
        A = (alpha_m[:, None] * gamma).sum(axis=0)  # real >=0

        # C_kj = sum_m alpha_m^2 * beta_mk * gamma_mj  -> (K,K)
        alpha2 = (alpha_m ** 2)[:, None, None]          # (M,1,1)
        beta_mk = beta[:, :, None]                       # (M,K,1)
        gamma_mj = gamma[:, None, :]                     # (M,1,K)
        C = np.sum(alpha2 * beta_mk * gamma_mj, axis=0)   # (K,K)

        # B_kj = |sum_m alpha_m * corr_{m,k,j}|^2  -> (K,K)
        B = np.zeros((K, K), dtype=np.float64)
        for k in range(K):
            # corr[:,k,:] => (M,K)
            s = np.sum(alpha_m[:, None] * corr[:, k, :], axis=0)  # (K,)
            B[k, :] = np.abs(s) ** 2

        total = C + B  # (K,K) => E{|h^H w_j|^2}

        signal = np.abs(A) ** 2                      # (K,)
        denom = np.sum(total, axis=1) - signal + noise_power_W
        denom = np.maximum(denom, 1e-15)

        sinr = signal / denom
        sinr = np.maximum(sinr, 1e-15)

        rate_bps = self.prelog * self.Bw_Hz * np.log2(1.0 + sinr)
        return sinr, rate_bps

    # -------------------------------------------------------------------------
    # Rotina principal de uma rede (topologia) com Nbc blocos de coerência
    # -------------------------------------------------------------------------
    def simulate_one_network(self):
        """
        Retorna dicionários com amostras de SINR e taxa:
            sinr_samples[(prec, csi)]  -> array shape (Nbc * K,) (por compatibilidade com seus plots)
            rate_samples[(prec, csi)]  -> array shape (Nbc * K,)
        """
        M = self.num_APs
        K = self.num_UEs

        AP_positions, UE_positions = self.distribute_nodes()
        distances = self.calculate_distances(AP_positions, UE_positions)
        PL_dB = self.path_loss_CI_dB(distances)
        beta = self.path_loss_linear(PL_dB)  # beta_mk

        noise_power_W = self.calculate_noise_power()

        precoders = ["MRT"]
        csi_types = ["perfect", "statistical", "uatf_mmse"]
        if self.validate_mmse_mc:
            csi_types.append("mmse_mc")  # apenas validação

        sinr_samples = {(p, c): [] for p in precoders for c in csi_types}
        rate_samples = {(p, c): [] for p in precoders for c in csi_types}

        # --- regime principal: UatF MMSE (uma vez por topologia) ---
        pilot_index = self.assign_pilots(K)
        gamma, corr = self.compute_mmse_gamma_and_corr(beta, pilot_index, noise_power_W)
        sinr_uatf, rate_uatf = self.compute_uatf_sinr_mr_mmse(beta, gamma, corr, noise_power_W)

        # Repete Nbc vezes só para manter peso igual aos outros modos no ECDF
        sinr_samples[("MRT", "uatf_mmse")] = np.tile(sinr_uatf, self.Nbc)
        rate_samples[("MRT", "uatf_mmse")] = np.tile(rate_uatf, self.Nbc)

        # --- blocos para perfect/statistical (Monte Carlo instantâneo) ---
        for _ in range(self.Nbc):
            h = self.small_scale_fading(M, K)
            H_actual = self.overall_channel_coefficients(h, beta)  # M x K

            for csi in ["perfect", "statistical"]:
                if csi == "perfect":
                    H_eff = H_actual
                else:
                    # baseline simples: apenas larga-escala (sem fase)
                    H_eff = np.sqrt(beta).astype(np.complex128)

                V = self.compute_precoder_matrix(H_eff, "MRT")
                W = self.normalize_precoder(V)

                sinr_lin, _, rate_bps = self.compute_dl_sinr_and_rate(H_actual, W, noise_power_W)
                sinr_samples[("MRT", csi)].append(sinr_lin)
                rate_samples[("MRT", csi)].append(rate_bps)

        # --- validação Monte Carlo MMSE (opcional; custo maior) ---
        if self.validate_mmse_mc:
            # geração explícita de estimativas via modelo estatístico (sem construir Yp completo)
            # Para validação: cria ĝ = a * (sum_{i em mesmo piloto} g_i + noise_proj)
            # onde noise_proj ~ CN(0, noise) após correlação com piloto (||phi||=1).
            pilots = np.unique(pilot_index)
            # pré-compute psi por piloto p: psi_{m,p}
            psi = np.zeros((M, len(pilots)), dtype=np.float64)
            pilot_to_col = {p: idx for idx, p in enumerate(pilots)}
            for p in pilots:
                users = np.where(pilot_index == p)[0]
                beta_sum = np.sum(beta[:, users], axis=1)
                psi[:, pilot_to_col[p]] = self.tau_p * self.Pp_W * beta_sum + noise_power_W

            for _ in range(self.Nbc_validation):
                h = self.small_scale_fading(M, K)
                G = self.overall_channel_coefficients(h, beta)  # canal real (M x K)

                # ruído após correlação com piloto: CN(0, noise_power_W)
                nproj = (self.rng.normal(0, np.sqrt(noise_power_W/2), (M, K))
                         + 1j*self.rng.normal(0, np.sqrt(noise_power_W/2), (M, K)))

                # construir ĝ
                Ghat = np.zeros((M, K), dtype=np.complex128)
                for k in range(K):
                    p = pilot_index[k]
                    users = np.where(pilot_index == p)[0]
                    ytilde = np.sqrt(self.tau_p * self.Pp_W) * np.sum(G[:, users], axis=1) + nproj[:, k]
                    c_mk = (np.sqrt(self.tau_p * self.Pp_W) * beta[:, k]) / psi[:, pilot_to_col[p]]
                    Ghat[:, k] = c_mk * ytilde

                # Precoder MR baseado em ĝ
                V = self.compute_precoder_matrix(Ghat, "MRT")
                W = self.normalize_precoder(V)

                sinr_lin, _, rate_bps = self.compute_dl_sinr_and_rate(G, W, noise_power_W)
                sinr_samples[("MRT", "mmse_mc")].append(sinr_lin)
                rate_samples[("MRT", "mmse_mc")].append(rate_bps)

        # concatena listas
        for key in list(sinr_samples.keys()):
            if isinstance(sinr_samples[key], list):
                if len(sinr_samples[key]) > 0:
                    sinr_samples[key] = np.concatenate(sinr_samples[key], axis=0)
                    rate_samples[key] = np.concatenate(rate_samples[key], axis=0)
                else:
                    sinr_samples[key] = np.array([])
                    rate_samples[key] = np.array([])

        return sinr_samples, rate_samples

    # -------------------------------------------------------------------------
    # Rotina para Ncf redes (para CDFs globais)
    # -------------------------------------------------------------------------
    def run_multi_realizations_for_config(self, M, K, exp_label="", generate_plots=True):
        self.num_APs = M
        self.num_UEs = K

        precoders = ["MRT"]

        # CSI que existem no simulador (computação / armazenamento)
        csi_types = ["perfect", "statistical", "uatf_mmse"]
        if self.validate_mmse_mc:
            csi_types.append("mmse_mc")

        # CSI que serão desenhados/impressos (sempre definido!)
        # Se você definiu self.plot_csi_types no __init__, ele será usado aqui.
        plot_pref = getattr(self, "plot_csi_types", ["statistical", "uatf_mmse"])
        csi_types_plot = [c for c in plot_pref if c in csi_types]
        if not csi_types_plot:
            csi_types_plot = ["statistical", "uatf_mmse"]

        sinr_global = {(p, c): [] for p in precoders for c in csi_types}
        rate_global = {(p, c): [] for p in precoders for c in csi_types}

        for _ in range(self.Ncf):
            sinr_net, rate_net = self.simulate_one_network()
            for key in sinr_global.keys():
                sinr_global[key].append(sinr_net[key])
                rate_global[key].append(rate_net[key])

        for key in sinr_global.keys():
            if len(sinr_global[key]) > 0:
                sinr_global[key] = np.concatenate(sinr_global[key], axis=0)
                rate_global[key] = np.concatenate(rate_global[key], axis=0)
            else:
                sinr_global[key] = np.array([])
                rate_global[key] = np.array([])

        if self.verbose:
            print(f"\n=== Downlink: Configuração M={M}, K={K} {exp_label} ===")
            print(f"  tau_c={self.tau_c}, tau_p={self.tau_p}, tau_dl={self.tau_dl}, prelog={self.prelog:.4f}")
            for prec in precoders:
                # Se quiser imprimir tudo (inclusive perfect), troque csi_types_plot -> csi_types
                for csi in csi_types_plot:
                    key = (prec, csi)
                    if sinr_global[key].size == 0:
                        continue
                    sinr_dB = 10 * np.log10(sinr_global[key])
                    rate_bps = rate_global[key]
                    print(f"Modo: {prec}, CSI: {csi}")
                    print("  SINR médio [dB]:", float(np.mean(sinr_dB)))
                    print("  Taxa média [bps]:", float(np.mean(rate_bps)))
                    print("  Taxa média [Mbps]:", float(np.mean(rate_bps) / 1e6))

        # (plots individuais por (M,K) continuam opcionais)
        if self.do_plots and generate_plots:
            for prec in precoders:
                # SINR
                fig1 = plt.figure()
                for csi in csi_types_plot:
                    key = (prec, csi)
                    sinr_vals = sinr_global[key]
                    if sinr_vals.size == 0:
                        continue
                    sinr_dB_vals = 10 * np.log10(sinr_vals)
                    x_sinr, F_sinr = empirical_cdf(sinr_dB_vals)
                    plt.plot(x_sinr, F_sinr, label=f"CSI={csi}")
                plt.xlabel("SINR [dB]")
                plt.ylabel("F(x)")
                plt.title(f"CDF da SINR DL - {prec}, M={M}, K={K}, {exp_label}")
                plt.grid(True)
                plt.legend()
                if self.save_figs:
                    fig1.savefig(
                        f"{self.fig_prefix}_DL_{exp_label}_M{M}_K{K}_{prec}_cdf_sinr_selCSI.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                plt.close(fig1)

                # Taxa
                fig2 = plt.figure()
                for csi in csi_types_plot:
                    key = (prec, csi)
                    rate_vals = rate_global[key]
                    if rate_vals.size == 0:
                        continue
                    rate_Mbps = rate_vals / 1e6
                    x_rate, F_rate = empirical_cdf(rate_Mbps)
                    plt.plot(x_rate, F_rate, label=f"CSI={csi}")
                plt.xlabel("Taxa [Mbps]")
                plt.ylabel("F(x)")
                plt.title(f"CDF da taxa DL - {prec}, M={M}, K={K}, {exp_label}")
                plt.grid(True)
                plt.legend()
                if self.save_figs:
                    fig2.savefig(
                        f"{self.fig_prefix}_DL_{exp_label}_M{M}_K{K}_{prec}_cdf_rate_selCSI.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                plt.close(fig2)

        return sinr_global, rate_global

        # -------------------------------------------------------------------------
    # Método exigido pelo BaseCampaign
    # -------------------------------------------------------------------------
    def run_simulation(self):
        """
        Geração de gráficos AGREGADOS, mantendo o estilo:

        - varyM: M ∈ {100,150,200} com K = 20
        - varyK: K ∈ {10,20,30}  com M = 100

        Observação:
        - O simulador pode calcular vários tipos de CSI (csi_types),
          mas os gráficos são controlados por self.plot_csi_types (csi_types_plot).
        """
        # Lista de configurações simuladas
        configs = [
            ("varyM", 100, 20),
            ("varyM", 150, 20),
            ("varyM", 200, 20),
            ("varyK", 100, 10),
            ("varyK", 100, 20),
            ("varyK", 100, 30),
        ]

        # Executa simulação para cada configuração, armazenando resultados
        all_results = {}  # (exp_label, M, K) -> (sinr_global, rate_global)
        last_sinr_global = None
        last_rate_global = None

        for exp_label, M, K in configs:
            sinr_global, rate_global = self.run_multi_realizations_for_config(
                M=M,
                K=K,
                exp_label=exp_label,
                generate_plots=False,  # plots agregados serão feitos aqui
            )
            all_results[(exp_label, M, K)] = (sinr_global, rate_global)
            last_sinr_global = sinr_global
            last_rate_global = rate_global

        # -------------------------------------------------------------
        # Gráficos agregados
        # -------------------------------------------------------------
        if self.do_plots:
            prec = "MRT"

            # CSI que existem no simulador (podem estar nos dicionários)
            csi_types = ["perfect", "statistical", "uatf_mmse"]
            if getattr(self, "validate_mmse_mc", False):
                csi_types.append("mmse_mc")

            # CSI que serão plotados (por padrão: sem perfect)
            plot_pref = getattr(self, "plot_csi_types", ["statistical", "uatf_mmse"])
            csi_types_plot = [c for c in plot_pref if c in csi_types]
            if not csi_types_plot:
                csi_types_plot = ["statistical", "uatf_mmse"]

            # Estilos (somente para os que plotamos; fallback "-" se faltar)
            csi_linestyle = {
                "uatf_mmse": "-",      # principal (contínuo)
                "statistical": "--",   # baseline (tracejado)
                "mmse_mc": ":",        # validação (pontilhado)
                # "perfect": "-"  # intencionalmente fora (não plotar)
                }
            
            csi_label = {
                "statistical": "Stat.",
                "uatf_mmse": "UatF-MMSE",
                "mmse_mc": "MMSE-MC",
                }
                      
            lw = 2.2

            def beautify_plot(ax, xlabel, ylabel, title):
                ax.set_xlabel(xlabel, fontsize=11)
                ax.set_ylabel(ylabel, fontsize=11)
                ax.set_title(title, fontsize=12, fontweight="bold")

                # Grid leve (IEEE-friendly)
                ax.grid(True, alpha=0.25)

                # CDF sempre em [0, 1]
                ax.set_ylim(-0.02, 1.02)

                # Margem automática em x
                xmin, xmax = ax.get_xlim()
                dx = (xmax - xmin) * 0.05
                ax.set_xlim(xmin - dx, xmax + dx)

                # Legenda fora do gráfico (padrão paper)
                ax.legend(
                    fontsize=10,
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.22),
                    ncol=3,
                    framealpha=0.85,
                )


            # -------------------------------
            # 1) M = 100, variar K – CDF SINR
            # -------------------------------
            Ks = [10, 20, 30]
            color_map_K = {10: "tab:blue", 20: "tab:orange", 30: "tab:green"}

            fig, ax = plt.subplots(figsize=(8, 6))
            for K in Ks:
                sinr_global_cfg = None
                for (exp_label_, M_cfg, K_cfg), (sinr_g, _) in all_results.items():
                    if M_cfg == 100 and K_cfg == K:
                        sinr_global_cfg = sinr_g
                        break
                if sinr_global_cfg is None:
                    continue

                for csi in csi_types_plot:
                    key = (prec, csi)
                    sinr_vals = sinr_global_cfg.get(key, np.array([]))
                    if sinr_vals.size == 0:
                        continue
                    sinr_dB_vals = 10 * np.log10(sinr_vals)
                    x_sinr, F_sinr = empirical_cdf(sinr_dB_vals)
                    ax.plot(
                        x_sinr,
                        F_sinr,
                        linestyle=csi_linestyle.get(csi, "-"),
                        color=color_map_K[K],
                        linewidth=lw,
                        label=f"K={K}, {csi_label.get(csi, csi)}",
                        )

            beautify_plot(ax, "SINR [dB]", "CDF", "Downlink SINR CDF (M=100, K∈{10,20,30})")
            ax.set_xlim(-30, 40)
            if self.save_figs:
                fig.savefig(
                    f"{self.fig_prefix}_DL_varyK_M100_cdf_sinr_allK.png",
                    dpi=300,
                    bbox_inches="tight",
                )
            plt.close(fig)

            # -------------------------------
            # 2) M = 100, variar K – CDF Taxa
            # -------------------------------
            fig, ax = plt.subplots(figsize=(8, 6))
            for K in Ks:
                rate_global_cfg = None
                for (exp_label_, M_cfg, K_cfg), (_, rate_g) in all_results.items():
                    if M_cfg == 100 and K_cfg == K:
                        rate_global_cfg = rate_g
                        break
                if rate_global_cfg is None:
                    continue

                for csi in csi_types_plot:
                    key = (prec, csi)
                    rate_vals = rate_global_cfg.get(key, np.array([]))
                    if rate_vals.size == 0:
                        continue
                    rate_Mbps = rate_vals / 1e6
                    x_rate, F_rate = empirical_cdf(rate_Mbps)
                    ax.plot(
                        x_rate,
                        F_rate,
                        linestyle=csi_linestyle.get(csi, "-"),
                        color=color_map_K[K],
                        linewidth=lw,
                        label=f"K={K}, {csi_label.get(csi, csi)}",
                    )

            beautify_plot(ax, "Rate [Mbit/s]", "CDF", "Downlink Rate CDF (M=100, K∈{10,20,30})")
            ax.set_xlim(0, 250)
            if self.save_figs:
                fig.savefig(
                    f"{self.fig_prefix}_DL_varyK_M100_cdf_rate_allK.png",
                    dpi=300,
                    bbox_inches="tight",
                )
            plt.close(fig)

            # -------------------------------
            # 3) K = 20, variar M – CDF SINR
            # -------------------------------
            Ms = [100, 150, 200]
            color_map_M = {100: "tab:blue", 150: "tab:orange", 200: "tab:green"}

            fig, ax = plt.subplots(figsize=(8, 6))
            for M in Ms:
                sinr_global_cfg = None
                for (exp_label_, M_cfg, K_cfg), (sinr_g, _) in all_results.items():
                    if K_cfg == 20 and M_cfg == M:
                        sinr_global_cfg = sinr_g
                        break
                if sinr_global_cfg is None:
                    continue

                for csi in csi_types_plot:
                    key = (prec, csi)
                    sinr_vals = sinr_global_cfg.get(key, np.array([]))
                    if sinr_vals.size == 0:
                        continue
                    sinr_dB_vals = 10 * np.log10(sinr_vals)
                    x_sinr, F_sinr = empirical_cdf(sinr_dB_vals)
                    ax.plot(
                        x_sinr,
                        F_sinr,
                        linestyle=csi_linestyle.get(csi, "-"),
                        color=color_map_M[M],
                        linewidth=lw,
                        label=f"M={M}, {csi_label.get(csi, csi)}",
                    )

            beautify_plot(ax, "SINR [dB]", "CDF", "Downlink SINR CDF (K=20, M∈{100,150,200})")
            ax.set_xlim(-30, 40)
            if self.save_figs:
                fig.savefig(
                    f"{self.fig_prefix}_DL_varyM_K20_cdf_sinr_allM.png",
                    dpi=300,
                    bbox_inches="tight",
                )
            plt.close(fig)

            # -------------------------------
            # 4) K = 20, variar M – CDF Taxa
            # -------------------------------
            fig, ax = plt.subplots(figsize=(8, 6))
            for M in Ms:
                rate_global_cfg = None
                for (exp_label_, M_cfg, K_cfg), (_, rate_g) in all_results.items():
                    if K_cfg == 20 and M_cfg == M:
                        rate_global_cfg = rate_g
                        break
                if rate_global_cfg is None:
                    continue

                for csi in csi_types_plot:
                    key = (prec, csi)
                    rate_vals = rate_global_cfg.get(key, np.array([]))
                    if rate_vals.size == 0:
                        continue
                    rate_Mbps = rate_vals / 1e6
                    x_rate, F_rate = empirical_cdf(rate_Mbps)
                    ax.plot(
                        x_rate,
                        F_rate,
                        linestyle=csi_linestyle.get(csi, "-"),
                        color=color_map_M[M],
                        linewidth=lw,
                        label=f"M={M}, {csi_label.get(csi, csi)}",
                    )

            beautify_plot(ax, "Rate [Mbit/s]", "CDF", "Downlink Rate CDF (K=20, M∈{100,150,200})")
            ax.set_xlim(0, 250)
            if self.save_figs:
                fig.savefig(
                    f"{self.fig_prefix}_DL_varyM_K20_cdf_rate_allM.png",
                    dpi=300,
                    bbox_inches="tight",
                )
            plt.close(fig)

        return last_sinr_global, last_rate_global


    def run(self):
        return self.run_simulation()


if __name__ == "__main__":
    parameters = {
        "fc_GHz": 3.0,
        "Bw_MHz": 20.0,
        "noise_figure_dB": 9.0,
        "Lx": 1000.0,
        "Ly": 1000.0,
        "num_APs": 100,
        "num_UEs": 20,
        "n_PL": 2.8,
        "sigma_sf_dB": 8.0,
        "Pp_mW": 200.0,
        "Pdl_mW": 200.0,

        # Coerência:
        "Tc_ms": 1.0,
        "Bc_kHz": 200.0,

        # Pilotos:
        "Tau_p": 50,
        "pilot_assignment": "orthogonal_if_possible",

        "Nbc": 100,
        "Ncf": 300,
        "power_normalization_mode": "per_AP",
        "precoder_mode": "MRT",

        # Validação MMSE por Monte Carlo (opcional)
        "validate_mmse_mc": False,
        "Nbc_validation": 50,

        "do_plots": True,
        "save_figs": True,
        "fig_prefix": "cf_dl",
        "verbose": True,
        "interactive_plots": False,
        "plot_csi_types": ["statistical", "uatf_mmse"],


        # Reprodutibilidade (opcional)
        "seed": None,
    }

campaign = CellFreeDownlinkSimulationCampaign(parameters)
campaign.run()
print("\nSimulações de downlink concluídas.")
