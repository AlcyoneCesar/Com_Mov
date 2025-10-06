# === SCRIPT_VERSION: 6 ===
import sys
import os
from pathlib import Path
import traceback
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # pode ser necessário em algumas instalações



print("=== SCRIPT_VERSION: 6 ===")
print(f"__file__: {__file__}")
print(f"CWD     : {os.getcwd()}")
print(f"Python  : {sys.executable}")
print(f"Universidade de Brasília (UnB)")
print(f"Disciplina de Comunicações Móveis - 2025/2")
print(f"Prof. Dr. Higo Thaian Pereira da Silva")
print(f"Aluno: Alcyone César Pereira Silva")
print(f"Projeto 1: Atraso Multipercurso - Modelo de Canal")
print(f"Data: 03/10/2025")

# --------------------------
# Configuração da pasta de saída
# --------------------------
OUTPUT_DIR = Path(r"C:\New_sharc\Graficos_Canal")

def ensure_output_dir(p: Path) -> Path:
    try:
        p.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Pasta de saída pronta: {p}")
        return p
    except Exception as e:
        print(f"[ERRO] Não foi possível criar {p}:\n{e}")
        raise

OUTPUT_DIR = ensure_output_dir(OUTPUT_DIR)

# --------------------------
# Função de salvamento com verificação (com controle de fechamento)
# --------------------------
SHOW_ALL_AT_END = True  # <- True: abre tudo no plt.show() final

def save_figure(fig, filename: str, desc: str, close_after_save=None):
    """
    Salva a figura em OUTPUT_DIR. Se close_after_save=None, obedece SHOW_ALL_AT_END:
      - SHOW_ALL_AT_END=True  -> NÃO fecha agora; plt.show() no fim abre todas.
      - SHOW_ALL_AT_END=False -> fecha após salvar (comportamento clássico).
    """
    out_path = OUTPUT_DIR / filename
    try:
        fig.savefig(out_path, dpi=300)
        if close_after_save is None:
            close_after_save = not SHOW_ALL_AT_END
        if close_after_save:
            fig.clf()
            plt.close(fig)
        print(f"[OK] {desc} salvo em: {out_path}")
    except Exception:
        print(f"[ERRO] Falha ao salvar {desc} em {out_path}")
        traceback.print_exc()
        raise

def maybe_close(fig):
    """Fecha a figura só se SHOW_ALL_AT_END=False (compatível com blocos antigos)."""
    if not SHOW_ALL_AT_END:
        fig.clf()
        plt.close(fig)

# ==========================================================
# Parte 1/2 — Modelo UMa NLoS, Atrasos, PDP, Fator de sombra
# ==========================================================
fc = 3.0
N  = 100
M = 1

# Estatísticas de log10(DS [s]) para 'fc'
MEAN_LOG10_DS = -0.204 * np.log10(1 + fc) - 6.28
STD_LOG10_DS  = 0.39

rng = np.random.default_rng(42)  # semente fixa
log10_DS_sample = rng.normal(MEAN_LOG10_DS, STD_LOG10_DS)
DS = (10**log10_DS_sample)       # [s] DS ~ lognormal (média em log10)

# Atrasos ~ Exponencial(scale = r_tau*DS), deslocados para iniciar em 0
r_tau = 2.3
media_sigma_tau = r_tau * DS
tau_raw = rng.exponential(scale=media_sigma_tau, size=N)
tau = np.sort(tau_raw - np.min(tau_raw))     # [s] inicia em 0
tau_us = tau * 1e6

# Sombreamento lognormal em dB
sigma_qsi_db = 6.0
qsi_n_db = rng.normal(0.0, sigma_qsi_db, size=N)

# Decaimento proposto: exp( -tau * ((r_tau-1)/media_sigma_tau) )
decay_rate = (r_tau - 1.0) / media_sigma_tau
P = np.exp(-tau * decay_rate) * np.power(10.0, qsi_n_db / 10.0)

# Normalização de energia (K=0 ⇒ Rayleigh)
kr = 0.0
omega_c = np.sum(P)
if not np.isfinite(omega_c) or omega_c <= 0:
    omega_c = 1e-15
P = (1.0 / (kr + 1.0)) * (P / omega_c)      # soma(P)=1

# PDP vs atraso (linear)
fig1, ax1 = plt.subplots(figsize=(9, 4.5))
mk = ax1.stem(tau_us, P)
plt.setp(mk[0], markersize=4)
plt.setp(mk[1], linewidth=1.2)
ax1.set_xlabel("Atraso [µs]")
ax1.set_ylabel("Potência relativa por caminho (PDP)")
ax1.set_title("Perfil de Potência vs Atraso (UMa NLoS @ 3 GHz)")
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
save_figure(fig1, "pdp_vs_atraso_linear.png", "PDP vs atraso (linear)")

# DS recalculado a partir de tau e P
tau_bar = np.sum(P * tau)
DS_recalc = np.sqrt(np.sum(P * (tau - tau_bar)**2))
print(f"DS recalculado a partir dos atrasos e PDP: {DS_recalc*1e6:.3f} µs (referência: ~{DS*1e6:.3f} µs)")

# ==========================================================
# Parte 3 — Curvas de média de DS vs frequência (apenas referência)
# ==========================================================
fx = np.linspace(0.5, 100.0, N, dtype=float)  # GHz
MEAN_LOG10_DS_fx = -0.204 * np.log10(1.0 + fx) - 6.28
DS_mean_us = (10.0 ** MEAN_LOG10_DS_fx) * 1e6

fig_ds, ax_ds = plt.subplots(figsize=(9, 4.5))
ax_ds.semilogy(fx, DS_mean_us)
ax_ds.set_xlabel("Frequência [GHz]")
ax_ds.set_ylabel("DS médio [µs]")
ax_ds.set_title("DS médio vs frequência (UMa NLoS)")
ax_ds.grid(True, which="both", alpha=0.3)
fig_ds.tight_layout()
save_figure(fig_ds, "ds_medio_vs_freq.png", "DS médio vs frequência")

# ==========================================================
# Parte 4 — Direções de chegada (θ, φ) e vetores r_n (φ = ELEVAÇÃO)
# ==========================================================
# Geração simples de θ e φ (poderia vir da sua etapa anterior ajustada):
theta_n_final = rng.uniform(0.0, 2*np.pi, size=N)  # azimute
phi_n_final   = rng.uniform(-np.pi/2, np.pi/2, size=N)  # ELEVAÇÃO [-90, +90] rad
theta = np.asarray(theta_n_final, dtype=float)
phi   = np.asarray(phi_n_final,   dtype=float)


# Vetores de direção (φ = elevação): x=cosθ·cosφ, y=sinθ·cosφ, z=sinφ
r_n = np.stack((
     np.cos(theta) * np.cos(phi),  # x
     np.sin(theta) * np.cos(phi),  # y
     np.sin(phi)                   # z
), axis=1)
r_n /= np.linalg.norm(r_n, axis=1, keepdims=True)

# Correção do arquivo para gerar o gráfico 3D novamente (trecho reincluído)

# ------------------------------------
# plot 3D
# ------------------------------------
# from mpl_toolkits.mplot3d import Axes3D  # pode ser necessário em algumas instalações

# cria pasta de saída
outdir = Path(r"C:\New_sharc\Graficos_Canal")
outdir.mkdir(parents=True, exist_ok=True)

fig = plt.figure(figsize=(7.5, 7.5))
ax = fig.add_subplot(111, projection="3d")

# esfera unitária (wireframe) para referência
u, v = np.meshgrid(np.linspace(0, 2*np.pi, 80), np.linspace(0, np.pi, 40))
xs = np.cos(u) * np.sin(v)
ys = np.sin(u) * np.sin(v)
zs = np.cos(v)
ax.plot_wireframe(xs, ys, zs, rstride=3, cstride=3, linewidth=0.3, alpha=0.5)

# vetores saindo da origem
zeros = np.zeros(N)
ax.quiver(zeros, zeros, zeros, r_n[:,0], r_n[:,1], r_n[:,2],
          length=1.0, normalize=True, arrow_length_ratio=0.07)

# pontas dos vetores
ax.scatter(r_n[:,0], r_n[:,1], r_n[:,2], s=25)

# ajustes de visualização
ax.set_box_aspect((1,1,1))
ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("Direções de Chegada (DoA)")
ax.view_init(elev=22, azim=35)
ax.grid(True)
fig.tight_layout()
fig.savefig(outdir / "doa_3d.png", dpi=300)
print(f"[OK] Figura salva em: {outdir / 'doa_3d.png'}")

# Skyplot rápido (opcional)

rmax = np.pi/2
phi_clip = np.clip(phi + np.pi/2, 0, np.pi)  # mapeia elevação [-pi/2,pi/2] a [0,pi] para polar
P_norm = P / (P.sum() + 1e-15)

'''
fig_sky, ax_sky = plt.subplots(figsize=(6.8, 6.8), subplot_kw={'projection':'polar'})
ax_sky.set_theta_zero_location("N")
ax_sky.set_theta_direction(-1)
sc1 = ax_sky.scatter(theta % (2*np.pi), phi_clip, s=10 + 190*P_norm, c=P_norm, cmap="viridis", alpha=0.95)
ax_sky.set_title("Skyplot (θ azimute, raio ~ elevação)")
cb1 = fig_sky.colorbar(sc1, ax=ax_sky, pad=0.1); cb1.set_label("Potência (normalizada)")
fig_sky.tight_layout()
save_figure(fig_sky, "doa_skyplot.png", "Skyplot")
'''

# Projeção XY
'''
x = r_n[:,0]; y = r_n[:,1]
fig_xy, ax_xy = plt.subplots(figsize=(6.8, 6.8))
ang = np.linspace(0, 2*np.pi, 361)
ax_xy.plot(np.cos(ang), np.sin(ang), lw=1, alpha=0.5)
sc2 = ax_xy.scatter(x, y, s=10 + 190*P_norm, c=P_norm, cmap="viridis", alpha=0.95, edgecolor="k", linewidth=0.3)
ax_xy.set_aspect("equal", adjustable="box")
ax_xy.set_xlim(-1.05, 1.05); ax_xy.set_ylim(-1.05, 1.05)
ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y")
ax_xy.set_title("Projeção no plano XY (ponderado por potência)")
cb2 = fig_xy.colorbar(sc2, ax=ax_xy); cb2.set_label("Potência (normalizada)")
fig_xy.tight_layout()
save_figure(fig_xy, "doa_xy.png", "Projeção XY")
'''

# Rosa dos ventos (histograma)
nbins = 144
bins = np.linspace(0, 2*np.pi, nbins + 1)
hist, edges = np.histogram(theta % (2*np.pi), bins=bins, weights=P_norm)
centers = (edges[:-1] + edges[1:]) / 2
fig_rose, ax_rose = plt.subplots(figsize=(6.8, 6.8), subplot_kw={'projection':'polar'})
ax_rose.set_theta_zero_location("N")
ax_rose.set_theta_direction(-1)
width = (2*np.pi) / nbins
ax_rose.bar(centers, hist, width=width, bottom=0.0, align="center", alpha=0.85)
ax_rose.set_title(f"Rosa dos ventos (azimute) — {nbins} bins, ponderado por potência")
fig_rose.tight_layout()
save_figure(fig_rose, "doa_azimute_rose.png", "Rosa dos ventos")

# --- PATCH B+F: Leque polar (100 barras, cores vivas por potência) ---
theta_mod = (theta % (2*np.pi))
fig_leque, ax_leq = plt.subplots(figsize=(6.8, 6.8), subplot_kw={'projection':'polar'})
ax_leq.set_theta_zero_location("N")
ax_leq.set_theta_direction(-1)
width = 2*np.pi / (N * 1.15)
norm = plt.Normalize(vmin=float(P_norm.min()), vmax=float(P_norm.max()))
colors = plt.cm.viridis(norm(P_norm))
bars = ax_leq.bar(theta_mod, P_norm, width=width, bottom=0.0,
                  color=colors, edgecolor="k", linewidth=0.25)
cbar = fig_leque.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
                          ax=ax_leq, pad=0.1)
cbar.set_label("Potência (normalizada)")
ax_leq.set_title("Leque de azimutes (cores e altura ∝ potência)")
fig_leque.tight_layout()
save_figure(fig_leque, "doa_azimute_leque_100.png", "Leque de azimutes (100)")

# --- PATCH A: Elevação por caminho (100 componentes) ---
'''
elev_deg_all = np.rad2deg(phi)  # elevação em graus
fig_el, ax_el = plt.subplots(figsize=(9, 4.2))
mk_el = ax_el.stem(np.arange(1, N+1), elev_deg_all)
plt.setp(mk_el[0], markersize=3)
plt.setp(mk_el[1], linewidth=1.0)
ax_el.set_xlabel("Índice do caminho (n)")
ax_el.set_ylabel("Elevação [graus]")
ax_el.set_title("Elevação de cada componente multipercurso (100 caminhos)")
ax_el.grid(True, alpha=0.3)
ymin, ymax = float(elev_deg_all.min()), float(elev_deg_all.max())
pad = max(2.0, 0.05*(ymax - ymin + 1e-9))
ax_el.set_ylim(ymin - pad, ymax + pad)
fig_el.tight_layout()
save_figure(fig_el, "doa_elevacao_por_caminho.png", "Elevação por caminho")
'''

# ==========================================================
# Parte 5 — Desvios Doppler (φ = elevação, coerente com r_n)
# ==========================================================
v_rx    = 10.0        # m/s
theta_v = np.pi/3     # azimute do movimento
phi_v   = np.pi/4     # ELEVAÇÃO do movimento
c       = 3e8         # m/s
f_c     = 3e9         # Hz
lambda_c = c / f_c
f_max  = v_rx / lambda_c

# v_hat coerente com r_n (φ = elevação)
v_hat = np.array([
    np.cos(theta_v) * np.cos(phi_v),
    np.sin(theta_v) * np.cos(phi_v),
    np.sin(phi_v)
], dtype=float)
v_hat /= np.linalg.norm(v_hat)

# Garantia de normalização r_n
r_n = np.asarray(r_n, dtype=float)
r_n /= np.linalg.norm(r_n, axis=1, keepdims=True)

# Desvio Doppler por caminho
f_D = (v_rx / lambda_c) * (r_n @ v_hat)  # (N,)
print(f"[INFO] f_max teórico = {f_max:.3f} Hz | N caminhos = {f_D.size}")
print(f"[INFO] f_D min/max = {f_D.min():.3f} / {f_D.max():.3f} Hz | média = {f_D.mean():.3f} Hz")

# Espectro Doppler (x=índice)
'''
fig_d1, ax_d1 = plt.subplots(figsize=(9, 4.5))
mkd1 = ax_d1.stem(np.arange(f_D.size), f_D)
plt.setp(mkd1[0], markersize=4); plt.setp(mkd1[1], linewidth=1.2)
ax_d1.axhline(0, lw=0.8, color="k", alpha=0.6)
ax_d1.set_xlabel("Índice do caminho (n)")
ax_d1.set_ylabel("Desvio Doppler f_D [Hz]")
ax_d1.set_title("Espectro Doppler por caminho (x = índice)")
ax_d1.grid(True, alpha=0.3)
fig_d1.tight_layout()
save_figure(fig_d1, "espectro_doppler_idx.png", "Espectro Doppler (x=índice)")
'''

# Espectro Doppler (x=f_D)
fig_d2, ax_d2 = plt.subplots(figsize=(9, 4.5))
mkd2 = ax_d2.stem(f_D, np.arange(f_D.size))
plt.setp(mkd2[0], markersize=4); plt.setp(mkd2[1], linewidth=1.2)
ax_d2.set_xlabel("Desvio Doppler f_D [Hz]")
ax_d2.set_ylabel("Índice do caminho (n)")
ax_d2.set_title("Espectro Doppler (x = f_D)")
ax_d2.grid(True, alpha=0.3)
fig_d2.tight_layout()
save_figure(fig_d2, "espectro_doppler_fx.png", "Espectro Doppler (x=f_D)")

# Histograma de f_D
'''
fig_d3, ax_d3 = plt.subplots(figsize=(9, 4.5))
ax_d3.hist(f_D, bins=40, edgecolor="k", alpha=0.85)
ax_d3.set_xlabel("f_D [Hz]"); ax_d3.set_ylabel("Contagem")
ax_d3.set_title("Histograma dos desvios Doppler")
ax_d3.grid(True, alpha=0.3)
fig_d3.tight_layout()
save_figure(fig_d3, "doppler_hist.png", "Histograma Doppler")
'''

# ==========================================================
# Parte 6 — Fading: h(t) a partir de f_D e P
# ==========================================================
# Eixo de tempo do canal
Ntime = 1000
Fs = max(20.0 * f_max, 1000.0)   # Hz
dt = 1.0 / Fs
t_fad = np.arange(Ntime) * dt

# Fases iniciais (se não houver): uma por caminho
phi_nt_medio = 2*np.pi * rng.random(len(P))   # (N,)

# Fase ao longo do tempo (N, Ntime)
phi_nt = phi_nt_medio[:, None] - 2*np.pi * np.outer(f_D, t_fad)

# Ganho complexo do canal (narrowband)
a_n = np.sqrt(P)
h_nt = a_n[:, None] * np.exp(1j * phi_nt)   # (N, Ntime)
h_t  = h_nt.sum(axis=0)                     # (Ntime,)

# ==========================================================
# Parte 6.1 — RX vs TX (zoom por δt) com cores contrastantes
# ==========================================================
deltas = [1e-7, 1e-5, 1e-3]   # [s]
for d in deltas:
    # grade densa apenas no intervalo [0, 5·δt]
    n_view = 100000 # Era 2000
    t_view = np.linspace(0.0, 5.0*d, n_view, dtype=float)

    # interpola h(t) da grade global t_fad para t_view
    h_real = np.interp(t_view, t_fad, h_t.real)
    h_imag = np.interp(t_view, t_fad, h_t.imag)
    h_view = h_real + 1j*h_imag

    # pulso retangular em t_view (TX)
    s_view = (t_view <= d).astype(float)

    # recebido (RX)
    r_view = h_view * s_view

    # RX vs TX (sobrepostos)
    fig_ov, ax_tx = plt.subplots(figsize=(9, 4.5))
    ax_tx.step(t_view, s_view, where="post", label="Pulso TX (ampl.=1)",
               color="tab:gray", linewidth=1.6)
    ax_tx.set_xlabel("t [s]")
    ax_tx.set_ylabel("Pulso TX [u.a.]")
    ax_tx.set_title(f"TX vs RX (zoom) — δt = {d:g} s")
    ax_tx.grid(True, alpha=0.3)

    ax_rx = ax_tx.twinx()
    ax_rx.plot(t_view, np.abs(r_view), label="|r(t)| (RX)",
               color="tab:blue", linewidth=1.8)
    ax_rx.set_ylabel("|r(t)| [u.a.]")

    ax_tx.set_xlim(0.0, 5.0*d)

    lines1, labels1 = ax_tx.get_legend_handles_labels()
    lines2, labels2 = ax_rx.get_legend_handles_labels()
    ax_tx.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig_ov.tight_layout()
    fname_ov = f"rx_vs_tx_zoom_dt_{d:g}s.png"
    save_figure(fig_ov, fname_ov, f"RX vs TX (zoom δt={d:g} s)")

    # |r(t)| e fase(r(t)) na mesma janela
    mask = s_view > 0
    fase = np.full_like(t_view, np.nan, dtype=float)
    if np.any(mask):
        fase[mask] = np.unwrap(np.angle(r_view[mask]))

    fig_mp, (ax_m, ax_p) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
    ax_m.plot(t_view, np.abs(r_view), linewidth=1.1)
    ax_m.set_ylabel("|r(t)| [u.a.]")
    ax_m.grid(True, alpha=0.3)
    ax_m.set_title(f"Módulo e fase do recebido r(t) (zoom) — δt = {d:g} s")

    ax_p.plot(t_view, fase, linewidth=1.1)
    ax_p.set_xlabel("t [s]")
    ax_p.set_ylabel("Fase {rad}")
    ax_p.grid(True, alpha=0.3)
    ax_p.set_xlim(0.0, 5.0*d)

    fig_mp.tight_layout()
    fname_mp = f"rx_mag_phase_zoom_dt_{d:g}s.png"
    save_figure(fig_mp, fname_mp, f"Módulo e fase de r(t) (zoom δt={d:g} s)")

    print(f"[OK] RX vs TX (zoom) salvo em: {OUTPUT_DIR / fname_ov}")
    print(f"[OK] |r(t)| e fase (zoom) salvo em: {OUTPUT_DIR / fname_mp}")

# ==========================================================
# Parte 7 — Autocorrelação temporal do canal ρ_TT(0; σ)
# ==========================================================
def acf_empirica_unbiased(x: np.ndarray, max_lags: int) -> np.ndarray:
    x = np.asarray(x, dtype=complex).ravel()
    Nloc = x.size
    max_lags = min(max_lags, Nloc-1)
    acf = np.empty(max_lags+1, dtype=complex)
    for k in range(max_lags+1):
        acf[k] = (x[:Nloc-k] * np.conj(x[k:])).mean()
    return acf

dt = float(t_fad[1] - t_fad[0])
dur_total = float(t_fad[-1] - t_fad[0] + dt)
tau_max = min(0.2, dur_total)
K = int(np.floor(tau_max / dt))
lags_s = np.arange(K+1) * dt

acf_emp = acf_empirica_unbiased(h_t, K)
rho_emp = acf_emp / (acf_emp[0] + 1e-15)
rho_theo = np.sum(P[None, :] * np.exp(1j * 2*np.pi * np.outer(lags_s, f_D)), axis=1)

'''
fig_r, ax_r = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
ax_r[0].plot(lags_s, np.real(rho_emp), label="Re{ρ̂_emp}", linewidth=1.2)
ax_r[0].plot(lags_s, np.real(rho_theo), '--', label="Re{ρ_theo}", linewidth=1.2)
ax_r[0].set_ylabel("Re{ρ(τ)}")
ax_r[0].set_title("Autocorrelação temporal do canal: parte real e módulo")
ax_r[0].grid(True, alpha=0.3); ax_r[0].legend()

ax_r[1].plot(lags_s, np.abs(rho_emp), label="|ρ̂_emp|", linewidth=1.2)
ax_r[1].plot(lags_s, np.abs(rho_theo), '--', label="|ρ_theo|", linewidth=1.2)
ax_r[1].set_xlabel("τ [s]")
ax_r[1].set_ylabel("|ρ(τ)|")
ax_r[1].grid(True, alpha=0.3); ax_r[1].legend()
fig_r.tight_layout()
save_figure(fig_r, "rho_temporal_real_mag.png", "Autocorrelação temporal (real e módulo)")
'''

# ==========================================================
# Parte 8 — Bc e Tc (ρTT(κ;0) e ρTT(0;σ))
# ==========================================================
def first_crossing_x(x, y, thr):
    yabs = np.abs(y)
    under = np.flatnonzero(yabs <= thr)
    if under.size == 0:
        return np.nan
    i = int(under[0])
    if i == 0:
        return x[0]
    x0, x1 = x[i-1], x[i]
    y0, y1 = yabs[i-1], yabs[i]
    if y1 == y0:
        return x1
    alpha = (thr - y0) / (y1 - y0)
    return x0 + alpha * (x1 - x0)

# (b) ρTT(κ;0) a partir de PDP
try:
    DS_for_range = float(DS)
except NameError:
    tau_bar_tmp = (P * tau).sum()
    DS_for_range = np.sqrt(max((P * (tau - tau_bar_tmp)**2).sum(), 1e-18))

kappa_max = min(200e6, max(5e6, 10.0 / DS_for_range))
kappa_Hz = np.linspace(0.0, kappa_max, 4000)
rho_B = (P[None, :] * np.exp(-1j * 2 * np.pi * np.outer(kappa_Hz, tau))).sum(axis=1)

targets_B = [0.95, 0.90]
Bc_vals = {}
for thr in targets_B:
    Bc = first_crossing_x(kappa_Hz, rho_B, thr)
    if np.isnan(Bc):
        kappa_Hz_ext = np.linspace(0.0, 10 * kappa_max, 8000)
        rho_B_ext = (P[None, :] * np.exp(-1j * 2 * np.pi * np.outer(kappa_Hz_ext, tau))).sum(axis=1)
        Bc = first_crossing_x(kappa_Hz_ext, rho_B_ext, thr)
        if not np.isnan(Bc):
            kappa_Hz, rho_B = kappa_Hz_ext, rho_B_ext
            kappa_max = 10 * kappa_max
    Bc_vals[thr] = Bc  # Hz

# (c) ρTT(0;σ) para v=5 e v=50 m/s
v_hat = np.array([
    np.cos(theta_v) * np.cos(phi_v),
    np.sin(theta_v) * np.cos(phi_v),
    np.sin(phi_v)
], dtype=float)
v_hat /= np.linalg.norm(v_hat)
r_n = np.asarray(r_n, float)
r_n /= np.linalg.norm(r_n, axis=1, keepdims=True)

def rho_T_sigma(Pw, fD, sigma_s):
    return (Pw[None, :] * np.exp(1j * 2 * np.pi * np.outer(sigma_s, fD))).sum(axis=1)

speeds = [5.0, 50.0]
targets_T = [0.95, 0.90]
Tc_results = {}
for v in speeds:
    fD = (v / lambda_c) * (r_n @ v_hat)
    fmax = np.max(np.abs(fD)) + 1e-12
    sigma_max = 10.0 / fmax
    Ns = 4000
    sigma_s = np.linspace(0.0, sigma_max, Ns)
    rho_T = rho_T_sigma(P, fD, sigma_s)
    Tc_vals = {thr: first_crossing_x(sigma_s, rho_T, thr) for thr in targets_T}
    Tc_results[v] = (sigma_s, rho_T, fmax, Tc_vals)

# --- Plot Bc com janela útil e legenda limpa ---


fig_bc, ax_bc = plt.subplots(figsize=(9, 4.8))
ax_bc.plot(kappa_Hz * 1e-6, np.abs(rho_B), lw=1.6, label="|ρTT(κ; 0)|")
leg_handles = [Line2D([0],[0], color='C0', lw=1.6, label='|ρTT(κ; 0)|')]
for thr, color in zip([0.95, 0.90], ["C1", "C2"]):
    Bc = Bc_vals.get(thr, np.nan)
    if np.isfinite(Bc):
        ax_bc.axvline(Bc * 1e-6, color=color, ls="--", lw=1.2)
        leg_handles.append(Line2D([0],[0], color=color, ls="--", lw=1.2,
                                  label=f"Bc@ρ={thr} ≈ {Bc*1e-6:.3f} MHz"))
ax_bc.set_xlabel("κ [MHz]")
ax_bc.set_ylabel("|ρTT(κ; 0)|")
ax_bc.set_title("Correlação em frequência e Bandas de Coerência")
ax_bc.grid(True, alpha=0.3)
ax_bc.legend(handles=leg_handles, loc="upper right")

xmax_B = np.nanmax([Bc_vals.get(0.95, np.nan), Bc_vals.get(0.90, np.nan)])
if np.isfinite(xmax_B):
    ax_bc.set_xlim(0.0, 1.25 * xmax_B * 1e-6)
else:
    Bc01 = first_crossing_x(kappa_Hz, rho_B, 0.1)
    if np.isfinite(Bc01):
        ax_bc.set_xlim(0.0, 1.1 * Bc01 * 1e-6)
fig_bc.tight_layout()
save_figure(fig_bc, "freq_correlation_and_Bc.png", "Correlação em frequência e Bc")

print("\n[b] Bandas de coerência (primeiro cruzamento):")
for thr in targets_B:
    Bc = Bc_vals[thr]
    if np.isnan(Bc):
        print(f"  ρB={thr}: não cruzou na faixa analisada (κ_max={kappa_max*1e-6:.1f} MHz).")
    else:
        print(f"  ρB={thr}: Bc ≈ {Bc*1e-6:.3f} MHz ({Bc/1e3:.1f} kHz)")

# --- Plot Tc com janela útil e TÍTULO com valores de Tc (ρ=0.95, 0.90) ---
fig_tc, ax_tc = plt.subplots(figsize=(9, 4.8))
Tc90_vals = []
for v, color in zip([5.0, 50.0], ["C0", "C3"]):
    sigma_s, rho_T, fmax_v, Tc_vals = Tc_results[v]
    ax_tc.plot(sigma_s * 1e3, np.abs(rho_T), lw=1.6, label=f"|ρTT(0; σ)|, v={v:g} m/s")
    Tc90 = Tc_vals.get(0.90, np.nan)
    if np.isfinite(Tc90):
        ax_tc.axvline(Tc90 * 1e3, color=color, ls="--", lw=1.2)
        Tc90_vals.append(Tc90)
    Tc95 = Tc_vals.get(0.95, np.nan)
    if np.isfinite(Tc95):
        ax_tc.axvline(Tc95 * 1e3, color=color, ls=":", lw=1.2)

def _fmt_ms(x):
    return "–" if (x is None or not np.isfinite(x)) else f"{x*1e3:.2f} ms"

def _get_tc(v, thr):
    Tc = Tc_results[v][3].get(thr, np.nan)
    return None if not np.isfinite(Tc) else Tc

Tc5_95  = _get_tc(5.0, 0.95)
Tc5_90  = _get_tc(5.0, 0.90)
Tc50_95 = _get_tc(50.0, 0.95)
Tc50_90 = _get_tc(50.0, 0.90)

ax_tc.set_xlabel("σ [ms]")
ax_tc.set_ylabel("|ρTT(0; σ)|")
ax_tc.set_title(
    "Correlação temporal e Tempos de Coerência\n"
    f"ρ=0.95 → Tc: v=5 m/s {_fmt_ms(Tc5_95)}, v=50 m/s {_fmt_ms(Tc50_95)} | "
    f"ρ=0.90 → Tc: v=5 m/s {_fmt_ms(Tc5_90)}, v=50 m/s {_fmt_ms(Tc50_90)}"
)
ax_tc.grid(True, alpha=0.3)

leg_tc = [Line2D([0],[0], color='C0', lw=1.6, label='|ρTT(0; σ)|, v=5 m/s'),
          Line2D([0],[0], color='C3', lw=1.6, label='|ρTT(0; σ)|, v=50 m/s'),
          Line2D([0],[0], color='k', ls='--', lw=1.2, label='Tc @ ρ=0.90'),
          Line2D([0],[0], color='k', ls=':',  lw=1.2, label='Tc @ ρ=0.95')]
ax_tc.legend(handles=leg_tc, loc="upper right")

if Tc90_vals:
    ax_tc.set_xlim(0.0, 1.5 * max(Tc90_vals) * 1e3)
fig_tc.tight_layout()
save_figure(fig_tc, "time_correlation_Tc_v5_v50.png", "Correlação temporal e Tc")

print("\n[c] Tempos de coerência (primeiro cruzamento):")
for v in [5.0, 50.0]:
    sigma_s, rho_T, fmax_v, Tc_vals = Tc_results[v]
    for thr in targets_T:
        Tc = Tc_vals[thr]
        if np.isnan(Tc):
            print(f"  v={v:g} m/s, ρT={thr}: não cruzou na faixa (σ_max={sigma_s[-1]:.3f} s).")
        else:
            print(f"  v={v:g} m/s, ρT={thr}: Tc ≈ {Tc*1e3:.2f} ms  (fmax≈{fmax_v:.2f} Hz)")

# ==========================================================
# Informações finais
# ==========================================================
print(f"\nDS médio teórico (mu): {10**MEAN_LOG10_DS*1e6:.3f} µs")
print(f"DS sorteado nesta realização:       {DS*1e6:.3f} µs")
print(f"[INFO] Confirme os arquivos em: {OUTPUT_DIR}")
print(f"f_max teórico = {f_max:.2f} Hz")
print("Desvios Doppler (10 primeiros) [Hz]:", np.round(f_D[:10], 2))

# Abra todas as figuras no final (se SHOW_ALL_AT_END=True)
plt.show()
# Fim do script
