#!/usr/bin/env python3
"""Streamlit app: Pipeline complet Market Data → Heston NN Calibration → Monte Carlo → IV Surfaces."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="🚀 Heston Full Pipeline | Advanced Options Analytics", layout="wide")
st.title("🚀 Pipeline Heston Complet: \nMarket Data → Heston params NN Calibration → IV Surfaces from Carr-Madan → Monte Carlo pricing")

st.markdown(
    """
    **Analyse complète de volatilité stochastique en une seule interface !**
    
    1️⃣ Téléchargement des données de marché en temps réel depuis yfinance
    
    2️⃣ Calibration automatique des paramètres Heston via réseau de neurones PyTorch
    
    3️⃣ Génération de heatmaps de prix par simulation Monte Carlo
    
    4️⃣ Inversion Black-Scholes pour surfaces d'IV 3D interactives
    
    **Comparez prix analytiques vs Monte Carlo et découvrez le smile de volatilité !**
    """
)

# Import du module Heston torch
from heston_torch import HestonParams, carr_madan_call_torch, carr_madan_put_torch

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cpu")
MIN_IV_MATURITY = 0.1


def heston_mc_pricer(
    S0: float, K: float, T: float, r: float,
    v0: float, theta: float, kappa: float, sigma_v: float, rho: float,
    n_paths: int = 50000, n_steps: int = 100, option_type: str = "call"
) -> float:
    """Pricer Monte Carlo pour options européennes sous Heston."""
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    
    # Initialisation
    S = np.full(n_paths, S0)
    v = np.full(n_paths, v0)
    
    # Simulation
    for _ in range(n_steps):
        Z1 = np.random.randn(n_paths)
        Z2 = np.random.randn(n_paths)
        Z_S = Z1
        Z_v = rho * Z1 + math.sqrt(1 - rho**2) * Z2
        
        # Euler pour S
        S = S * np.exp((r - 0.5 * np.maximum(v, 0)) * dt + np.sqrt(np.maximum(v, 0)) * sqrt_dt * Z_S)
        
        # Euler pour v avec troncation
        v = v + kappa * (theta - np.maximum(v, 0)) * dt + sigma_v * np.sqrt(np.maximum(v, 0)) * sqrt_dt * Z_v
        v = np.maximum(v, 0)  # Troncation
    
    # Payoff
    if option_type == "call":
        payoff = np.maximum(S - K, 0)
    else:
        payoff = np.maximum(K - S, 0)
    
    return math.exp(-r * T) * np.mean(payoff)


def fetch_spot(symbol: str) -> float:
    """Récupère le prix spot actuel."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1d")
    if hist.empty:
        raise RuntimeError("Unable to retrieve spot price.")
    return float(hist["Close"].iloc[-1])


def _select_monthly_expirations(expirations, years_ahead: float = 2.5) -> list[str]:
    today = pd.Timestamp.utcnow().date()
    limit_date = today + pd.Timedelta(days=365 * years_ahead)
    monthly: Dict[Tuple[int, int], Tuple[pd.Timestamp, str]] = {}
    for exp in expirations:
        exp_ts = pd.Timestamp(exp)
        exp_date = exp_ts.date()
        if not (today < exp_date <= limit_date):
            continue
        key = (exp_date.year, exp_date.month)
        if key not in monthly or exp_ts < monthly[key][0]:
            monthly[key] = (exp_ts, exp)
    return [item[1] for item in sorted(monthly.values(), key=lambda x: x[0])]


@st.cache_data(show_spinner=True)
def download_options(symbol: str, option_type: str, years_ahead: float = 2.5) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    spot = fetch_spot(symbol)
    expirations = ticker.options
    if not expirations:
        raise RuntimeError(f"No option expirations found for {symbol}")
    selected = _select_monthly_expirations(expirations, years_ahead)
    rows: list[dict] = []
    now = pd.Timestamp.utcnow().tz_localize(None)
    for expiry in selected:
        expiry_dt = pd.Timestamp(expiry)
        T = max((expiry_dt - now).total_seconds() / (365.0 * 24 * 3600), 0.0)
        chain = ticker.option_chain(expiry)
        data = chain.calls if option_type == "call" else chain.puts
        price_col = "C_mkt" if option_type == "call" else "P_mkt"
        for _, row in data.iterrows():
            rows.append(
                {
                    "S0": spot,
                    "K": float(row["strike"]),
                    "T": T,
                    price_col: float(row["lastPrice"]),
                    "iv_market": float(row.get("impliedVolatility", float("nan"))),
                }
            )
    df = pd.DataFrame(rows)
    try:
        out_dir = Path("data")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{symbol}_{option_type}_options_{ts}.csv"
        df.to_csv(out_path, index=False)
    except Exception:
        pass
    return df


def prices_from_unconstrained(
    u: torch.Tensor, S0_t: torch.Tensor, K_t: torch.Tensor, T_t: torch.Tensor, r: float, q: float
) -> torch.Tensor:
    params = HestonParams.from_unconstrained(u[0], u[1], u[2], u[3], u[4])
    prices = []
    for S0_i, K_i, T_i in zip(S0_t, K_t, T_t):
        price_i = carr_madan_call_torch(S0_i, r, q, T_i, params, K_i)
        prices.append(price_i)
    return torch.stack(prices)


def loss(
    u: torch.Tensor, S0_t: torch.Tensor, K_t: torch.Tensor, T_t: torch.Tensor, C_mkt_t: torch.Tensor, r: float, q: float
) -> torch.Tensor:
    model_prices = prices_from_unconstrained(u, S0_t, K_t, T_t, r, q)
    diff = model_prices - C_mkt_t
    return 0.5 * (diff**2).mean()


def calibrate_heston_nn(
    df: pd.DataFrame,
    r: float,
    q: float,
    max_points: int = 1000,
    max_iters: int = 200,
    lr: float = 5e-3,
    progress_callback: Callable[[int, int], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> dict:
    """Calibration via réseau de neurones PyTorch."""
    if df.empty:
        raise ValueError("DataFrame vide.")
    
    df_clean = df.dropna(subset=["S0", "K", "T", "C_mkt"])
    df_clean = df_clean[df_clean["T"] > 0]
    df_clean = df_clean[df_clean["C_mkt"] > 0]
    
    if len(df_clean) == 0:
        raise ValueError("Aucun point valide après nettoyage.")
    
    if len(df_clean) > max_points:
        df_clean = df_clean.sample(n=max_points, random_state=42)
    
    S0_t = torch.tensor(df_clean["S0"].values, dtype=torch.float64, device=DEVICE)
    K_t = torch.tensor(df_clean["K"].values, dtype=torch.float64, device=DEVICE)
    T_t = torch.tensor(df_clean["T"].values, dtype=torch.float64, device=DEVICE)
    C_mkt_t = torch.tensor(df_clean["C_mkt"].values, dtype=torch.float64, device=DEVICE)
    
    # Initialisation
    u = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64, device=DEVICE, requires_grad=True)
    optimizer = torch.optim.Adam([u], lr=lr)
    
    for iteration in range(max_iters):
        optimizer.zero_grad()
        loss_val = loss(u, S0_t, K_t, T_t, C_mkt_t, r, q)
        loss_val.backward()
        optimizer.step()
        
        if progress_callback:
            progress_callback(iteration + 1, max_iters)
        if log_callback:
            log_callback(f"Iter {iteration + 1}/{max_iters} | Loss = {loss_val.item():.6f}")
    
    params = HestonParams.from_unconstrained(u[0], u[1], u[2], u[3], u[4])
    
    return {
        "kappa": float(params.kappa.cpu().detach()),
        "theta": float(params.theta.cpu().detach()),
        "sigma": float(params.sigma.cpu().detach()),
        "rho": float(params.rho.cpu().detach()),
        "v0": float(params.v0.cpu().detach()),
    }


def heston_monte_carlo(S0: float, r: float, q: float, T: float, params: dict, K_grid: np.ndarray, n_paths: int = 50000) -> Tuple[np.ndarray, np.ndarray]:
    """Simulation Monte Carlo du modèle Heston pour calculer les prix d'options."""
    # Pas de temps basé sur T * 252 (jours de trading)
    n_steps = max(int(T ), 10)  # Au moins 10 pas
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    kappa = params['kappa']
    theta = params['theta']
    sigma = params['sigma']
    rho = params['rho']
    v0 = params['v0']
    
    # Initialisation
    S = np.full(n_paths, S0)
    v = np.full(n_paths, v0)
    
    # Simulation
    for _ in range(n_steps):
        z1 = np.random.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(n_paths)
        
        v_pos = np.maximum(v, 0)
        S = S * np.exp((r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * z1)
        v = v + kappa * (theta - v_pos) * dt + sigma * np.sqrt(v_pos) * sqrt_dt * z2
        v = np.maximum(v, 0)
    
    # Payoffs
    discount = np.exp(-r * T)
    call_prices = np.array([discount * np.maximum(S - K, 0).mean() for K in K_grid])
    put_prices = np.array([discount * np.maximum(K - S, 0).mean() for K in K_grid])
    
    return call_prices, put_prices


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Prix d'un call Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    from scipy.stats import norm
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Prix d'un put Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    from scipy.stats import norm
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol_option(price: float, S: float, K: float, T: float, r: float, option_type: str = 'call', tol: float = 1e-6, max_iter: int = 100) -> float:
    """Calcul de la volatilité implicite par Newton-Raphson."""
    if T < MIN_IV_MATURITY:
        return np.nan
    
    intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    if price <= intrinsic:
        return np.nan
    
    sigma = 0.3  # Initial guess
    
    for _ in range(max_iter):
        if option_type == 'call':
            price_est = bs_call(S, K, T, r, sigma)
        else:
            price_est = bs_put(S, K, T, r, sigma)
        
        diff = price_est - price
        
        if abs(diff) < tol:
            return sigma
        
        # Vega
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        from scipy.stats import norm
        vega = S * norm.pdf(d1) * math.sqrt(T)
        
        if vega < 1e-10:
            return np.nan
        
        sigma = sigma - diff / vega
        
        if sigma <= 0:
            return np.nan
    
    return np.nan


# Sidebar configuration
st.sidebar.header("⚙️ Configuration de base")

ticker = st.sidebar.text_input("Ticker", value="SPY")
rf_rate = st.sidebar.number_input("Taux sans risque (r)", value=0.02, step=0.01, format="%.3f")
div_yield = st.sidebar.number_input("Dividende (q)", value=0.00, step=0.01, format="%.3f")
years_ahead = st.sidebar.number_input("Horizon (années)", value=2.5, min_value=0.1, max_value=5.0, step=0.1)

# Paramètres principaux sur l'écran
st.header("⚙️ Paramètres de modélisation")

col_nn, col_mc, col_grid = st.columns(3)

with col_nn:
    st.subheader("🎯 Calibration NN")
    max_iters = st.number_input("Itérations NN", value=10, min_value=10, max_value=1000, step=10, key="max_iters")
    st.caption("ℹ️ Learning rate fixé à 0.05")
    st.caption("ℹ️ Max points = 1000")

with col_mc:
    st.subheader("📊 Monte Carlo")
    n_paths = st.number_input("Nombre de trajectoires", value=10000, min_value=1000, max_value=200000, step=1000, key="n_paths")
    st.caption("ℹ️ Pas de temps = T × 252 (jours de trading)")

with col_grid:
    st.subheader("🔢 Grille de calcul (IV surfaces analytiques)")
    span = st.number_input("Span autour de S0 (±)", value=20.0, min_value=5.0, max_value=200.0, step=5.0, key="span")
    step_strike = st.number_input("Step strike", value=1.0, min_value=1.0, max_value=20.0, step=1.0, key="step_strike")
    n_maturities = st.number_input("Step T", value=40, min_value=3, max_value=1000, step=1, key="n_maturities")

run_button = st.button("🚀 Lancer l'analyse complète", type="primary", width="stretch")

st.divider()

if run_button:
    try:
        st.info(f"📡 Téléchargement des données pour {ticker}...")
        
        # Étape 1: Téléchargement des données
        calls_df = download_options(ticker, "call", years_ahead)
        puts_df = download_options(ticker, "put", years_ahead)
        S0_ref = fetch_spot(ticker)
        
        st.success(f"✓ {len(calls_df)} calls et {len(puts_df)} puts téléchargés | S0 = {S0_ref:.2f}")
        
        # Étape 2: Calibration Heston NN
        st.info("🧠 Calibration des paramètres Heston via réseau de neurones...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.expander("📜 Logs de calibration", expanded=True)
        log_placeholder = log_container.empty()
        log_messages = []
        
        def progress_cb(current: int, total: int) -> None:
            progress_bar.progress(current / total)
            status_text.text(f"⏳ Itération {current}/{total} ({100*current/total:.1f}%)")
        
        def log_cb(msg: str) -> None:
            log_messages.append(msg)
            log_placeholder.text("\n".join(log_messages))  # Affiche tous les messages
        
        calib = calibrate_heston_nn(
            calls_df,
            r=rf_rate,
            q=div_yield,
            max_points=1000,
            max_iters=max_iters,
            lr=0.05,
            progress_callback=progress_cb,
            log_callback=log_cb,
        )
        progress_bar.empty()
        status_text.empty()
        
        st.success("✓ Calibration terminée!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Paramètres Heston calibrés")
            st.dataframe(pd.Series(calib, name="Valeur").to_frame())
        
        # Étape 3: Calcul immédiat des IV Surfaces analytiques Carr-Madan
        st.info("📐 Calcul des IV Surfaces analytiques (Carr-Madan FFT)...")
        
        # Grilles pour Carr-Madan
        K_grid = np.arange(S0_ref - span, S0_ref + span + step_strike, step_strike)
        T_grid = np.linspace(0.1, years_ahead, n_maturities)
        
        params_cm = HestonParams(
            kappa=torch.tensor(calib['kappa'], dtype=torch.float64),
            theta=torch.tensor(calib['theta'], dtype=torch.float64),
            sigma=torch.tensor(calib['sigma'], dtype=torch.float64),
            rho=torch.tensor(calib['rho'], dtype=torch.float64),
            v0=torch.tensor(calib['v0'], dtype=torch.float64),
        )
        
        # Calcul des prix analytiques Carr-Madan
        call_prices_cm = np.zeros((len(T_grid), len(K_grid)))
        put_prices_cm = np.zeros((len(T_grid), len(K_grid)))
        
        cm_progress = st.progress(0)
        Ks_t = torch.tensor(K_grid, dtype=torch.float64)
        
        for i, T_val in enumerate(T_grid):
            call_anal = carr_madan_call_torch(S0_ref, rf_rate, div_yield, float(T_val), params_cm, Ks_t)
            put_anal = carr_madan_put_torch(S0_ref, rf_rate, div_yield, float(T_val), params_cm, Ks_t)
            call_prices_cm[i, :] = call_anal.detach().cpu().numpy()
            put_prices_cm[i, :] = put_anal.detach().cpu().numpy()
            cm_progress.progress((i + 1) / len(T_grid))
        
        cm_progress.empty()
        st.success("✓ Prix analytiques calculés!")
        
        # Inversion BS immédiate pour IV surfaces
        st.info("🔄 Calcul des IV Surfaces BS (depuis prix Carr-Madan)...")
        
        call_iv_cm = np.zeros_like(call_prices_cm)
        put_iv_cm = np.zeros_like(put_prices_cm)
        
        iv_progress = st.progress(0)
        
        for i, T_val in enumerate(T_grid):
            for j, K_val in enumerate(K_grid):
                call_iv_cm[i, j] = implied_vol_option(
                    call_prices_cm[i, j], S0_ref, K_val, T_val, rf_rate, "call"
                )
                put_iv_cm[i, j] = implied_vol_option(
                    put_prices_cm[i, j], S0_ref, K_val, T_val, rf_rate, "put"
                )
            iv_progress.progress((i + 1) / len(T_grid))
        
        iv_progress.empty()
        st.success("✓ IV Surfaces analytiques calculées!")
        
        # Affichage des IV surfaces analytiques
        with col2:
            st.subheader("📈 Grille de calcul ")
            st.write(f"**Strikes:** {K_grid[0]:.1f} → {K_grid[-1]:.1f} ({len(K_grid)} points)")
            st.write(f"**Maturités:** {T_grid[0]:.2f} → {T_grid[-1]:.2f} ans ({len(T_grid)} points)")
        
        st.subheader("🌊 IV Surfaces 3D (Carr-Madan Analytique)")
        
        KK_cm, TT_cm = np.meshgrid(K_grid, T_grid)
        
        fig_iv_calls_cm = go.Figure(data=[go.Surface(
            x=KK_cm,
            y=TT_cm,
            z=call_iv_cm,
            colorscale='Viridis',
            colorbar=dict(title="IV")
        )])
        fig_iv_calls_cm.update_layout(
            title=f"IV Surface Calls BS (Carr-Madan Analytique) - {ticker}",
            scene=dict(
                xaxis=dict(title="Strike K"),
                yaxis=dict(title="Maturité T (années)"),
                zaxis=dict(title="Implied Volatility")
            ),
            height=600
        )
        
        fig_iv_puts_cm = go.Figure(data=[go.Surface(
            x=KK_cm,
            y=TT_cm,
            z=put_iv_cm,
            colorscale='Viridis',
            colorbar=dict(title="IV")
        )])
        fig_iv_puts_cm.update_layout(
            title=f"IV Surface Puts BS (Carr-Madan Analytique) - {ticker}",
            scene=dict(
                xaxis=dict(title="Strike K"),
                yaxis=dict(title="Maturité T (années)"),
                zaxis=dict(title="Implied Volatility")
            ),
            height=600
        )
        
        col_iv1, col_iv2 = st.columns(2)
        with col_iv1:
            st.plotly_chart(fig_iv_calls_cm, width="stretch")
        with col_iv2:
            st.plotly_chart(fig_iv_puts_cm, width="stretch")
        
        # Étape 4: Monte Carlo Heston - Grilles de prix (méthode notebook)
        #TODO : pour la heatmap, je veux les prix en fonction de S et K pour un T donné dans la sidebar
        st.info("🎲 Pricing Heston par Monte Carlo...")
        
        log_text = st.empty()
        log_text.write(f"Grille K: {len(K_grid)} points de {K_grid[0]:.1f} à {K_grid[-1]:.1f}")
        log_text.write(f"Grille T: {len(T_grid)} points de {T_grid[0]:.1f} à {T_grid[-1]:.1f} années")
        log_text.write(f"Total: {len(K_grid) * len(T_grid)} prix à calculer\n")
        
        call_prices_mc = np.zeros((len(T_grid), len(K_grid)))
        put_prices_mc = np.zeros((len(T_grid), len(K_grid)))
        
        total_calcs = len(T_grid) * len(K_grid)
        calc_count = 0
        
        log_text.write("Démarrage du pricing Monte Carlo...")
        mc_progress = st.progress(0)
        
        for i, T_val in enumerate(T_grid):
            for j, K_val in enumerate(K_grid):
                call_prices_mc[i, j] = heston_mc_pricer(
                    S0_ref, K_val, T_val, rf_rate,
                    calib['v0'], calib['theta'], calib['kappa'], calib['sigma'], calib['rho'],
                    n_paths=n_paths, n_steps=int(T_val * 252), option_type="call"
                )
                put_prices_mc[i, j] = heston_mc_pricer(
                    S0_ref, K_val, T_val, rf_rate,
                    calib['v0'], calib['theta'], calib['kappa'], calib['sigma'], calib['rho'],
                    n_paths=n_paths, n_steps=int(T_val * 252), option_type="put"
                )
                calc_count += 2
                if calc_count % 20 == 0 or calc_count == total_calcs * 2:
                    pct = 100 * calc_count / (total_calcs * 2)
                    log_text.write(f"  Progression: {pct:.1f}% ({calc_count}/{total_calcs * 2} prix calculés)")
                    mc_progress.progress(pct / 100)
        
        mc_progress.empty()
        st.success("✓ Pricing Monte Carlo terminé!")
        
        # Affichage des heatmaps MC
        st.subheader("🔥 Heatmaps des prix Monte Carlo Heston")
        
        fig_call_mc = go.Figure(data=go.Heatmap(
            z=call_prices_mc,
            x=K_grid,
            y=T_grid,
            colorscale='Viridis',
            colorbar=dict(title="Prix Call Heston")
        ))
        fig_call_mc.update_layout(
            title=f"Heatmap Prix Calls Heston (MC) - {ticker}",
            xaxis_title="Strike K",
            yaxis_title="Maturité T (années)",
            height=500
        )
        
        fig_put_mc = go.Figure(data=go.Heatmap(
            z=put_prices_mc,
            x=K_grid,
            y=T_grid,
            colorscale='Viridis',
            colorbar=dict(title="Prix Put Heston")
        ))
        fig_put_mc.update_layout(
            title=f"Heatmap Prix Puts Heston (MC) - {ticker}",
            xaxis_title="Strike K",
            yaxis_title="Maturité T (années)",
            height=500
        )
        
        col_mc1, col_mc2 = st.columns(2)
        with col_mc1:
            st.plotly_chart(fig_call_mc, width="stretch")
        with col_mc2:
            st.plotly_chart(fig_put_mc, width="stretch")
        
        # Étape 5: Inversion BS pour IV surfaces MC (méthode notebook)
        st.info("🔄 Calcul des IV Surfaces BS (depuis prix MC)...")
        
        call_iv_mc = np.zeros_like(call_prices_mc)
        put_iv_mc = np.zeros_like(put_prices_mc)
        
        log_text.write("Inversion BS pour Calls et Puts MC...")
        iv_mc_progress = st.progress(0)
        
        for i, T_val in enumerate(T_grid):
            for j, K_val in enumerate(K_grid):
                call_iv_mc[i, j] = implied_vol_option(
                    call_prices_mc[i, j], S0_ref, K_val, T_val, rf_rate, "call"
                )
                put_iv_mc[i, j] = implied_vol_option(
                    put_prices_mc[i, j], S0_ref, K_val, T_val, rf_rate, "put"
                )
            iv_mc_progress.progress((i + 1) / len(T_grid))
        
        iv_mc_progress.empty()
        st.success("✓ Inversion MC terminée")
        
        # Affichage 3D MC (méthode notebook)
        st.subheader("🌊 IV Surfaces 3D (Monte Carlo)")
        
        KK_mc, TT_mc = np.meshgrid(K_grid, T_grid)
        
        fig_iv_calls_mc = go.Figure(data=[go.Surface(
            x=KK_mc,
            y=TT_mc,
            z=call_iv_mc,
            colorscale='Plasma',
            colorbar=dict(title="IV")
        )])
        fig_iv_calls_mc.update_layout(
            title=f"IV Surface Calls BS (depuis Heston MC) - {ticker}",
            scene=dict(
                xaxis=dict(title="Strike K"),
                yaxis=dict(title="Maturité T (années)"),
                zaxis=dict(title="Implied Volatility")
            ),
            height=600
        )
        
        fig_iv_puts_mc = go.Figure(data=[go.Surface(
            x=KK_mc,
            y=TT_mc,
            z=put_iv_mc,
            colorscale='Plasma',
            colorbar=dict(title="IV")
        )])
        fig_iv_puts_mc.update_layout(
            title=f"IV Surface Puts BS (depuis Heston MC) - {ticker}",
            scene=dict(
                xaxis=dict(title="Strike K"),
                yaxis=dict(title="Maturité T (années)"),
                zaxis=dict(title="Implied Volatility")
            ),
            height=600
        )
        
        col_iv1, col_iv2 = st.columns(2)
        with col_iv1:
            st.plotly_chart(fig_iv_calls_mc, width="stretch")
        with col_iv2:
            st.plotly_chart(fig_iv_puts_mc, width="stretch")
        
        # Comparaison Analytique vs MC
        st.subheader("🔬 Comparaison: Monte Carlo vs Carr-Madan Analytique")
        
        # Choisir une maturité pour comparer
        T_compare = T_grid[len(T_grid)//2]
        idx_T = len(T_grid)//2
        
        # Graphiques de comparaison (prix déjà calculés)
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(x=K_grid, y=call_prices_mc[idx_T, :], mode='lines+markers', name='MC Call', line=dict(color='blue')))
        fig_compare.add_trace(go.Scatter(x=K_grid, y=call_prices_cm[idx_T, :], mode='lines', name='Carr-Madan Call', line=dict(color='red', dash='dash')))
        fig_compare.add_trace(go.Scatter(x=K_grid, y=put_prices_mc[idx_T, :], mode='lines+markers', name='MC Put', line=dict(color='green')))
        fig_compare.add_trace(go.Scatter(x=K_grid, y=put_prices_cm[idx_T, :], mode='lines', name='Carr-Madan Put', line=dict(color='orange', dash='dash')))
        fig_compare.update_layout(
            title=f"Comparaison MC vs Analytique (T={T_compare:.2f} ans)",
            xaxis_title="Strike K",
            yaxis_title="Prix",
            height=500,
            showlegend=True
        )
        st.plotly_chart(fig_compare, width="stretch")
        
        st.balloons()
        st.success("🎉 Analyse complète terminée avec succès!")
        
    except Exception as exc:
        st.error(f"❌ Erreur lors de l'analyse: {exc}")
        import traceback
        st.code(traceback.format_exc())
