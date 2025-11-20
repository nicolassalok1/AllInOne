#!/usr/bin/env python3
"""Streamlit app basé sur le notebook heston_iv_surfaces.ipynb."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import torch
from scipy.interpolate import griddata

from heston_torch import HestonParams, carr_madan_call_torch

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cpu")
MIN_IV_MATURITY = 0.1
CALIB_T_BAND = None

st.set_page_config(page_title="🚀 Heston IV Surfaces", layout="wide")
st.title("🚀 Surface IV Heston : CBOE → Calibration NN → Carr-Madan")

st.write(
    """**Pipeline issu du notebook `heston_iv_surfaces.ipynb`** :
1️⃣ Téléchargement des options CBOE (données retardées)
2️⃣ Calibration Heston (NN Carr-Madan) ciblée sur la zone d'analyse
3️⃣ Surfaces IV Carr-Madan vs Marché + heatmaps de prix
"""
)

# ---------------------------------------------------------------------------
# Téléchargement CBOE
# ---------------------------------------------------------------------------


def download_options_cboe(symbol: str, option_type: str) -> tuple[pd.DataFrame, float]:
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol.upper()}.json"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data", {})
    options = data.get("options", [])
    spot = float(data.get("current_price") or data.get("close") or np.nan)
    now = pd.Timestamp.utcnow().tz_localize(None)
    pattern = re.compile(rf"^{symbol.upper()}(?P<expiry>\d{{6}})(?P<cp>[CP])(?P<strike>\d+)$")

    rows: list[dict] = []
    for opt in options:
        m = pattern.match(opt.get("option", ""))
        if not m:
            continue
        cp = m.group("cp")
        if (option_type == "call" and cp != "C") or (option_type == "put" and cp != "P"):
            continue
        expiry_dt = pd.to_datetime(m.group("expiry"), format="%y%m%d")
        T = (expiry_dt - now).total_seconds() / (365.0 * 24 * 3600)
        if T <= 0:
            continue
        T = round(T, 2)
        if T <= MIN_IV_MATURITY:
            continue
        strike = int(m.group("strike")) / 1000.0
        bid = float(opt.get("bid") or 0.0)
        ask = float(opt.get("ask") or 0.0)
        last = float(opt.get("last_trade_price") or 0.0)
        mid = np.nan
        if bid > 0 and ask > 0:
            mid = 0.5 * (bid + ask)
        elif last > 0:
            mid = last
        if np.isnan(mid) or mid <= 0:
            continue
        mid = round(mid, 2)
        iv_val = opt.get("iv", np.nan)
        iv_val = float(iv_val) if iv_val not in (None, "") else np.nan
        rows.append({
            "S0": spot,
            "K": strike,
            "T": T,
            ("C_mkt" if option_type == "call" else "P_mkt"): mid,
            "iv_market": iv_val,
        })

    df = pd.DataFrame(rows)
    df = df[df["T"] > MIN_IV_MATURITY]
    return df, spot


@st.cache_data(show_spinner=False)
def load_cboe_data(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Télécharge et met en cache les données CBOE calls/puts + spot moyen."""
    calls_df, spot_calls = download_options_cboe(symbol, "call")
    puts_df, spot_puts = download_options_cboe(symbol, "put")
    S0_ref = float(np.nanmean([spot_calls, spot_puts]))
    return calls_df, puts_df, S0_ref


# ---------------------------------------------------------------------------
# Calibration NN Carr-Madan
# ---------------------------------------------------------------------------


def prices_from_unconstrained(u, S0_t, K_t, T_t, r, q):
    params = HestonParams.from_unconstrained(u[0], u[1], u[2], u[3], u[4])
    prices = []
    for S0_i, K_i, T_i in zip(S0_t, K_t, T_t):
        prices.append(carr_madan_call_torch(S0_i, r, q, T_i, params, K_i))
    return torch.stack(prices)


def loss(u, S0_t, K_t, T_t, C_t, r, q, weights=None):
    model_prices = prices_from_unconstrained(u, S0_t, K_t, T_t, r, q)
    diff = model_prices - C_t
    if weights is not None:
        return 0.5 * (weights * diff**2).mean()
    return 0.5 * (diff**2).mean()


def calibrate_heston_nn(df, r, q, max_iters, lr, spot_override, progress_callback=None):
    df_clean = df.dropna(subset=["S0", "K", "T", "C_mkt"])
    df_clean = df_clean[(df_clean["T"] > MIN_IV_MATURITY) & (df_clean["C_mkt"] > 0.05)]
    df_clean = df_clean[df_clean.get("iv_market", 0) > 0]
    if df_clean.empty:
        raise ValueError("Pas de points pour la calibration")

    S0_ref = spot_override if spot_override is not None else float(df_clean["S0"].median())
    moneyness = df_clean["K"].values / S0_ref

    S0_t = torch.tensor(df_clean["S0"].values, dtype=torch.float64, device=DEVICE)
    K_t = torch.tensor(df_clean["K"].values, dtype=torch.float64, device=DEVICE)
    T_t = torch.tensor(df_clean["T"].values, dtype=torch.float64, device=DEVICE)
    C_t = torch.tensor(df_clean["C_mkt"].values, dtype=torch.float64, device=DEVICE)

    weights_np = 1.0 / (np.abs(moneyness - 1.0) + 1e-3)
    weights_np = np.clip(weights_np / weights_np.mean(), 0.5, 5.0)
    weights_t = torch.tensor(weights_np, dtype=torch.float64, device=DEVICE)

    u = torch.zeros(5, dtype=torch.float64, device=DEVICE, requires_grad=True)
    optimizer = torch.optim.Adam([u], lr=lr)

    for iteration in range(max_iters):
        optimizer.zero_grad()
        loss_val = loss(u, S0_t, K_t, T_t, C_t, r, q, weights=weights_t)
        loss_val.backward()
        optimizer.step()
        if progress_callback:
            # On passe aussi la valeur du loss courant au callback
            progress_callback(iteration + 1, max_iters, float(loss_val.detach().cpu()))

    return HestonParams.from_unconstrained(u[0], u[1], u[2], u[3], u[4])


# ---------------------------------------------------------------------------
# Black-Scholes tools
# ---------------------------------------------------------------------------


def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    from scipy.stats import norm
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    from scipy.stats import norm
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol_option(price, S, K, T, r, option_type="call"):
    if T < MIN_IV_MATURITY:
        return np.nan
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    if price <= intrinsic:
        return np.nan
    sigma = 0.3
    for _ in range(100):
        est = bs_call(S, K, T, r, sigma) if option_type == "call" else bs_put(S, K, T, r, sigma)
        diff = est - price
        if abs(diff) < 1e-6:
            return sigma
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        from scipy.stats import norm
        vega = S * norm.pdf(d1) * math.sqrt(T)
        if vega < 1e-8:
            return np.nan
        sigma -= diff / vega
        if sigma <= 0:
            return np.nan
    return np.nan


def build_market_surface(df, price_col, opt_type, KK_cm, TT_cm):
    df = df.dropna(subset=[price_col]).copy()
    df = df[(df["T"] >= MIN_IV_MATURITY) & (df[price_col] > 0)]
    df["iv_calc"] = df.apply(lambda row: implied_vol_option(row[price_col], row["S0"], row["K"], row["T"], rf_rate, opt_type), axis=1)
    df = df.dropna(subset=["iv_calc"])
    if len(df) < 5:
        return None
    pts = df[["K", "T"]].to_numpy()
    vals = df["iv_calc"].to_numpy()
    surf = griddata(pts, vals, (KK_cm, TT_cm), method="linear")
    if surf is None or np.all(np.isnan(surf)):
        surf = griddata(pts, vals, (KK_cm, TT_cm), method="nearest")
    else:
        mask = np.isnan(surf)
        if mask.any():
            surf[mask] = griddata(pts, vals, (KK_cm[mask], TT_cm[mask]), method="nearest")
    return surf


def build_market_price_grid(df, price_col, KK_cm, TT_cm):
    df = df.dropna(subset=[price_col]).copy()
    df = df[(df["T"] >= MIN_IV_MATURITY) & (df[price_col] > 0)]
    if len(df) < 5:
        return None
    pts = df[["K", "T"]].to_numpy()
    vals = df[price_col].to_numpy()
    grid = griddata(pts, vals, (KK_cm, TT_cm), method="linear")
    if grid is None or np.all(np.isnan(grid)):
        grid = griddata(pts, vals, (KK_cm, TT_cm), method="nearest")
    else:
        mask = np.isnan(grid)
        if mask.any():
            grid[mask] = griddata(pts, vals, (KK_cm[mask], TT_cm[mask]), method="nearest")
    return grid


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.sidebar.header("⚙️ Configuration de base")
rf_rate = st.sidebar.number_input("Taux sans risque (r)", value=0.02, step=0.01, format="%.3f")
div_yield = st.sidebar.number_input("Dividende (q)", value=0.00, step=0.01, format="%.3f")
span_mc = st.sidebar.number_input("Span autour de S0 pour les grilles K", value=20.0, min_value=5.0, max_value=100.0, step=5.0)
n_maturities = 40
CALIB_T_BAND = None

st.header("⚙️ Paramètres de modélisation")
ticker = st.text_input("Ticker (sous-jacent)", value="SPY")

# État persistant pour les données CBOE et la maturité cible
if "calls_df" not in st.session_state:
    st.session_state.calls_df = None
    st.session_state.puts_df = None
    st.session_state.S0_ref = None
    st.session_state.calib_T_target = None

# Bouton dédié pour récupérer les données du ticker (dans le panel principal)
# Couleur différente de celle du bouton d'analyse (qui reste en type="primary")
fetch_btn = st.button("Récupérer les données du ticker", use_container_width=True)
st.divider()

if fetch_btn:
    try:
        calls_df, puts_df, S0_ref = load_cboe_data(ticker)
        st.session_state.calls_df = calls_df
        st.session_state.puts_df = puts_df
        st.session_state.S0_ref = S0_ref

        # Pastilles de statut sous le bouton une fois le téléchargement terminé
        st.info(f"📡 Données CBOE chargées pour {ticker} (cache)")
        st.success(f"{len(calls_df)} calls, {len(puts_df)} puts | S0 ≈ {S0_ref:.2f}")
    except Exception as exc:
        st.error(f"❌ Erreur lors du téléchargement des données CBOE : {exc}")

# Récupération de l'état courant
calls_df = st.session_state.calls_df
puts_df = st.session_state.puts_df
S0_ref = st.session_state.S0_ref
calib_T_target = st.session_state.calib_T_target

# Partie calibration NN (cachée tant que les données n'ont pas été récupérées)
if calls_df is not None and puts_df is not None and S0_ref is not None:
    col_nn, col_modes = st.columns(2)

    # Colonne gauche : bande T et maturité cible
    with col_nn:
        st.subheader("🎯 Calibration NN Carr-Madan")
        calib_T_band = st.number_input(
            "Largeur bande T (±)",
            value=0.04,
            min_value=0.01,
            max_value=0.5,
            step=0.01,
            format="%.2f",
            key="calib_T_band",
        )

        unique_T = sorted(calls_df["T"].round(2).unique())
        if unique_T:
            calib_T_target = st.selectbox(
                "Maturité T cible pour la calibration (Time to Maturity)",
                unique_T,
                index=unique_T.index(calib_T_target) if calib_T_target in unique_T and calib_T_target is not None else 0,
                format_func=lambda x: f"{x:.2f}",
            )
            st.session_state.calib_T_target = calib_T_target
        else:
            st.warning("Pas de maturités disponibles dans les données CBOE.")
            calib_T_target = None

    # Colonne droite : choix du mode de calibration via boutons
    with col_modes:
        st.subheader("⚙️ Modes de calibration NN")
        mode = st.radio(
            "Choisir un mode",
            ["Rapide", "Bonne", "Excellente"],
            index=1,
            horizontal=True,
        )
        if mode == "Rapide":
            max_iters = 300
            learning_rate = 0.01
        elif mode == "Bonne":
            max_iters = 1000
            learning_rate = 0.005
        else:  # Précision
            max_iters = 2000
            learning_rate = 0.001

        # Affiche les hyperparamètres associés au mode sélectionné
        st.markdown(
            f"**Itérations NN** : `{max_iters}`  \n"
            f"**Learning rate** : `{learning_rate}`"
        )

    if calib_T_target is not None:
        CALIB_T_BAND = (
            max(MIN_IV_MATURITY, calib_T_target - calib_T_band),
            calib_T_target + calib_T_band,
        )
    else:
        CALIB_T_BAND = None
else:
    CALIB_T_BAND = None

# Bouton d'analyse NN : on ne l'affiche que si les données ont été fetch avec succès
run_button = False
if calls_df is not None and puts_df is not None and S0_ref is not None:
    run_button = st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True)
    st.divider()

if run_button:
    # Vérifications préalables : données et T cible doivent être disponibles
    if calls_df is None or puts_df is None or S0_ref is None:
        st.error("Veuillez d'abord cliquer sur « Récupérer les données du ticker ».")
    elif calib_T_target is None or CALIB_T_BAND is None:
        st.error("Veuillez choisir une maturité T cible après avoir chargé les données.")
    else:
        try:
            # Rappel visuel des infos marché sous le bouton d'analyse
            st.info(f"📡 Données CBOE chargées pour {ticker} (cache)")
            st.success(f"{len(calls_df)} calls, {len(puts_df)} puts | S0 ≈ {S0_ref:.2f}")
            st.write(f"Maturité T cible pour la calibration (Time to Maturity) : {calib_T_target:.2f} ans")

            st.info("🧠 Calibration ciblée...")
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            loss_log: list[float] = []

            def progress_cb(current: int, total: int, loss_val: float) -> None:
                progress_bar.progress(current / total)
                status_text.text(f"⏳ Iter {current}/{total} | Loss = {loss_val:.6f}")
                loss_log.append(loss_val)

            calib_slice = calls_df[
                (calls_df["T"].round(2).between(*CALIB_T_BAND)) &
                (calls_df["K"].between(S0_ref - span_mc, S0_ref + span_mc)) &
                (calls_df["C_mkt"] > 0.05) &
                (calls_df["iv_market"] > 0)
            ]
            if len(calib_slice) < 5:
                calib_slice = calls_df.copy()

            params_cm = calibrate_heston_nn(
                calib_slice,
                r=rf_rate,
                q=div_yield,
                max_iters=int(max_iters),
                lr=learning_rate,
                spot_override=S0_ref,
                progress_callback=progress_cb,
            )
            progress_bar.empty()
            status_text.empty()

            params_dict = {
                "kappa": float(params_cm.kappa.detach()),
                "theta": float(params_cm.theta.detach()),
                "sigma": float(params_cm.sigma.detach()),
                "rho": float(params_cm.rho.detach()),
                "v0": float(params_cm.v0.detach()),
            }
            st.success("✓ Calibration terminée")
            st.dataframe(pd.Series(params_dict, name="Paramètre").to_frame())

            # Log du loss pendant la calibration (courbe)
            if loss_log:
                st.subheader("📉 Évolution du loss pendant la calibration")
                st.line_chart(pd.DataFrame({"loss": loss_log}))

            # ---------------------------------------------------------------
            st.info("📐 Surfaces analytiques Carr-Madan")
            K_grid = np.arange(S0_ref - span_mc, S0_ref + span_mc + 1, 1)
            t_min = max(MIN_IV_MATURITY, CALIB_T_BAND[0])
            t_max = max(t_min + 0.05, min(2.0, CALIB_T_BAND[1]))
            T_grid = np.linspace(t_min, t_max, n_maturities)
            K_grid = np.unique(K_grid)
            T_grid = np.unique(T_grid)
            Ks_t = torch.tensor(K_grid, dtype=torch.float64)

            call_prices_cm = np.zeros((len(T_grid), len(K_grid)))
            put_prices_cm = np.zeros_like(call_prices_cm)
            for i, T_val in enumerate(T_grid):
                call_vals = carr_madan_call_torch(S0_ref, rf_rate, div_yield, float(T_val), params_cm, Ks_t)
                discount = torch.exp(-torch.tensor(rf_rate * T_val, dtype=torch.float64))
                forward = torch.exp(-torch.tensor(div_yield * T_val, dtype=torch.float64))
                put_vals = call_vals - S0_ref * forward + Ks_t * discount
                call_prices_cm[i, :] = call_vals.detach().cpu().numpy()
                put_prices_cm[i, :] = put_vals.detach().cpu().numpy()

            call_iv_cm = np.zeros_like(call_prices_cm)
            put_iv_cm = np.zeros_like(put_prices_cm)
            for i, T_val in enumerate(T_grid):
                for j, K_val in enumerate(K_grid):
                    call_iv_cm[i, j] = implied_vol_option(call_prices_cm[i, j], S0_ref, K_val, T_val, rf_rate, "call")
                    put_iv_cm[i, j] = implied_vol_option(put_prices_cm[i, j], S0_ref, K_val, T_val, rf_rate, "put")

            KK_cm, TT_cm = np.meshgrid(K_grid, T_grid, indexing="xy")

            # Détermination des bornes Z pour les surfaces d'IV (0 → vol max + 0.1)
            call_iv_max = float(np.nanmax(call_iv_cm)) if call_iv_cm.size > 0 else 0.0
            put_iv_max = float(np.nanmax(put_iv_cm)) if put_iv_cm.size > 0 else 0.0

            # Surfaces marché (IV recalculée) – calculées avant pour ajuster les bornes Z
            surf_call_market = build_market_surface(calls_df, "C_mkt", "call", KK_cm, TT_cm)
            surf_put_market = build_market_surface(puts_df, "P_mkt", "put", KK_cm, TT_cm)

            if surf_call_market is not None:
                call_iv_max = max(call_iv_max, float(np.nanmax(surf_call_market)))
            if surf_put_market is not None:
                put_iv_max = max(put_iv_max, float(np.nanmax(surf_put_market)))

            call_iv_zmax = call_iv_max + 0.1
            put_iv_zmax = put_iv_max + 0.1

            fig_call_cm = go.Figure(
                data=[
                    go.Surface(
                        x=KK_cm,
                        y=TT_cm,
                        z=call_iv_cm,
                        colorscale="Viridis",
                        cmin=0.0,
                        cmax=call_iv_zmax,
                    )
                ]
            )
            fig_call_cm.update_layout(
                title=f"IV Surface Calls (Carr-Madan) - {ticker}",
                scene=dict(
                    xaxis_title="K",
                    yaxis_title="T",
                    zaxis_title="IV",
                    zaxis=dict(range=[0.0, call_iv_zmax]),
                ),
                height=600,
            )

            fig_put_cm = go.Figure(
                data=[
                    go.Surface(
                        x=KK_cm,
                        y=TT_cm,
                        z=put_iv_cm,
                        colorscale="Viridis",
                        cmin=0.0,
                        cmax=put_iv_zmax,
                    )
                ]
            )
            fig_put_cm.update_layout(
                title=f"IV Surface Puts (Carr-Madan) - {ticker}",
                scene=dict(
                    xaxis_title="K",
                    yaxis_title="T",
                    zaxis_title="IV",
                    zaxis=dict(range=[0.0, put_iv_zmax]),
                ),
                height=600,
            )

            # Surfaces marché (IV recalculée)
            fig_call_market = None
            fig_put_market = None
            if surf_call_market is not None:
                fig_call_market = go.Figure(
                    data=[
                        go.Surface(
                            x=KK_cm,
                            y=TT_cm,
                            z=surf_call_market,
                            colorscale="Plasma",
                            cmin=0.0,
                            cmax=call_iv_zmax,
                        )
                    ]
                )
                fig_call_market.update_layout(
                    title=f"IV Surface Calls (Marché) - {ticker}",
                    scene=dict(
                        xaxis_title="K",
                        yaxis_title="T",
                        zaxis_title="IV",
                        zaxis=dict(range=[0.0, call_iv_zmax]),
                    ),
                    height=600,
                )
            if surf_put_market is not None:
                fig_put_market = go.Figure(
                    data=[
                        go.Surface(
                            x=KK_cm,
                            y=TT_cm,
                            z=surf_put_market,
                            colorscale="Plasma",
                            cmin=0.0,
                            cmax=put_iv_zmax,
                        )
                    ]
                )
                fig_put_market.update_layout(
                    title=f"IV Surface Puts (Marché) - {ticker}",
                    scene=dict(
                        xaxis_title="K",
                        yaxis_title="T",
                        zaxis_title="IV",
                        zaxis=dict(range=[0.0, put_iv_zmax]),
                    ),
                    height=600,
                )

            # Heatmaps prix (Carr-Madan & Marché)
            market_call_grid = build_market_price_grid(calls_df, "C_mkt", KK_cm, TT_cm)
            market_put_grid = build_market_price_grid(puts_df, "P_mkt", KK_cm, TT_cm)

            tab_calls, tab_puts = st.tabs(["📈 Calls", "📉 Puts"])

            # Échelle de couleurs commune avec padding ±10 autour du range des prix modèle
            pad = 10.0
            call_min = float(call_prices_cm.min())
            call_max = float(call_prices_cm.max())
            call_zmin = call_min - pad
            call_zmax = call_max + pad

            put_min = float(put_prices_cm.min())
            put_max = float(put_prices_cm.max())
            put_zmin = put_min - pad
            put_zmax = put_max + pad

            fig_heat_call_cm = go.Figure(
                data=[
                    go.Heatmap(
                        z=call_prices_cm,
                        x=K_grid,
                        y=T_grid,
                        colorscale="Viridis",
                        colorbar=dict(title="Prix des Call Heston"),
                        zmin=call_zmin,
                        zmax=call_zmax,
                    )
                ]
            )
            fig_heat_put_cm = go.Figure(
                data=[
                    go.Heatmap(
                        z=put_prices_cm,
                        x=K_grid,
                        y=T_grid,
                        colorscale="Viridis",
                        colorbar=dict(title="Prix des Put Heston"),
                        zmin=put_zmin,
                        zmax=put_zmax,
                    )
                ]
            )

            # Onglet Calls : IV et prix Carr-Madan côte à côte, puis marché
            with tab_calls:
                st.subheader("Carr-Madan : IV & prix (Calls)")
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(fig_call_cm, use_container_width=True)
                with c2:
                    st.plotly_chart(fig_heat_call_cm, use_container_width=True)

                st.subheader("Marché : IV & prix (Calls)")
                c3, c4 = st.columns(2)
                with c3:
                    if fig_call_market:
                        st.plotly_chart(fig_call_market, use_container_width=True)
                    else:
                        st.info("Pas assez de points marché pour la surface call.")
                with c4:
                    if market_call_grid is not None:
                        fig_heat_call_mkt = go.Figure(
                            data=[
                                go.Heatmap(
                                    z=market_call_grid,
                                    x=K_grid,
                                    y=T_grid,
                                    colorscale="Plasma",
                                    colorbar=dict(title="Prix des Call Marché"),
                                    zmin=call_zmin,
                                    zmax=call_zmax,
                                )
                            ]
                        )
                        st.plotly_chart(fig_heat_call_mkt, use_container_width=True)
                    else:
                        st.info("Pas assez de points marché pour la heatmap call.")

                # Description textuelle de ce que l'utilisateur voit sur les heatmaps de calls
                st.markdown(
                    f"""
**Lecture des heatmaps (Calls)**  
- Titre : comparaison des prix de calls Carr-Madan et marché  
- Axe des abscisses (**K**) : strikes autour de S₀ ≈ `{S0_ref:.2f}`  
- Axe des ordonnées (**T**) : maturité en années  
- Couleur : niveau de **prix du call** pour chaque couple (K, T)  
- Paramètres utilisés : `S0 = {S0_ref:.2f}`, `r = {rf_rate:.3f}`, `q = {div_yield:.3f}`
"""
                )

            # Onglet Puts : IV et prix Carr-Madan côte à côte, puis marché
            with tab_puts:
                st.subheader("Carr-Madan : IV & prix (Puts)")
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(fig_put_cm, use_container_width=True)
                with c2:
                    st.plotly_chart(fig_heat_put_cm, use_container_width=True)

                st.subheader("Marché : IV & prix (Puts)")
                c3, c4 = st.columns(2)
                with c3:
                    if fig_put_market:
                        st.plotly_chart(fig_put_market, use_container_width=True)
                    else:
                        st.info("Pas assez de points marché pour la surface put.")
                with c4:
                    if market_put_grid is not None:
                        fig_heat_put_mkt = go.Figure(
                            data=[
                                go.Heatmap(
                                    z=market_put_grid,
                                    x=K_grid,
                                    y=T_grid,
                                    colorscale="Plasma",
                                    colorbar=dict(title="Prix des Put Marché"),
                                    zmin=put_zmin,
                                    zmax=put_zmax,
                                )
                            ]
                        )
                        st.plotly_chart(fig_heat_put_mkt, use_container_width=True)
                    else:
                        st.info("Pas assez de points marché pour la heatmap put.")

                # Description textuelle de ce que l'utilisateur voit sur les heatmaps de puts
                st.markdown(
                    f"""
**Lecture des heatmaps (Puts)**  
- Titre : comparaison des prix de puts Carr-Madan et marché  
- Axe des abscisses (**K**) : strikes autour de S₀ ≈ `{S0_ref:.2f}`  
- Axe des ordonnées (**T**) : maturité en années  
- Couleur : niveau de **prix du put** pour chaque couple (K, T)  
- Paramètres utilisés : `S0 = {S0_ref:.2f}`, `r = {rf_rate:.3f}`, `q = {div_yield:.3f}`
"""
                )

            st.balloons()
            st.success("🎉 Analyse terminée")

        except Exception as exc:
            st.error(f"❌ Erreur : {exc}")
            import traceback
            st.code(traceback.format_exc())
