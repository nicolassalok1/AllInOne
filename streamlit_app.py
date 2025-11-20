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
CALIB_T_BAND = (0.82, 0.90)

st.set_page_config(page_title="🚀 Heston IV Surfaces", layout="wide")
st.title("🚀 Surface IV Heston : CBOE → Calibration NN → Carr-Madan")

st.write(
    """**Pipeline issu du notebook `heston_iv_surfaces.ipynb`** :
1️⃣ Téléchargement des options CBOE (données retardées)
2️⃣ Calibration Heston (NN Carr-Madan) ciblée sur la zone d'analyse
3️⃣ Surfaces IV Carr-Madan vs Marché + heatmaps de prix
4️⃣ Reconstruction des surfaces IV à partir des heatmaps
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
            progress_callback(iteration + 1, max_iters)

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


def iv_surface_from_prices(price_grid, option_type):
    iv_grid = np.full_like(price_grid, np.nan)
    for i, T_val in enumerate(T_grid):
        for j, K_val in enumerate(K_grid):
            iv_grid[i, j] = implied_vol_option(price_grid[i, j], S0_ref, K_val, T_val, rf_rate, option_type)
    return iv_grid


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.sidebar.header("⚙️ Configuration de base")
ticker = st.sidebar.text_input("Ticker", value="SPY")
rf_rate = st.sidebar.number_input("Taux sans risque (r)", value=0.02, step=0.01, format="%.3f")
div_yield = st.sidebar.number_input("Dividende (q)", value=0.00, step=0.01, format="%.3f")
span_mc = st.sidebar.number_input("Span autour de S0 pour les grilles K", value=20.0, min_value=5.0, max_value=100.0, step=5.0)
n_maturities = 40

st.header("⚙️ Paramètres de modélisation")
col_nn, _ = st.columns(2)
with col_nn:
    st.subheader("🎯 Calibration NN Carr-Madan")
    max_iters = st.number_input("Itérations NN", value=1000, min_value=100, max_value=5000, step=100)
    learning_rate = st.number_input("Learning rate", value=0.005, min_value=0.0005, max_value=0.05, step=0.0005, format="%.4f")

run_button = st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True)
st.divider()

if run_button:
    try:
        st.info(f"📡 Téléchargement des données CBOE pour {ticker}...")
        calls_df, spot_calls = download_options_cboe(ticker, "call")
        puts_df, spot_puts = download_options_cboe(ticker, "put")
        S0_ref = float(np.nanmean([spot_calls, spot_puts]))
        st.success(f"✓ {len(calls_df)} calls et {len(puts_df)} puts téléchargés | S0 ≈ {S0_ref:.2f}")

        st.info("🧠 Calibration ciblée...")
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def progress_cb(current: int, total: int) -> None:
            progress_bar.progress(current / total)
            status_text.text(f"⏳ Iter {current}/{total}")

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

        # ---------------------------------------------------------------
        st.info("📐 Surfaces analytiques Carr-Madan")
        K_grid = np.arange(S0_ref - span_mc, S0_ref + span_mc + 1, 1)
        T_grid = np.linspace(0.1, 2.0, n_maturities)
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
        fig_call_cm = go.Figure(data=[go.Surface(x=KK_cm, y=TT_cm, z=call_iv_cm, colorscale='Viridis')])
        fig_call_cm.update_layout(title=f"IV Surface Calls (Carr-Madan) - {ticker}", scene=dict(xaxis_title='K', yaxis_title='T', zaxis_title='IV'), height=600)
        fig_put_cm = go.Figure(data=[go.Surface(x=KK_cm, y=TT_cm, z=put_iv_cm, colorscale='Viridis')])
        fig_put_cm.update_layout(title=f"IV Surface Puts (Carr-Madan) - {ticker}", scene=dict(xaxis_title='K', yaxis_title='T', zaxis_title='IV'), height=600)

        # Surfaces marché (IV recalculée)
        surf_call_market = build_market_surface(calls_df, "C_mkt", "call", KK_cm, TT_cm)
        surf_put_market = build_market_surface(puts_df, "P_mkt", "put", KK_cm, TT_cm)
        fig_call_market = None
        fig_put_market = None
        if surf_call_market is not None:
            fig_call_market = go.Figure(data=[go.Surface(x=KK_cm, y=TT_cm, z=surf_call_market, colorscale='Plasma')])
            fig_call_market.update_layout(title=f"IV Surface Calls (Marché) - {ticker}", scene=dict(xaxis_title='K', yaxis_title='T', zaxis_title='IV'), height=600)
        if surf_put_market is not None:
            fig_put_market = go.Figure(data=[go.Surface(x=KK_cm, y=TT_cm, z=surf_put_market, colorscale='Plasma')])
            fig_put_market.update_layout(title=f"IV Surface Puts (Marché) - {ticker}", scene=dict(xaxis_title='K', yaxis_title='T', zaxis_title='IV'), height=600)

        st.subheader("🌊 IV Surfaces Carr-Madan vs Marché")
        col_call, col_put = st.columns(2)
        with col_call:
            st.plotly_chart(fig_call_cm, use_container_width=True)
            if fig_call_market:
                st.plotly_chart(fig_call_market, use_container_width=True)
            else:
                st.info("Pas assez de points marché pour la surface call.")
        with col_put:
            st.plotly_chart(fig_put_cm, use_container_width=True)
            if fig_put_market:
                st.plotly_chart(fig_put_market, use_container_width=True)
            else:
                st.info("Pas assez de points marché pour la surface put.")

        # Heatmaps prix (Carr-Madan & Marché)
        market_call_grid = build_market_price_grid(calls_df, "C_mkt", KK_cm, TT_cm)
        market_put_grid = build_market_price_grid(puts_df, "P_mkt", KK_cm, TT_cm)

        st.subheader("🔥 Heatmaps Prix (Carr-Madan vs Marché)")
        fig_heat_call_cm = go.Figure(data=[go.Heatmap(z=call_prices_cm, x=K_grid, y=T_grid, colorscale='Viridis', colorbar=dict(title='Call CM'))])
        fig_heat_put_cm = go.Figure(data=[go.Heatmap(z=put_prices_cm, x=K_grid, y=T_grid, colorscale='Viridis', colorbar=dict(title='Put CM'))])
        st.plotly_chart(fig_heat_call_cm, use_container_width=True)
        st.plotly_chart(fig_heat_put_cm, use_container_width=True)
        if market_call_grid is not None:
            fig_heat_call_mkt = go.Figure(data=[go.Heatmap(z=market_call_grid, x=K_grid, y=T_grid, colorscale='Plasma', colorbar=dict(title='Call Marché'), zmin=call_prices_cm.min(), zmax=call_prices_cm.max())])
            st.plotly_chart(fig_heat_call_mkt, use_container_width=True)
        if market_put_grid is not None:
            fig_heat_put_mkt = go.Figure(data=[go.Heatmap(z=market_put_grid, x=K_grid, y=T_grid, colorscale='Plasma', colorbar=dict(title='Put Marché'), zmin=put_prices_cm.min(), zmax=put_prices_cm.max())])
            st.plotly_chart(fig_heat_put_mkt, use_container_width=True)

        # Surfaces IV depuis les heatmaps prix
        st.subheader("🔄 IV surfaces recalculées depuis les heatmaps")
        call_iv_from_prices = iv_surface_from_prices(call_prices_cm, 'call')
        put_iv_from_prices = iv_surface_from_prices(put_prices_cm, 'put')
        fig_call_iv_heat = go.Figure(data=[go.Surface(x=KK_cm, y=TT_cm, z=call_iv_from_prices, colorscale='Viridis')])
        fig_call_iv_heat.update_layout(title=f"IV Carr-Madan (depuis heatmap prix) - {ticker}", scene=dict(xaxis_title='K', yaxis_title='T', zaxis_title='IV'), height=600)
        st.plotly_chart(fig_call_iv_heat, use_container_width=True)
        fig_put_iv_heat = go.Figure(data=[go.Surface(x=KK_cm, y=TT_cm, z=put_iv_from_prices, colorscale='Viridis')])
        fig_put_iv_heat.update_layout(title=f"IV Carr-Madan Puts (depuis heatmap prix) - {ticker}", scene=dict(xaxis_title='K', yaxis_title='T', zaxis_title='IV'), height=600)
        st.plotly_chart(fig_put_iv_heat, use_container_width=True)

        if market_call_grid is not None:
            call_iv_market_surface = iv_surface_from_prices(market_call_grid, 'call')
            fig_call_iv_market_heat = go.Figure(data=[go.Surface(x=KK_cm, y=TT_cm, z=call_iv_market_surface, colorscale='Plasma')])
            fig_call_iv_market_heat.update_layout(title=f"IV Marché (depuis heatmap prix) - Calls {ticker}", scene=dict(xaxis_title='K', yaxis_title='T', zaxis_title='IV'), height=600)
            st.plotly_chart(fig_call_iv_market_heat, use_container_width=True)

        if market_put_grid is not None:
            put_iv_market_surface = iv_surface_from_prices(market_put_grid, 'put')
            fig_put_iv_market_heat = go.Figure(data=[go.Surface(x=KK_cm, y=TT_cm, z=put_iv_market_surface, colorscale='Plasma')])
            fig_put_iv_market_heat.update_layout(title=f"IV Marché (depuis heatmap prix) - Puts {ticker}", scene=dict(xaxis_title='K', yaxis_title='T', zaxis_title='IV'), height=600)
            st.plotly_chart(fig_put_iv_market_heat, use_container_width=True)

        st.balloons()
        st.success("🎉 Analyse terminée")

    except Exception as exc:
        st.error(f"❌ Erreur : {exc}")
        import traceback
        st.code(traceback.format_exc())
