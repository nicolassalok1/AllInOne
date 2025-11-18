#!/usr/bin/env python3
"""Streamlit app: compare Heston vs Black-Scholes heatmaps and IV surfaces after calibration."""

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

st.set_page_config(page_title="Heston vs Black-Scholes Heatmaps & IVs", layout="wide")
st.title("Heston Calibration (yfinance) → Heatmaps & IV Surfaces")
st.write(
    "Télécharge des options via `yfinance`, calibre Heston sur les calls avec PyTorch, puis compare "
    "les heatmaps Call/Put Heston avec celles de Black-Scholes. Ajoute aussi les surfaces d'IV "
    "Market, BS et Heston (calls & puts) sur les strikes observés."
)

# Import du module Heston torch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "Heston" / "NN"))

from heston_torch import HestonParams, carr_madan_call_torch  # type: ignore  # noqa: E402

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cpu")
MIN_IV_MATURITY = 0.1


def fetch_spot(symbol: str) -> float:
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
    return pd.DataFrame(rows)


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


def calibrate_heston_from_calls(
    calls_df: pd.DataFrame,
    r: float,
    q: float,
    max_points: int,
    max_iters: int,
    lr: float,
    progress_callback: Callable[[int, int, float], None] | None = None,
    log_callback: Callable[[int, float], None] | None = None,
) -> tuple[dict[str, float], list[float], pd.DataFrame]:
    df = calls_df[["S0", "K", "T", "C_mkt"]].dropna().copy()
    n_total = len(df)
    if n_total > max_points:
        df = df.sort_values("T")
        idx = np.linspace(0, n_total - 1, max_points, dtype=int)
        df = df.iloc[idx]
    df = df.reset_index(drop=True)

    S0_t = torch.tensor(df["S0"].to_numpy(), dtype=torch.float64, device=DEVICE)
    K_t = torch.tensor(df["K"].to_numpy(), dtype=torch.float64, device=DEVICE)
    T_t = torch.tensor(df["T"].to_numpy(), dtype=torch.float64, device=DEVICE)
    C_mkt_t = torch.tensor(df["C_mkt"].to_numpy(), dtype=torch.float64, device=DEVICE)

    u = torch.tensor([1.0, -3.0, -0.5, -0.5, -3.0], dtype=torch.float64, device=DEVICE, requires_grad=True)
    optimizer = torch.optim.Adam([u], lr=lr)
    history: list[float] = []

    for it in range(max_iters):
        optimizer.zero_grad()
        L = loss(u, S0_t, K_t, T_t, C_mkt_t, r, q)
        L.backward()
        optimizer.step()
        curr_loss = float(L.detach().cpu())
        history.append(curr_loss)
        if progress_callback is not None:
            progress_callback(it, max_iters, curr_loss)
        if log_callback is not None:
            log_callback(it, curr_loss)

    with torch.no_grad():
        params_fin = HestonParams.from_unconstrained(u[0], u[1], u[2], u[3], u[4])
    calib = {
        "kappa": float(params_fin.kappa.cpu()),
        "theta": float(params_fin.theta.cpu()),
        "sigma": float(params_fin.sigma.cpu()),
        "rho": float(params_fin.rho.cpu()),
        "v0": float(params_fin.v0.cpu()),
    }
    summary = pd.DataFrame(
        {
            "used_quotes": [len(df)],
            "total_quotes": [n_total],
            "loss_final": [history[-1] if history else float("nan")],
        }
    )
    return calib, history, summary


def params_from_calib(calib: dict[str, float]) -> HestonParams:
    return HestonParams(
        kappa=torch.tensor(calib["kappa"], dtype=torch.float64, device=DEVICE),
        theta=torch.tensor(calib["theta"], dtype=torch.float64, device=DEVICE),
        sigma=torch.tensor(calib["sigma"], dtype=torch.float64, device=DEVICE),
        rho=torch.tensor(calib["rho"], dtype=torch.float64, device=DEVICE),
        v0=torch.tensor(calib["v0"], dtype=torch.float64, device=DEVICE),
    )


def bs_price(S0: float, K: float, T: float, vol: float, r: float) -> float:
    if T <= 0.0 or vol <= 0.0:
        return max(0.0, S0 - K * math.exp(-r * T))
    sqrt_T = math.sqrt(T)
    v = vol * sqrt_T
    d1 = (math.log(S0 / K) + (r + 0.5 * vol * vol) * T) / v
    d2 = d1 - v
    nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
    return S0 * nd1 - K * math.exp(-r * T) * nd2


def implied_vol(price: float, S0: float, K: float, T: float, r: float, tol: float = 1e-6, max_iter: int = 100) -> float:
    intrinsic = max(0.0, S0 - K * math.exp(-r * T))
    if price <= intrinsic + 1e-12:
        return 0.0
    low, high = 1e-6, 1.0
    p_high = bs_price(S0, K, T, high, r)
    while p_high < price and high < 5.0:
        high *= 2.0
        p_high = bs_price(S0, K, T, high, r)
    if p_high < price:
        return float("nan")
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        p_mid = bs_price(S0, K, T, mid, r)
        if abs(p_mid - price) < tol:
            return mid
        if p_mid > price:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def bs_price_option(S0: float, K: float, T: float, vol: float, r: float, option_type: str) -> float:
    if T <= 0.0 or vol <= 0.0:
        intrinsic_call = max(0.0, S0 - K * math.exp(-r * T))
        intrinsic_put = max(0.0, K * math.exp(-r * T) - S0)
        return intrinsic_call if option_type == "call" else intrinsic_put
    sqrt_T = math.sqrt(T)
    vol_sqrt_T = vol * sqrt_T
    d1 = (math.log(S0 / K) + (r + 0.5 * vol * vol) * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    discount = math.exp(-r * T)
    if option_type == "call":
        return S0 * 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0))) - K * discount * 0.5 * (
            1.0 + math.erf(d2 / math.sqrt(2.0))
        )
    return K * discount * 0.5 * (1.0 + math.erf(-d2 / math.sqrt(2.0))) - S0 * 0.5 * (
        1.0 + math.erf(-d1 / math.sqrt(2.0))
    )


def implied_vol_option(price: float, S0: float, K: float, T: float, r: float, option_type: str) -> float:
    if T <= 0.0 or price <= 0.0 or S0 <= 0.0 or K <= 0.0:
        return 0.0
    intrinsic = bs_price_option(S0, K, T, 0.0, r, option_type)
    if price <= intrinsic + 1e-8:
        return 0.0
    vol_low, vol_high = 1e-6, 1.0
    price_high = bs_price_option(S0, K, T, vol_high, r, option_type)
    while price_high < price and vol_high < 5.0:
        vol_high *= 2.0
        price_high = bs_price_option(S0, K, T, vol_high, r, option_type)
    if price_high < price:
        return float("nan")
    for _ in range(100):
        vol_mid = 0.5 * (vol_low + vol_high)
        price_mid = bs_price_option(S0, K, T, vol_mid, r, option_type)
        if abs(price_mid - price) < 1e-6:
            return vol_mid
        if price_mid > price:
            vol_high = vol_mid
        else:
            vol_low = vol_mid
    return 0.5 * (vol_low + vol_high)


def add_iv_columns(
    df: pd.DataFrame,
    price_col: str,
    option_type: str,
    r: float,
    q: float,
    params_tensor: HestonParams,
) -> pd.DataFrame:
    df_out = df.dropna(subset=[price_col]).copy()
    iv_bs_vals: list[float] = []
    iv_heston_vals: list[float] = []
    with torch.no_grad():
        for row in df_out.itertuples(index=False):
            market_price = float(getattr(row, price_col))
            iv_bs_vals.append(implied_vol_option(market_price, row.S0, row.K, row.T, r, option_type))
            S_tensor = torch.tensor(row.S0, dtype=torch.float64, device=DEVICE)
            K_tensor = torch.tensor(row.K, dtype=torch.float64, device=DEVICE)
            T_tensor = torch.tensor(row.T, dtype=torch.float64, device=DEVICE)
            call_price = carr_madan_call_torch(S_tensor, r, q, T_tensor, params_tensor, K_tensor)
            call_val = float(call_price.cpu())
            if option_type == "call":
                price_heston = call_val
            else:
                price_heston = call_val - row.S0 + row.K * math.exp(-r * row.T)
            iv_heston_vals.append(implied_vol_option(price_heston, row.S0, row.K, row.T, r, option_type))
    df_out["iv_bs"] = iv_bs_vals
    df_out["iv_heston"] = iv_heston_vals
    if "iv_market" not in df_out.columns:
        df_out["iv_market"] = np.nan
    return df_out


def prepare_iv_surface(
    df: pd.DataFrame, iv_column: str, spot: float, strike_span: float, min_maturity: float = MIN_IV_MATURITY
) -> pd.DataFrame:
    lower = spot - strike_span
    upper = spot + strike_span
    df_sel = df[(df["K"] >= lower) & (df["K"] <= upper) & (df["T"] >= min_maturity)].copy()
    df_sel = df_sel.dropna(subset=[iv_column])
    if df_sel.empty:
        raise ValueError(f"Aucune donnée pour construire la surface {iv_column}.")
    df_sel = df_sel.sort_values(["T", "K"])
    surface = df_sel.pivot_table(index="T", columns="K", values=iv_column, aggfunc="mean")
    surface = surface.sort_index().sort_index(axis=1)
    surface = surface.interpolate(axis=1, limit_direction="both").interpolate(axis=1, limit_direction="both")
    if surface.empty:
        raise ValueError(f"Surface vide pour {iv_column}.")
    return surface


def plot_iv_surface(surface: pd.DataFrame, spot: float, title_suffix: str) -> go.Figure:
    k_vals = surface.columns.to_numpy(dtype=float)
    t_vals = surface.index.to_numpy(dtype=float)
    KK, TT = np.meshgrid(k_vals, t_vals)
    data = surface.to_numpy(dtype=float)
    if np.isnan(data).any():
        mean_val = np.nanmean(data)
        if np.isnan(mean_val):
            mean_val = 0.0
        data = np.nan_to_num(data, nan=mean_val)
    fig = go.Figure(
        data=[
            go.Surface(
                x=KK,
                y=TT,
                z=data,
                colorscale="Viridis",
                colorbar=dict(title="IV"),
                showscale=True,
            )
        ]
    )
    fig.update_layout(
        title=title_suffix,
        scene=dict(
            xaxis=dict(title=f"Strike K (spot ≈ {spot:.2f})"),
            yaxis=dict(title="Maturité T (années)"),
            zaxis=dict(title="Implied Volatility"),
        ),
        height=550,
    )
    return fig


def compute_heston_heatmaps(
    calib: dict[str, float], r: float, q: float, S0_ref: float, span: float, points: int, maturity: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    S_grid = np.linspace(S0_ref - span, S0_ref + span, points)
    K_grid = np.linspace(S0_ref - span, S0_ref + span, points)
    call_map = np.zeros((len(S_grid), len(K_grid)))
    put_map = np.zeros_like(call_map)
    params_tensor = params_from_calib(calib)

    with torch.no_grad():
        T_tensor = torch.tensor(maturity, dtype=torch.float64, device=DEVICE)
        for i, S_val in enumerate(S_grid):
            S_tensor = torch.tensor(S_val, dtype=torch.float64, device=DEVICE)
            for j, K_val in enumerate(K_grid):
                K_tensor = torch.tensor(K_val, dtype=torch.float64, device=DEVICE)
                call_p = carr_madan_call_torch(S_tensor, r, q, T_tensor, params_tensor, K_tensor)
                call_val = float(call_p.cpu())
                call_map[i, j] = call_val
                put_map[i, j] = call_val - S_val + K_val * math.exp(-r * maturity)
    return S_grid, K_grid, call_map, put_map


def compute_bs_heatmaps(
    S_grid: np.ndarray, K_grid: np.ndarray, maturity: float, r: float, vol: float
) -> tuple[np.ndarray, np.ndarray]:
    call_map = np.zeros((len(S_grid), len(K_grid)))
    put_map = np.zeros_like(call_map)
    for i, S_val in enumerate(S_grid):
        for j, K_val in enumerate(K_grid):
            call_val = bs_price(S_val, K_val, maturity, vol, r)
            call_map[i, j] = call_val
            put_map[i, j] = call_val - S_val + K_val * math.exp(-r * maturity)
    return call_map, put_map


def plot_heatmap(matrix: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=np.round(x_grid, 2),
            y=np.round(y_grid, 2),
            colorscale="Viridis",
            colorbar=dict(title=title),
        )
    )
    fig.update_layout(xaxis_title="Strike K", yaxis_title="Spot S₀", yaxis_autorange="reversed", title=title)
    return fig


with st.sidebar:
    st.header("Paramètres")
    ticker = st.text_input("Ticker", value="SPY").strip().upper()
    rf_rate = st.slider("Taux sans risque r", min_value=-0.01, max_value=0.10, value=0.02, step=0.005)
    years_ahead = 2.5
    max_quotes = 300
    max_iters = 80
    lr = 5e-3
    heatmap_span = st.number_input("Écart autour de S₀ (±)", min_value=10.0, max_value=200.0, value=100.0, step=5.0)
    heatmap_points = st.slider("Points par axe", min_value=5, max_value=31, value=21, step=2)
    heatmap_maturity = st.number_input("Maturité heatmap (années)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    run_button = st.button("Télécharger, calibrer & comparer")


if run_button:
    if not ticker:
        st.warning("Merci de renseigner un ticker.")
    else:
        try:
            calls_df = download_options(ticker, "call", years_ahead=years_ahead)
            puts_df = download_options(ticker, "put", years_ahead=years_ahead)
            st.write(f"{len(calls_df)} calls et {len(puts_df)} puts téléchargés pour {ticker}.")
            S0_ref = float(calls_df["S0"].median())

            progress_bar = st.progress(0.0, text="Calibration Heston...")
            log_box = st.empty()
            log_messages: list[str] = []

            def progress_cb(iter_idx: int, total: int, loss_val: float) -> None:
                fraction = (iter_idx + 1) / total
                progress_bar.progress(fraction, text=f"Calibration... loss={loss_val:.3e}")

            def log_cb(iter_idx: int, loss_val: float) -> None:
                log_messages.append(f"Iter {iter_idx:03d} | loss = {loss_val:.6e}")
                log_box.text("\n".join(log_messages[-10:]))

            calib, history, summary = calibrate_heston_from_calls(
                calls_df,
                r=rf_rate,
                q=0.0,
                max_points=max_quotes,
                max_iters=max_iters,
                lr=lr,
                progress_callback=progress_cb,
                log_callback=log_cb,
            )
            progress_bar.empty()
            log_box.empty()

            st.subheader("Résumé calibration")
            st.dataframe(summary)
            st.subheader("Paramètres Heston calibrés")
            st.dataframe(pd.Series(calib, name="params").to_frame())

            st.subheader("Historique de loss")
            loss_df = pd.DataFrame({"iteration": range(len(history)), "loss": history})
            st.line_chart(loss_df.set_index("iteration"))

            st.subheader("Heatmaps Heston vs Black-Scholes")
            S_grid, K_grid, call_heston, put_heston = compute_heston_heatmaps(
                calib,
                r=rf_rate,
                q=0.0,
                S0_ref=S0_ref,
                span=heatmap_span,
                points=heatmap_points,
                maturity=heatmap_maturity,
            )
            params_tensor = params_from_calib(calib)
            with torch.no_grad():
                atm_call = carr_madan_call_torch(
                    torch.tensor(S0_ref, dtype=torch.float64, device=DEVICE),
                    rf_rate,
                    0.0,
                    torch.tensor(heatmap_maturity, dtype=torch.float64, device=DEVICE),
                    params_tensor,
                    torch.tensor(S0_ref, dtype=torch.float64, device=DEVICE),
                )
            vol_bs = implied_vol(float(atm_call.cpu()), S0_ref, S0_ref, heatmap_maturity, rf_rate)
            call_bs, put_bs = compute_bs_heatmaps(S_grid, K_grid, heatmap_maturity, rf_rate, vol_bs)

            summary_heatmap = pd.DataFrame(
                {
                    "Reference spot": [S0_ref],
                    "Rate": [rf_rate],
                    "Maturity T": [heatmap_maturity],
                    "Strike range": [f"{K_grid[0]:.2f} → {K_grid[-1]:.2f} ({len(K_grid)} pts)"],
                    "Spot range": [f"{S_grid[0]:.2f} → {S_grid[-1]:.2f} ({len(S_grid)} pts)"],
                    "κ": [calib["kappa"]],
                    "θ": [calib["theta"]],
                    "σ": [calib["sigma"]],
                    "ρ": [calib["rho"]],
                    "v₀": [calib["v0"]],
                    "σ_BS_ATM": [vol_bs],
                }
            )
            st.dataframe(summary_heatmap)

            fig_call_heston = plot_heatmap(call_heston, K_grid, S_grid, "Call Price (Heston)")
            fig_put_heston = plot_heatmap(put_heston, K_grid, S_grid, "Put Price (Heston)")
            fig_call_bs = plot_heatmap(call_bs, K_grid, S_grid, "Call Price (Black-Scholes)")
            fig_put_bs = plot_heatmap(put_bs, K_grid, S_grid, "Put Price (Black-Scholes)")

            col_call_h, col_call_bs = st.columns(2)
            with col_call_h:
                st.subheader("Call Heston")
                st.plotly_chart(fig_call_heston, use_container_width=True)
            with col_call_bs:
                st.subheader("Call Black-Scholes")
                st.plotly_chart(fig_call_bs, use_container_width=True)

            col_put_h, col_put_bs = st.columns(2)
            with col_put_h:
                st.subheader("Put Heston")
                st.plotly_chart(fig_put_heston, use_container_width=True)
            with col_put_bs:
                st.subheader("Put Black-Scholes")
                st.plotly_chart(fig_put_bs, use_container_width=True)

            st.subheader("Surfaces d'IV (Market vs BS vs Heston)")

            def _surface_or_none(df: pd.DataFrame, iv_col: str, label: str) -> pd.DataFrame | None:
                try:
                    return prepare_iv_surface(df, iv_col, S0_ref, heatmap_span)
                except ValueError as exc:
                    st.warning(f"{label}: {exc}")
                    return None

            calls_with_iv = add_iv_columns(calls_df, "C_mkt", "call", rf_rate, 0.0, params_tensor)
            puts_with_iv = add_iv_columns(puts_df, "P_mkt", "put", rf_rate, 0.0, params_tensor)

            surfaces_calls = [
                ("Call Market IV", _surface_or_none(calls_with_iv, "iv_market", "Call Market IV")),
                ("Call BS IV", _surface_or_none(calls_with_iv, "iv_bs", "Call BS IV")),
                ("Call Heston IV", _surface_or_none(calls_with_iv, "iv_heston", "Call Heston IV")),
            ]
            surfaces_puts = [
                ("Put Market IV", _surface_or_none(puts_with_iv, "iv_market", "Put Market IV")),
                ("Put BS IV", _surface_or_none(puts_with_iv, "iv_bs", "Put BS IV")),
                ("Put Heston IV", _surface_or_none(puts_with_iv, "iv_heston", "Put Heston IV")),
            ]

            call_cols = st.columns(3)
            for col, (title, surface_obj) in zip(call_cols, surfaces_calls):
                with col:
                    st.caption(title)
                    if surface_obj is not None:
                        st.plotly_chart(plot_iv_surface(surface_obj, S0_ref, title), use_container_width=True)
                    else:
                        st.info("Surface indisponible.")

            put_cols = st.columns(3)
            for col, (title, surface_obj) in zip(put_cols, surfaces_puts):
                with col:
                    st.caption(title)
                    if surface_obj is not None:
                        st.plotly_chart(plot_iv_surface(surface_obj, S0_ref, title), use_container_width=True)
                    else:
                        st.info("Surface indisponible.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Échec du téléchargement ou de la calibration: {exc}")
else:
    st.info("Configure les paramètres dans la barre latérale puis clique sur **Télécharger, calibrer & comparer**.")
