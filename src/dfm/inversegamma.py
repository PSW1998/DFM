from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import f as f_dist
import matplotlib.pyplot as plt


def scaled_f_predictive(x):
    return x

@dataclass(frozen=True)
class IGDFitResult:
    alpha_star: float
    beta_star: float
    max_loglik: float
    alpha_grid: np.ndarray
    beta_grid: np.ndarray
    loglik_grid: np.ndarray | None = None

def igdf_loglik(alpha: float, beta: float, y: np.ndarray, *, n0: float = 49.0, s0: float = 1/5000.0, clip: float = 1e-12) -> float:
    if alpha <= 0 or not (0 < beta < 1): return -np.inf
    y = np.asarray(y, float)
    if y.ndim != 1 or y.size == 0: return -np.inf
    y = np.clip(y, clip, None)
    n_prev, s_prev = float(n0), max(float(s0), clip)
    loglik = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        for yt in y:
            dfn = 2.0 * alpha
            dfd = 2.0 * beta * n_prev
            if dfn <= 0 or dfd <= 0: return -np.inf
            scale = max(s_prev, clip)
            z = yt / scale
            if z <= 0 or not np.isfinite(z): return -np.inf
            loglik += f_dist.logpdf(z, dfn, dfd) - np.log(scale)
            n_new = beta * n_prev + alpha
            s_new = (beta * n_prev * s_prev + alpha * yt) / n_new  # uses OLD n_prev in numerator
            n_prev, s_prev = n_new, max(s_new, clip)
    return float(loglik)

def fit_igdf_hyperparams(
    y, *,
    alpha_grid=None, beta_grid=None,
    n0: float = 49.0, s0: float = 1/5000.0, clip: float = 1e-12,
    warmup: int = 0, return_surface: bool = True
) -> IGDFitResult:
    y = np.asarray(list(y), float)
    if warmup > 0: y = y[warmup:]
    if y.ndim != 1 or y.size == 0: raise ValueError("`y` must be 1D and non-empty after warmup.")
    if alpha_grid is None: alpha_grid = np.linspace(1.0, 2.5, 11)
    if beta_grid  is None: beta_grid  = np.linspace(0.60, 0.95, 11)
    alpha_grid = np.asarray(list(alpha_grid), float)
    beta_grid  = np.asarray(list(beta_grid), float)
    loglik_grid = np.full((alpha_grid.size, beta_grid.size), -np.inf)
    for i, a in enumerate(alpha_grid):
        for j, b in enumerate(beta_grid):
            loglik_grid[i, j] = igdf_loglik(a, b, y, n0=n0, s0=s0, clip=clip)
    i_star, j_star = np.unravel_index(np.nanargmax(loglik_grid), loglik_grid.shape)
    return IGDFitResult(
        alpha_star=float(alpha_grid[i_star]),
        beta_star=float(beta_grid[j_star]),
        max_loglik=float(loglik_grid[i_star, j_star]),
        alpha_grid=alpha_grid,
        beta_grid=beta_grid,
        loglik_grid=loglik_grid if return_surface else None,
    )

def plot_igdf_loglik_surface(
    fit: IGDFitResult,
    *,
    levels: int | list[int] = 12,
    show_mle: bool = True,
    title: str | None = None
):
    """
    Heatmap + contour plot of the log-likelihood surface over (alpha, beta).
    Returns (fig, ax).
    """
    if fit.loglik_grid is None:
        raise ValueError("fit.loglik_grid is None. Call fit_igdf_hyperparams(..., return_surface=True).")

    A, B = np.meshgrid(fit.beta_grid, fit.alpha_grid)  # note: X-axis = beta, Y-axis = alpha
    Z = fit.loglik_grid

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[fit.beta_grid.min(), fit.beta_grid.max(),
                fit.alpha_grid.min(), fit.alpha_grid.max()]
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("log-likelihood", rotation=90)

    # Contours (same grid but explicit coordinates)
    CS = ax.contour(
        B, A, Z,
        levels=levels if isinstance(levels, int) else levels
    )
    ax.clabel(CS, inline=True, fontsize=8, fmt="%.1f")

    if show_mle:
        ax.scatter(fit.beta_star, fit.alpha_star, s=60, marker="x")
        ax.annotate(
            fr"MLE: $\alpha^*={fit.alpha_star:.3g}$, $\beta^*={fit.beta_star:.3g}$",
            (fit.beta_star, fit.alpha_star),
            xytext=(8, 8), textcoords="offset points"
        )

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title(title or "IGDF log-likelihood surface")
    ax.set_xlim(fit.beta_grid.min(), fit.beta_grid.max())
    ax.set_ylim(fit.alpha_grid.min(), fit.alpha_grid.max())
    ax.grid(False)
    fig.tight_layout()
    return fig, ax


def fit_and_plot_igdf(
    y: np.ndarray,
    *,
    alpha_grid=None,
    beta_grid=None,
    n0: float = 49.0,
    s0: float = 1/5000.0,
    clip: float = 1e-12,
    warmup: int = 0,
    levels: int = 12
):
    """
    Convenience: fit hyperparameters on a grid and immediately plot the surface.
    Returns (fit_result, (fig, ax)).
    """
    fit = fit_igdf_hyperparams(
        y,
        alpha_grid=alpha_grid,
        beta_grid=beta_grid,
        n0=n0,
        s0=s0,
        clip=clip,
        warmup=warmup,
        return_surface=True,
    )
    fig_ax = plot_igdf_loglik_surface(fit, levels=levels)
    return fit, fig_ax

