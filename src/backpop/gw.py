"""Utilities for LVK gravitational-wave posterior-sample likelihoods."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy import units as u
from astropy.constants import c
from astropy.cosmology import Planck15 as COSMOLOGY
from scipy.stats import gaussian_kde


@dataclass
class GWLikelihood:
    """Container for a 2D source-frame GW KDE likelihood."""

    kde: gaussian_kde
    q_bounds: tuple[float, float]
    mc_bounds: tuple[float, float]
    raw_samples: np.ndarray
    approximant: str

    def logpdf(self, mass_1: float, mass_2: float) -> float:
        """Evaluate ``log p(mc_source, q)`` for source-frame component masses."""
        m_hi = max(mass_1, mass_2)
        m_lo = min(mass_1, mass_2)
        if not np.isfinite(m_hi) or not np.isfinite(m_lo) or m_hi <= 0.0 or m_lo <= 0.0:
            return -np.inf

        q = m_lo / m_hi
        if q < self.q_bounds[0] or q > self.q_bounds[1]:
            return -np.inf

        mc = (m_hi * m_lo) ** (3.0 / 5.0) / (m_hi + m_lo) ** (1.0 / 5.0)
        density = self.kde(np.array([[mc], [q]]))[0]
        if density <= 0.0 or not np.isfinite(density):
            return -np.inf
        return float(np.log(density))


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def zofdL(luminosity_distance_mpc: np.ndarray) -> np.ndarray:
    """Convert luminosity distance in Mpc to redshift using Planck15 cosmology."""
    d_l = np.asarray(luminosity_distance_mpc, dtype=float)
    finite = np.isfinite(d_l)
    if not np.any(finite):
        return np.full_like(d_l, np.nan, dtype=float)

    max_d_l = float(np.nanmax(d_l[finite]))
    z_max = 0.5
    while COSMOLOGY.luminosity_distance(z_max).to_value(u.Mpc) < max_d_l:
        z_max *= 2.0

    z_grid = np.linspace(0.0, z_max, 20000)
    d_grid = COSMOLOGY.luminosity_distance(z_grid).to_value(u.Mpc)
    return np.interp(d_l, d_grid, z_grid)


def ddLdz(redshift: np.ndarray) -> np.ndarray:
    """Return dD_L/dz in Mpc for the same cosmology used by :func:`zofdL`."""
    z = np.asarray(redshift, dtype=float)
    transverse = COSMOLOGY.comoving_transverse_distance(z).to_value(u.Mpc)
    hubble_distance = (c / COSMOLOGY.H0).to_value(u.Mpc)
    return transverse + (1.0 + z) * hubble_distance / COSMOLOGY.efunc(z)


def _read_samples(samples_source: str | Any, fetch_kwargs: dict[str, Any] | None = None) -> Any:
    """Read a local PESummary file or fetch public samples by event name."""
    fetch_kwargs = dict(fetch_kwargs or {})
    if not isinstance(samples_source, str):
        return samples_source

    expanded = os.path.expanduser(samples_source)
    if os.path.exists(expanded):
        from pesummary.io import read

        return read(expanded, package="gw")

    from pesummary.gw.fetch import fetch_open_samples

    return fetch_open_samples(samples_source, **fetch_kwargs)


def _samples_array(samples: Any, *names: str) -> np.ndarray:
    for name in names:
        if name in samples:
            return np.asarray(samples[name], dtype=float)
    raise KeyError(f"None of the sample parameters {names} were present in the PESummary samples")


def _hdi(values: np.ndarray, prob: float = 0.999) -> tuple[float, float]:
    import arviz as az

    lo, hi = az.hdi(values, hdi_prob=prob)
    return float(lo), float(hi)


def load_gw_data(
    samples_source: str | Any,
    approximant: str = "C01:Mixed",
    use_pe_weights: bool = True,
    fetch_kwargs: dict[str, Any] | None = None,
) -> GWLikelihood:
    """Load LVK posterior samples and build a 2D source-frame ``(mc, q)`` KDE.

    ``samples_source`` may be either a path to a local PESummary-compatible HDF5
    file, an already-read PESummary object, or an event string such as
    ``"GW150914"``.  Event strings are downloaded with
    :func:`pesummary.gw.fetch.fetch_open_samples`; pass options such as
    ``catalog``, ``path``, ``unpack`` and ``outdir`` through ``fetch_kwargs``.
    """
    data = _read_samples(samples_source, fetch_kwargs=fetch_kwargs)

    available = list(data.samples_dict.keys())
    if approximant not in available:
        fallback = available[0]
        print(
            f"[load_gw_data] '{approximant}' not found; using '{fallback}'. "
            f"Available: {available}"
        )
        approximant = fallback

    samples = data.samples_dict[approximant]
    m1_det = _samples_array(samples, "mass_1", "mass1", "m1")
    m2_det = _samples_array(samples, "mass_2", "mass2", "m2")
    d_l = _samples_array(samples, "luminosity_distance", "luminosity_distance_Mpc", "dist")

    redshift = zofdL(d_l)
    m1_src = m1_det / (1.0 + redshift)
    m2_src = m2_det / (1.0 + redshift)

    m_hi = np.maximum(m1_src, m2_src)
    m_lo = np.minimum(m1_src, m2_src)
    mc_src = (m_hi * m_lo) ** (3.0 / 5.0) / (m_hi + m_lo) ** (1.0 / 5.0)
    q_src = m_lo / m_hi

    valid = np.isfinite(mc_src) & np.isfinite(q_src) & np.isfinite(redshift) & (mc_src > 0.0) & (q_src > 0.0)
    if not np.any(valid):
        raise ValueError("No finite LVK posterior samples remain after source-frame conversion")

    mc_src = mc_src[valid]
    q_src = q_src[valid]
    d_l = d_l[valid]
    redshift = redshift[valid]
    m_hi = m_hi[valid]

    if use_pe_weights:
        jacobian = d_l**2 * (1.0 + redshift) ** 2 * ddLdz(redshift) * m_hi**2 / mc_src
        weights = 1.0 / jacobian
        weights = weights / np.sum(weights)
    else:
        weights = np.ones(len(mc_src), dtype=float) / len(mc_src)

    q_bounds = _hdi(q_src)
    mc_bounds = _hdi(mc_src)
    raw_samples = np.column_stack([mc_src, q_src])
    kde = gaussian_kde(raw_samples.T, weights=weights)

    print(
        f"[load_gw_data] 2D KDE ({approximant}): "
        f"mc=[{mc_bounds[0]:.3f},{mc_bounds[1]:.3f}] M_sun, "
        f"q=[{q_bounds[0]:.4f},{q_bounds[1]:.4f}]"
    )
    return GWLikelihood(kde=kde, q_bounds=q_bounds, mc_bounds=mc_bounds, raw_samples=raw_samples,
                        approximant=approximant)


def load_gw_data_from_config(config: dict[str, Any]) -> GWLikelihood | None:
    """Build a GW likelihood from the optional ``[backpop.gw]`` config section."""
    gw_config = config.get("gw", {})
    if not _as_bool(gw_config.get("enabled"), default=False):
        return None

    source = gw_config.get("samples_path") or gw_config.get("event") or gw_config.get("source")
    if not source:
        raise ValueError("[backpop.gw] requires one of samples_path, event, or source")

    fetch_kwargs: dict[str, Any] = {}
    for key in ("catalog", "path", "outdir", "version"):
        value = gw_config.get(key)
        if value not in (None, "", "None"):
            fetch_kwargs[key] = value
    for key in ("unpack", "read_file", "delete_on_exit"):
        if key in gw_config:
            fetch_kwargs[key] = _as_bool(gw_config[key])

    if "delete_on_exit" not in fetch_kwargs:
        fetch_kwargs["delete_on_exit"] = False

    return load_gw_data(
        source,
        approximant=gw_config.get("approximant", "C01:Mixed"),
        use_pe_weights=_as_bool(gw_config.get("use_pe_weights"), default=True),
        fetch_kwargs=fetch_kwargs,
    )
