from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from analysis.interfaces import EpisodeAnalyzer, StepContext


# ---------------------------------------------------------------------------
# Grid visualisation helpers (used by both save() and redraw_plots())
# ---------------------------------------------------------------------------

def _plot_congestion_grid(env, mean_fail_rho: npt.NDArray, out_dir: Path) -> None:
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from visualization import GridPlottingArgs, visualize_grid

    n_line = env.n_line
    cmap = plt.colormaps["RdYlGn"]
    rho_vmax = float(mean_fail_rho.max()) if mean_fail_rho.max() > 0 else 1.0

    # High rho → red: invert the RdYlGn map (high value = left/red side)
    line_colors = []
    for i in range(n_line):
        rho_norm = max(0.0, float(mean_fail_rho[i])) / rho_vmax
        color = cmap(1.0 - rho_norm)
        line_colors.append(plt.matplotlib.colors.rgb2hex(color[:3]))

    fig, ax = plt.subplots(figsize=(14, 8))
    visualize_grid(
        GridPlottingArgs(
            env=env,
            node_size=700,
            font_size=12,
            node_color="white",
            line_colors=line_colors,
            line_widths=[3.0] * n_line,
            show_legend=False,
        ),
        ax=ax,
    )
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0.0, vmax=rho_vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Mean ρ at failure")
    ax.set_title("Mean congestion at failure (grid view)")
    fig.savefig(out_dir / "congestion_grid.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "congestion_grid.svg", bbox_inches="tight")
    plt.close(fig)


def _plot_connectivity_grid(env, mean_fail_status: npt.NDArray, out_dir: Path) -> None:
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from visualization import GridPlottingArgs, visualize_grid

    n_line = env.n_line
    cmap = plt.colormaps["RdYlGn"]

    # Use fixed [0, 1] range so colour directly encodes the absolute rate:
    # 0.0 = always disconnected (red), 1.0 = always connected (green).
    line_colors = []
    for i in range(n_line):
        cr = float(mean_fail_status[i])
        color = cmap(cr)
        line_colors.append(plt.matplotlib.colors.rgb2hex(color[:3]))

    fig, ax = plt.subplots(figsize=(14, 8))
    visualize_grid(
        GridPlottingArgs(
            env=env,
            node_size=700,
            font_size=12,
            node_color="white",
            line_colors=line_colors,
            line_widths=[3.0] * n_line,
            show_legend=False,
        ),
        ax=ax,
    )
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Mean connection rate at failure")
    ax.set_title("Line connectivity at failure (grid view)")
    fig.savefig(out_dir / "connectivity_grid.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "connectivity_grid.svg", bbox_inches="tight")
    plt.close(fig)


def redraw_plots(data_dir: Path, env_name: Optional[str] = None) -> None:
    """Regenerate all congestion plots from saved .npy files.

    :param data_dir: directory written by :meth:`CongestionProfileAnalyzer.save`
                     (i.e. ``out_dir/congestion``).
    :param env_name: Grid2Op environment name used to build the grid layout.
                     Falls back to the ``env_name.txt`` file in *data_dir* when omitted.
    """
    import grid2op

    d = data_dir
    if env_name is None:
        env_name_file = d / "env_name.txt"
        if not env_name_file.exists():
            raise FileNotFoundError(
                "env_name not given and env_name.txt not found in data_dir"
            )
        env_name = env_name_file.read_text().strip()

    mean_rho = np.load(d / "mean_rho_per_line.npy")
    n_lines = mean_rho.shape[0]
    line_idx = np.arange(n_lines)

    # --- bar charts (no env needed) ---
    fig, ax = plt.subplots(figsize=(max(8, n_lines // 2), 4))
    ax.bar(line_idx, mean_rho, color="steelblue")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="Thermal limit")
    ax.set_xlabel("Line index")
    ax.set_ylabel("Mean ρ")
    ax.set_title("Mean line loading over all test steps")
    ax.legend()
    plt.tight_layout()
    fig.savefig(d / "mean_rho_per_line.png", dpi=150, bbox_inches="tight")
    fig.savefig(d / "mean_rho_per_line.svg", bbox_inches="tight")
    plt.close(fig)

    rho_at_failure_path = d / "rho_at_failure.npy"
    status_at_failure_path = d / "line_status_at_failure.npy"
    if rho_at_failure_path.exists() and status_at_failure_path.exists():
        mean_fail_rho = np.load(rho_at_failure_path)
        mean_fail_status = np.load(status_at_failure_path)

        fig, axes = plt.subplots(1, 2, figsize=(max(14, n_lines), 4))
        axes[0].bar(line_idx, mean_fail_rho, color="tomato")
        axes[0].axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="Thermal limit")
        axes[0].set_xlabel("Line index")
        axes[0].set_ylabel("Mean ρ at failure")
        axes[0].legend()
        axes[1].bar(line_idx, mean_fail_status * 100, color="steelblue")
        axes[1].axhline(100.0, color="green", linestyle="--", linewidth=0.8)
        axes[1].set_xlabel("Line index")
        axes[1].set_ylabel("Mean connection rate (%)")
        axes[1].set_title("Line connectivity at failure")
        plt.tight_layout()
        fig.savefig(d / "rho_and_connectivity_at_failure.png", dpi=150, bbox_inches="tight")
        fig.savefig(d / "rho_and_connectivity_at_failure.svg", bbox_inches="tight")
        plt.close(fig)

        # --- grid visualisations ---
        env = grid2op.make(env_name)
        _plot_congestion_grid(env, mean_fail_rho, d)
        _plot_connectivity_grid(env, mean_fail_status, d)
    else:
        print("No rho_at_failure.npy / line_status_at_failure.npy found — grid plots skipped.")


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class CongestionProfileAnalyzer(EpisodeAnalyzer):
    def __init__(self, env_name: Optional[str] = None) -> None:
        self._env_name = env_name
        self._all_rho: List[npt.NDArray] = []
        self._all_status: List[npt.NDArray] = []
        self._failure_rhos: List[npt.NDArray] = []
        self._failure_statuses: List[npt.NDArray] = []
        self._pending_failure_rho: Optional[npt.NDArray] = None
        self._pending_failure_status: Optional[npt.NDArray] = None

    def on_episode_start(self, episode_id: str, max_steps: int) -> None:
        self._pending_failure_rho = None
        self._pending_failure_status = None

    def on_step(self, ctx: StepContext) -> None:
        self._all_rho.append(ctx.obs.rho.copy())
        self._all_status.append(ctx.obs.line_status.copy())
        if ctx.done:
            self._pending_failure_rho = ctx.obs.rho.copy()
            self._pending_failure_status = ctx.obs.line_status.copy()

    def on_episode_end(self, steps_survived: int, max_steps: int) -> None:
        if steps_survived < max_steps and self._pending_failure_rho is not None:
            self._failure_rhos.append(self._pending_failure_rho)
            self._failure_statuses.append(self._pending_failure_status)
        self._pending_failure_rho = None
        self._pending_failure_status = None

    def save(self, out_dir: Path) -> None:
        import grid2op

        d = out_dir / "congestion"
        d.mkdir(parents=True, exist_ok=True)

        if not self._all_rho:
            return

        if self._env_name:
            (d / "env_name.txt").write_text(self._env_name)

        rho_all = np.stack(self._all_rho, axis=0)
        mean_rho = rho_all.mean(axis=0)
        np.save(d / "mean_rho_per_line.npy", mean_rho)

        status_all = np.stack(self._all_status, axis=0)
        np.save(d / "mean_line_status.npy", status_all.mean(axis=0))

        n_lines = mean_rho.shape[0]
        line_idx = np.arange(n_lines)

        fig, ax = plt.subplots(figsize=(max(8, n_lines // 2), 4))
        ax.bar(line_idx, mean_rho, color="steelblue")
        ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="Thermal limit")
        ax.set_xlabel("Line index")
        ax.set_ylabel("Mean ρ")
        ax.set_title("Mean line loading over all test steps")
        ax.legend()
        plt.tight_layout()
        fig.savefig(d / "mean_rho_per_line.png", dpi=150, bbox_inches="tight")
        fig.savefig(d / "mean_rho_per_line.svg", bbox_inches="tight")
        plt.close(fig)

        if self._failure_rhos:
            fail_rho = np.stack(self._failure_rhos, axis=0)
            fail_status = np.stack(self._failure_statuses, axis=0)
            mean_fail_rho = fail_rho.mean(axis=0)
            mean_fail_status = fail_status.mean(axis=0)
            np.save(d / "rho_at_failure.npy", mean_fail_rho)
            np.save(d / "line_status_at_failure.npy", mean_fail_status)

            fig, axes = plt.subplots(1, 2, figsize=(max(14, n_lines), 4))
            axes[0].bar(line_idx, mean_fail_rho, color="tomato")
            axes[0].axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="Thermal limit")
            axes[0].set_xlabel("Line index")
            axes[0].set_ylabel("Mean ρ at failure")
            axes[0].set_title(f"Congestion at failure ({len(self._failure_rhos)} episodes)")
            axes[0].legend()
            axes[1].bar(line_idx, mean_fail_status * 100, color="steelblue")
            axes[1].axhline(100.0, color="green", linestyle="--", linewidth=0.8)
            axes[1].set_xlabel("Line index")
            axes[1].set_ylabel("Mean connection rate (%)")
            axes[1].set_title("Line connectivity at failure")
            plt.tight_layout()
            fig.savefig(d / "rho_and_connectivity_at_failure.png", dpi=150, bbox_inches="tight")
            fig.savefig(d / "rho_and_connectivity_at_failure.svg", bbox_inches="tight")
            plt.close(fig)

            if self._env_name:
                env = grid2op.make(self._env_name)
                _plot_congestion_grid(env, mean_fail_rho, d)
                _plot_connectivity_grid(env, mean_fail_status, d)
