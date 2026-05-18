"""Play 用：机体平面 APF 调试面板（左图右表，圆形遮罩局部地图）。"""

from __future__ import annotations

import os
from typing import Any, cast

import numpy as np
import torch
from isaaclab.utils.math import quat_apply_inverse


def unwrap_env_for_apf_collision_grid(env: Any) -> Any | None:
    """沿 gym / Isaac 包装链查找暴露 `apf_points_w` 的环境。"""
    stack = [env]
    seen: set[int] = set()
    while stack:
        e = stack.pop()
        eid = id(e)
        if eid in seen:
            continue
        seen.add(eid)
        if hasattr(e, "apf_points_w") and hasattr(e, "apf_slot_valid"):
            return e
        # 兼容：若只有内部缓存字段，也认为是可用核心环境，并补齐别名。
        if hasattr(e, "_obstacle_collision_points_w") and hasattr(e, "_obstacle_collision_slot_valid"):
            if not hasattr(e, "apf_points_w"):
                e.apf_points_w = e._obstacle_collision_points_w
            if not hasattr(e, "apf_slot_valid"):
                e.apf_slot_valid = e._obstacle_collision_slot_valid
            return e
        for name in ("unwrapped", "env"):
            nxt = getattr(e, name, None)
            if nxt is not None and nxt is not e:
                stack.append(nxt)
    return None


def _to_float(x: torch.Tensor | float) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


# 地图速度箭头可视化增益（仅影响显示长度，与真实 m/s 无关；箭头太短可改大，如 4.0）
ARROW_DISPLAY_GAIN = 1.0
# 地图速度箭头线宽（pt）：统一用 FancyArrowPatch，v_cmd = 2 × 其它
_ARROW_BASE_LINEWIDTH = 2.6
_CMD_ARROW_LINEWIDTH = _ARROW_BASE_LINEWIDTH * 2.0
# 右侧横向条：固定 3 行槽位，条高一致（ωz 占前两行，与 v_cmd / v_out 对齐）
_HBAR_SLOT_Y = (2.0, 1.0, 0.0)
_HBAR_BAR_HEIGHT = 0.42
_HBAR_YLIM = (-0.55, 2.55)


def _body_xy_to_plot_xy(bx: np.ndarray | float, by: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """机体 base：+x 前、+y 左 → 绘图：前向上、右向右（plot_x = −y, plot_y = x）。"""
    return -np.asarray(by, dtype=np.float64), np.asarray(bx, dtype=np.float64)


def _body_vel_to_plot_uv(vx: float, vy: float) -> tuple[float, float]:
    return float(-vy), float(vx)


def _grid_body_to_plot(z_body: np.ndarray) -> np.ndarray:
    """栅格 z[i_y, j_x]（imshow 的 x←body_x, y←body_y）旋转为「前方向上」。"""
    # 目标坐标约定与散点一致：plot_x = -body_y, plot_y = body_x
    # 因此需要将 body 栅格做转置后“按列反转”，而不是按行反转。
    # 若按行反转会得到 (plot_x=+body_y, plot_y=-body_x)，与散点镜像相反。
    return np.transpose(z_body)[:, ::-1]


def _make_wz_arc_arrow(
    wz: float,
    arc_radius: float,
    *,
    color: str,
    wz_ref: float = 1.0,
) -> Any | None:
    """绕机体 +z 的角速度圆弧箭头（plot 平面：+wz 逆时针，−wz 顺时针）。"""
    if abs(wz) < 1e-4:
        return None

    from matplotlib.patches import FancyArrowPatch

    max_span = 1.05
    min_span = 0.28
    span = float(np.clip(abs(wz) / max(wz_ref, 1e-6) * max_span, min_span, max_span))

    # 以前方 (+y plot) 为弧段中点；+wz 沿逆时针扫过 span
    mid = np.pi / 2
    if wz >= 0.0:
        a0, a1 = mid - span * 0.5, mid + span * 0.5
        bend = 0.58 * arc_radius
    else:
        a0, a1 = mid + span * 0.5, mid - span * 0.5
        bend = -0.58 * arc_radius

    p0 = (arc_radius * np.cos(a0), arc_radius * np.sin(a0))
    p1 = (arc_radius * np.cos(a1), arc_radius * np.sin(a1))
    return FancyArrowPatch(
        p0,
        p1,
        connectionstyle=f"arc3,rad={bend}",
        arrowstyle="-|>",
        color=color,
        linewidth=2.4,
        mutation_scale=14,
        alpha=0.93,
        zorder=10,
    )


class ApfPlayDebugPanel:
    """左：圆形遮罩局部地图 + 速度箭头；右：线/角速度横向条形图。"""

    _C_CMD = "#5b9cf5"
    _C_CMD_MAP = "#79c0ff"
    _C_DV = "#f47067"
    _C_OUT = "#ffe14a"
    _C_ACT = "#56d364"
    _C_COLL = "#79c0ff"
    _C_WZ_CMD = "#a371f7"
    _C_WZ_MEAS = "#56d364"

    _BG = "#0d1117"
    _PANEL = "#161b22"
    _BORDER = "#30363d"
    _RING = "#58a6ff"
    _GRID = "#21262d"
    _TEXT = "#e6edf3"
    _MUTED = "#8b949e"

    def __init__(
        self,
        *,
        radius_m: float = 1.0,
        cell_m: float = 0.05,
        env_id: int = 0,
        update_every: int = 1,
    ) -> None:
        self.radius_m = float(radius_m)
        self.cell_m = float(cell_m)
        self.env_id = int(env_id)
        self.update_every = max(1, int(update_every))
        self._step_count = 0

        if self.radius_m <= 0 or self.cell_m <= 0:
            raise ValueError("radius_m and cell_m must be positive.")
        self.n = int(round(2 * self.radius_m / self.cell_m))
        if self.n < 1:
            raise ValueError("grid has no cells; check radius_m / cell_m.")

        h = self.cell_m
        r = self.radius_m
        xc = torch.linspace(-r + 0.5 * h, r - 0.5 * h, self.n)
        yc = torch.linspace(-r + 0.5 * h, r - 0.5 * h, self.n)
        yg, xg = torch.meshgrid(yc, xc, indexing="ij")
        self._circ_mask_np = ((xg * xg + yg * yg) <= r * r + 1e-9).numpy()

        import matplotlib

        # 关键：优先保留当前“可交互”后端，避免在运行时强切后端导致空白 canvas。
        # 仅当当前后端是非交互（如 Agg）时，才尝试切换到 QtAgg/TkAgg。
        forced_backend = os.environ.get("APF_MPL_BACKEND", "").strip()
        if forced_backend:
            try:
                matplotlib.use(forced_backend, force=True)
            except Exception:
                pass
        else:
            current_backend = str(matplotlib.get_backend()).lower()
            if "agg" in current_backend:
                for candidate in ("QtAgg", "TkAgg"):
                    try:
                        matplotlib.use(candidate, force=True)
                        break
                    except Exception:
                        continue
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
        from matplotlib.axes import Axes
        from matplotlib.gridspec import GridSpec
        from matplotlib.patches import Circle

        self._plt = plt
        self._plt.ion()
        self._fig = plt.figure(figsize=(13.0, 7.5), facecolor=self._BG)
        gs = GridSpec(
            1, 2,
            figure=self._fig,
            width_ratios=[1.75, 1.0],
            left=0.05,
            right=0.98,
            top=0.91,
            bottom=0.07,
            wspace=0.12,
        )

        self._ax_map = cast(Axes, self._fig.add_subplot(gs[0, 0]))
        self._ax_right = cast(Axes, self._fig.add_subplot(gs[0, 1]))
        self._ax_right.axis("off")

        self._fig.suptitle(
            f"Proprioception Play  ·  env {self.env_id}",
            fontsize=13,
            fontweight="600",
            color=self._TEXT,
            y=0.97,
        )

        # —— 左侧地图（方形坐标系 + 圆形可视区遮罩）——
        r = self.radius_m
        pad = r * 0.06
        self._ax_map.set_facecolor(self._BG)
        self._ax_map.set_xlim(-r - pad, r + pad)
        self._ax_map.set_ylim(-r - pad, r + pad)
        self._ax_map.set_aspect("equal")
        self._ax_map.set_title("Local map (+x forward ↑)", fontsize=11, fontweight="600", color=self._TEXT, pad=12)
        self._ax_map.set_xlabel("lateral  −y → right (m)", color=self._MUTED, fontsize=9)
        self._ax_map.set_ylabel("forward  +x (m)", color=self._MUTED, fontsize=9)
        self._ax_map.tick_params(colors=self._MUTED, labelsize=8)
        for spine in self._ax_map.spines.values():
            spine.set_color(self._BORDER)

        self._clip_circle = Circle((0.0, 0.0), r, transform=self._ax_map.transData)
        self._ax_map.add_patch(
            Circle((0, 0), r, facecolor=self._PANEL, edgecolor="none", zorder=0)
        )
        self._ring = Circle(
            (0, 0), r, fill=False, edgecolor=self._RING, linewidth=2.4, linestyle="-", zorder=20,
        )
        self._ax_map.add_patch(self._ring)

        heat_cmap = mcolors.LinearSegmentedColormap.from_list(
            "apf_heat", ["#161b22", "#1f3a5f", "#388bfd", "#a371f7"], N=64
        )
        heat_cmap.set_bad(color=self._BG, alpha=1.0)
        z0 = np.ma.masked_where(~self._circ_mask_np, np.zeros((self.n, self.n)))
        self._im = self._ax_map.imshow(
            z0,
            origin="lower",
            extent=(-r, r, -r, r),
            cmap=heat_cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            zorder=2,
        )
        self._im.set_clip_path(self._clip_circle)

        (self._scatter_pts,) = self._ax_map.plot(
            [], [], "o", color=self._C_COLL, markersize=8,
            markeredgecolor="#ffffff", markeredgewidth=0.7, label="collision", zorder=5,
        )
        self._scatter_pts.set_clip_path(self._clip_circle)
        self._ax_map.plot(0, 0, "o", color=self._PANEL, markersize=12, zorder=6, clip_on=True)
        self._ax_map.plot(0, 0, "o", color="#ffffff", markersize=6, zorder=7, clip_on=True)
        self._ax_map.annotate(
            "",
            xy=(0.0, 0.22 * r),
            xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="-|>", color="#ffffff", lw=1.6, mutation_scale=12),
            zorder=9,
        )
        self._ax_map.text(0.0, 0.26 * r, "+x", ha="center", va="bottom", color="#ffffff", fontsize=8, zorder=9)

        # 速度箭头统一 FancyArrowPatch（pt 线宽），避免 quiver 与 patch 视觉不一致
        self._vel_arrow_patches: dict[str, Any] = {}

        from matplotlib.lines import Line2D

        leg_handles = [
            Line2D([], [], color=self._C_COLL, marker="o", linestyle="", markersize=6, label="collision"),
            Line2D([], [], color=self._C_CMD_MAP, linestyle="--", linewidth=_CMD_ARROW_LINEWIDTH, label="v_cmd"),
            Line2D([], [], color=self._C_DV, linewidth=_ARROW_BASE_LINEWIDTH, label="Δv"),
            Line2D([], [], color=self._C_OUT, linewidth=_ARROW_BASE_LINEWIDTH, label="v_out"),
            Line2D([], [], color=self._C_ACT, linewidth=_ARROW_BASE_LINEWIDTH, label="v_meas"),
        ]
        leg = self._ax_map.legend(
            handles=leg_handles,
            loc="lower left",
            fontsize=8,
            frameon=True,
            ncol=2,
            facecolor=self._PANEL,
            edgecolor=self._BORDER,
            labelcolor=self._TEXT,
        )
        leg.get_frame().set_alpha(0.94)

        # ωz 圆弧箭头（每帧在 step 里刷新）
        self._wz_arc_cmd: Any | None = None
        self._wz_arc_meas: Any | None = None
        self._wz_arc_label_cmd: Any | None = None
        self._wz_arc_label_meas: Any | None = None

        # —— 右侧：横向速度条 + 数值标注 ——
        self._hbar_axes: list[Any] = []
        self._hbar_artists: list[list[Any]] = []
        self._hbar_value_texts: list[list[Any]] = []
        panel_specs = [
            ("vx  (m/s)", (0, 1, 2), ["v_cmd", "v_out", "v_meas"], [self._C_CMD, self._C_OUT, self._C_ACT]),
            ("vy  (m/s)", (0, 1, 2), ["v_cmd", "v_out", "v_meas"], [self._C_CMD, self._C_OUT, self._C_ACT]),
            # ωz 仅 2 项：槽位 0、1 → y=2、1，与 v_cmd、v_out 同行，条宽与上两图一致
            ("ωz  (rad/s)", (0, 1), ["command", "measured"], [self._C_CMD, self._C_ACT]),
        ]
        tops = [0.72, 0.39, 0.06]
        panel_h = 0.24
        for (title, slot_idx, labels, colors), top in zip(panel_specs, tops, strict=True):
            ax = self._ax_right.inset_axes([0.0, top, 1.0, panel_h])
            self._style_hbar_axis(ax, title)
            y_pos = [_HBAR_SLOT_Y[i] for i in slot_idx]
            bars = ax.barh(
                y_pos, [0.0] * len(labels), height=_HBAR_BAR_HEIGHT, color=colors, edgecolor="none", zorder=3,
            )
            ax.set_ylim(_HBAR_YLIM)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, color=self._MUTED, fontsize=9)
            val_texts = []
            for bar in bars:
                t = ax.text(
                    0.0,
                    bar.get_y() + bar.get_height() * 0.5,
                    "",
                    va="center",
                    ha="left",
                    fontsize=8,
                    color=self._TEXT,
                    zorder=4,
                )
                val_texts.append(t)
            self._hbar_axes.append(ax)
            self._hbar_artists.append(list(bars))
            self._hbar_value_texts.append(val_texts)

        self._info_text = self._ax_right.text(
            0.0, 0.94, "",
            transform=self._ax_right.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            color=self._TEXT,
            family="monospace",
            linespacing=1.45,
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor=self._PANEL,
                edgecolor=self._BORDER,
                linewidth=1.0,
            ),
        )

        try:
            mgr = self._fig.canvas.manager
            if mgr is not None and hasattr(mgr, "set_window_title"):
                mgr.set_window_title("Proprioception · APF Debug")
        except Exception:
            pass

        self._plt.show(block=False)
        # 强制首帧 draw，避免部分桌面环境只创建窗口壳体但不渲染 canvas（灰屏）。
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        self._plt.pause(0.05)
        # 关键：很多桌面环境会把新窗口放在后台，强制前置一次提升可见性。
        self._raise_window_to_front()

    def _raise_window_to_front(self) -> None:
        """尽量把 matplotlib 窗口提到前台（兼容 Tk/Qt，失败静默）。"""
        try:
            mgr = self._fig.canvas.manager
            if mgr is None:
                return
            win = getattr(mgr, "window", None)
            if win is None:
                return

            # Tk 窗口路径
            for fn in ("deiconify", "lift", "focus_force"):
                if hasattr(win, fn):
                    try:
                        getattr(win, fn)()
                    except Exception:
                        pass
            if hasattr(win, "attributes"):
                try:
                    win.attributes("-topmost", True)
                    if hasattr(win, "after"):
                        win.after(120, lambda: win.attributes("-topmost", False))
                except Exception:
                    pass

            # Qt 窗口路径
            for fn in ("showNormal", "raise_", "activateWindow"):
                if hasattr(win, fn):
                    try:
                        getattr(win, fn)()
                    except Exception:
                        pass
        except Exception:
            pass

    def _style_hbar_axis(self, ax: Any, title: str) -> None:
        ax.set_facecolor(self._PANEL)
        ax.set_title(title, fontsize=10, fontweight="600", color=self._TEXT, loc="left", pad=6)
        ax.axvline(0.0, color=self._BORDER, linewidth=1.0, zorder=1)
        ax.tick_params(axis="x", colors=self._MUTED, labelsize=8)
        ax.grid(axis="x", color=self._GRID, linestyle="-", linewidth=0.55, alpha=0.9)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(self._BORDER)

    def close(self) -> None:
        if getattr(self, "_fig", None) is not None:
            self._plt.close(self._fig)
            self._fig = None

    def _build_collision_grid(self, core: Any, robot: Any, e: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        valid = core.apf_slot_valid[e]
        if not torch.any(valid):
            return torch.zeros((self.n, self.n), dtype=torch.float32), torch.zeros((0, 2), device=device)

        pts_w = core.apf_points_w[e, valid, :]
        root_p = robot.data.root_pos_w[e, :3]
        root_q = robot.data.root_quat_w[e, :]
        off_w = pts_w - root_p.unsqueeze(0)
        q = root_q.unsqueeze(0).expand(off_w.shape[0], -1)
        off_b = quat_apply_inverse(q, off_w)
        xy = off_b[:, :2]

        r, h, n = self.radius_m, self.cell_m, self.n
        j = torch.floor((xy[:, 0] + r) / h).long()
        i = torch.floor((xy[:, 1] + r) / h).long()
        pr = torch.linalg.norm(xy, dim=-1) <= r + 1e-5
        inside = pr & (i >= 0) & (i < n) & (j >= 0) & (j < n)
        z = torch.zeros((n, n), dtype=torch.float32, device=device)
        if torch.any(inside):
            z[i[inside], j[inside]] = 1.0
        circ = torch.from_numpy(self._circ_mask_np).to(device=device)
        z = z * circ.to(dtype=z.dtype)
        return z, xy[pr]

    def _update_map_velocity_arrow(
        self,
        key: str,
        vx: float,
        vy: float,
        *,
        color: str,
        linestyle: str = "-",
        linewidth: float = _ARROW_BASE_LINEWIDTH,
        zorder: int = 8,
        mutation_scale: float = 14,
    ) -> None:
        """机体平面速度箭头（FancyArrowPatch，linewidth 单位 pt）。"""
        old = self._vel_arrow_patches.pop(key, None)
        if old is not None:
            try:
                old.remove()
            except (ValueError, AttributeError):
                pass

        if (vx * vx + vy * vy) ** 0.5 < 1e-6:
            return

        u, v = _body_vel_to_plot_uv(vx, vy)
        plot_speed = (u * u + v * v) ** 0.5
        if plot_speed < 1e-6:
            return
        length = plot_speed * ARROW_DISPLAY_GAIN
        ex = u / plot_speed * length
        ey = v / plot_speed * length

        from matplotlib.patches import FancyArrowPatch

        patch = FancyArrowPatch(
            (0.0, 0.0),
            (ex, ey),
            arrowstyle="-|>",
            linestyle=linestyle,
            color=color,
            linewidth=linewidth,
            alpha=0.95,
            mutation_scale=mutation_scale,
            zorder=zorder,
        )
        patch.set_clip_path(self._clip_circle)
        self._ax_map.add_patch(patch)
        self._vel_arrow_patches[key] = patch

    def _update_hbar_panel(
        self,
        ax: Any,
        bars: list[Any],
        values: list[float],
        value_texts: list[Any],
        *,
        fmt: str = "{:+.3f}",
    ) -> None:
        lim = max(0.35, min(2.0, max(abs(v) for v in values) * 1.35))
        pad = lim * 0.04
        for bar, val, txt in zip(bars, values, value_texts, strict=True):
            bar.set_width(val)
            y = bar.get_y() + bar.get_height() * 0.5
            label = fmt.format(val)
            if abs(val) < 1e-5:
                txt.set_position((pad, y))
                txt.set_ha("left")
            elif val >= 0.0:
                txt.set_position((val + pad, y))
                txt.set_ha("left")
            else:
                txt.set_position((val - pad, y))
                txt.set_ha("right")
            txt.set_text(label)
        ax.set_xlim(-lim, lim)

    def _clear_wz_arc(self, attr: str) -> None:
        obj = getattr(self, attr, None)
        if obj is not None:
            try:
                obj.remove()
            except (ValueError, AttributeError):
                pass
            setattr(self, attr, None)

    def _update_wz_rotation_arcs(self, wz_cmd: float, wz_meas: float) -> None:
        """在地图上绘制 ωz 指令/实测圆弧箭头（不同半径与颜色）。"""
        for attr in (
            "_wz_arc_cmd",
            "_wz_arc_meas",
            "_wz_arc_label_cmd",
            "_wz_arc_label_meas",
        ):
            self._clear_wz_arc(attr)

        r = self.radius_m
        specs = (
            ("_wz_arc_cmd", "_wz_arc_label_cmd", wz_cmd, 0.42 * r, self._C_WZ_CMD, "ωz cmd"),
            ("_wz_arc_meas", "_wz_arc_label_meas", wz_meas, 0.54 * r, self._C_WZ_MEAS, "ωz meas"),
        )
        for arc_attr, label_attr, wz, arc_r, color, tag in specs:
            patch = _make_wz_arc_arrow(wz, arc_r, color=color)
            if patch is None:
                continue
            patch.set_clip_path(self._clip_circle)
            self._ax_map.add_patch(patch)
            setattr(self, arc_attr, patch)

            # 标签放在弧段中点外侧
            span = float(np.clip(abs(wz) * 0.85, 0.28, 1.05))
            mid = np.pi / 2
            ang = mid + (span * 0.55 if wz >= 0.0 else -span * 0.55)
            lx = (arc_r + 0.08 * r) * np.cos(ang)
            ly = (arc_r + 0.08 * r) * np.sin(ang)
            txt = self._ax_map.text(
                lx,
                ly,
                f"{tag}\n{wz:+.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=color,
                zorder=11,
            )
            txt.set_clip_path(self._clip_circle)
            setattr(self, label_attr, txt)

    def step(self, gym_env: Any) -> None:
        if getattr(self, "_fig", None) is None:
            return
        self._step_count += 1
        if (self._step_count - 1) % self.update_every != 0:
            return

        core = unwrap_env_for_apf_collision_grid(gym_env)
        if core is None:
            return
        # 兼容：若 reset 事件尚未写入别名，这里按内部字段兜底补齐。
        if (not hasattr(core, "apf_points_w")) and hasattr(core, "_obstacle_collision_points_w"):
            core.apf_points_w = core._obstacle_collision_points_w
        if (not hasattr(core, "apf_slot_valid")) and hasattr(core, "_obstacle_collision_slot_valid"):
            core.apf_slot_valid = core._obstacle_collision_slot_valid
        # 关键兼容：不同 IsaacLab 封装下，机器人可能挂在 core.robot 或 core.scene["robot"]。
        # 若这里只取 core.robot，会导致提前 return，从而右侧数值一直不刷新。
        robot = getattr(core, "robot", None)
        if robot is None and hasattr(core, "scene"):
            try:
                robot = core.scene["robot"]
            except Exception:
                robot = None
        if robot is None:
            return

        e = self.env_id
        if e < 0 or e >= core.num_envs:
            return

        device = core.device
        z, pts_xy = self._build_collision_grid(core, robot, e, device)
        z_np = z.detach().cpu().numpy()
        z_plot = _grid_body_to_plot(z_np)
        mask_plot = _grid_body_to_plot(self._circ_mask_np.astype(np.float32)) > 0.5
        self._im.set_data(np.ma.masked_where(~mask_plot, z_plot))

        if pts_xy.numel() == 0:
            self._scatter_pts.set_data([], [])
        else:
            xy_cpu = pts_xy.detach().cpu().numpy()
            px, py = _body_xy_to_plot_xy(xy_cpu[:, 0], xy_cpu[:, 1])
            self._scatter_pts.set_data(px, py)

        # 命令数值兼容读取：
        # - 首选 APF 缓存（有 APF 语义）
        # - 缺失时回退到 command_manager 当前命令，避免右侧数值面板空白
        if hasattr(core, "apf_v_cmd_b"):
            v_cmd = core.apf_v_cmd_b[e]
        elif hasattr(core, "_apf_v_cmd_b"):
            v_cmd = core._apf_v_cmd_b[e]
        else:
            v_cmd = core.command_manager.get_command("base_velocity")[e]

        if hasattr(core, "apf_v_out_b"):
            v_out = core.apf_v_out_b[e]
        elif hasattr(core, "_apf_v_out_b"):
            v_out = core._apf_v_out_b[e]
        else:
            v_out = v_cmd

        if hasattr(core, "apf_delta_v_b"):
            dv = core.apf_delta_v_b[e]
        elif hasattr(core, "_apf_delta_v_b"):
            dv = core._apf_delta_v_b[e]
        else:
            dv = torch.zeros(2, device=device, dtype=torch.float32)
        v_act = robot.data.root_lin_vel_b[e]
        wz_act = robot.data.root_ang_vel_b[e, 2]

        # v_cmd 先画（底层）；v_out 最后画，黄色压在蓝色虚线上
        self._update_map_velocity_arrow(
            "cmd",
            _to_float(v_cmd[0]),
            _to_float(v_cmd[1]),
            color=self._C_CMD_MAP,
            linestyle="--",
            linewidth=_CMD_ARROW_LINEWIDTH,
            zorder=8,
            mutation_scale=16,
        )
        self._update_map_velocity_arrow("dv", _to_float(dv[0]), _to_float(dv[1]), color=self._C_DV, zorder=9)
        self._update_map_velocity_arrow("act", _to_float(v_act[0]), _to_float(v_act[1]), color=self._C_ACT, zorder=9)
        self._update_map_velocity_arrow(
            "out",
            _to_float(v_out[0]),
            _to_float(v_out[1]),
            color=self._C_OUT,
            linewidth=_ARROW_BASE_LINEWIDTH,
            zorder=11,
        )

        wz_cmd_f = _to_float(v_cmd[2])
        wz_meas_f = _to_float(wz_act)
        self._update_wz_rotation_arcs(wz_cmd_f, wz_meas_f)

        vx_vals = [_to_float(v_cmd[0]), _to_float(v_out[0]), _to_float(v_act[0])]
        vy_vals = [_to_float(v_cmd[1]), _to_float(v_out[1]), _to_float(v_act[1])]
        wz_vals = [wz_cmd_f, wz_meas_f]

        self._update_hbar_panel(
            self._hbar_axes[0], self._hbar_artists[0], vx_vals, self._hbar_value_texts[0], fmt="{:+.3f}"
        )
        self._update_hbar_panel(
            self._hbar_axes[1], self._hbar_artists[1], vy_vals, self._hbar_value_texts[1], fmt="{:+.3f}"
        )
        self._update_hbar_panel(
            self._hbar_axes[2], self._hbar_artists[2], wz_vals, self._hbar_value_texts[2], fmt="{:+.3f}"
        )

        n_col = int(core.apf_slot_valid[e].sum().item())
        dv_norm = (_to_float(dv[0]) ** 2 + _to_float(dv[1]) ** 2) ** 0.5
        self._info_text.set_text(
            f"collisions {n_col}   |Δv| {dv_norm:.3f} m/s\n"
            f"vx  cmd {vx_vals[0]:+.3f}  out {vx_vals[1]:+.3f}  meas {vx_vals[2]:+.3f} m/s\n"
            f"vy  cmd {vy_vals[0]:+.3f}  out {vy_vals[1]:+.3f}  meas {vy_vals[2]:+.3f} m/s\n"
            f"ωz  cmd {wz_vals[0]:+.3f}  meas {wz_vals[1]:+.3f} rad/s"
        )

        fig = self._fig
        assert fig is not None
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        self._plt.pause(0.001)


ApfCollisionGridMapWindow = ApfPlayDebugPanel
