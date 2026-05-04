import os
import base64
from pathlib import Path

import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.io import export_png
from bokeh.models import ColumnDataSource, HoverTool

from spatialtissuepy.synthetic.physicell import PhysiCellSimulation


SCRIPT_DIR = Path(__file__).resolve().parent
# Repo root: <repo>/tme-trajectory-landscape/figures/figure_2/<this file>
REPO_ROOT = SCRIPT_DIR.parents[2]
OUTPUT_DIR = SCRIPT_DIR / "output"

ICON_DIR = REPO_ROOT / "docs" / "biorender_icons" / "icons" / "png"
ICON_MAP = {
    "M0_macrophage": str(ICON_DIR / "m0.png"),
    "M1_macrophage": str(ICON_DIR / "m1.png"),
    "M2_macrophage": str(ICON_DIR / "m2.png"),
    "effector_T_cell": str(ICON_DIR / "cd8_effector.png"),
    "exhausted_T_cell": str(ICON_DIR / "cd8_exhausted.png"),
    "malignant_epithelial_cell": str(ICON_DIR / "malignant_epithelial_cell.png"),
}

SIM_DIR = REPO_ROOT / "data" / "abm" / "raw"

# Transparent red-circle SVG fallback so the plot still renders if an icon file is missing.
PLACEHOLDER_DATA_URI = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMCIgaGVpZ2h0PSIxMCI+"
    "PGNpcmNsZSBjeD0iNSIgY3k9IjUiIHI9IjUiIGZpbGw9InJlZCIvPjwvc3ZnPg=="
)

# export_png needs a headless browser+driver. Flip off after first failure so we still
# produce HTML for every snapshot.
_png_export_enabled = True


def image_to_base64(path: str) -> str:
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return PLACEHOLDER_DATA_URI
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    mime = "image/svg+xml" if path.lower().endswith(".svg") else "image/png"
    return f"data:{mime};base64,{encoded}"


def render_snapshot(df: pd.DataFrame, title: str, output_html: str) -> None:
    global _png_export_enabled

    df = df.copy()
    df["file_path"] = df["cell_type"].map(ICON_MAP)
    df["icon_image"] = df["file_path"].apply(lambda p: image_to_base64(str(p)))
    source = ColumnDataSource(df)

    p = figure(
        width=1000,
        height=1000,
        title=title,
        match_aspect=True,
        tools="pan,wheel_zoom,reset,save",
    )
    p.background_fill_color = "white"
    p.background_fill_alpha = 1.0
    p.border_fill_color = "white"
    p.border_fill_alpha = 1.0
    p.outline_line_color = None
    p.xgrid.visible = False
    p.ygrid.visible = False

    p.image_url(
        url="icon_image",
        x="x",
        y="y",
        w=40,
        h=40,
        w_units="screen",
        h_units="screen",
        anchor="center",
        source=source,
    )
    p.add_tools(HoverTool(tooltips=[
        ("Cell Type", "@cell_type"),
        ("Position", "(@x, @y)"),
    ]))

    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    output_file(output_html)
    save(p)

    png_path = output_html.replace(".html", ".png")
    if _png_export_enabled:
        try:
            export_png(p, filename=png_path)
            print(f"Saved {output_html} and {png_path}")
            return
        except Exception as e:
            _png_export_enabled = False
            print(
                f"PNG export failed ({type(e).__name__}: {e}).\n"
                "Disabling PNG export for remaining snapshots. To enable PNG output, install\n"
                "a headless browser + driver, e.g.:\n"
                "  conda install -c conda-forge firefox geckodriver"
            )
    print(f"Saved {output_html} (HTML only)")


def render_state_snapshot(sim_path: str, timestep: int, title: str, output_html: str) -> None:
    sim = PhysiCellSimulation.from_output_folder(sim_path)
    df = sim.get_timestep(timestep).to_dataframe()
    render_snapshot(df, title, output_html)


def render_demo(output_html: str) -> None:
    df = pd.DataFrame({
        "x": [100, 250, 400, 550, 700, 850],
        "y": [100, 250, 400, 550, 700, 850],
        "cell_type": [
            "M0_macrophage",
            "M1_macrophage",
            "M2_macrophage",
            "effector_T_cell",
            "exhausted_T_cell",
            "malignant_epithelial_cell",
        ],
    })
    render_snapshot(df, "Cell Plot (Embedded Images)", output_html)


STATE_SNAPSHOTS = [
    (str(SIM_DIR / "sim_014"), 690, "Example Snapshot of State 1", str(OUTPUT_DIR / "figure_2_panel_f_state1_sim014_t690.html")),
    (str(SIM_DIR / "sim_014"), 40,  "Example Snapshot of State 2", str(OUTPUT_DIR / "figure_2_panel_f_state2_sim014_t40.html")),
    (str(SIM_DIR / "sim_000"), 50,  "Example Snapshot of State 3", str(OUTPUT_DIR / "figure_2_panel_f_state3_sim000_t50.html")),
    (str(SIM_DIR / "sim_000"), 690, "Example Snapshot of State 6", str(OUTPUT_DIR / "figure_2_panel_f_state6_sim000_t690.html")),
    (str(SIM_DIR / "sim_003"), 450, "Example Snapshot of State 4", str(OUTPUT_DIR / "figure_2_panel_f_state4_sim003_t450.html")),
    (str(SIM_DIR / "sim_003"), 250, "Example Snapshot of State 4", str(OUTPUT_DIR / "figure_2_panel_f_state4_sim003_t250.html")),
]


def main() -> None:
    try:
        render_demo(str(OUTPUT_DIR / "figure_2_panel_f_demo.html"))
    except Exception as e:
        print(f"Failed demo snapshot: {type(e).__name__}: {e}")
    for sim_path, timestep, title, output_html in STATE_SNAPSHOTS:
        try:
            render_state_snapshot(sim_path, timestep, title, output_html)
        except Exception as e:
            # One failure (missing sim folder, bad timestep, ...) shouldn't block the rest.
            print(f"Failed {output_html}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
