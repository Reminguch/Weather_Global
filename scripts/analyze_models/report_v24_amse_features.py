"""Assemble the four-model AMSE feature report from validated saved metrics.

Run after amse_feature_report_data.py and the two plot_v24_amse_feature_*.py
generators, using scripts/graphcast_env.sh. No model inference is performed.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import textwrap

os.environ.setdefault("MPLCONFIGDIR", "/tmp/amse_feature_report_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
NAME = "v24_amse_feature_report_20261004"
DATA = ROOT / "plots/analyze_models/data/resolution_eval" / NAME
IMAGES = ROOT / "plots/analyze_models/images/resolution_eval" / NAME
REPORT = ROOT / "docs/reports/v24_amse_features_2026-10-04.pdf"
GALLERY = REPORT.with_suffix(".html")
MODEL_ORDER = ["original_gc", "mse", "amse_lambda0p1", "full_amse"]
LABELS = ["Original GC", "MSE", "AMSE λ=0.1", "Full AMSE"]
COLORS = ["#606871", "#0072B2", "#009E73", "#D55E00"]
INK = "#162C3E"
MUTED = "#526575"
LEADS = [24, 72, 168, 240]
FEATURE_NAMES = {
    "T2m": "2 m temperature", "U10": "10 m zonal wind", "V10": "10 m meridional wind",
    "MSLP": "Mean sea-level pressure", "TP6": "6 h precipitation", "T850": "850 hPa temperature",
    "Q700": "700 hPa specific humidity", "U850": "850 hPa zonal wind", "V850": "850 hPa meridional wind",
    "WS10": "10 m wind speed", "WS850": "850 hPa wind speed",
}


def relative(path):
    return os.path.relpath(path, REPORT.parent)


def paragraph(fig, text, y, *, x=.065, width=116, size=11, color=INK, line=.031):
    for part in text.split("\n"):
        lines = textwrap.wrap(part, width=width, break_long_words=False, break_on_hyphens=False) or [""]
        for value in lines:
            fig.text(x, y, value, fontsize=size, color=color, va="top")
            y -= line
        y -= .010
    return y


def page(title, subtitle, number):
    fig = plt.figure(figsize=(13, 9), facecolor="white")
    fig.text(.065, .94, title, fontsize=22, weight="bold", color=INK, va="top")
    fig.text(.065, .885, subtitle, fontsize=11, color=MUTED, va="top")
    fig.text(.065, .027, "AMSE feature comparison · 4 October 2026 · Saved SWA evaluations", fontsize=9, color=MUTED)
    fig.text(.955, .027, str(number), fontsize=9, color=MUTED, ha="right")
    return fig


def table(fig, rows, labels, bbox, *, widths=None, size=11):
    ax = fig.add_axes(bbox)
    ax.axis("off")
    result = ax.table(cellText=rows, colLabels=labels, colWidths=widths,
                      loc="upper left", cellLoc="left", bbox=[0, 0, 1, 1])
    result.auto_set_font_size(False)
    result.set_fontsize(size)
    for (r, c), cell in result.get_celld().items():
        cell.set_edgecolor("#DCE3E9")
        cell.set_linewidth(.5)
        cell.PAD = .12
        if r == 0:
            cell.set_facecolor("#EAF0F5")
            cell.get_text().set_weight("bold")
        elif r % 2:
            cell.set_facecolor("#F7F9FB")
        cell.get_text().set_color(INK)
    return result


def read_spectra():
    with np.load(DATA / "spectra.npz", allow_pickle=False) as z:
        values = {key: z[key] for key in z.files}
    assert values["model_ids"].tolist() == MODEL_ORDER
    index = [values["lead_hours"].tolist().index(h) for h in LEADS]
    truth = values["truth_power"][index, :, 1:]
    power = values["power"][:, index, :, 1:]
    cross = values["cross_power"][:, index, :, 1:]
    error = 100 * np.abs(power - truth[None]) / truth[None]
    coherence = cross / np.sqrt(power * truth[None])
    assert np.isfinite(error).all() and np.isfinite(coherence).all()
    return values, error, coherence


def front_pages(document, spectra, error, coherence):
    path = DATA / "report_front_matter.pdf"
    features = spectra["variables"].tolist()
    averages = error.mean(axis=(1, 2, 3))
    coherent = coherence.mean(axis=(1, 2, 3))
    with PdfPages(path) as pdf:
        fig = page("AMSE forecast quality across weather features",
                   "Original GC · MSE · AMSE λ=0.1 · Full AMSE", 1)
        y = paragraph(fig,
            "This report compares spectral power, forecast–truth coherence and deterministic forecast error "
            "using completed evaluations of the same 1° residual-Mamba architecture. The three trained "
            "models use stochastic weight averaging over updates 500, 1000, 1500 and 2000.", .815)
        y = paragraph(fig,
            "Full AMSE restores substantially more missing spectral power, but it also loses pattern "
            "alignment and much of the MSE-trained model's deterministic accuracy advantage. AMSE λ=0.1 "
            "stays much closer to MSE: its spectral changes are smaller and its forecast-loss performance is similar.", y-.015)
        rows = [[LABELS[i], f"{averages[i]:.2f}%", f"{coherent[i]:.4f}"] for i in range(4)]
        table(fig, rows, ["Model", "Mean relative power error", "Mean spectral coherence"],
              [.065, .405, .87, .215], widths=[.35, .34, .31])
        paragraph(fig,
            "These spectral summaries give equal weight to 11 fields, zonal wavenumbers 1–180 and "
            "days 1, 3, 7 and 10. Raw spectra are pooled across eight 2023 starts before ratios are formed. "
            "The summaries include wind speed alongside its components and are secondary diagnostics, "
            "not the original GraphCast training loss.", .36, size=10.5, width=122)
        paragraph(fig,
            "The report contains a methods guide, spectral summary tables, exact GraphCast loss, "
            "global RMSE by variable, and three spectral plots for each field. The downloadable tables "
            "include every saved lead and all 83 global-RMSE channels.", .22, size=10.5, width=122)
        paragraph(fig,
            "Evidence scope: 32 historical 2022 development starts and eight exploratory 2023 starts, "
            "reported separately. No uncertainty intervals or new forecasts are included.", .105,
            size=10, width=125, color=MUTED)
        pdf.savefig(fig); fig.savefig(IMAGES / "report_overview.png", dpi=160); plt.close(fig)

        fig = page("How to read the comparison", "Definitions and fixed model references", 2)
        y = .815
        points = [
            "Original GC is the baseline evaluation paired with the MSE SWA checkpoint from the original "
            "full-AMSE campaign. MSE is that same campaign's trained control. AMSE λ=0.1 comes from the "
            "later λ=0.1 campaign; Full AMSE comes from the original campaign. These four references are "
            "fixed throughout. Repeated GC evaluations are not averaged or silently substituted.",
            "The separate λ=0.1 campaign has another matched MSE control. It is excluded from the four-model "
            "figures as requested, but retained in the source audit. Small cross-campaign differences may "
            "reflect run-to-run variability; the repeatability appendix records those differences.",
            "Spectral diagnostics use zonal Fourier transforms of scalar fields within 30–60° latitude in "
            "both hemispheres. The zonal mean is removed at each latitude. Latitude weights are cosine "
            "weights, and raw powers and cospectra are averaged over starts before forming ratios. Curves "
            "are unsmoothed. Larger zonal wavenumber means finer scale, but wavelength depends on latitude.",
            "Power error (%) = 100 × |forecast power − truth power| / truth power; lower is better. "
            "Coherence = real cospectrum / sqrt(forecast power × truth power); higher is better. ERA5 is "
            "a verification reference, not an additional forecast model.",
            "Global RMSE is sqrt(mean squared physical error over starts and the global cosine-weighted "
            "grid). Upper-air variable-group RMSE additionally pools squared errors equally across 13 "
            "levels. Specific-level RMSE uses the stated pressure level. Neither metric is identical "
            "to pressure-weighted, difference-standard-deviation-normalized original GraphCast loss.",
            "Exact GraphCast loss reduction is 100 × (1 − aggregate model loss / aggregate reference "
            "loss). Aggregation precedes the ratio. Displayed-reference and own-paired-reference "
            "comparisons are labeled separately. Wind-speed global RMSE was not saved; component "
            "RMSEs cannot reconstruct it. WS10 and WS850 have spectral diagnostics only.",
        ]
        for i, text in enumerate(points):
            y = paragraph(fig, text, y, size=10.4, width=126, line=.027)
            y -= .010
        pdf.savefig(fig); plt.close(fig)

        for metric, values, description, number in [
            ("Relative power error by field", error, "Mean over days 1, 3, 7, 10 and k = 1–180 · percent · lower is better", 3),
            ("Spectral coherence by field", coherence, "Mean over days 1, 3, 7, 10 and k = 1–180 · higher is better", 4),
        ]:
            fig = page(metric, description, number)
            means = values.mean(axis=(1, 3))
            rows = [[f"{code}  {FEATURE_NAMES[code]}"] +
                    [f"{means[m, vi]:.2f}" if number == 3 else f"{means[m, vi]:.4f}" for m in range(4)]
                    for vi, code in enumerate(features)]
            table(fig, rows, ["Field"] + LABELS, [.065, .235, .87, .59],
                  widths=[.40, .14, .14, .16, .16], size=10.5)
            paragraph(fig,
                "Each entry gives equal weight to the retained spectral bins, rather than weighting by "
                "their physical energy. Do not interpret these means as global RMSE or GraphCast loss. "
                "The individual field plots show scale and lead dependence that these averages conceal.",
                .18, size=10.5, width=122)
            pdf.savefig(fig); plt.close(fig)

        display_variables = [
            ("2m_temperature", "2 m temperature", "K", 1),
            ("10m_u_component_of_wind", "10 m zonal wind", "m/s", 1),
            ("10m_v_component_of_wind", "10 m meridional wind", "m/s", 1),
            ("mean_sea_level_pressure", "Sea-level pressure", "hPa", .01),
            ("total_precipitation_6hr", "6 h precipitation", "mm", 1000),
            ("temperature", "Temperature, all levels", "K", 1),
            ("specific_humidity", "Specific humidity, all levels", "g/kg", 1000),
            ("u_component_of_wind", "Zonal wind, all levels", "m/s", 1),
            ("v_component_of_wind", "Meridional wind, all levels", "m/s", 1),
            ("geopotential", "Geopotential, all levels", "m²/s²", 1),
            ("vertical_velocity", "Vertical velocity, all levels", "Pa/s", 1),
        ]
        for number, dataset in enumerate(("exact32", "spectral8"), 5):
            block = document["evaluations"][dataset]
            fig = page(f"Rollout RMSE by variable in {block['year']}",
                       f"{block['n_samples']} starts · all 40 leads pooled in squared-error space · lower is better", number)
            rows, changes = [], []
            for key, label, units, factor in display_variables:
                rmses = [float(np.sqrt(np.mean(np.square(block["models"][mid]["per_variable_per_step"][key]["rmse"]))))*factor
                         for mid in MODEL_ORDER]
                rows.append([f"{label} ({units})"] + [f"{value:.4g}" for value in rmses])
                changes.append((label, 100*(rmses[3]/rmses[1]-1), 100*(rmses[2]/rmses[1]-1)))
            table(fig, rows, ["Variable group"] + LABELS, [.065, .235, .87, .59],
                  widths=[.40, .14, .14, .16, .16], size=10.5)
            largest = sorted(changes, key=lambda r:r[1], reverse=True)[:3]
            notes = "; ".join(f"{label.lower()} +{change:.1f}%" for label,change,_ in largest)
            paragraph(fig, "Largest Full AMSE rollout-RMSE increases versus MSE: " + notes + ".", .18,
                      size=10.5, width=122)
            paragraph(fig, "These are physical-unit RMSEs, not the exact GraphCast loss. Upper-air groups pool all "
                      "13 levels equally. Small AMSE λ=0.1 differences are inconclusive without uncertainty estimates.",
                      .10, size=10, width=125, color=MUTED)
            pdf.savefig(fig); plt.close(fig)

        fig = page("Why sharper power can coexist with larger error", "Measured mechanism and comparison with the AMSE paper", 7)
        y = paragraph(fig,
            "Power matching does not ensure that a feature is correctly located or timed. In each spectral "
            "mode, ordinary squared error equals an amplitude-mismatch term plus a decorrelation term. "
            "Restoring the amplitude of uncertain patterns can increase error even if coherence stays fixed.", .815)
        fig.text(.08, y-.012,
                 r"$\mathrm{MSE}_k=(\sqrt{P_f}-\sqrt{P_t})^2+2\sqrt{P_fP_t}(1-\rho_k)$",
                 fontsize=18, color=INK)
        y -= .09
        vi = features.index("T850")
        li = spectra["lead_hours"].tolist().index(240)
        t = spectra["truth_power"][li, vi, 1:]
        rows = []
        for m in range(4):
            p = spectra["power"][m, li, vi, 1:]
            c = spectra["cross_power"][m, li, vi, 1:]
            amp = np.sum((np.sqrt(p)-np.sqrt(t))**2)
            decor = np.sum(2*(np.sqrt(p*t)-c))
            rows.append([LABELS[m], f"{amp:.3f}", f"{decor:.3f}", f"{amp+decor:.3f}"])
        table(fig, rows, ["Day 10 T850", "Amplitude error (K²)", "Decorrelation (K²)", "Total anomaly MSE (K²)"],
              [.065, y-.225, .87, .22], widths=[.26, .23, .25, .26], size=10.5)
        y = paragraph(fig,
            "These are midlatitude zonal-anomaly errors, not global T850 RMSE. The decorrelation term "
            "increases with both larger power and reduced coherence. AMSE includes a coherence penalty; "
            "finite network capacity, optimization and autoregressive dynamics still permit worse "
            "alignment on evaluation cases.", y-.255, size=10.5, width=122)
        y = paragraph(fig,
            "The original paper's main forecast-skill comparison uses nine-member lagged ensembles: "
            "CRPS improves while ensemble-mean RMSE changes little. This is a different target from "
            "individual-forecast RMSE. Its training also differs: 0.25° GraphCast, 32,500 updates, and a "
            "single-step-to-12-step curriculum. Here a frozen 1° GraphCast supports a residual branch "
            "trained for 2,000 updates with stopped physical-feedback gradients. These are plausible "
            "contributors, not isolated causal findings.", y-.005, size=10.5, width=122)
        fig.text(.065, .074, "Source: Subich et al., sections 2.4 and 3.2 · arxiv.org/html/2501.19374v2",
                 fontsize=9, color="#00638C", url="https://arxiv.org/html/2501.19374v2")
        pdf.savefig(fig); plt.close(fig)
    return path


def load_manifest(path):
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        value = value.get("figures", value.get("plots", value))
    if not isinstance(value, list):
        raise ValueError(f"Unexpected figure manifest in {path}")
    return value


def path_of(record, extension):
    value = record.get(extension) or record.get(f"{extension}_path")
    if value is None and isinstance(record.get("paths"), dict):
        value = record["paths"].get(extension)
    if value is None and isinstance(record.get("files"), dict):
        value = record["files"].get(extension)
    if value is None:
        raise ValueError(f"Missing {extension} in {record}")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def appendix(document, start_page):
    path = DATA / "report_appendix.pdf"
    with PdfPages(path) as pdf:
        fig = page("Reference repeatability and interpretation limits", "The figures contain exactly four forecast models", start_page)
        rows = []
        for dataset, block in document["evaluations"].items():
            reference = block["models"]["original_gc"]["exact_graphcast_loss"]
            for mid in MODEL_ORDER:
                record = block["models"][mid]
                loss = record["exact_graphcast_loss"]
                paired = record.get("paired_baseline", {}).get("exact_graphcast_loss", reference)
                common_reduction = 100*(1-loss["rollout"]/reference["rollout"])
                paired_reduction = 100*(1-loss["rollout"]/paired["rollout"])
                rows.append([str(block["year"]), document["models"][mid]["label"],
                             f"{loss['rollout']:.6f}", f"{common_reduction:+.3f}%", f"{paired_reduction:+.3f}%"])
        table(fig, rows, ["Year", "Model", "Exact GC rollout loss", "vs displayed GC", "vs own paired GC"],
              [.065, .365, .87, .44], widths=[.08, .25, .23, .22, .22], size=10.5)
        paragraph(fig,
            "Repeated original-GC forecasts exhibit numerical drift. All primary plots use one explicitly "
            "identified Original GC run, without averaging. The rightmost column retains each model's "
            "paired reference to show the practical effect of that choice. The complete audit also records "
            "the extra MSE control and per-lead baseline differences.", .32, size=10.5, width=122)
        paragraph(fig,
            "The spectral sample contains only eight previously inspected starts. The 2022 set contains "
            "32 historical development starts. No confidence intervals, ensemble metrics or unseen "
            "test-set claims are made. Pressure-level, latitude-domain and weighting differences must be "
            "respected when comparing spectral and global-error charts.", .16, size=10.5, width=122)
        pdf.savefig(fig); plt.close(fig)
        fig = page("Reproducibility and artifact guide", "All source evaluations and checkpoints are preserved", start_page+1)
        y = paragraph(fig,
            "Primary checkpoint policy: SWA of updates 500, 1000, 1500 and 2000. Inference uses 40 "
            "six-hour leads, cold initial recurrent state and full corrected feedback. Source manifests "
            "identify the original GraphCast checkpoint, data, initialization dates and metric settings.", .81)
        y = paragraph(fig,
            "The report_data.json and spectra.npz files contain the validated four-model inputs. CSV files "
            "provide all plotted spectral summaries, all 40 leads of global RMSE and every one of the "
            "83 output channels. Source and checkpoint identities are recorded by SHA256. Standalone "
            "PNG and vector PDF files are provided for each figure.", y-.025)
        y = paragraph(fig, "Data and audit directory:", y-.025, size=11)
        y = paragraph(fig, str(DATA.relative_to(ROOT)), y, size=9.4, width=120, color=MUTED)
        y = paragraph(fig, "Figure directory:", y-.025, size=11)
        y = paragraph(fig, str(IMAGES.relative_to(ROOT)), y, size=9.4, width=120, color=MUTED)
        y = paragraph(fig, "Generators:", y-.025, size=11)
        for name in ["amse_feature_report_data.py", "plot_v24_amse_feature_report.py",
                     "plot_v24_amse_feature_errors.py", "report_v24_amse_features.py"]:
            y = paragraph(fig, "scripts/analyze_models/"+name, y, size=9.4, width=120, color=MUTED)
        paragraph(fig,
            "Rebuild in that order after sourcing scripts/graphcast_env.sh. The process reads saved "
            "evaluations, validates them, generates plots and assembles this report. It does not submit "
            "training or evaluation jobs.", y-.025, size=10.5, width=122)
        pdf.savefig(fig); plt.close(fig)
    return path


def gallery(records, spectral_error):
    items = []
    for i, record in enumerate(records):
        png = path_of(record, "png")
        title = record.get("title", record.get("description", png.stem))
        items.append(f'<article id="figure-{i}"><h2>{html.escape(title)}</h2>'
                     f'<a href="{html.escape(relative(path_of(record,"pdf")))}">Download vector PDF</a>'
                     f'<img loading="lazy" src="{html.escape(relative(png))}" alt="{html.escape(title)}"></article>')
    nav = "".join(f'<li><a href="#figure-{i}">{html.escape(r.get("title",r.get("description",path_of(r,"png").stem)))}</a></li>'
                  for i,r in enumerate(records))
    means = spectral_error.mean(axis=(1,2,3))
    summary = "".join(f"<tr><td>{html.escape(label)}</td><td>{means[i]:.2f}%</td></tr>" for i,label in enumerate(LABELS))
    output = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AMSE forecast quality across weather features</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;color:#162c3e;background:#f4f7fa;margin:0}}main{{max-width:1150px;margin:auto;padding:32px}}h1{{font-size:32px}}article,section{{background:white;padding:24px;margin:24px 0;border-radius:8px}}img{{width:100%;height:auto}}a{{color:#00638c}}td,th{{padding:8px 20px 8px 0;text-align:left}}nav ul{{columns:2;list-style:none;padding:0}}nav li{{margin-bottom:8px}}@media(max-width:700px){{nav ul{{columns:1}}main{{padding:12px}}article{{padding:8px}}}}</style>
<main><h1>AMSE forecast quality across weather features</h1><p>4 October 2026 · Original GC · MSE · AMSE λ=0.1 · Full AMSE</p>
<p><a href="{REPORT.name}">Download the comprehensive PDF report</a></p>
<section><p>Full AMSE restores substantially more spectral power, while MSE and AMSE λ=0.1 preserve more deterministic forecast skill. These are completed SWA evaluations; the report contains no new inference.</p>
<table><tr><th>Model</th><th>Mean relative power error</th></tr>{summary}</table>
<p>Equal weighting of 11 spectral fields, k=1–180, and days 1, 3, 7, 10. Eight exploratory 2023 starts; raw powers are pooled before ratios. Global RMSE and exact GraphCast loss have different domains and weights. The 32-start 2022 evaluation is reported separately.</p>
<p>One fixed Original GC and one fixed MSE reference are shown. Repeatability and paired-baseline comparisons are documented in the PDF and <a href="{html.escape(relative(DATA/'report_data.json'))}">data audit</a>. Wind-speed global RMSE was not saved; WS10 and WS850 have spectral results only.</p></section>
<nav><h2>Figures</h2><ul>{nav}</ul></nav>{''.join(items)}</main></html>'''
    GALLERY.write_text(output)


def bookmarks(errors, spectral):
    marks = []
    def entry(title, number, children=None):
        title_hex = (b"\xfe\xff"+title.encode("utf-16-be")).hex()
        count = f" /Count {-children}" if children else ""
        marks.append(f"[ /Title <{title_hex}> /Page {number}{count} /OUT pdfmark")
    titles = ["Summary", "Models and metrics", "Power error by field", "Coherence by field",
              "2022 rollout RMSE", "2023 rollout RMSE", "Power and error mechanism"]
    for number, title in enumerate(titles, 1):
        entry(title, number)
    number = 8
    entry("Forecast error figures", number, len(errors))
    for record in errors:
        title = record.get("title", record.get("description", path_of(record,"png").stem))
        if record.get("dataset"):
            datasets = record["dataset"] if isinstance(record["dataset"], list) else [record["dataset"]]
            title += " · " + " / ".join({"exact32":"2022", "spectral8":"2023"}.get(d, d) for d in datasets)
        entry(title, number)
        number += 1
    entry("Spectral figures by field", number, len(spectral))
    for record in spectral:
        entry(record.get("title",path_of(record,"png").stem), number)
        number += 1
    entry("Reference repeatability audit", number)
    entry("Reproducibility", number+1)
    path = DATA / "report_bookmarks.ps"
    path.write_text("\n".join(marks)+"\n")
    return path


def main():
    document = json.loads((DATA / "report_data.json").read_text())
    spectra, error, coherence = read_spectra()
    plt.rcParams.update({"font.family":"DejaVu Sans", "pdf.fonttype":42, "ps.fonttype":42})
    IMAGES.mkdir(parents=True, exist_ok=True)
    front = front_pages(document, spectra, error, coherence)
    errors = load_manifest(DATA / "error_figures_manifest.json")
    spectral = load_manifest(IMAGES / "figures_manifest.json")
    figures = errors + spectral
    end = appendix(document, 8+len(figures))
    outlines = bookmarks(errors, spectral)
    inputs = [front]+[path_of(record,"pdf") for record in figures]+[end]
    assert all(path.is_file() for path in inputs)
    temporary = DATA / "assembled_report.pdf"
    subprocess.run(["gs", "-q", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pdfwrite",
                    "-dCompatibilityLevel=1.7", "-dPDFSETTINGS=/prepress",
                    "-sOutputFile="+str(temporary), *map(str,inputs), str(outlines)], check=True)
    REPORT.write_bytes(temporary.read_bytes())
    gallery(figures, error)
    manifest = {"report":str(REPORT.relative_to(ROOT)), "gallery":str(GALLERY.relative_to(ROOT)),
                "figure_count":len(figures), "expected_pages":9+len(figures),
                "inputs":[{"path":str(p.relative_to(ROOT)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs],
                "generator":str(Path(__file__).relative_to(ROOT))}
    (DATA / "report_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:v for k,v in manifest.items() if k!="inputs"},indent=2))


if __name__ == "__main__":
    main()
