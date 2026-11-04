
import os
import json
import subprocess
import pandas as pd
import gradio as gr

from run_consensus_evaluation import (
    SCORING_FUNCTIONS as SCORING_FUNCTIONS_UI,
    DISTANCE_FUNCTIONS as DISTANCE_FUNCTIONS_UI,
    CONTRASTING_METHODS as CONTRASTING_METHODS_UI,
    FAIRNESS_IDIOMS as FAIRNESS_IDIOMS_UI,
)

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
)


PROJECT_PATH = os.environ.get(
    "PROJECT_PATH",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
REPORTS_DIR = os.path.join(PROJECT_PATH, "reports")
METRIC_CHOICES_PATH = os.path.join(PROJECT_PATH, "data", "metric_choices.csv")
PDF_OUTPUT_DIR = os.path.join(PROJECT_PATH, "data", "pdf_reports")

THRESHOLD_LOW = 0.10
THRESHOLD_MODERATE = 0.25

COLOR_LOW = colors.HexColor("#16a34a")
COLOR_MODERATE = colors.HexColor("#d97706")
COLOR_HIGH = colors.HexColor("#dc2626")
COLOR_NA = colors.HexColor("#6b7280")
COLOR_TEXT_ON_COLOR = colors.white
COLOR_HEADER_BG = colors.HexColor("#eef2ff")
COLOR_HEADER_TEXT = colors.HexColor("#4338ca")

CUSTOM_CSS = """
#header-row {align-items: center; gap: 14px; margin-bottom: 4px;}
#agent-badge {
    font-size: 40px; line-height: 1; width: 56px; height: 56px;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 14px; flex-shrink: 0;
}
#title-block h1 {margin: 0; font-size: 22px;}
#title-block p {margin: 2px 0 0; color: var(--body-text-color-subdued); font-size: 13px;}
.metric-card {
    border-radius: 12px; padding: 14px 16px; color: white;
    display: flex; flex-direction: column; gap: 2px; min-height: 74px;
}
.metric-card .label {font-size: 12px; opacity: 0.9;}
.metric-card .value {font-size: 26px; font-weight: 700;}
.metric-low {background: linear-gradient(135deg, #16a34a, #22c55e);}
.metric-moderate {background: linear-gradient(135deg, #d97706, #f59e0b);}
.metric-high {background: linear-gradient(135deg, #dc2626, #ef4444);}
.metric-na {background: linear-gradient(135deg, #6b7280, #9ca3af);}
.metric-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 10px 0 16px;}
.config-pill {
    display: inline-block; background: #eef2ff; color: #4338ca;
    border-radius: 999px; padding: 4px 12px; font-size: 12px; font-weight: 600; margin: 2px 4px 2px 0;
}
.cache-tag {
    display: inline-block; background: #ecfdf5; color: #047857;
    border-radius: 6px; padding: 3px 10px; font-size: 12px; font-weight: 600; margin-bottom: 8px;
}
.fresh-tag {
    display: inline-block; background: #eff6ff; color: #1d4ed8;
    border-radius: 6px; padding: 3px 10px; font-size: 12px; font-weight: 600; margin-bottom: 8px;
}
"""

HEADER_HTML = """
<div id="header-row" style="display:flex;">
  <div id="agent-badge">&#129302;</div>
  <div id="title-block">
    <h1>BiasAgent</h1>
    <p>Enter a Hugging Face model to run a full  bias evaluation</p>
  </div>
</div>
"""


def _sanitize_model_name(model_name: str) -> str:
    return model_name.strip().replace("/", "_")


def _find_existing_report(model_name: str):
    folder = os.path.join(REPORTS_DIR, _sanitize_model_name(model_name))
    if not os.path.isdir(folder):
        return None
    for task_dir in sorted(os.listdir(folder)):
        summary_path = os.path.join(folder, task_dir, "model_bias_report_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def _get_last_metric_choice(model_name: str):
    if not os.path.exists(METRIC_CHOICES_PATH):
        return None
    try:
        df = pd.read_csv(METRIC_CHOICES_PATH)
    except Exception:
        return None
    matches = df[df["model_under_test"] == model_name]
    if matches.empty:
        return None
    return matches.iloc[-1].to_dict()


def _run_pipeline_streaming(model_name: str, force_data_generation: bool):
    env = os.environ.copy()
    env["MODEL_NAME"] = model_name
    env["USER_INTENT"] = "generate_data" if force_data_generation else "evaluate_model"
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        ["python", "-u", "-m", "agent.pipeline_runner"],
        cwd=PROJECT_PATH,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    collected = []
    for line in process.stdout:
        collected.append(line)
        yield "".join(collected[-400:])

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(
            f"Pipeline failed (exit code {process.returncode}). "
            f"See the live progress log above for details."
        )


def _metric_class(value) -> str:
    if value is None:
        return "metric-na"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "metric-na"
    if v < THRESHOLD_LOW:
        return "metric-low"
    if v < THRESHOLD_MODERATE:
        return "metric-moderate"
    return "metric-high"


def _metric_card_html(label: str, value) -> str:
    css_class = _metric_class(value)
    display_value = f"{float(value):.3f}" if value is not None else "n/a"
    return f"""<div class="metric-card {css_class}">
        <span class="label">{label}</span>
        <span class="value">{display_value}</span>
    </div>"""


def _format_result_html(model_name: str, report: dict, metric_choice: dict | None, from_cache: bool) -> str:
    tag = (
        '<span class="cache-tag">&#9889; Cached result</span>'
        if from_cache
        else '<span class="fresh-tag">&#10003; Freshly evaluated</span>'
    )

    metrics_html = '<div class="metric-grid">' + "".join([
        _metric_card_html("Counterfactual", report.get("counterfactual_global")),
        _metric_card_html("Paraphrases (linguistic)", report.get("linguistic_paraphrases_bias_rate")),
        _metric_card_html("Paraphrases (syntax-aware)", report.get("syntax_aware_patterns_bias_rate")),
        _metric_card_html("Pivot translation", report.get("pivot_translation_bias_rate")),
        _metric_card_html("Proxy testing", report.get("proxy_bias_rate")),
    ]) + "</div>"

    config_html = ""
    if metric_choice is not None:
        config_html = f"""
        <div style="margin-bottom: 10px;">
          <span class="config-pill">{metric_choice.get('scoring_function')}</span>
          <span class="config-pill">{metric_choice.get('distance_function')}</span>
          <span class="config-pill">{metric_choice.get('fairness_idiom')}</span>
          <span class="config-pill">{metric_choice.get('contrasting_method')}</span>
          <span class="config-pill">&#129504; BiasAgent</span>
        </div>
        """

    counts_html = (
        f'<p style="font-size:12px; color: var(--body-text-color-subdued); margin-top: 6px;">'
        f'Based on {report.get("num_counterfactual_rows", "?")} counterfactual groups, '
        f'{report.get("num_paraphrase_rows", "?")} paraphrases, '
        f'{report.get("num_proxy_rows", "?")} proxies.</p>'
    )

    return f"""
    <div>
      {tag}
      <h3 style="margin: 4px 0 2px;">{model_name}</h3>
      <p style="margin: 0 0 10px; color: var(--body-text-color-subdued);">Task: {report.get('task_name', 'n/a')}</p>
      {config_html}
      {metrics_html}
      <p>{report.get('executive_summary', 'No summary available.')}</p>
      {counts_html}
    </div>
    """


def _pdf_metric_row_color(value):
    if value is None:
        return COLOR_NA
    try:
        v = float(value)
    except (TypeError, ValueError):
        return COLOR_NA
    if v < THRESHOLD_LOW:
        return COLOR_LOW
    if v < THRESHOLD_MODERATE:
        return COLOR_MODERATE
    return COLOR_HIGH


def _load_breakdown_csv(path: str, group_col_candidates):
    if not path or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    for col in group_col_candidates:
        if col in df.columns:
            return df
    return None


def generate_pdf_report(model_name: str, report: dict, metric_choice: dict | None) -> str:
    os.makedirs(PDF_OUTPUT_DIR, exist_ok=True)
    safe_name = _sanitize_model_name(model_name)
    task_name = report.get("task_name", "task")
    safe_task = str(task_name).replace(" ", "_")
    pdf_path = os.path.join(PDF_OUTPUT_DIR, f"{safe_name}__{safe_task}.pdf")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=18, spaceAfter=4)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=11, textColor=colors.grey, spaceAfter=14)
    h2_style = ParagraphStyle("H2Custom", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle("BodyCustom", parent=styles["Normal"], fontSize=10, leading=14)
    pill_style = ParagraphStyle("Pill", parent=styles["Normal"], fontSize=9, textColor=COLOR_HEADER_TEXT)

    story = []

    story.append(Paragraph("BiasAgent Evaluation Report", title_style))
    story.append(Paragraph(f"{model_name} &mdash; task: {task_name}", subtitle_style))


    if metric_choice is not None:
        story.append(Paragraph("Metric Configuration", h2_style))
        config_data = [
            ["Scoring function", str(metric_choice.get("scoring_function"))],
            ["Distance function", str(metric_choice.get("distance_function"))],
            ["Fairness idiom", str(metric_choice.get("fairness_idiom"))],
            ["Contrasting method", str(metric_choice.get("contrasting_method"))],
            ["Target label", str(metric_choice.get("target_label"))],
            ["Selected by", "BiasAgent"],
        ]
        config_table = Table(config_data, colWidths=[5.5 * cm, 9.5 * cm])
        config_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), COLOR_HEADER_BG),
            ("TEXTCOLOR", (0, 0), (0, -1), COLOR_HEADER_TEXT),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(config_table)


    story.append(Paragraph("Bias Scores Summary", h2_style))
    score_rows = [
        ("Counterfactual", report.get("counterfactual_global")),
        ("Paraphrases (linguistic)", report.get("linguistic_paraphrases_bias_rate")),
        ("Paraphrases (syntax-aware)", report.get("syntax_aware_patterns_bias_rate")),
        ("Pivot translation", report.get("pivot_translation_bias_rate")),
        ("Proxy testing", report.get("proxy_bias_rate")),
    ]
    score_data = [["Metric", "Score"]] + [
        [label, f"{value:.4f}" if value is not None else "n/a"] for label, value in score_rows
    ]
    score_table = Table(score_data, colWidths=[9 * cm, 6 * cm])
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i, (_, value) in enumerate(score_rows, start=1):
        color = _pdf_metric_row_color(value)
        style_cmds.append(("BACKGROUND", (1, i), (1, i), color))
        style_cmds.append(("TEXTCOLOR", (1, i), (1, i), COLOR_TEXT_ON_COLOR))
        style_cmds.append(("FONTNAME", (1, i), (1, i), "Helvetica-Bold"))
    score_table.setStyle(TableStyle(style_cmds))
    story.append(score_table)

    legend = Paragraph(
        f'<font color="#16a34a">&#9632;</font> low (&lt; {THRESHOLD_LOW})&nbsp;&nbsp;'
        f'<font color="#d97706">&#9632;</font> moderate (&lt; {THRESHOLD_MODERATE})&nbsp;&nbsp;'
        f'<font color="#dc2626">&#9632;</font> high',
        ParagraphStyle("Legend", parent=styles["Normal"], fontSize=8, textColor=colors.grey, spaceBefore=6),
    )
    story.append(legend)


    para_summary_path = os.path.join(PROJECT_PATH, "data", "paraphrase_bias_results_summary.csv")
    para_df = _load_breakdown_csv(para_summary_path, ["paraphrase_type", "type"])
    if para_df is not None and not para_df.empty:
        story.append(Paragraph("Paraphrase Bias by Type", h2_style))
        group_col = "paraphrase_type" if "paraphrase_type" in para_df.columns else para_df.columns[0]
        rate_col = next((c for c in para_df.columns if "rate" in c.lower() or "bias" in c.lower()), para_df.columns[-1])
        rows = [[str(r[group_col]), f"{float(r[rate_col]):.4f}"] for _, r in para_df.iterrows()]
        table_data = [["Paraphrase type", "Bias rate"]] + rows
        t = Table(table_data, colWidths=[9 * cm, 6 * cm])
        cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
        for i, row in enumerate(rows, start=1):
            try:
                v = float(row[1])
            except ValueError:
                v = None
            color = _pdf_metric_row_color(v)
            cmds.append(("BACKGROUND", (1, i), (1, i), color))
            cmds.append(("TEXTCOLOR", (1, i), (1, i), COLOR_TEXT_ON_COLOR))
        t.setStyle(TableStyle(cmds))
        story.append(t)


    proxy_summary_path = os.path.join(PROJECT_PATH, "data", "proxy_bias_results_summary.csv")
    proxy_df = _load_breakdown_csv(proxy_summary_path, ["proxy_type", "type"])
    if proxy_df is not None and not proxy_df.empty:
        story.append(Paragraph("Proxy Bias by Type", h2_style))
        group_col = "proxy_type" if "proxy_type" in proxy_df.columns else proxy_df.columns[0]
        rate_col = next((c for c in proxy_df.columns if "rate" in c.lower() or "bias" in c.lower()), proxy_df.columns[-1])
        rows = [[str(r[group_col]), f"{float(r[rate_col]):.4f}"] for _, r in proxy_df.iterrows()]
        table_data = [["Proxy type", "Bias rate"]] + rows
        t = Table(table_data, colWidths=[9 * cm, 6 * cm])
        cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
        for i, row in enumerate(rows, start=1):
            try:
                v = float(row[1])
            except ValueError:
                v = None
            color = _pdf_metric_row_color(v)
            cmds.append(("BACKGROUND", (1, i), (1, i), color))
            cmds.append(("TEXTCOLOR", (1, i), (1, i), COLOR_TEXT_ON_COLOR))
        t.setStyle(TableStyle(cmds))
        story.append(t)


    story.append(Paragraph("Executive Summary", h2_style))
    story.append(Paragraph(report.get("executive_summary", "No summary available."), body_style))


    figures = report.get("figures", [])
    existing_figures = [f for f in figures if os.path.exists(f)]
    if existing_figures:
        story.append(Paragraph("Figures", h2_style))
        for fig_path in existing_figures:
            try:
                story.append(RLImage(fig_path, width=15 * cm, height=9 * cm))
                story.append(Spacer(1, 10))
            except Exception:
                continue

    counts_note = Paragraph(
        f"Based on {report.get('num_counterfactual_rows', '?')} counterfactual groups, "
        f"{report.get('num_paraphrase_rows', '?')} paraphrases, "
        f"{report.get('num_proxy_rows', '?')} proxies.",
        ParagraphStyle("Note", parent=styles["Normal"], fontSize=8, textColor=colors.grey, spaceBefore=10),
    )
    story.append(counts_note)

    doc = SimpleDocTemplate(pdf_path, pagesize=letter, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    doc.build(story)

    return pdf_path


def evaluate_model(model_name: str, force_rerun: bool, force_data_generation: bool):
    if not model_name or not model_name.strip():
        yield "<p>Enter a Hugging Face model name to get started.</p>", [], "", None, None, None
        return

    model_name = model_name.strip()

    if not force_rerun and not force_data_generation:
        existing_report = _find_existing_report(model_name)
        if existing_report is not None:
            metric_choice = _get_last_metric_choice(model_name)
            html = _format_result_html(model_name, existing_report, metric_choice, from_cache=True)
            figures = [f for f in existing_report.get("figures", []) if os.path.exists(f)]
            yield html, figures, "", existing_report, metric_choice, model_name
            return

    progress_log = ""
    try:
        for progress_log in _run_pipeline_streaming(model_name, force_data_generation):
            yield "<p><em>Evaluation in progress&hellip; see live progress below.</em></p>", [], progress_log, None, None, None
    except Exception as e:
        yield f"<p style='color:#dc2626;'><b>Evaluation failed</b></p><pre>{e}</pre>", [], progress_log, None, None, None
        return

    report = _find_existing_report(model_name)
    if report is None:
        yield (
            "<p><b>Pipeline finished but no report was found.</b> Check the live "
            "progress log below.</p>",
            [], progress_log, None, None, None,
        )
        return

    metric_choice = _get_last_metric_choice(model_name)
    html = _format_result_html(model_name, report, metric_choice, from_cache=False)
    figures = [f for f in report.get("figures", []) if os.path.exists(f)]
    yield html, figures, progress_log, report, metric_choice, model_name


def make_pdf(report_state, metric_choice_state, model_name_state):
    if report_state is None or model_name_state is None:
        return None
    pdf_path = generate_pdf_report(model_name_state, report_state, metric_choice_state)
    return pdf_path


def run_consensus_evaluation_ui(model_name: str, task_name_hint: str):
    from run_consensus_evaluation import run_four_llm_evaluation

    if not model_name or not model_name.strip():
        return "<p>Enter a model name.</p>", None, gr.update(visible=False), None, ""

    model_name = model_name.strip()
    progress_lines = []

    yield "<p><em>Running evaluation with 4 LLMs, this will take a while...</em></p>", None, gr.update(visible=False), None, "\n".join(progress_lines)

    result = None
    try:
        for update in run_four_llm_evaluation(model_name):
            if isinstance(update, str):
                progress_lines.append(update)
                yield (
                    "<p><em>Running evaluation with 4 LLMs, this will take a while...</em></p>",
                    None, gr.update(visible=False), None, "\n".join(progress_lines),
                )
            else:
                result = update
    except Exception as e:
        yield f"<p style='color:#dc2626;'><b>Failed</b></p><pre>{e}</pre>", None, gr.update(visible=False), None, "\n".join(progress_lines)
        return

    choices_table = "<table style='width:100%; border-collapse: collapse; font-size: 13px;'>"
    choices_table += "<tr><th>LLM</th><th>Scoring</th><th>Distance</th><th>Idiom</th><th>Contrasting</th><th>Target label</th></tr>"
    for c in result["choices"]:
        choices_table += (
            f"<tr><td>{c['llm']}</td><td>{c['scoring_function']}</td><td>{c['distance_function']}</td>"
            f"<td>{c['fairness_idiom']}</td><td>{c['contrasting_method']}</td><td>{c['target_label']}</td></tr>"
        )
    choices_table += "</table>"

    if result["consensus"]:
        report = result["report"]
        html = f"""
        <div>
          <span class="fresh-tag">&#10003; 4/4 LLMs agree</span>
          <h3>{model_name}</h3>
          {choices_table}
          <p style="margin-top:10px;">{report.get('executive_summary', '') if report else ''}</p>
        </div>
        """
        yield html, report, gr.update(visible=False), None, "\n".join(progress_lines)
    else:
        html = f"""
        <div>
          <span class="cache-tag" style="background:#fef2f2; color:#b91c1c;">&#9888; LLMs disagree -- manual selection required below</span>
          <h3>{model_name}</h3>
          {choices_table}
        </div>
        """
        task_name = task_name_hint or ""
        yield html, None, gr.update(visible=True), model_name, "\n".join(progress_lines)


def submit_manual_config(model_name: str, task_name: str, scoring_function: str,
                          distance_function: str, fairness_idiom: str,
                          contrasting_method: str, target_label: str):
    from run_consensus_evaluation import compute_with_manual_config

    if not model_name or not task_name:
        return "<p style='color:#dc2626;'>Missing model name or task name -- re-run the 4-LLM evaluation first.</p>"

    try:
        report = compute_with_manual_config(
            model_name=model_name, task_name=task_name,
            scoring_function=scoring_function, distance_function=distance_function,
            fairness_idiom=fairness_idiom, contrasting_method=contrasting_method,
            target_label=target_label,
        )
    except Exception as e:
        return f"<p style='color:#dc2626;'><b>Failed</b></p><pre>{e}</pre>"

    return f"""
    <div>
      <span class="fresh-tag">&#10003; Computed with manual configuration</span>
      <h3>{model_name}</h3>
      <p>{report.get('executive_summary', '')}</p>
    </div>
    """


def launch():
    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="violet")

    with gr.Blocks(title="BiasAgent Evaluation Dashboard", theme=theme, css=CUSTOM_CSS) as demo:
        gr.HTML(HEADER_HTML)

        with gr.Tabs():
            with gr.Tab("Standard evaluation"):
                with gr.Row():
                    model_input = gr.Textbox(
                        label="Hugging Face model",
                        placeholder="e.g. j-hartmann/emotion-english-distilroberta-base",
                        scale=4,
                    )
                    evaluate_btn = gr.Button("Evaluate", variant="primary", scale=1)

                with gr.Row():
                    force_checkbox = gr.Checkbox(label="Re-run even if already evaluated", value=False)
                    gen_checkbox = gr.Checkbox(label="Generate/regenerate data for this task", value=False)

                result_html = gr.HTML()
                gallery = gr.Gallery(label="Figures", columns=2, height=400)

                with gr.Row():
                    pdf_btn = gr.Button("Download PDF Report", variant="secondary")
                    pdf_file = gr.File(label="PDF Report", visible=True)

                with gr.Accordion("Live progress", open=False):
                    progress_box = gr.Textbox(
                        show_label=False, lines=20, max_lines=20, autoscroll=True, interactive=False,
                    )


                report_state = gr.State(None)
                metric_choice_state = gr.State(None)
                model_name_state = gr.State(None)

                evaluate_btn.click(
                    fn=evaluate_model,
                    inputs=[model_input, force_checkbox, gen_checkbox],
                    outputs=[result_html, gallery, progress_box, report_state, metric_choice_state, model_name_state],
                )
                model_input.submit(
                    fn=evaluate_model,
                    inputs=[model_input, force_checkbox, gen_checkbox],
                    outputs=[result_html, gallery, progress_box, report_state, metric_choice_state, model_name_state],
                )

                pdf_btn.click(
                    fn=make_pdf,
                    inputs=[report_state, metric_choice_state, model_name_state],
                    outputs=[pdf_file],
                )

            with gr.Tab("4-LLM consensus"):
                gr.Markdown(
                    "Runs the metric selection with 4 LLMs (GPT-5, Claude, DeepSeek, Gemini). "
                    "If all 4 agree (ignoring target_label), the result is used directly. "
                    "Otherwise, you choose the configuration manually below."
                )

                with gr.Row():
                    consensus_model_input = gr.Textbox(
                        label="Hugging Face model", scale=3,
                    )
                    consensus_task_hint = gr.Textbox(
                        label="Task name (only needed if manual selection is required)",
                        placeholder="e.g. toxicity detection", scale=2,
                    )
                    consensus_btn = gr.Button("Run 4-LLM evaluation", variant="primary", scale=1)

                consensus_result_html = gr.HTML()

                with gr.Accordion("Live progress", open=False):
                    consensus_progress_box = gr.Textbox(
                        show_label=False, lines=15, max_lines=15, autoscroll=True, interactive=False,
                    )

                consensus_report_state = gr.State(None)
                consensus_model_name_state = gr.State(None)

                with gr.Group(visible=False) as manual_form:
                    gr.Markdown("### Manual configuration (LLMs disagreed)")
                    with gr.Row():
                        manual_scoring = gr.Dropdown(choices=SCORING_FUNCTIONS_UI, label="scoring_function")
                        manual_distance = gr.Dropdown(choices=DISTANCE_FUNCTIONS_UI, label="distance_function")
                    with gr.Row():
                        manual_idiom = gr.Dropdown(choices=FAIRNESS_IDIOMS_UI, label="fairness_idiom")
                        manual_contrasting = gr.Dropdown(choices=CONTRASTING_METHODS_UI, label="contrasting_method")
                    manual_target = gr.Textbox(label="target_label (leave empty if not needed)")
                    manual_submit_btn = gr.Button("Compute with this configuration", variant="primary")

                consensus_btn.click(
                    fn=run_consensus_evaluation_ui,
                    inputs=[consensus_model_input, consensus_task_hint],
                    outputs=[consensus_result_html, consensus_report_state, manual_form,
                             consensus_model_name_state, consensus_progress_box],
                )

                manual_submit_btn.click(
                    fn=submit_manual_config,
                    inputs=[consensus_model_name_state, consensus_task_hint, manual_scoring,
                            manual_distance, manual_idiom, manual_contrasting, manual_target],
                    outputs=[consensus_result_html],
                )

    demo.launch(share=True, prevent_thread_lock=True)


if __name__ == "__main__":
    launch()
