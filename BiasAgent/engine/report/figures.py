import os
from engine.report.charts import save_bar_chart


def create_counterfactual_bias_figure(cf_by_bias, figures_dir):
    if len(cf_by_bias) == 0:
        return None

    fig_path = os.path.join(figures_dir, "fig1_counterfactual_by_bias_type.png")

    save_bar_chart(
        cf_by_bias,
        x_col="bias_type",
        y_col="score",
        title="Counterfactual Baseline Bias by Bias Type",
        output_path=fig_path,
        xlabel="Bias type",
        ylabel="Bias score",
    )

    return fig_path


def create_paraphrase_family_figure(family_df, figures_dir):
    fig_path = os.path.join(figures_dir, "fig2_paraphrase_family_bias.png")

    save_bar_chart(
        family_df,
        x_col="paraphrase_family",
        y_col="bias_rate",
        title="Average Bias Score by Paraphrase Family",
        output_path=fig_path,
        xlabel="Paraphrase family",
        ylabel="Bias score",
    )

    return fig_path


def create_proxy_figure(proxy_by_type, figures_dir):
    fig_path = os.path.join(figures_dir, "fig3_proxy_bias_by_type.png")

    save_bar_chart(
        proxy_by_type,
        x_col="proxy_type",
        y_col="bias_rate",
        title="Proxy Bias Score by Proxy Type",
        output_path=fig_path,
        xlabel="Proxy type",
        ylabel="Bias score",
    )

    return fig_path


def create_paraphrase_detail_figure(para_by_type, figures_dir):
    fig_path = os.path.join(figures_dir, "fig4_detailed_paraphrase_type_bias.png")

    save_bar_chart(
        para_by_type,
        x_col="paraphrase_type",
        y_col="bias_rate",
        title="Detailed Bias Score by Paraphrase Type",
        output_path=fig_path,
        xlabel="Paraphrase type",
        ylabel="Bias score",
    )

    return fig_path


def create_overall_comparison_figure(overall_df, figures_dir):
    fig_path = os.path.join(figures_dir, "fig5_overall_comparison.png")

    save_bar_chart(
        overall_df,
        x_col="method_family",
        y_col="bias_rate",
        title="Overall Bias Comparison",
        output_path=fig_path,
        xlabel="Evaluation family",
        ylabel="Bias score (metric selected for the baseline)",
    )

    return fig_path
