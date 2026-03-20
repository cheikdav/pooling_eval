"""Tab for paired state evaluation with ground truth CIs."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from pathlib import Path

from common import get_method_display_name, sort_methods, compute_batch_constants, _run_bootstrap, _metric_vals


@st.cache_data
def _load_paired_batch_pivots(paired_predictions_path, predictions_path, adjust_constant, gamma, truncation_coefficient):
    """Load paired predictions and return (s1_pivot, s2_pivot): pair_idx × batch_name.

    Shared intermediate cached by plot_paired_log_metric_evolution and the bootstrap.
    """
    df = pd.read_parquet(paired_predictions_path)

    if adjust_constant:
        results_dir = str(Path(predictions_path).parent.parent.parent)
        regular_df = pd.read_parquet(predictions_path)
        batch_constants = compute_batch_constants(regular_df, results_dir, gamma, truncation_coefficient)
        if batch_constants is not None:
            df = df.copy()
            constants = df['batch_name'].map(batch_constants).fillna(0)
            df['s1_predicted'] += constants
            df['s2_predicted'] += constants

    s1_pivot = df.pivot_table(index='pair_idx', columns='batch_name', values='s1_predicted', aggfunc='mean').dropna()
    s2_pivot = df.pivot_table(index='pair_idx', columns='batch_name', values='s2_predicted', aggfunc='mean').dropna()
    return s1_pivot, s2_pivot


@st.cache_data
def _paired_bootstrap_stderr(s1_pivot, s2_pivot, metric_name, mode, s1_gt, s2_gt, diff_gt, scale_mode,
                              n_bootstrap=200, subsample=5000, seed=0):
    """Bootstrap SE for paired evolution plot metrics by resampling batch columns."""
    EPS = 1e-10
    METRIC_KEY_MAP = {'Variance': 'variance', 'Squared Bias': 'ground_truth_error_squared', 'MSE': 'mse'}
    metric_key = METRIC_KEY_MAP[metric_name]

    common_pairs = np.array(sorted(set(s1_pivot.index) & set(s2_pivot.index)))
    if len(common_pairs) == 0:
        return 0.0

    if len(common_pairs) > subsample:
        rng_sub = np.random.default_rng(seed + 1)
        common_pairs = common_pairs[np.sort(rng_sub.choice(len(common_pairs), size=subsample, replace=False))]

    arrays = {
        's1': s1_pivot.loc[common_pairs].values,
        's2': s2_pivot.loc[common_pairs].values,
    }
    pair_arr = common_pairs

    def compute_fn(resampled):
        rs1, rs2 = resampled['s1'], resampled['s2']
        if mode == 'full':
            vals = np.concatenate([
                _metric_vals(rs1, metric_key, s1_gt[pair_arr], EPS),
                _metric_vals(rs2, metric_key, s2_gt[pair_arr], EPS),
            ])
        else:
            vals = _metric_vals(rs1 - rs2, metric_key, diff_gt[pair_arr], EPS)
        if scale_mode == 'log':
            vals = np.log10(vals + EPS)
        return float(vals.mean())

    k = arrays['s1'].shape[1]
    bootstrap_samples = _run_bootstrap(arrays, compute_fn, k, n_bootstrap, seed)
    return float(np.std(bootstrap_samples)) if len(bootstrap_samples) >= 2 else 0.0


def sort_predictions(predictions_data):
    """Sort predictions_data dict by METHOD_ORDER, unknown methods go last."""
    ordered = sort_methods(predictions_data.keys())
    return {m: predictions_data[m] for m in ordered}


def load_paired_inputs_for_n_episodes(filtered_metadata, methods, n_episodes, adjust_constant=False):
    """Load paired ground-truth data and paired predictions for one training size."""
    filtered_for_n_ep = filtered_metadata[filtered_metadata['n_episodes'] == n_episodes]
    if filtered_for_n_ep.empty:
        return None, {}

    first_row = filtered_for_n_ep.iloc[0]
    data_dir = Path(first_row['data_dir']) if 'data_dir' in first_row else Path(first_row['predictions_path']).parents[2].parent / "data"
    paired_states_file = data_dir / "paired_states.npz"

    if not paired_states_file.exists():
        return None, {}

    paired_data = np.load(paired_states_file, allow_pickle=True)

    predictions_data = {}
    for _, row in filtered_for_n_ep.iterrows():
        if row['method'] not in methods:
            continue

        method = row['method']
        predictions_path = Path(row['predictions_path'])
        paired_predictions_path = predictions_path.parent / "paired_predictions.parquet"

        if not paired_predictions_path.exists():
            continue

        pred_df = pd.read_parquet(paired_predictions_path)

        if adjust_constant:
            results_dir_str = str(predictions_path.parent.parent.parent)
            gamma = row.get('policy_gamma', 0.99)
            truncation_coefficient = row.get('truncation_coefficient', 5.0)

            regular_df = pd.read_parquet(predictions_path)
            batch_constants = compute_batch_constants(regular_df, results_dir_str, gamma, truncation_coefficient)
            print(f"[DEBUG adjust_constant paired] method={method}, batch_constants={'None' if batch_constants is None else f'{len(batch_constants)} batches'}")

            if batch_constants is not None:
                s1_before = pred_df['s1_predicted'].mean()
                constants_series = pred_df['batch_name'].map(batch_constants)
                mask = constants_series.notna()
                pred_df = pred_df.copy()
                pred_df.loc[mask, 's1_predicted'] += constants_series[mask]
                pred_df.loc[mask, 's2_predicted'] += constants_series[mask]
                print(f"[DEBUG adjust_constant paired] method={method}, s1_mean BEFORE={s1_before:.4f}, AFTER={pred_df['s1_predicted'].mean():.4f}, matched={mask.sum()}/{len(mask)}")
            else:
                print(f"[DEBUG adjust_constant paired] SKIPPED: ground truth not found")

        predictions_data[method] = pred_df

    return paired_data, predictions_data


def _compute_metric_values_full_mode(pred_df, paired_data, metric_name):
    """Compute per-state metric values for full mode."""
    if metric_name == 'MSE':
        pair_indices = pred_df['pair_idx'].to_numpy(dtype=int)
        s1_gt = paired_data['s1_mean'][pair_indices]
        s2_gt = paired_data['s2_mean'][pair_indices]

        s1_sq_err = (pred_df['s1_predicted'].to_numpy() - s1_gt) ** 2
        s2_sq_err = (pred_df['s2_predicted'].to_numpy() - s2_gt) ** 2

        s1_state_mse = pd.Series(s1_sq_err).groupby(pred_df['pair_idx']).mean().values
        s2_state_mse = pd.Series(s2_sq_err).groupby(pred_df['pair_idx']).mean().values
        return np.concatenate([s1_state_mse, s2_state_mse])

    if metric_name == 'Squared Bias':
        avg_pred = pred_df.groupby('pair_idx')[['s1_predicted', 's2_predicted']].mean()
        s1 = (avg_pred['s1_predicted'].values - paired_data['s1_mean']) ** 2
        s2 = (avg_pred['s2_predicted'].values - paired_data['s2_mean']) ** 2
        return np.concatenate([s1, s2])

    s1 = pred_df.groupby('pair_idx')['s1_predicted'].var().fillna(0).values
    s2 = pred_df.groupby('pair_idx')['s2_predicted'].var().fillna(0).values
    return np.concatenate([s1, s2])


def _compute_metric_values_difference_mode(pred_df, diff_means, metric_name):
    """Compute per-pair metric values for difference mode."""
    if metric_name == 'MSE':
        pair_indices = pred_df['pair_idx'].to_numpy(dtype=int)
        diff_gt = diff_means[pair_indices]
        sq_err = (pred_df['diff_predicted'].to_numpy() - diff_gt) ** 2
        return pd.Series(sq_err).groupby(pred_df['pair_idx']).mean().values

    if metric_name == 'Squared Bias':
        avg_pred = pred_df.groupby('pair_idx')['diff_predicted'].mean().values
        return (avg_pred - diff_means) ** 2

    return pred_df.groupby('pair_idx')['diff_predicted'].var().fillna(0).values


def plot_paired_log_metric_evolution(filtered_metadata, methods, adjust_constant=False, mode='full', scale_mode='log'):
    """Plot evolution of mean metrics across training sizes for paired evaluations."""
    n_episodes_values = sorted(filtered_metadata['n_episodes'].unique())
    if scale_mode == 'log':
        metric_specs = [
            ('MSE', 'Mean log10(MSE)'),
            ('Squared Bias', 'Mean log10(Bias²)'),
            ('Variance', 'Mean log10(Variance)'),
        ]
    else:
        metric_specs = [
            ('MSE', 'Mean MSE'),
            ('Squared Bias', 'Mean Bias²'),
            ('Variance', 'Mean Variance'),
        ]

    records = []
    for n_ep in n_episodes_values:
        paired_data, predictions_data = load_paired_inputs_for_n_episodes(
            filtered_metadata, methods, n_ep, adjust_constant=adjust_constant
        )
        if paired_data is None or not predictions_data:
            continue

        s1_gt = paired_data['s1_mean']
        s2_gt = paired_data['s2_mean']
        diff_gt = paired_data['diff_mean']

        # Load pivots and compute bootstrap stderrs per method
        filtered_for_n_ep = filtered_metadata[filtered_metadata['n_episodes'] == n_ep]
        pivots_by_method = {}
        for _, row in filtered_for_n_ep.iterrows():
            if row['method'] not in predictions_data:
                continue
            paired_path = Path(row['predictions_path']).parent / "paired_predictions.parquet"
            if paired_path.exists():
                pivots_by_method[row['method']] = _load_paired_batch_pivots(
                    str(paired_path), row['predictions_path'],
                    adjust_constant, row.get('policy_gamma', 0.99),
                    row.get('truncation_coefficient', 5.0)
                )

        for method, pred_df in sort_predictions(predictions_data).items():
            for metric_name, _ in metric_specs:
                if mode == 'full':
                    values = _compute_metric_values_full_mode(pred_df, paired_data, metric_name)
                else:
                    values = _compute_metric_values_difference_mode(pred_df, paired_data['diff_mean'], metric_name)

                if len(values) == 0:
                    continue

                if scale_mode == 'log':
                    transformed_values = np.log10(values + 1e-10)
                else:
                    transformed_values = values

                if method in pivots_by_method:
                    s1_pivot, s2_pivot = pivots_by_method[method]
                    stderr = _paired_bootstrap_stderr(
                        s1_pivot, s2_pivot, metric_name, mode,
                        s1_gt, s2_gt, diff_gt, scale_mode
                    )
                else:
                    stderr = float(np.std(transformed_values) / np.sqrt(len(transformed_values)))

                records.append({
                    'Method': get_method_display_name(method),
                    'n_episodes': n_ep,
                    'metric': metric_name,
                    'mean_value': float(np.mean(transformed_values)),
                    'stderr': stderr
                })

    if not records:
        st.warning("No paired data available to plot metric evolution across training sizes.")
        return

    evolution_df = pd.DataFrame(records)
    cols = st.columns(3)

    for col, (metric_name, y_label) in zip(cols, metric_specs):
        with col:
            fig = go.Figure()
            metric_df = evolution_df[evolution_df['metric'] == metric_name]

            for method in sort_methods(methods):
                method_display = get_method_display_name(method)
                method_data = metric_df[metric_df['Method'] == method_display]
                if method_data.empty:
                    continue

                fig.add_trace(go.Scatter(
                    x=method_data['n_episodes'],
                    y=method_data['mean_value'],
                    error_y=dict(type='data', array=method_data['stderr']),
                    mode='lines+markers',
                    name=method_display,
                    marker=dict(size=8)
                ))

            fig.update_layout(
                title=y_label,
                xaxis_title="Training Episodes",
                yaxis_title=f"{y_label} (± stderr)",
                height=380,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, width='stretch')


def render_tab(filtered_metadata, methods, baseline_method, adjust_constant=False):
    """Render the paired states tab.

    Args:
        filtered_metadata: DataFrame with experiment metadata
        methods: List of methods to display
        baseline_method: Baseline method name (not used here, passed for consistency)
        adjust_constant: If True, add constant so mean(predictions) = mean(ground_truth)
    """
    st.header("🎯 Paired State Evaluation")

    # Mode selection
    mode = st.radio(
        "Evaluation Mode:",
        options=['full', 'difference'],
        format_func=lambda x: {
            'full': 'Full Dataset (Individual States)',
            'difference': 'Difference Dataset (V(s₁) - V(s₂))'
        }[x],
        help="Full: evaluate V(s) for each state independently | Difference: evaluate V(s₁) - V(s₂) for pairs"
    )

    scale_mode = st.radio(
        "Scale Mode:",
        options=['log', 'normal'],
        format_func=lambda x: 'Log Scale' if x == 'log' else 'Normal Scale',
        horizontal=True,
        key='paired_scale_mode'
    )

    st.markdown("---")

    # Training size selection
    n_episodes_values = sorted(filtered_metadata['n_episodes'].unique())
    selected_n_ep = st.selectbox(
        "Training data size:",
        n_episodes_values,
        format_func=lambda x: f"{x} episodes",
        key="paired_n_ep"
    )

    # Load paired state data
    filtered_for_n_ep = filtered_metadata[filtered_metadata['n_episodes'] == selected_n_ep]

    if filtered_for_n_ep.empty:
        st.error(f"No data for {selected_n_ep} episodes")
        return

    paired_data, predictions_data = load_paired_inputs_for_n_episodes(
        filtered_metadata, methods, selected_n_ep, adjust_constant=adjust_constant
    )

    if paired_data is None:
        st.warning(f"No paired state data found. Generate it with: `uv run -m src.generate_data --config <config> --generate-paired`")
        return

    if not predictions_data:
        st.warning("No paired prediction data available yet. Run evaluation to generate paired predictions: `uv run -m src.evaluate --config <config>`")
        st.info("Note: Regular evaluation batch predictions exist, but paired state predictions require a separate data generation step.")
        # Still show ground truth statistics even without predictions

    if mode == 'full':
        render_full_dataset_mode(paired_data, predictions_data, methods, selected_n_ep, filtered_metadata, adjust_constant, scale_mode)
    else:
        render_difference_mode(paired_data, predictions_data, methods, selected_n_ep, filtered_metadata, adjust_constant, scale_mode)


def render_full_dataset_mode(paired_data, predictions_data, methods, n_episodes, filtered_metadata, adjust_constant, scale_mode):
    """Render full dataset mode: V(s) for each state independently."""

    st.subheader("Individual State Evaluation")
    st.markdown("Evaluates V(s) for each state independently against ground truth (mean ± CI)")

    # Extract ground truth for all states (s1 and s2 combined)
    n_pairs = len(paired_data['pair_indices'])

    # Combine s1 and s2 into single dataset
    all_gt_means = np.concatenate([paired_data['s1_mean'], paired_data['s2_mean']])
    all_gt_ci_lower = np.concatenate([paired_data['s1_ci_lower'], paired_data['s2_ci_lower']])
    all_gt_ci_upper = np.concatenate([paired_data['s1_ci_upper'], paired_data['s2_ci_upper']])

    # For now, show ground truth statistics
    st.markdown("### Ground Truth Statistics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total States", len(all_gt_means))
    with col2:
        st.metric("Mean Value", f"{np.mean(all_gt_means):.2f}")
    with col3:
        avg_ci_width = np.mean(all_gt_ci_upper - all_gt_ci_lower)
        st.metric("Avg CI Width", f"{avg_ci_width:.2f}")

    # Histogram of ground truth values
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=all_gt_means,
        nbinsx=30,
        name="Ground Truth Values",
        marker_color='blue',
        opacity=0.7
    ))
    fig.update_layout(
        title=f"Distribution of Ground Truth Values ({n_episodes} episodes)",
        xaxis_title="V(s) - Ground Truth Mean",
        yaxis_title="Count",
        height=400
    )
    st.plotly_chart(fig, width='stretch')

    # CI width distribution
    ci_widths = all_gt_ci_upper - all_gt_ci_lower
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ci_widths,
        nbinsx=30,
        name="CI Widths",
        marker_color='green',
        opacity=0.7
    ))
    fig.update_layout(
        title="Distribution of Confidence Interval Widths",
        xaxis_title="CI Width (95%)",
        yaxis_title="Count",
        height=400
    )
    st.plotly_chart(fig, width='stretch')

    # If we have predictions, show evaluation metrics
    if predictions_data:
        predictions_data = sort_predictions(predictions_data)
        st.markdown("### Prediction Evaluation")

        # Compute metrics for each method
        metrics_data = []
        for method, pred_df in predictions_data.items():
            avg_pred = pred_df.groupby('pair_idx')[['s1_predicted', 's2_predicted']].mean()
            all_pred = np.concatenate([avg_pred['s1_predicted'].values, avg_pred['s2_predicted'].values])

            errors = all_pred - all_gt_means
            mse = np.mean(errors**2)
            mae = np.mean(np.abs(errors))

            metrics_data.append({
                'Method': get_method_display_name(method),
                'MSE': mse,
                'MAE': mae,
            })

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df.style.format({
            'MSE': '{:.4f}',
            'MAE': '{:.4f}',
        }), width='stretch')

        # Histograms: per-state MSE, squared bias, variance distributions
        metric_specs = [
            ('MSE', 'Per-State MSE Distribution'),
            ('Squared Bias', 'Per-State Squared Bias Distribution'),
            ('Variance', 'Per-State Variance Distribution'),
        ]
        for metric_name, base_title in metric_specs:
            scale_suffix = ' (Log Scale)' if scale_mode == 'log' else ' (Normal Scale)'
            title = f"{base_title}{scale_suffix}"
            x_col_name = f"log10({metric_name})" if scale_mode == 'log' else metric_name
            st.markdown(f"### {title}")

            # Add explanation for MSE
            if metric_name == 'MSE':
                st.info("**Note:** MSE histogram shows per-state mean squared error, where each point is "
                        "the batch-average of $(V_i(s) - V_{true}(s))^2$ for a fixed state.")
            hist_rows = []
            for method, pred_df in predictions_data.items():
                if metric_name == 'MSE':
                    values = _compute_metric_values_full_mode(pred_df, paired_data, metric_name)

                elif metric_name == 'Squared Bias':
                    values = _compute_metric_values_full_mode(pred_df, paired_data, metric_name)

                else:  # Variance
                    values = _compute_metric_values_full_mode(pred_df, paired_data, metric_name)

                for v in values:
                    if scale_mode == 'log':
                        plotted_value = np.log10(v + 1e-10)
                    else:
                        plotted_value = v
                    hist_rows.append({'Method': get_method_display_name(method), x_col_name: plotted_value})
            hist_df = pd.DataFrame(hist_rows)
            col1, col2 = st.columns([3, 1])
            with col1:
                fig = px.histogram(hist_df, x=x_col_name, color='Method', nbins=40, opacity=0.7,
                                   barmode='overlay', title=f"{title} ({n_episodes} episodes)")
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')
            with col2:
                st.markdown("**Statistics**")
                summary = hist_df.groupby('Method')[x_col_name].agg(mean='mean', std='std').reset_index()
                st.dataframe(summary.style.format({'mean': '{:.4f}', 'std': '{:.4f}'}),
                             width='stretch', hide_index=True)

        evolution_header = "### Evolution of Mean Log Metrics Across Training Sizes" if scale_mode == 'log' else "### Evolution of Mean Metrics Across Training Sizes"
        st.markdown(evolution_header)
        plot_paired_log_metric_evolution(
            filtered_metadata,
            methods,
            adjust_constant=adjust_constant,
            mode='full',
            scale_mode=scale_mode
        )

        # Scatter plot: Predictions vs Ground Truth for each method
        st.markdown("### Predictions vs Ground Truth")

        # Separate x and y axis ranges
        x_min, x_max = all_gt_ci_lower.min(), all_gt_ci_upper.max()
        x_pad = (x_max - x_min) * 0.10
        x_range = [x_min - x_pad, x_max + x_pad]

        y_min, y_max = x_min, x_max  # start from GT range for y too
        for pred_df in predictions_data.values():
            avg_pred = pred_df.groupby('pair_idx')[['s1_predicted', 's2_predicted']].mean()
            preds = np.concatenate([avg_pred['s1_predicted'].values, avg_pred['s2_predicted'].values])
            y_min = min(y_min, preds.min())
            y_max = max(y_max, preds.max())
        y_pad = (y_max - y_min) * 0.10
        y_range = [y_min - y_pad, y_max + y_pad]

        # Diagonal line spans the overlap of both ranges
        diag_min = min(x_range[0], y_range[0])
        diag_max = max(x_range[1], y_range[1])

        # Precompute horizontal error bar arrays (asymmetric: distance from mean to lower/upper)
        gt_err_lower = all_gt_means - all_gt_ci_lower
        gt_err_upper = all_gt_ci_upper - all_gt_means

        for method, pred_df in predictions_data.items():
            avg_pred = pred_df.groupby('pair_idx')[['s1_predicted', 's2_predicted']].mean()
            all_pred = np.concatenate([avg_pred['s1_predicted'].values, avg_pred['s2_predicted'].values])

            fig = go.Figure()

            # Scatter with horizontal 95% CI error bars on ground truth
            fig.add_trace(go.Scatter(
                x=all_gt_means,
                y=all_pred,
                mode='markers',
                marker=dict(size=5, opacity=0.5),
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=gt_err_upper,
                    arrayminus=gt_err_lower,
                    thickness=1,
                    width=0,
                    color='rgba(100,100,100,0.3)'
                ),
                name=get_method_display_name(method)
            ))

            # Add diagonal line
            fig.add_trace(go.Scatter(
                x=[diag_min, diag_max],
                y=[diag_min, diag_max],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Perfect Prediction',
                showlegend=True
            ))

            fig.update_layout(
                title=f"{get_method_display_name(method)} - Predictions vs Ground Truth",
                xaxis_title="Ground Truth Mean (± 95% CI)",
                yaxis_title="Predicted Value",
                xaxis=dict(range=x_range),
                yaxis=dict(range=y_range),
                height=400
            )
            st.plotly_chart(fig, width='stretch')


def render_difference_mode(paired_data, predictions_data, methods, n_episodes, filtered_metadata, adjust_constant, scale_mode):
    """Render difference mode: V(s₁) - V(s₂) for pairs."""

    st.subheader("Paired Difference Evaluation")
    st.markdown("Evaluates V(s₁) - V(s₂) for state pairs against ground truth (mean ± CI)")

    # Extract difference ground truth
    diff_means = paired_data['diff_mean']
    diff_ci_lower = paired_data['diff_ci_lower']
    diff_ci_upper = paired_data['diff_ci_upper']
    n_pairs = len(diff_means)

    # Ground truth statistics
    st.markdown("### Ground Truth Statistics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pairs", n_pairs)
    with col2:
        st.metric("Mean Difference", f"{np.mean(diff_means):.2f}")
    with col3:
        st.metric("Std Difference", f"{np.std(diff_means):.2f}")
    with col4:
        avg_ci_width = np.mean(diff_ci_upper - diff_ci_lower)
        st.metric("Avg CI Width", f"{avg_ci_width:.2f}")

    # Histogram of ground truth differences
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=diff_means,
        nbinsx=30,
        name="Ground Truth Differences",
        marker_color='purple',
        opacity=0.7
    ))

    # Add reference line at 0
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Zero Difference"
    )

    fig.update_layout(
        title=f"Distribution of V(s₁) - V(s₂) Ground Truth ({n_episodes} episodes)",
        xaxis_title="V(s₁) - V(s₂) - Ground Truth Mean",
        yaxis_title="Count",
        height=400
    )
    st.plotly_chart(fig, width='stretch')

    # Scatter plot: Individual state values
    st.markdown("### Individual State Values (s₁ vs s₂)")

    s1_means = paired_data['s1_mean']
    s2_means = paired_data['s2_mean']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s1_means,
        y=s2_means,
        mode='markers',
        marker=dict(
            size=8,
            color=diff_means,
            colorscale='RdBu',
            showscale=True,
            colorbar=dict(title="V(s₁)-V(s₂)")
        ),
        text=[f"Pair {i}<br>V(s₁)={s1:.2f}<br>V(s₂)={s2:.2f}<br>Diff={d:.2f}"
              for i, (s1, s2, d) in enumerate(zip(s1_means, s2_means, diff_means))],
        hovertemplate='%{text}<extra></extra>'
    ))

    # Add diagonal line
    min_val = min(s1_means.min(), s2_means.min())
    max_val = max(s1_means.max(), s2_means.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='V(s₁) = V(s₂)',
        showlegend=True
    ))

    fig.update_layout(
        title="Ground Truth: V(s₁) vs V(s₂)",
        xaxis_title="V(s₁) - Ground Truth Mean",
        yaxis_title="V(s₂) - Ground Truth Mean",
        height=500,
        width=500
    )
    st.plotly_chart(fig, width='stretch')

    # CI width distribution
    ci_widths = diff_ci_upper - diff_ci_lower
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ci_widths,
        nbinsx=30,
        name="CI Widths for Differences",
        marker_color='green',
        opacity=0.7
    ))
    fig.update_layout(
        title="Distribution of Confidence Interval Widths for Differences",
        xaxis_title="CI Width (95%) for V(s₁) - V(s₂)",
        yaxis_title="Count",
        height=400
    )
    st.plotly_chart(fig, width='stretch')

    # If we have predictions, show evaluation metrics
    if predictions_data:
        predictions_data = sort_predictions(predictions_data)
        st.markdown("### Prediction Evaluation for Differences")

        # Compute metrics for each method
        metrics_data = []
        for method, pred_df in predictions_data.items():
            avg_pred = pred_df.groupby('pair_idx')['diff_predicted'].mean().values

            errors = avg_pred - diff_means
            mse = np.mean(errors**2)
            mae = np.mean(np.abs(errors))

            metrics_data.append({
                'Method': get_method_display_name(method),
                'MSE': mse,
                'MAE': mae,
            })

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df.style.format({
            'MSE': '{:.4f}',
            'MAE': '{:.4f}',
        }), width='stretch')

        # Histograms: per-pair MSE, squared bias, variance distributions
        metric_specs = [
            ('MSE', 'Per-Pair MSE Distribution'),
            ('Squared Bias', 'Per-Pair Squared Bias Distribution'),
            ('Variance', 'Per-Pair Variance Distribution'),
        ]
        for metric_name, base_title in metric_specs:
            scale_suffix = ' (Log Scale)' if scale_mode == 'log' else ' (Normal Scale)'
            title = f"{base_title}{scale_suffix}"
            x_col_name = f"log10({metric_name})" if scale_mode == 'log' else metric_name
            st.markdown(f"### {title}")

            # Add explanation for MSE
            if metric_name == 'MSE':
                st.info("**Note:** MSE histogram shows per-pair mean squared error, where each point is "
                        "the batch-average of $(V_i(s_1)-V_i(s_2) - (V_{true}(s_1)-V_{true}(s_2)))^2$ for a fixed pair.")
            hist_rows = []
            for method, pred_df in predictions_data.items():
                if metric_name == 'MSE':
                    values = _compute_metric_values_difference_mode(pred_df, diff_means, metric_name)

                elif metric_name == 'Squared Bias':
                    values = _compute_metric_values_difference_mode(pred_df, diff_means, metric_name)

                else:  # Variance
                    values = _compute_metric_values_difference_mode(pred_df, diff_means, metric_name)

                for v in values:
                    if scale_mode == 'log':
                        plotted_value = np.log10(v + 1e-10)
                    else:
                        plotted_value = v
                    hist_rows.append({'Method': get_method_display_name(method), x_col_name: plotted_value})
            hist_df = pd.DataFrame(hist_rows)
            col1, col2 = st.columns([3, 1])
            with col1:
                fig = px.histogram(hist_df, x=x_col_name, color='Method', nbins=40, opacity=0.7,
                                   barmode='overlay', title=f"{title} ({n_episodes} episodes)")
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')
            with col2:
                st.markdown("**Statistics**")
                summary = hist_df.groupby('Method')[x_col_name].agg(mean='mean', std='std').reset_index()
                st.dataframe(summary.style.format({'mean': '{:.4f}', 'std': '{:.4f}'}),
                             width='stretch', hide_index=True)

        evolution_header = "### Evolution of Mean Log Metrics Across Training Sizes" if scale_mode == 'log' else "### Evolution of Mean Metrics Across Training Sizes"
        st.markdown(evolution_header)
        plot_paired_log_metric_evolution(
            filtered_metadata,
            methods,
            adjust_constant=adjust_constant,
            mode='difference',
            scale_mode=scale_mode
        )

        # Scatter plot: Predicted differences vs Ground Truth for each method
        st.markdown("### Predicted Differences vs Ground Truth")

        # Separate x and y axis ranges
        x_min, x_max = diff_ci_lower.min(), diff_ci_upper.max()
        x_pad = (x_max - x_min) * 0.10
        x_range = [x_min - x_pad, x_max + x_pad]

        y_min, y_max = x_min, x_max
        for pred_df in predictions_data.values():
            avg_pred = pred_df.groupby('pair_idx')['diff_predicted'].mean().values
            y_min = min(y_min, avg_pred.min())
            y_max = max(y_max, avg_pred.max())
        y_pad = (y_max - y_min) * 0.10
        y_range = [y_min - y_pad, y_max + y_pad]

        diag_min = min(x_range[0], y_range[0])
        diag_max = max(x_range[1], y_range[1])

        # Precompute horizontal error bar arrays (asymmetric)
        diff_err_lower = diff_means - diff_ci_lower
        diff_err_upper = diff_ci_upper - diff_means

        for method, pred_df in predictions_data.items():
            avg_pred = pred_df.groupby('pair_idx')['diff_predicted'].mean().values

            fig = go.Figure()

            # Scatter with horizontal 95% CI error bars on ground truth
            fig.add_trace(go.Scatter(
                x=diff_means,
                y=avg_pred,
                mode='markers',
                marker=dict(size=5, opacity=0.5),
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=diff_err_upper,
                    arrayminus=diff_err_lower,
                    thickness=1,
                    width=0,
                    color='rgba(100,100,100,0.3)'
                ),
                name=get_method_display_name(method)
            ))

            # Add diagonal line
            fig.add_trace(go.Scatter(
                x=[diag_min, diag_max],
                y=[diag_min, diag_max],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Perfect Prediction',
                showlegend=True
            ))

            # Add zero line
            fig.add_hline(y=0, line_dash="dot", line_color="red", opacity=0.5)
            fig.add_vline(x=0, line_dash="dot", line_color="red", opacity=0.5)

            fig.update_layout(
                title=f"{get_method_display_name(method)} - V(s₁) - V(s₂) Predictions vs Ground Truth",
                xaxis_title="Ground Truth Difference Mean (± 95% CI)",
                yaxis_title="Predicted Difference",
                xaxis=dict(range=x_range),
                yaxis=dict(range=y_range),
                height=400
            )
            st.plotly_chart(fig, width='stretch')
