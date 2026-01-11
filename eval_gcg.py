"""
Run GCG on specific advbench samples, using two different implementations.
"""
import math
import os
import pandas as pd
import wandb
from attack_evaluate.evaluate_jailbreakness import evaluate_triggers
from nanogcg import GCGConfig, GCG
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import typer
from typing import List
from tropt.loss.base import PrefillCELoss
from tropt.models.huggingface.lm import LMHFModel
from tropt.optimizer.base import OptimizerResult
from tropt.optimizer.gcg_optimizer import GCGOptimizer
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.tracker.base import WandbTracker

import numpy as np
import matplotlib.pyplot as plt

# Default list from original script
DEFAULT_INDICES = [ 
    225, 89, 155, 78, 283, 162, 236, 389, 568, 695,
    128, 342, 456, 509, 634, 721, 
]
INITIAL_TRIGGER = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
RANDOM_SEED = 42
WANDB_ENTITY = "matanbt"
WANDB_PROJECT = "gcg_eval"
DATASET_PATH = "./advbench_plus.csv"
app = typer.Typer()

# TODO these runs should be:
# {google/gemma-2-2b-it,Qwen/Qwen3-8B} x {3 seeds} x {15 samples}

@app.command()
def nanogcg(
    model_name: str = typer.Option("google/gemma-2-2b-it", help="HuggingFace model identifier"),
    sample_indices: List[int] = typer.Option(
        DEFAULT_INDICES, 
        help="Specific indices to run. Usage: --sample-indices 225 --sample-indices 89"
    ),
    seed: int = typer.Option(RANDOM_SEED, help="Random seed for reproducibility"),
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load CSV
    df = pd.read_csv(DATASET_PATH)
    df['message_id'] = range(len(df))

    # Load model and tokenizer
    print(f"Loading model {model_name} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype='bfloat16',
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loaded model {model_name} with dtype {model.dtype}.")

    for message_id in sample_indices:
        # Filter dataframe for the specific index
        row = df.loc[df.message_id == message_id].iloc[0]
        
        print(f"running: advbench[{model_name.split('/')[-1]}, {message_id}]")

        # Load data safely accessing the scalar values
        messages = row['message_template'].replace('{{OPTIMIZED_TRIGGER}}', '{optim_str}')
        target = row['target_response_prefix']
        run_name = f'nanogcg[{model_name.split("/")[-1]},m={message_id}]'
        if seed != RANDOM_SEED:
            run_name += f'_s={seed}'
        wandb_metadata = dict(
            optimized_message_id=message_id,
            name=run_name,
            model_name=model_name,
            implementation='nanogcg',
            method='gcg',
            random_seed=seed,
        )

        # Set GCG parameters
        config = GCGConfig(
            optim_str_init=INITIAL_TRIGGER,
            seed=seed,
            num_steps=500,
            search_width=512,
            topk=256,
            n_replace=1,
            use_prefix_cache=False,
            wandb_log=True
        )
        
        # Initialize wandb
        wandb.init(
            name=wandb_metadata.get('name', "nanogcg"),
            tags=['nanogcg', 'gcg'],
            project=WANDB_PROJECT, 
            entity=WANDB_ENTITY,
            config={**wandb_metadata},
        )

        # Run GCG
        gcg = GCG(model, tokenizer, config)
        result = gcg.run(messages, target, wandb_metadata)

        wandb.finish()


@app.command()
def tropt(
    model_name: str = typer.Option("google/gemma-2-2b-it", help="HuggingFace model identifier"),
    sample_indices: List[int] = typer.Option(
        DEFAULT_INDICES, 
        help="Specific indices to run. Usage: --sample-indices 225 --sample-indices 89"
    ),
    method: str = typer.Option("gcg", help="Which optimier to use"),
    seed: int = typer.Option(RANDOM_SEED, help="Random seed for reproducibility"),
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load CSV
    df = pd.read_csv(DATASET_PATH)
    df['message_id'] = range(len(df))

    # Load model and tokenizer
    model = LMHFModel(
        model_name=model_name,
        device=device,
        forward_pass_batch_size=1024,
        use_prefix_cache=False,
        dtype='bfloat16',
    )
    loss = PrefillCELoss()

    for message_id in sample_indices:

       # Filter dataframe for the specific index
        row = df.loc[df.message_id == message_id].iloc[0]
        
        print(f"running: advbench[{model_name.split('/')[-1]}, {message_id}]")

        # Load data safely accessing the scalar values
        messages = row['message_template']
        target = row['target_response_prefix']
        run_name = f'tropt[{method},{model_name.split("/")[-1]},m={message_id}]'
        if seed != RANDOM_SEED:
            run_name += f'_s={seed}'
        wandb_metadata = dict(
            optimized_message_id=message_id,
            name=run_name,
            model_name=model_name,
            implementation='tropt',
            method=method,
            random_seed=seed,
        )

        # Run GCG
        tracker = WandbTracker(
            wandb_metadata.get('name', "tropt"),
            tags=['tropt', method],
            project_name=WANDB_PROJECT, 
            entity=WANDB_ENTITY,
            config_dump=wandb_metadata
        )
        if method == 'gcg':
            optimizer = GCGOptimizer(
                model=model,
                loss=loss,
                seed=seed,
                tracker=tracker,
                # Set parameters from the paper:
                num_steps=500,
                n_candidates=512,
                sample_topk=256,
                sample_n_replace=1,
                token_constraints=TokenConstraints(
                    disallow_non_ascii=True, disallow_special_tokens=True
                ),
                use_retokenize=True,
            )
        elif method == 'gslt':
            # TODO 
            from tropt.optimizer.gaslite_optimizer import GASLITEOptimizer
            optimizer = GASLITEOptimizer(
                model=model,
                loss=loss,
                seed=seed,
                tracker=tracker,
                
                num_steps=100,
                n_grad=10,
                n_flip=0.3,
                n_candidates=128,
                token_constraints=TokenConstraints(
                    disallow_non_ascii=True, disallow_special_tokens=True
                ),
                use_retokenize=True,
            )
        elif method == 'qgslt+2':
            from tropt.optimizer.gasliteplus_optimizer import GASLITEPlusOptimizer
            optimizer = GASLITEPlusOptimizer(
                model=model,
                loss=loss,
                seed=seed,
                tracker=tracker,
                
                num_steps=100,
                n_grad=5,
                n_flip=10,
                n_candidates=128,
                buffer_size=10,
                decline_n_flip_from_step=0.5,
                early_stopping_patience=100,  # effectively disabled
                n_bulk_flips=10,
                flip_pos_method="ordered",
                token_constraints=TokenConstraints(
                    disallow_non_ascii=True, disallow_special_tokens=True
                ),
                use_retokenize=True,
            )

        result = optimizer.optimize_trigger(
            texts=[messages],
            targets=dict(target_outputs=[target]),
            initial_trigger=INITIAL_TRIGGER,
        )
        tracker.finish()

    
@app.command()
def eval_jailbreak_results(
    model_name: str = typer.Option("google/gemma-2-2b-it", help="HuggingFace model identifier"),
    output_path: str = typer.Option("evaluation_results.csv", help="Where to save the results"),
):

    api = wandb.Api()
    runs = api.runs(
        path=f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"state": "finished"}
    )

    metadata_df = []
    
    for run in runs:
        # Extract the optimized trigger. 
        trigger = run.summary['best_trigger_str']
        metadata_df.append(dict(
            trigger=trigger,
            target_model=run.config['model_name'],
            optimized_message_id=run.config['optimized_message_id'],
            method=run.config['method'],
            implementation=run.config['implementation'],
            random_seed=run.config.get('random_seed', None),
        ))

    metadata_df = pd.DataFrame(metadata_df)
    metadata_df['trigger_id'] = range(len(metadata_df))

    # Load Model
    print(f"Loading model {model_name} for evaluation...")
    model = LMHFModel(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Run Evaluation
    eval_df = evaluate_triggers(  # TODO get from tropts
        model=model,
        trigger_strs=metadata_df['trigger'].tolist(),
        trigger_ids=metadata_df['trigger_id'].tolist(),
        eval_dataset_path=DATASET_PATH,
        batch_size=16
    )
    eval_df['eval_model'] = model_name

    # Merge the DFs by the trigger index
    final_df = pd.merge(
        eval_df, 
        metadata_df, 
        on='trigger_id', 
        how='left'
    )

    # Save
    final_df.to_csv(output_path, index=False)
    print(f"Evaluation complete. Results saved to {output_path}")

# ---------------------------------------------
# Plots
# ---------------------------------------------
@app.command()
def analyze_loss_step_progress(
    message_id: int = typer.Option(225, help="The message ID to analyze (defaults to first in list)"),
    model_name: str = typer.Option("google/gemma-2-2b-it", help="Model name filter"),
    output_plot: str = typer.Option("gcg_comparison.png", help="Path to save the resulting plot"),
    max_time_mins: int = typer.Option(None, help="Cutoff time in minutes (optional)"),
):
    """
    Plots Loss vs Time comparison for NanoGCG and TROPT for a specific message,
    averaging across seeds and showing STD shadows.
    """
    print(f"Fetching runs for Message ID: {message_id}, Model: {model_name}...")
    
    api = wandb.Api()
    
    # "project" is implied by the path in api.runs(), so we remove it from filters.
    filters = {
        "config.optimized_message_id": message_id,
        "config.model_name": model_name,
        "state": "finished"
    }
    
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters=filters)
    
    print(f"Found {len(runs)} runs.")
    
    # Store data structure: data['nanogcg'] = [ (times_array, loss_array), ... ]
    data = {'nanogcg': [], 'tropt': []}
    
    print(f"Found {len(runs)} runs total. Downloading history...")
    
    for run in runs:
        impl = run.config.get('implementation')
        if impl not in data: 
            continue
            
        # Fetch history (loss and runtime). _runtime is in seconds.
        # We ask for a high number of samples to get granular step data
        hist = run.history(keys=['loss', '_runtime'], samples=10000)
        
        if 'loss' in hist and '_runtime' in hist:
            # Clean data: drop NaNs just in case
            hist = hist.dropna(subset=['loss', '_runtime'])
            
            # Convert seconds to minutes
            times = hist['_runtime'].values / 60.0
            losses = hist['loss'].values
            
            # Sort by time just in case wandb returns unsorted
            sort_idx = np.argsort(times)
            data[impl].append((times[sort_idx], losses[sort_idx]))

    # --- Plotting ---
    plt.figure(figsize=(6, 4))
    colors = {'nanogcg': '#1f77b4', 'tropt': '#ff7f0e'} # Blue and Orange
    
    # Determine common time grid
    # We find the max duration across all runs to define the x-axis limits
    all_max_times = []
    for impl_data in data.values():
        for t, _ in impl_data:
            all_max_times.append(t.max())
            
    if not all_max_times:
        print("No valid history data found.")
        return

    # Create a common time axis (e.g., 500 points from 0 to max_time)
    limit_time = max_time_mins if max_time_mins else max(all_max_times)
    common_time_grid = np.linspace(0, limit_time, 500)

    for impl, runs_data in data.items():
        if not runs_data:
            print(f"No runs found for implementation: {impl}")
            continue
            
        interpolated_losses = []
        
        for t, l in runs_data:
            # Interpolate loss onto the common time grid
            # We use np.interp (linear interpolation)
            # For times past the run's end, we can either pad with NaN or the last value.
            # Here we assign NaN to avoid misleading flat lines if one seed died early.
            
            interp_val = np.interp(common_time_grid, t, l, left=np.nan, right=np.nan)
            
            # If the run ended early, np.interp with right=NaN will fill the tail with NaN
            # However, usually optimizations 'converge', so extending the last value 
            # might be valid, but 'NaN' is safer for strictly comparing "time spent".
            interpolated_losses.append(interp_val)
            
        # Convert to numpy array: shape (num_seeds, 500)
        interpolated_losses = np.array(interpolated_losses)
        
        # Compute Mean and STD (ignoring NaNs to handle different run lengths)
        mean_loss = np.nanmean(interpolated_losses, axis=0)
        std_loss = np.nanstd(interpolated_losses, axis=0)
        
        # Plot Mean
        plt.plot(common_time_grid, mean_loss, label=f"{impl} (mean)", color=colors[impl], linewidth=2)
        
        # Plot STD Shadow
        plt.fill_between(
            common_time_grid, 
            mean_loss - std_loss, 
            mean_loss + std_loss, 
            color=colors[impl], 
            alpha=0.2,
            label=f"{impl} (±1 std)"
        )

    plt.xlabel("Time (Minutes)")
    plt.ylabel("Loss")
    plt.title(f"GCG Optimization Progress: {model_name.split('/')[-1]} (Msg ID: {message_id})")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(output_plot)
    print(f"Analysis complete. Plot saved to {output_plot}")

@app.command()
def analyze_loss_step_all_grid(
    model_name: str = typer.Option("google/gemma-2-2b-it", help="Model name filter"),
    output_plot: str = typer.Option("gcg_grid_comparison.png", help="Path to save the resulting plot"),
    max_time_mins: int = typer.Option(None, help="Cutoff time in minutes (optional)"),
    sample_indices: List[int] = typer.Option(DEFAULT_INDICES, help="Indices to include in the grid")
):
    """
    Plots a grid of Loss vs Time comparisons for all specified messages in one image.
    """
    # 1. Setup Grid
    num_plots = len(sample_indices)
    cols = 4
    rows = math.ceil(num_plots / cols)
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() # Flatten for easy indexing
    
    colors = {'nanogcg': '#1f77b4', 'tropt': '#ff7f0e'} 
    api = wandb.Api()
    
    print(f"Generating grid for {num_plots} messages...")

    # 2. Iterate through every message ID
    for i, message_id in enumerate(sample_indices):
        ax = axes[i]
        print(f"Processing Msg ID: {message_id} ({i+1}/{num_plots})...")
        
        # --- Fetch Data (Same logic as single plot) ---
        filters = {
            "config.optimized_message_id": int(message_id),
            "config.model_name": model_name,
            # "state": "finished" # Optional
        }
        # Use path kwarg to avoid filter issues
        runs = api.runs(path=f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters=filters)
        
        data = {'nanogcg': [], 'tropt': []}
        all_max_times = []
        
        for run in runs:
            impl = run.config.get('implementation')
            if impl not in data: continue
            
            # Fetch history, requesting enough samples
            hist = run.history(keys=['loss', '_runtime'], samples=5000)
            if 'loss' in hist and '_runtime' in hist:
                hist = hist.dropna(subset=['loss', '_runtime'])
                times = hist['_runtime'].values / 60.0
                losses = hist['loss'].values
                
                # Sort by time
                sort_idx = np.argsort(times)
                t_sorted = times[sort_idx]
                l_sorted = losses[sort_idx]
                
                if len(t_sorted) > 0:
                    data[impl].append((t_sorted, l_sorted))
                    all_max_times.append(t_sorted.max())

        # --- Plotting on specific Subplot (ax) ---
        if not all_max_times:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Msg {message_id}")
            continue

        limit_time = max_time_mins if max_time_mins else max(all_max_times)
        # Reduced resolution slightly for grid to keep file size managed
        common_time_grid = np.linspace(0, limit_time, 300) 

        for impl, runs_data in data.items():
            if not runs_data: continue
            
            interpolated_losses = []
            for t, l in runs_data:
                # Interpolate
                interp_val = np.interp(common_time_grid, t, l, left=np.nan, right=np.nan)
                interpolated_losses.append(interp_val)
            
            interpolated_losses = np.array(interpolated_losses)
            
            # Compute Stats
            mean_loss = np.nanmean(interpolated_losses, axis=0)
            std_loss = np.nanstd(interpolated_losses, axis=0)
            
            # Plot
            ax.plot(common_time_grid, mean_loss, label=impl, color=colors[impl], linewidth=1.5)
            ax.fill_between(
                common_time_grid, 
                mean_loss - std_loss, 
                mean_loss + std_loss, 
                color=colors[impl], 
                alpha=0.2
            )

        ax.set_title(f"Msg ID: {message_id}")
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Only add labels to edges to reduce clutter
        if i % cols == 0:
            ax.set_ylabel("Loss")
        if i >= (rows - 1) * cols:
            ax.set_xlabel("Time (m)")

    # 3. Cleanup
    # Hide empty subplots if the grid isn't full
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Add a single global legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"Grid analysis complete. Saved to {output_plot}")

@app.command()
def compare_asr_csv(
    csv_path: str = typer.Option("evaluation_results.csv", help="Path to the results CSV"),
):
    """
    Analyzes the evaluation CSV and prints a Markdown table comparing NanoGCG vs TROPT.
    """
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # 1. Normalize Column Names
    # We look for common names for the loss column returned by evaluate_triggers
    loss_col = next((c for c in ['loss', 'eval_loss', 'target_loss'] if c in df.columns), None)
    success_col = next((c for c in ['success', 'is_success'] if c in df.columns), None)
    
    if not loss_col:
        print("Error: Could not find a 'loss' column in the CSV.")
        return

    print(f"Analyzing {len(df)} records using loss column: '{loss_col}'...\n")

    # 2. Global Summary Table
    # Group by implementation to get mean/std across ALL runs
    summary = df.groupby('implementation')[loss_col].agg(['mean', 'std', 'count'])
    if success_col:
        summary['success_rate'] = df.groupby('implementation')[success_col].mean() * 100

    print("### 📊 Global Summary")
    print("| Implementation | Mean Loss | Std Dev | Samples |" + (" Success Rate |" if success_col else ""))
    print("| :--- | :--- | :--- | :--- |" + (" :--- |" if success_col else ""))
    
    for impl, row in summary.iterrows():
        success_str = f" {row['success_rate']:.1f}% |" if success_col else ""
        print(f"| **{impl}** | {row['mean']:.4f} | {row['std']:.4f} | {int(row['count'])} |{success_str}")
    print("\n" + "-"*40 + "\n")

    # 3. Per-Message Breakdown (Pivot Table)
    # We average the 3 seeds for each message_id
    grouped = df.groupby(['optimized_message_id', 'implementation'])[loss_col].mean().reset_index()
    
    # Pivot: Index=MsgID, Columns=Implementation, Values=Loss
    pivot = grouped.pivot(index='optimized_message_id', columns='implementation', values=loss_col)
    
    # Calculate Delta (assuming we have both keys)
    if 'nanogcg' in pivot.columns and 'tropt' in pivot.columns:
        pivot['delta'] = pivot['nanogcg'] - pivot['tropt'] # Positive = TROPT is better (lower loss)
        pivot['winner'] = pivot['delta'].apply(lambda x: 'TROPT' if x > 0 else 'NanoGCG')
    
    print("### 🧐 Per-Message Breakdown (Avg over seeds)")
    headers = "| Msg ID | NanoGCG Loss | TROPT Loss | Delta (Nano - TROPT) | Winner |"
    print(headers)
    print("| :--- | :--- | :--- | :--- | :--- |")

    for msg_id, row in pivot.iterrows():
        n_loss = row.get('nanogcg', float('nan'))
        t_loss = row.get('tropt', float('nan'))
        
        # Formatting for missing data
        n_str = f"{n_loss:.4f}" if not pd.isna(n_loss) else "N/A"
        t_str = f"{t_loss:.4f}" if not pd.isna(t_loss) else "N/A"
        
        delta_str = ""
        winner_str = ""
        
        if not pd.isna(n_loss) and not pd.isna(t_loss):
            delta = n_loss - t_loss
            # Bold the winner logic
            if delta > 0.001: # TROPT wins significantly
                delta_str = f"🟢 +{delta:.4f}"
                winner_str = "**TROPT**"
            elif delta < -0.001: # NanoGCG wins significantly
                delta_str = f"🔴 {delta:.4f}"
                winner_str = "NanoGCG"
            else:
                delta_str = "0.0000"
                winner_str = "Tie"
        
        print(f"| {msg_id} | {n_str} | {t_str} | {delta_str} | {winner_str} |")

    print("\n*Note: Positive Delta (🟢) means TROPT achieved lower loss.*")


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

@app.command()
def analyze_results(
    input_path: str= typer.Option("evaluation_results.csv", help="Path to the evaluation results CSV"),
    output_csv_path: str= typer.Option("universality_scores.csv", help="Output CSV file path"),
    plot_path: str= typer.Option("universality_plot.png", help="Output image file path")):
    # 1. Load Data
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File {input_path} not found.")
        return

    # 2. Calculate Universality Score
    # Group by implementation and trigger_id, then average the metric across messages
    universality_df = df.groupby(['implementation', 'trigger_id'])['strongreject_finetuned'].mean().reset_index()
    universality_df.rename(columns={'strongreject_finetuned': 'universality_score'}, inplace=True)

    # 3. Save to CSV
    universality_df.to_csv(output_csv_path, index=False)
    print(f"Universality scores saved to: {output_csv_path}")

    # 4. Generate Plot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # Violin plot for distribution shape
    sns.violinplot(
        data=universality_df,
        x='implementation',
        y='universality_score',
        inner=None,  # Remove inner bars to keep it clean for the strip plot
        color='lightblue'
    )

    # Strip plot to show individual data points
    sns.stripplot(
        data=universality_df,
        x='implementation',
        y='universality_score',
        color='black',
        alpha=0.6,
        jitter=True
    )

    plt.title('Universality Score Distribution by GCG Implementation', fontsize=14)
    plt.ylabel('Universality Score (Avg StrongReject)', fontsize=12)
    plt.xlabel('Implementation', fontsize=12)
    
    # Save Plot
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Violin plot saved to: {plot_path}")


if __name__ == "__main__":
    app()
