"""
Run GCG on specific advbench samples, using two different implementations.
"""
import pandas as pd
import wandb
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


@app.command()
def nanogcg(
    model_name: str = typer.Option("google/gemma-2-2b-it", help="HuggingFace model identifier"),
    sample_indices: List[int] = typer.Option(
        DEFAULT_INDICES, 
        help="Specific indices to run. Usage: --sample-indices 225 --sample-indices 89"
    ),
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load CSV
    df = pd.read_csv(DATASET_PATH)
    df['message_id'] = range(len(df))

    # Load model and tokenizer
    print(f"Loading model {model_name} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
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
        wandb_metadata = dict(
            optimized_message_id=message_id,
            name=f'nanogcg[{model_name.split("/")[-1]},m={message_id}]',
            model_name=model_name,
            implementation='nanogcg',
            method='gcg',
        )

        # Set GCG parameters
        config = GCGConfig(
            optim_str_init=INITIAL_TRIGGER,
            seed=RANDOM_SEED,
            num_steps=500,
            search_width=512,
            topk=256,
            n_replace=1,
<<<<<<< HEAD
=======
            use_prefix_cache=False,
>>>>>>> 2d8697e (Add comprison script)
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

        print(result)

@app.command()
def tropt(
    model_name: str = typer.Option("google/gemma-2-2b-it", help="HuggingFace model identifier"),
    sample_indices: List[int] = typer.Option(
        DEFAULT_INDICES, 
        help="Specific indices to run. Usage: --sample-indices 225 --sample-indices 89"
    ),
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
    )
    loss = PrefillCELoss()

    for message_id in sample_indices:

       # Filter dataframe for the specific index
        row = df.loc[df.message_id == message_id].iloc[0]
        
        print(f"running: advbench[{model_name.split('/')[-1]}, {message_id}]")

        # Load data safely accessing the scalar values
        messages = row['message_template']
        target = row['target_response_prefix']
        wandb_metadata = dict(
            optimized_message_id=message_id,
            name=f'tropt[{model_name.split("/")[-1]},m={message_id}]',
            model_name=model_name,
            implementation='tropt',
            method='gcg',
        )

        # Run GCG
        tracker = WandbTracker(
            wandb_metadata.get('name', "tropt"),
            tags=['tropt', 'gcg'],
            project_name=WANDB_PROJECT, 
            entity=WANDB_ENTITY,
            config_dump=wandb_metadata
        )
        optimizer = GCGOptimizer(
            model=model,
            loss=loss,
            seed=RANDOM_SEED,
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

        result = optimizer.optimize_trigger(
            texts=[messages],
            targets=dict(target_outputs=[target]),
            initial_trigger=INITIAL_TRIGGER,
        )
        tracker.finish()

        print(result)
    
@app.command()
def eval_results(
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
        batch_size=64
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


if __name__ == "__main__":
    app()
