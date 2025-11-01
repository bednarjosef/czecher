import wandb

def log_step_wandb(step_eval: dict, lr: float, epochs: int):
    # step_eval = { 'threshold': 0.5, 'precision': 0, 'recall': 0, 'f1': -1, 'loss': 0.5}
    run = wandb.init(
        entity="czecher-team",
        project="czecher-commas",
        config={
            "learning_rate": lr,
            "architecture": "Transformer",
            "dataset": "dataset_500k",
            "epochs": epochs,
        },
    )
    pass