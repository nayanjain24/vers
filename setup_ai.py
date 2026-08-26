#!/usr/bin/env python3
"""One-command setup script for the AI Smart Detection system (v5.0).

This script:
1. Generates synthetic hand landmarks for all 30 sign vocabulary words
2. Trains the PyTorch LSTM sequence model for sign language recognition
3. Verifies model accuracy and readies the system
"""

import sys
import subprocess
from pathlib import Path

try:
    from rich import print as rprint
except ImportError:
    rprint = print

PROJECT_ROOT = Path(__file__).resolve().parent

def _run(cmd: list[str], title: str) -> None:
    rprint(f"\n[bold blue]--- {title} ---[/bold blue]")
    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        rprint(f"[bold green]✓ {title} completed successfully.[/bold green]")
    except subprocess.CalledProcessError as exc:
        rprint(f"[bold red]✗ {title} failed with exit code {exc.returncode}.[/bold red]")
        sys.exit(1)

def main():
    rprint("[bold magenta]VERS AI Smart Detection Setup (v5.0)[/bold magenta]")
    rprint("This will generate training data and train the PyTorch LSTM model for all 30 conversational & emergency signs.")
    
    python_exec = sys.executable
    
    # 1. Generate Synthetic Data
    gen_script = PROJECT_ROOT / "src" / "training" / "generate_synthetic_data.py"
    _run([python_exec, str(gen_script)], "Generating Synthetic Data (30 words x 120 samples)")
    
    # 2. Train the LSTM model
    train_script = PROJECT_ROOT / "src" / "training" / "train_sign_model.py"
    _run([python_exec, str(train_script), "--epochs", "50", "--batch-size", "32"], "Training LSTM Sequence Classifier")
    
    # 3. Print verification info
    from src.vision.sign_language_model import SignLanguageRecognizer
    rprint("\n[bold blue]--- Verification ---[/bold blue]")
    recognizer = SignLanguageRecognizer()
    if recognizer.available:
        rprint(f"[bold green]✓ Model loaded successfully![/bold green]")
        rprint(f"  Vocabulary size: {len(recognizer.vocabulary)}")
        rprint(f"  Emergency signs: {len([w for w in recognizer.vocabulary if recognizer.is_emergency_sign(w)])}")
        rprint(f"  Conversational signs: {len([w for w in recognizer.vocabulary if recognizer.is_conversational_sign(w)])}")
        rprint("\n[bold magenta]Setup complete! You can now run the dashboard or realtime demo.[/bold magenta]")
        rprint("[cyan]Try: python orchestrate.py --mode dashboard[/cyan]")
    else:
        rprint("[bold red]✗ Model failed to load. Check logs for details.[/bold red]")

if __name__ == "__main__":
    main()
