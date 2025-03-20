"""
Main module for fractal analysis of images.
Handles configuration, image processing, and results display.
"""
import hydra
from omegaconf import DictConfig
import sys
import pandas as pd
from pathlib import Path
sys.dont_write_bytecode = True

from src.fractal_analyzer.fractal_analyzer import FractalAnalyzer
from src.utils import (
    console, 
    verbose_print,
    create_result_table, 
    find_image_files,
    display_error,
    save_results
)
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, SpinnerColumn


@hydra.main(config_path="./config", config_name="main", version_base="1.2")
def main(config: DictConfig):
    """Process images with fractal analysis based on configuration."""
    
    # --- SETUP ---
    # Display header
    console.print(Panel.fit(
        f"🔍 [bold]Fractal Analysis[/bold]", 
        subtitle=f"Target: [cyan]{config.data.path}[/cyan]",
        border_style="blue"
    ))
    
    # Show verbosity level in verbose mode
    verbose_print(f"[dim]Verbosity level: {config.display.verbosity}[/dim]", 
                 min_level=2, current_level=config.display.verbosity)
    
    # --- PATH VALIDATION ---
    # Validate data path
    data_path = Path(config.data.path)
    if not data_path.exists():
        console.print(Panel(
            f"Data path not found: [bold]{data_path}[/bold]", 
            title="❌ Error", 
            border_style="red"
        ))
        return
    
    # Create results directory
    results_path = Path(config.results.path)
    results_path.mkdir(parents=True, exist_ok=True)
    verbose_print(f"[dim]Results will be saved to: {results_path}[/dim]", 
                 min_level=2, current_level=config.display.verbosity)

    console.print("\n")
    console.rule("[bold green] Starting Images Processing [/bold green]")

    # Create custom progress bar columns
    progress_columns = [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ]

    # Loop through each TASK_** folder
    for task_folder in data_path.glob("TASK*"):
        if task_folder.is_dir():
            t = task_folder.name
            console.print(f"[bold]Processing task:[/bold] [cyan]{t}[/cyan]")

            t_feature_dicts = []
            ids = []

            # Get all image files in this task folder
            image_files = [f for f in task_folder.glob("*.*") 
                          if f.suffix.lower() in config.data.extensions]
            
            # Skip empty folders
            if not image_files:
                console.print(f"[yellow]No images found in {t}[/yellow]")
                continue
                
            # Create progress bar for this task
            with Progress(*progress_columns, console=console) as progress:
                # Create a task with the total number of images
                task_id = progress.add_task(f"Analyzing {t}", total=len(image_files))
                
                # Process each image with progress tracking
                for image_path in image_files:
                    # Extract ID from filename
                    id = image_path.name.split(".")[0]
                    ids.append(id)
                    
                    # Update description to show current file
                    progress.update(task_id, description=f"Analyzing {image_path.name}")

                    # --- FRACTAL ANALYSIS ---
                    # Show box sizes in normal or verbose mode
                    box_sizes = config.fractal.box_sizes
                    if config.display.verbosity >= 1:
                        box_table = Table(show_header=False, box=None)
                        box_table.add_row("[bold cyan]Box Sizes:[/bold cyan]", ", ".join(str(size) for size in box_sizes))
                        console.print(box_table)

                    # Initialize analyzer
                    analyzer = FractalAnalyzer(
                                    box_sizes=config.fractal.box_sizes,
                                    min_black_ratio=config.fractal.get('min_black_ratio', 0.1),
                                    config=dict(config)
                                )
                    verbose_print(f"[dim]Initialized FractalAnalyzer with {len(box_sizes)} box sizes[/dim]",
                                 min_level=2, current_level=config.display.verbosity)

                    # Run analysis
                    try:
                        verbose_print(f"Processing: [bold cyan]{image_path.name}[/bold cyan]",
                                     min_level=1, current_level=config.display.verbosity)
                        features = analyzer.analyze_image(image_path)
                        verbose_print(f"[dim]Analysis complete with {len(features)} features extracted[/dim]",
                                     min_level=2, current_level=config.display.verbosity)

                        # Categorize features
                        fractal_features = {k: v for k, v in features.items() if k.startswith('fractal_')}
                        lacunarity_features = {k: v for k, v in features.items() if k.startswith('lacunarity_')}

                        # Display result tables if not in minimal mode
                        if config.display.verbosity >= 1:
                            # Create and display tables using utility function
                            fractal_table = create_result_table("Fractal Features", fractal_features)
                            console.print(fractal_table)

                            lacunarity_table = create_result_table("Lacunarity Features", lacunarity_features)
                            console.print(lacunarity_table)

                        # Add features to task list
                        t_feature_dicts.append(features)

                        # Display summary
                        if config.display.verbosity >= 1:
                            summary = Table.grid()
                            summary.add_row(
                                f"Extracted [bold cyan]{len(fractal_features)}[/bold cyan] fractal features and "
                                f"[bold cyan]{len(lacunarity_features)}[/bold cyan] lacunarity features"
                            )
                            console.print(Panel(summary, title="📊 Analysis Summary", border_style="blue"))

                    except Exception as e:
                        # Display error with appropriate verbosity
                        display_error(e, config.display.theme, config.display.verbosity)
                    
                    # Advance the progress bar
                    progress.advance(task_id)

            # After processing all images, convert the list of dictionaries into a DataFrame
            # Check if t_feature_dicts exists
            labels = list(map(lambda x: 1 if "PT" in x else 0, ids))
            if 't_feature_dicts' in locals() and t_feature_dicts:
                if len(ids) == len(t_feature_dicts):
                    # Create DataFrame from t_feature_dicts
                    df = pd.DataFrame(t_feature_dicts)

                    # Add 'Id' as the first column
                    df.insert(0, 'Id', ids)

                    # Add 'Class' as the last column
                    df['Class'] = labels

                    # Save the DataFrame
                    output_file = results_path / (t + '_feature_dicts.csv')
                    df.to_csv(output_file, index=False)
                    console.print(f"[green]Results saved to:[/green] [bold cyan]{output_file}[/bold cyan]")
                else:
                    console.print("[yellow]Warning:[/yellow] The number of ids does not match the number of rows in t_feature_dicts.")
            else:
                console.print("[yellow]Warning:[/yellow] No feature dictionaries were created.")
                
        # return
                

if __name__ == "__main__":
    main()