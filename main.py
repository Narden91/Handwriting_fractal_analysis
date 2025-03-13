"""
Main module for fractal analysis of images.
Handles configuration, image processing, and results display.
"""
import hydra
from omegaconf import DictConfig
import sys
from pathlib import Path

# Don't create __pycache__ folders
sys.dont_write_bytecode = True

# Import FractalAnalyzer and utility functions
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


@hydra.main(config_path="./config", config_name="main", version_base="1.2")
def main(config: DictConfig):
    """Process images with fractal analysis based on configuration."""
    
    # --- SETUP ---
    # Display header
    console.print(Panel.fit(
        f"🔍 [bold]Fractal Analysis Process[/bold]", 
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
    
    # --- IMAGE FINDING ---
    # Check for specified test image
    test_image_path = data_path / config.data.test_image
    
    # If test image doesn't exist, find alternatives
    if not test_image_path.exists():
        with console.status("[yellow]Looking for image files...[/yellow]", spinner="dots"):
            verbose_print("[yellow]Test image not found at[/yellow] " + str(test_image_path), 
                         min_level=1, current_level=config.display.verbosity)
            verbose_print("[yellow]Searching for any image file in the data directory...[/yellow]", 
                         min_level=1, current_level=config.display.verbosity)
            
            # Find image files using utility function
            found_image, found_image_path, image_tree = find_image_files(
                data_path, 
                config.data.extensions, 
                config.data.max_display,
                config.display.verbosity
            )
            
            if found_image:
                test_image_path = found_image_path
                # Only show the tree in normal or verbose mode
                if config.display.verbosity >= 1:
                    console.print(image_tree)
                console.print(f"[green]✓ Using image:[/green] [bold cyan]{test_image_path}[/bold cyan]")
            else:
                console.print(Panel(
                    "No image files found in the data directory!", 
                    title="❌ Error", 
                    border_style="red"
                ))
                return
    else:
        verbose_print(f"[green]✓ Using specified test image:[/green] [bold cyan]{test_image_path}[/bold cyan]", 
                     min_level=1, current_level=config.display.verbosity)
    
    # --- FRACTAL ANALYSIS ---
    # Show box sizes in normal or verbose mode
    box_sizes = config.fractal.box_sizes
    if config.display.verbosity >= 1:
        box_table = Table(show_header=False, box=None)
        box_table.add_row("[bold cyan]Box Sizes:[/bold cyan]", ", ".join(str(size) for size in box_sizes))
        console.print(box_table)
    
    # Initialize analyzer
    analyzer = FractalAnalyzer(box_sizes=box_sizes)
    verbose_print(f"[dim]Initialized FractalAnalyzer with {len(box_sizes)} box sizes[/dim]", 
                 min_level=2, current_level=config.display.verbosity)
    
    # Run analysis
    try:
        with console.status(f"[bold blue]Analyzing image...[/bold blue]", spinner=config.display.spinner):
            verbose_print(f"Processing: [bold cyan]{test_image_path.name}[/bold cyan]", 
                         min_level=1, current_level=config.display.verbosity)
            features = analyzer.analyze_image(test_image_path)
            verbose_print(f"[dim]Analysis complete with {len(features)} features extracted[/dim]", 
                         min_level=2, current_level=config.display.verbosity)
        
        # --- RESULTS DISPLAY ---
        console.print("\n")
        console.rule("[bold green]Fractal Analysis Results[/bold green]")
        
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
        
        # --- SAVE RESULTS ---
        save_results(
            features, 
            results_path, 
            config.results.output_file,
            config.display.verbosity
        )
        
        # Display summary
        summary = Table.grid()
        summary.add_row(
            f"Extracted [bold cyan]{len(fractal_features)}[/bold cyan] fractal features and "
            f"[bold cyan]{len(lacunarity_features)}[/bold cyan] lacunarity features"
        )
        console.print(Panel(summary, title="📊 Analysis Summary", border_style="blue"))
        
    except Exception as e:
        # Display error with appropriate verbosity
        display_error(e, config.display.theme, config.display.verbosity)


if __name__ == "__main__":
    main()