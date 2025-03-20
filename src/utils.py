"""
Utility functions for the fractal analysis application.
"""
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.box import SIMPLE
from rich.tree import Tree
from rich.syntax import Syntax
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, SpinnerColumn
from pathlib import Path
import traceback


# Initialize console object that can be imported by other modules
console = Console()

def verbose_print(message, min_level=1, current_level=1):
    """Print message only if verbosity level is high enough.
    
    Args:
        message: The message to print
        min_level: Minimum verbosity level required to print this message
        current_level: Current verbosity level from config
    """
    if current_level >= min_level:
        console.print(message)

def create_result_table(title, features, title_style="bold cyan"):
    """Create a formatted table for displaying features.
    
    Args:
        title: Table title
        features: Dictionary of feature names and values
        title_style: Style for the table title
        
    Returns:
        Rich Table object with formatted features
    """
    table = Table(title=title, box=SIMPLE, title_style=title_style, show_lines=True)
    table.add_column("Feature", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in features.items():
        table.add_row(
            key, 
            f"{value:.6f}" if isinstance(value, float) else str(value)
        )
    
    return table

def find_image_files(data_path, extensions, max_display, verbosity):
    """Find image files in the data directory.
    
    Args:
        data_path: Path to search for images
        extensions: List of file extensions to look for
        max_display: Maximum number of files to display per extension
        verbosity: Current verbosity level
        
    Returns:
        (found_image, test_image_path, image_tree) tuple
    """
    found_image = False
    test_image_path = None
    image_tree = Tree("📁 [bold]Data Directory[/bold]")
    
    for ext in extensions:
        verbose_print(f"[dim]Searching for {ext} files...[/dim]", 
                      min_level=2, current_level=verbosity)
        
        image_files = list(data_path.glob(f"**/*{ext}"))
        if image_files:
            test_image_path = image_files[0] if not test_image_path else test_image_path
            ext_branch = image_tree.add(f"[green]{ext} files[/green]")
            for img in image_files[:max_display]:
                ext_branch.add(f"[green]{img.relative_to(data_path)}[/green]")
            if len(image_files) > max_display:
                ext_branch.add(f"[dim]...and {len(image_files) - max_display} more[/dim]")
            found_image = True
            verbose_print(f"[dim]Found {len(image_files)} {ext} files[/dim]", 
                         min_level=2, current_level=verbosity)
    
    return found_image, test_image_path, image_tree

def display_error(error, theme="monokai", verbosity=1):
    """Display an error with appropriate verbosity.
    
    Args:
        error: The exception object
        theme: Syntax highlighting theme
        verbosity: Current verbosity level
    """
    console.print(Panel(
        f"[bold]Error analyzing image:[/bold]\n{str(error)}", 
        title="❌ Exception", 
        border_style="red"
    ))
    
    if verbosity >= 1:
        console.print("\n[bold red]Traceback:[/bold red]")
        error_trace = traceback.format_exc()
        syntax = Syntax(error_trace, "python", theme=theme, line_numbers=True)
        console.print(syntax)
    else:
        console.print("[dim]Set verbosity level higher to see full traceback[/dim]")

def save_results(features, results_path, output_file, verbosity):
    """Save analysis results to a CSV file.
    
    Args:
        features: Dictionary of features
        results_path: Path to save results
        output_file: Output filename
        verbosity: Current verbosity level
        
    Returns:
        Path to the saved file
    """
    import pandas as pd
    
    df = pd.DataFrame([features])
    results_file = results_path / output_file
    df.to_csv(results_file, index=False)
    
    verbose_print(f"[dim]DataFrame created with {df.shape[1]} columns[/dim]", 
                 min_level=2, current_level=verbosity)
    
    console.print(Panel(
        f"Results exported to CSV file:\n[bold cyan]{results_file}[/bold cyan]", 
        title="💾 Data Saved", 
        border_style="green"
    ))
    
    return results_file

def create_progress_columns():
    """Create standard progress bar columns for consistent display.
    
    Returns:
        List of Rich progress columns
    """
    return [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ]

def process_with_progress(iterable, description, callback_fn):
    """Process items with a progress bar.
    
    Args:
        iterable: Items to process
        description: Progress bar description
        callback_fn: Function to call for each item
        
    Returns:
        List of results from callback_fn
    """
    results = []
    
    with Progress(*create_progress_columns(), console=console) as progress:
        task_id = progress.add_task(description, total=len(iterable))
        
        for item in iterable:
            # Update description to show current item
            progress.update(task_id, description=f"{description}: {item}")
            
            # Process the item
            result = callback_fn(item)
            results.append(result)
            
            # Advance the progress bar
            progress.advance(task_id)
    
    return results