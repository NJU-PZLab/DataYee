"""Command-line entry point for the DataYee desktop application."""


def main():
    """Start the DataYee Qt application."""
    from UI import main as run_ui

    return run_ui()
