def green_bold(text):
    """
    Format text with green bold ANSI color codes for terminal output.

    Parameters
    ----------
    text : str
        Text string to format

    Returns
    -------
    str
        Text wrapped with ANSI escape codes for green bold formatting
    """
    return f"\033[1;32m{text}\033[0m"
