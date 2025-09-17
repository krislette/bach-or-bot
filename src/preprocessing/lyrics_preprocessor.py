
import re

class LyricsPreprocessor:
    """
    A preprocessing class for cleaning and preparing song lyrics 
    for LLM2Vec.

    Parameters
    ----------
    keep_case : bool, optional (default=True)
        If False, converts all lyrics to lowercase.
    
    keep_punctuation : bool, optional (default=True)
        If False, removes all punctuation from lyrics.
    
    Usage
    -----
    >>> preprocessor = LyricsPreprocessor(keep_case=False, keep_punctuation=False)
    >>> processed = preprocessor("Hello, world!\n[Chorus]\nSing along")
    >>> print(processed)
    "Hello, world! Sing along"
    """
    def __init__(self, keep_case=True, keep_punctuation=True):
        self.keep_case = keep_case
        self.keep_punctuation= keep_punctuation

    def __call__(self, lyrics: str):
        """
        Preprocess the input lyrics text.

        Steps:
        1. Removes empty lines or lines with metadata (e.g., [Chorus], (Verse)).
        2. Applies case handling and punctuation removal based on settings.
        3. Builds a cleaned lyrics string.

        Parameters
        ----------
        lyrics : str
            Raw lyrics text.

        Returns
        -------
        str
        
        a cleaned lyric string
        """
        lyrics_cleaned = ""

        # Split lyrics by lines
        lyric_array = lyrics.split('\n')

        for line in lyric_array:
            line = line.strip()

            # Skip unimportant lines like [Chorus] or (Verse)
            if not line or re.match(r'^\[.*\]$', line) or re.match(r'^\(.*\)$', line):
                continue
            
            # Case handling
            if not self.keep_case:
                line = line.lower()

            # Punctuation handling
            if not self.keep_punctuation:
                line = re.sub(r'[^\w\s]', '', line)
            
            # Normalize to lowercase and split into words
            words = line.split()
            
            lyrics_cleaned += ' '.join(words) + ' '

        lyrics_cleaned = lyrics_cleaned.strip()

        return lyrics_cleaned
    