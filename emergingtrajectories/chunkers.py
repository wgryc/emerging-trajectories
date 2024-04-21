"""
chunkers.py is used to chunk facts using different strategies. Emerging Trajectories started by chunking via GPT-4, but we can also appreciate using sentences, paragraphs, or other verbatim approaches. We'll be adding more chunkers as time goes on.

Chunkers should simply take a piece of content and chunk it into a list of facts. As such, they are provided as functions, for now.
"""


def chunker_gpt4(content: str) -> list[str]:
    """
    Chunker using GPT-4. This is the default chunker for Emerging Trajectories.

    Args:
        content (str): The content to chunk.

    Returns:
        list[str]: The list of facts.
    """
    pass
