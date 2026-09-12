"""Text cleaning — copied from resume-api/app/utils/text.py."""
import re


def clean_resume(text: str) -> str:
    """
    Cleans the input resume text by removing URLs, special characters,
    and extra whitespace, and converting to lowercase.
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove non-alphanumeric characters (keeps only letters and spaces)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
