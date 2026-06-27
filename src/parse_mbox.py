# extracts emails from .mbox into structured format using pandas dataframe

import argparse
import mailbox
import re
from datetime import datetime
from email.utils import parsedate_to_datetime
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

def strip_html(html_text: str) -> str:
    # removing HTML tags from email body and handling whitespace to return readable text
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")

    # removing script and style elements 
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(seprator="")
    
    