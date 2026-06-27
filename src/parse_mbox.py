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
    
    # removing extra whitespace and newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text
    
def get_body(msg) -> str:
    """
    extracts the body of an email message.

    an email can be:
    - simple (text/plain or text/html)
    - multipart (text/plain and text/html)
    text/plain is preferred over text/html if both are present.
    """

    if msg.is_multipart():
        plain_parts = []
        html_parts = []
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))

            # skip attachments as we only want the email body
            if "attachment" in contet_disposition:
                continue

            try:
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                charset = part.get_content_charset or "utf-8"
                decoded = payload.decode(charset, errors="replace")
            except (LookupError, ValueError):
                continue

            if content_type == "text/plain":
                plain_parts.append(decoded)
            elif content_type == "text/html":
                html_parts.append(decoded)
        
        if plain_parts:
            return " ".join(plain_parts).strip()
        elif html_parts:
            return strip_html(" ".join(html_parts))
        else:
            return ""
        
    else:
        # not multipart, single body which could be plain or html
        try:
            payload = msg.get_payload(decode=True)
            if payload is None:
                return ""
            charset = msg.get_content_charset() or "utf-8"
            decoded = payload.decode(charset, errors="replace")
        except (LookupError, ValueError):
            return ""
        
        if msg.get_content_type() == "text/html":
            return strip_html(decoded)
        return decoded.strip()
    
def parse_date(date_str: str):
    # parses the email date header into a datetime object. returns none if unparseable
    if not date_str:
        return None
    try:
        return parsedate_to_datetime(date_str)
    except (TypeError, ValueError):
        return None
    
def parse_mbox (input_path: str, max_body_chars: int = 5000) -> pd.DataFrame:
    # parses the full .mbox file into a pandas dataframe with one row per email

    print(f"Opening mbox file: {input_path}")
    print("(This can take a few minutes for large files)")
    mbox = mailbox.mbox(input_path)

    records = []
    skipped = 0