# takes parsed data from parse_mbox and produces labeld data
# flags candidate job/application emails using sender/subject heuristics
# outputs CSV with candidates for manual review

import argparse
import re
import pandas as pd

# collection of known ATS and job-platforms
JOB_SENDER_DOMAINS = [
    "greenhouse.io",
    "lever.co",
    "myworkday.com",
    "workday.com",
    "icims.com",
    "smartrecruiters.com",
    "jobvite.com",
    "ashbyhq.com",
    "taleo.net",
    "successfactors.com",
    "indeedemail.com",
    "linkedin.com",  
    "handshake.com",
    "ziprecruiter.com",
    "glassdoor.com",
]

# keywords for subject line heuristics
JOB_SUBJECT_KEYWORDS = [
    "application", "applied", "interview", "assessment", "coding challenge",
    "online assessment", "phone screen", "next steps in your application",
    "thank you for applying", "your application to", "your application for",
    "internship", "recruiting", "recruiter", "hiring team", "technical screen",
]

def extract_category(label_str: str) -> str:
    # pulling the Gmail tab category out of the raw X-Gmail-Labels string
    # e.g. "CATEGORY_PERSONAL" or "CATEGORY_UPDATES"

    if not isinstance(label_str, str):
        return "Primary"
    
    # finding all matches of the pattern "category <word>" in the label string
    matches = re.findall(r"category (\w+)", label_str)

    if not matches:
        return "Primary" # no tag means it's in the primary inbox
    
    tab_categories = {"Promotions", "Updates", "Social", "Forums", "Personal"}
    for m in matches:
        if m in tab_categories:
            return m
        
    return matches[0]

