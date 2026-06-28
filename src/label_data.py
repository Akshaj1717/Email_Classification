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

