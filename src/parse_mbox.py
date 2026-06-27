# extracts emails from .mbox into structured format using pandas dataframe

import argparse
import mailbox
import re
from datetime import datetime
from email.utils import parsedate_to_datetime
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

