#!/usr/bin/env python3

'''functions taken from https://github.com/lfurrer/disease-normalization/tree/master/tzlink'''

import re

def tzlink_tokenizer(string):
    tokenizer =  _find_pattern()
    return tokenizer(string)

def _find_pattern():
    pattern = re.compile(
        r'''\d+|            # match contiguous runs of digits
            [^\W\d_]+|      # or letters
            (?:[^\w\s]|_)+  # or other non-whitespace characters
            ''', re.VERBOSE)
    return pattern.findall

