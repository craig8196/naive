"""Temporary class to create a stopword list."""

import os
import sys
import re

def generate_stopword_set_from_file(filepath):
    result = set()
    with open(filepath, 'r') as f:
        for word in re.findall(r'[a-zA-Z]+', f.read()):
            result.add(word.lower())
    return result

def generate_word_list_from_file(filepath):
    result = []
    with open(filepath, 'r') as f:
        for word in re.findall(r'[a-zA-Z]+', f.read()):
            result.append(word.lower())
    return result 
