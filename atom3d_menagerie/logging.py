"""
Simple utilities to help the user verify that the program is behaving as it 
should be.
"""

from functools import partial
import sys

info = partial(print, "INFO:", file=sys.stderr)
