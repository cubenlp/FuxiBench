"""
Xinyun Package - Poem to 平仄 Conversion

A package that provides API to convert poem sentences into 平仄 (level and oblique tones)
based on the 2005 Chinese Poetry Society's 14-rhyme system.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .converter import PoemConverter
from .pinyin_utils import PinyinUtils

__all__ = ["PoemConverter", "PinyinUtils"]