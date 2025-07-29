"""
Pinyin utilities for the Xinyun package

This module provides utilities for working with Chinese pinyin and tone detection.
"""

import re
from typing import Dict, List, Tuple
from pypinyin import pinyin, Style


class PinyinUtils:
    """Utility class for pinyin operations and tone detection."""
    
    # Define the 14-rhyme system (2005 Chinese Poetry Society)
    RHYME_GROUPS = {
        # 一麻 a, ia, ua
        'a': ['a', 'ia', 'ua'],
        # 二波 o, e, uo
        'o': ['o', 'e', 'uo'],
        # 三皆 ie, üe
        'ie': ['ie', 've'],
        # 四开 ai, uai
        'ai': ['ai', 'uai'],
        # 五微 ei, ui
        'ei': ['ei', 'ui'],
        # 六豪 ao, iao
        'ao': ['ao', 'iao'],
        # 七尤 ou, iu
        'ou': ['ou', 'iu'],
        # 八寒 an, ian, uan, üan
        'an': ['an', 'ian', 'uan', 'van'],
        # 九文 en, in, un, ün
        'en': ['en', 'in', 'un', 'vn'],
        # 十唐 ang, iang, uang
        'ang': ['ang', 'iang', 'uang'],
        # 十一庚 eng, ing, ong, iong
        'eng': ['eng', 'ing', 'ong', 'iong'],
        # 十二齐 i, ü, er
        'i': ['i', 'v', 'er'],
        # 十三支 -i (zhi, chi, shi, ri, zi, ci, si)
        'zhi': ['zhi', 'chi', 'shi', 'ri', 'zi', 'ci', 'si'],
        # 十四姑 u
        'u': ['u']
    }
    
    # Characters that belong to the 13th rhyme (zhi group)
    ZHI_GROUP_CHARS = {'知', '吃', '师', '诗', '日', '资', '词', '思', '丝', '之', '只', '是', '时', '十', '石', '实', '识', '事', '子', '字', '此', '次', '自', '司', '四'}
    
    @staticmethod
    def get_pinyin_with_tone(text: str) -> List[List[str]]:
        """Get pinyin with tone marks for Chinese text."""
        return pinyin(text, style=Style.TONE3)
    
    @staticmethod
    def get_pinyin_no_tone(text: str) -> List[List[str]]:
        """Get pinyin without tone marks for Chinese text."""
        return pinyin(text, style=Style.NORMAL)
    
    @staticmethod
    def extract_tone(pinyin_syllable: str) -> str:
        """Extract tone number from pinyin syllable."""
        # Extract tone number (1-5, where 5 is neutral)
        tone_match = re.search(r'(\d)$', pinyin_syllable)
        return tone_match.group(1) if tone_match else '5'
    
    @staticmethod
    def extract_vowel(pinyin_syllable: str) -> str:
        """Extract vowel part from pinyin syllable."""
        # Remove tone number and any consonants to get vowel
        vowel_part = re.sub(r'^[bpmfdtnlgkhjqxzcsrywvchshrz]*', '', pinyin_syllable)
        vowel_part = re.sub(r'\d$', '', vowel_part)
        return vowel_part
    
    @staticmethod
    def is_level_tone(tone: str) -> bool:
        """Check if a tone is level (平) or oblique (仄)."""
        # Level tones: 1 (阴平), 2 (阳平)
        # Oblique tones: 3 (上声), 4 (去声)
        # Neutral tone (5) is treated as level
        return tone in ['1', '2', '5']
    
    @staticmethod
    def get_rhyme_group(vowel: str) -> str:
        """Determine which rhyme group a vowel belongs to."""
        # Special handling for zhi group
        if vowel in ['zhi', 'chi', 'shi', 'ri', 'zi', 'ci', 'si']:
            return 'zhi'
        
        # Check each rhyme group
        for group, vowels in PinyinUtils.RHYME_GROUPS.items():
            if vowel in vowels:
                return group
        
        # Fallback: try to find partial match
        for group, vowels in PinyinUtils.RHYME_GROUPS.items():
            for v in vowels:
                if v in vowel or vowel in v:
                    return group
        
        return 'unknown'
    
    @staticmethod
    def is_chinese_char(char: str) -> bool:
        """Check if a character is Chinese."""
        return '\u4e00' <= char <= '\u9fff'
    
    @staticmethod
    def process_text(text: str) -> List[Tuple[str, str, str]]:
        """Process text and return (char, pinyin, tone) tuples."""
        result = []
        pinyin_list = PinyinUtils.get_pinyin_with_tone(text)
        
        for i, char in enumerate(text):
            if PinyinUtils.is_chinese_char(char):
                pinyin_syllable = pinyin_list[i][0] if i < len(pinyin_list) else ''
                tone = PinyinUtils.extract_tone(pinyin_syllable)
                result.append((char, pinyin_syllable, tone))
            else:
                result.append((char, '', ''))
        
        return result