"""
Poem converter for the Xinyun package

This module provides the main functionality to convert poem sentences into 平仄 patterns.
"""

from typing import Dict, List, Tuple, Optional
from .pinyin_utils import PinyinUtils
from .exceptions import InputError, ConversionError, ValidationError


class PoemConverter:
    """Main class for converting poem text to 平仄 patterns."""
    
    def __init__(self):
        self.utils = PinyinUtils()
    
    def convert_character(self, char: str) -> str:
        """Convert a single Chinese character to 平 or 仄."""
        if not char:
            raise InputError("Character cannot be empty")
        
        if not self.utils.is_chinese_char(char):
            return ''
        
        try:
            pinyin_list = self.utils.get_pinyin_with_tone(char)
            if not pinyin_list or not pinyin_list[0]:
                return ''
            
            pinyin_syllable = pinyin_list[0][0]
            tone = self.utils.extract_tone(pinyin_syllable)
            
            # Determine if it's level (平) or oblique (仄)
            return '平' if self.utils.is_level_tone(tone) else '仄'
        except Exception as e:
            raise ConversionError(f"Failed to convert character '{char}': {e}")
    
    def convert_line(self, line: str) -> str:
        """Convert a line of poem text to 平仄 pattern."""
        if not line:
            raise InputError("Line cannot be empty")
        
        try:
            result = []
            for char in line:
                pingze = self.convert_character(char)
                if pingze:
                    result.append(pingze)
                else:
                    # Keep non-Chinese characters as-is
                    result.append(char)
            
            return ''.join(result)
        except Exception as e:
            raise ConversionError(f"Failed to convert line '{line}': {e}")
    
    def convert_poem(self, text: str) -> str:
        """Convert a multi-line poem to 平仄 pattern."""
        if not text:
            raise InputError("Text cannot be empty")
        
        try:
            lines = text.split('\n')
            converted_lines = []
            
            for line in lines:
                converted_line = self.convert_line(line)
                converted_lines.append(converted_line)
            
            return '\n'.join(converted_lines)
        except Exception as e:
            raise ConversionError(f"Failed to convert poem: {e}")
    
    def convert_with_details(self, text: str) -> Dict[str, List[Dict]]:
        """Convert poem with detailed information for each character."""
        if not text:
            raise InputError("Text cannot be empty")
        
        try:
            result = {
                'input': text,
                'output': self.convert_poem(text),
                'details': []
            }
            
            lines = text.split('\n')
            for line_num, line in enumerate(lines, 1):
                line_details = []
                processed_chars = self.utils.process_text(line)
                
                for char, pinyin_syllable, tone in processed_chars:
                    if self.utils.is_chinese_char(char):
                        is_level = self.utils.is_level_tone(tone)
                        vowel = self.utils.extract_vowel(pinyin_syllable)
                        rhyme_group = self.utils.get_rhyme_group(vowel)
                        
                        char_detail = {
                            'character': char,
                            'pinyin': pinyin_syllable,
                            'tone': tone,
                            'is_level': is_level,
                            'pingze': '平' if is_level else '仄',
                            'vowel': vowel,
                            'rhyme_group': rhyme_group,
                            'line': line_num,
                            'position': len(line_details) + 1
                        }
                    else:
                        char_detail = {
                            'character': char,
                            'pinyin': '',
                            'tone': '',
                            'is_level': False,
                            'pingze': char,
                            'vowel': '',
                            'rhyme_group': '',
                            'line': line_num,
                            'position': len(line_details) + 1
                        }
                    
                    line_details.append(char_detail)
                
                result['details'].append({
                    'line_number': line_num,
                    'original': line,
                    'converted': self.convert_line(line),
                    'characters': line_details
                })
            
            return result
        except Exception as e:
            raise ConversionError(f"Failed to convert poem with details: {e}")
    
    def get_statistics(self, text: str) -> Dict[str, any]:
        """Get statistics about the poem's 平仄 pattern."""
        if not text:
            raise InputError("Text cannot be empty")
        
        try:
            details = self.convert_with_details(text)
            
            total_chars = 0
            level_chars = 0
            oblique_chars = 0
            rhyme_distribution = {}
            
            for line_details in details['details']:
                for char_info in line_details['characters']:
                    if char_info['is_level'] is not False:  # Exclude non-Chinese chars
                        total_chars += 1
                        if char_info['is_level']:
                            level_chars += 1
                        else:
                            oblique_chars += 1
                        
                        rhyme_group = char_info.get('rhyme_group', '')
                        if rhyme_group:
                            rhyme_distribution[rhyme_group] = rhyme_distribution.get(rhyme_group, 0) + 1
            
            return {
                'total_characters': total_chars,
                'level_characters': level_chars,
                'oblique_characters': oblique_chars,
                'level_percentage': (level_chars / total_chars * 100) if total_chars > 0 else 0,
                'oblique_percentage': (oblique_chars / total_chars * 100) if total_chars > 0 else 0,
                'rhyme_distribution': rhyme_distribution,
                'lines': len(text.split('\n'))
            }
        except Exception as e:
            raise ConversionError(f"Failed to get statistics: {e}")
    
    def validate_poem_format(self, text: str) -> List[str]:
        """Validate poem format and return any issues found."""
        if not text:
            raise InputError("Text cannot be empty")
        
        try:
            issues = []
            lines = text.split('\n')
            
            if not lines or not any(line.strip() for line in lines):
                issues.append("Poem is empty")
                return issues
            
            # Check for empty lines
            for i, line in enumerate(lines, 1):
                if not line.strip():
                    issues.append(f"Line {i} is empty")
            
            # Check for consistent line length (for classical poems)
            line_lengths = [len(line) for line in lines if line.strip()]
            if line_lengths:
                unique_lengths = set(line_lengths)
                if len(unique_lengths) > 1:
                    issues.append(f"Inconsistent line lengths: {unique_lengths}")
            
            return issues
        except Exception as e:
            raise ValidationError(f"Failed to validate poem format: {e}")