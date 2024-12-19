import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))


import re, math
from lingua import Language, LanguageDetectorBuilder


### CUSTOM LENGTH CALCULATION ###
# 预编译正则表达式
# 优先匹配混合字母和数字的字符串，其次是纯字母，最后是纯数字
COMBINED_PATTERN = re.compile(
    r'(?P<mixed>[a-zA-Z]+\d+|\d+[a-zA-Z]+)'  # 混合字母和数字
    r'|(?P<word>[a-zA-Z]+)'                  # 纯英文单词
    r'|(?P<digit>\d+)',                       # 纯数字
    re.UNICODE
)

def calculate_custom_length(text: str) -> int:
    """
    计算自定义长度：
    - 每个混合字母和数字的连续字符串计为0.1个字符/字符
    - 每个纯英文单词计为0.47个字符
    - 每个纯数字计为0.01个字符
    - 其他字符（如中文）计为1个字符
    最终结果取下整
    """
    total_length = 0
    last_index = 0  # 跟踪上一个匹配的结束位置

    for match in COMBINED_PATTERN.finditer(text):
        start, end = match.span()
        
        # 处理匹配之间的其他字符（未匹配到的部分，如中文等）
        if last_index < start:
            intermediate_text = text[last_index:start]
            total_length += len(intermediate_text)  # 每个字符计为1
        last_index = end  # 更新上一个匹配的结束位置

        if match.group('mixed'):
            # 混合字母和数字，按每个字符0.01计
            mixed_length = len(match.group('mixed')) * 0.01
            total_length += mixed_length
        elif match.group('word'):
            # 纯英文单词，按每个单词0.47计
            word_length = 0.47 * 1  # 每个单词计为0.47
            total_length += word_length
        elif match.group('digit'):
            # 纯数字，按每个数字0.01计
            digit_length = len(match.group('digit')) * 0.01
            total_length += digit_length

    # 处理最后一个匹配之后的其他字符
    if last_index < len(text):
        remaining_text = text[last_index:]
        total_length += len(remaining_text)  # 每个字符计为1

    # 取下整
    return math.floor(total_length)


### 语言检测 ###
languages = [
    Language.ENGLISH, 
    Language.CHINESE, 
]
detector = LanguageDetectorBuilder.from_languages(*languages).build()
def detect_language(text):
    try:
        lang = detector.detect_language_of(text).iso_code_639_1.name.lower()
        if lang in ['zh', 'en']:
            return lang
        else:
            return None
    except:
        return None