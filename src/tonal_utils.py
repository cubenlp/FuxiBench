from xinyun import PoemConverter, PinyinUtils

converter = PoemConverter()


poem = "床前明月光，疑是地上霜。举头望明月，低头思故乡。" 
for c in poem:
    y = converter.convert_character(c) 
    print(c, y)
# converter.convert_poem("床前明月光，疑是地上霜。举头望明月，低头思故乡。")  


print(converter.convert_line("红蓼渡头秋正雨"))

print(converter.convert_with_details("红蓼渡头秋正雨"))

print(converter.get_statistics("红蓼渡头秋正雨"))