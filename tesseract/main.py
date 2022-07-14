import pytesseract
from distortion import *
import csv
import cv2

filename = 'receipt.jpg'
img = cv2.imread(f'images/{filename}')

# img = rotation(img)
img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_output = img.copy()

print(pytesseract.get_languages())  # 현재 설치되어있는 모든 언어 출력
custom_config = r'--oem 1 --psm 1'  # config 설정


# 이미지에서 찾은 모든 글자를 반환
# string = pytesseract.image_to_string(img_input, config=custom_config)
# print(string)


# 이미지에 있는 글자를 인식한 결과 데이터를 dict의 구조로 반환, tesseract 3.05버전 이후
details = pytesseract.image_to_data(img_input, output_type=pytesseract.Output.DICT, config=custom_config, lang='eng')
# print(details.keys())
# ['level', 'page_num', 'block_num', 'par_num', 'line_num',
#  'word_num', 'left', 'top', 'width', 'height', 'conf', 'text']

total_boxes = len(details['text'])
for i in range(total_boxes):
    if int(details['conf'][i]) > 30:
        (x, y, w, h) = (details['left'][i], details['top'][i], details['width'][i], details['height'][i])
        img_output = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.putText(img_output, str(details['text'][i]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        # print(details['text'][i])

# display image
cv2.imwrite(f'./results/result_{filename[:-4]}.jpg', img_output)
cv2.imshow('captured text', img_output)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 이미지에서 찾은 텍스트를 txt 파일에 저장
parse_text = []
word_list = []
last_word = ''
for word in details['text']:
    if word != '':
        word_list.append(word)
        last_word = word
    if (last_word != '' and word == '') or (word == details['text'][-1]):
        parse_text.append(word_list)
        word_list = []

with open(f'./results/result_{filename[:-4]}.txt',  'w', newline="") as file:
    csv.writer(file, delimiter=" ").writerows(parse_text)


