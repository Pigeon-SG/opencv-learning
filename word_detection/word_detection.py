import cv2
import paddlehub as hub

word_img = cv2.imread("word3.jpg")

ocr = hub.Module(name="chinese_ocr_db_crnn_server")
result = ocr.recognize_text(images=[word_img],visualization = True)
print(result)
word_list = result[0]["data"]
print(len(word_list))
for i in range(len(word_list)):
    word = word_list[i]["text"]
    print(word)