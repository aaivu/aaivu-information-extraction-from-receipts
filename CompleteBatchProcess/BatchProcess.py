import argparse
import io
import tempfile
import os.path
import argparse
import os
import math
import time
from html import escape

from google.protobuf.json_format import MessageToJson
import json
from enum import Enum
from json import JSONEncoder
# from Abbyy.AbbyyOnlineSdk import ProcessingSettings, AbbyyOnlineSdk
#
# from Abbyy.AbbyyOnlineSdk import ProcessingSettings

from google.protobuf.json_format import MessageToJson
from PIL import Image, ImageDraw
from google.cloud import vision
from google.cloud.vision import types
import os



def google_ocr_line(annotation):
    breaks = vision.enums.TextAnnotation.DetectedBreak.BreakType
    paragraphs = []
    lines = []
    flag=False

    for page in annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                para = ""
                line = ""
                for word in paragraph.words:
                    for symbol in word.symbols:
                        line += symbol.text
                        print(line)
                        if symbol.property.detected_break.type == breaks.SPACE:
                            line += ' '
                        if symbol.property.detected_break.type == breaks.EOL_SURE_SPACE:
                            line += ' '
                            lines.append(line)
                            # print(line)
                            para += line
                            line = ''
                        if symbol.property.detected_break.type == breaks.LINE_BREAK:
                            lines.append(line)
                            # print(line)
                            para += line
                            line = ''
                paragraphs.append(para)

    print(paragraphs)
    # print(lines)
def google_ocr(image_file,completeName):
    # Instantiates a client]
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/thumilan/Desktop/LSTM-sample/CompleteBatchProcess/a.json"
    client = vision.ImageAnnotatorClient()

    # Loads the image into memory
    with io.open(image_file, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    # Performs label detection on the image file
    resp = client.document_text_detection(image=image)

    labels = resp.text_annotations

    document = resp.full_text_annotation
    # print(resp)

    # serialized = MessageToJson(resp)
    # with open('receipt.jpg.json', 'w') as json_file:
    #     json.dump(serialized, json_file)
    # bounds = get_document_bounds(response, FeatureType.WORD,document)
    # draw_boxes(image, bounds, 'yellow')

    # for label in labels:
    #    print(label.description)
    # save_file(labels[0].description, completeName)

    # print(format(document))
    # google_ocr_line(document)
    text_dict={}
    point_dict={}
    last_index=0
    for anno_idx, anno_json in enumerate(resp.text_annotations):
        if (anno_idx==0):
            continue
        box = anno_json.bounding_poly.vertices
        X=0
        Y=0
        for vertex in box:
            X=X+vertex.x
            Y=Y+vertex.y
        point=[X/4,Y/4]
        text=anno_json.description
        text_dict[anno_idx]=text
        point_dict[anno_idx]=point
        last_index=anno_idx


    Y=100
    deviation=50
    end=point_dict[last_index][1]+100
    print(end)
    index_list = list()
    while Y<end:
        if len((point_dict.keys()))<1:
            break
        for key in point_dict.keys():
            vertex=point_dict[key]
            if vertex[1]<=Y:
                index_list.append(key)
        for key in index_list:
            point_dict.pop(key,None)
        Y=Y+deviation
        index_list.append(1000)
    print(index_list)
    receipt=str()
    receipt=receipt+"\n"
    charec_list = [".","/"]
    del_list=["[","]"]
    for index in index_list:
        if (index==1000):
            print("\n")
            if (receipt[-1]=="\n"):
                continue
            receipt=receipt+"\n"
            continue
        print(text_dict[index])
        if (text_dict[index] in charec_list):
            receipt=receipt[0:len(receipt)-1]+text_dict[index]
            continue
        if (text_dict[index] in del_list):
            receipt = receipt[0:len(receipt) - 1]
            continue
        # if (receipt[-1]=="."):
        #     receipt = receipt + text_dict[index]
        #     continue
        receipt=receipt+text_dict[index]
        receipt=receipt+" "
    receipt=receipt[1:len(receipt)]
    save_file(receipt,completeName)




    # texts = resp.text_annotations
    # print('Texts:')
    #
    # for text in texts:
    #     # print('\n"{}"'.format(text.description))
    #     texts=([text
    #                  for text in text.description])
    #     vertices = ([(vertex.x, vertex.y)
    #                  for vertex in text.bounding_poly.vertices])
    #     print(texts)
        # print('bounds: {}'.format(','.join(vertices)))
    # resp = resp[0] if  "textAnnotations" in resp['responses'][0] else False
    # page = fromResponse(resp)
    # print(page)
    # savefile='data.hocr'
    # with (open(savefile, 'w', encoding="utf-8") if str == bytes else open(savefile, 'w')) as outfile:
    #     outfile.write(page.render().encode('utf-8') if str == bytes else page.render())
    #     outfile.close()

#     # print(resp)


def create_parser():
    parser = argparse.ArgumentParser(description="Recognize a file via web service")
    parser.add_argument('source_file')
    parser.add_argument('target_file')

    parser.add_argument('-l', '--language', default='English', help='Recognition language (default: %(default)s)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-txt', action='store_const', const='txt', dest='format', default='txt')
    group.add_argument('-pdf', action='store_const', const='pdfSearchable', dest='format')
    group.add_argument('-rtf', action='store_const', const='rtf', dest='format')
    group.add_argument('-docx', action='store_const', const='docx', dest='format')
    group.add_argument('-xml', action='store_const', const='xml', dest='format')

    return parser


def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False,   suffix='.png')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename


def save_file(data_text,file_name):
    file_object = open(file_name, "w")
    file_object.write(data_text)
    file_object.close()
## defining the directories which we have to work with



images_dir = r'/home/thumilan/Desktop/LSTM-sample/CompleteBatchProcess/testbills'
google_output_dir = r'/home/thumilan/Desktop/LSTM-sample/CompleteBatchProcess/GoogleOCROutput'
abbyy_output_dir = r'/home/thumilan/Desktop/LSTM-sample/CompleteBatchProcess/ABBYYOCROutput'




images_list= os.listdir(images_dir)


if not os.path.exists(google_output_dir):
    os.makedirs(google_output_dir)

if not os.path.exists(abbyy_output_dir):
    os.makedirs(abbyy_output_dir)

new_images_list=[images_dir + '/' + x for x in images_list]
i = 0
for img in new_images_list:
    image_name = images_list[i]
    google_ocr_text = google_output_dir + '/' + image_name[:-4] + "_GVOcr.txt"
    print('Processing image : ' + google_ocr_text)
    google_ocr(images_dir + '/' + image_name, google_ocr_text)
    i += 1

