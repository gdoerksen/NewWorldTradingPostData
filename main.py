import cv2 
import numpy as np
import pandas as pd
import os 
import pyautogui
from time import time
from PIL import ImageGrab, Image
import win32gui, win32ui, win32con
import pytesseract
from pytesseract import Output
from datetime import datetime, timezone
import re
import string
import time
from sklearn.cluster import KMeans
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

import items

def listWindowNames():
    def winEnumHandler(hwnd, ctx):
        if win32gui.IsWindowVisible(hwnd):
            print(hex(hwnd), win32gui.GetWindowText(hwnd))
    win32gui.EnumWindows(winEnumHandler, None)

def windowCapture():
    width = 3440
    height = 1440

    hwnd = None
    # hwnd = win32gui.FindWindow(None, windowname)
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, width, height)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0,0), (width,height) , dcObj, (0,0), win32con.SRCCOPY)

    # dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
    signedIntsArray = dataBitMap.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = ( height, width, 4 )

    # Cleanup
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    img = img[...,:3]

    img = np.ascontiguousarray(img)

    return img

def windowCaptureRealtime():
    loop_time = time()
    while(True):
        
        # screenshot = ImageGrab.grab()
        screenshot = windowCapture()
        # screenshot = np.array(screenshot)
        # screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

        cv2.imshow('Computer Vision', screenshot)
        print('FPS {}'.format( 1 / (time() - loop_time)))
        loop_time = time()

        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break

def windowCaptureSave(filename):
    # screenshot = windowCapture()
    # im = Image.fromarray(screenshot).convert('RGB')
    im = screenshot = ImageGrab.grab()
    im.save(filename + '.jpeg')
    # TODO Color channels are wrong currently, Orange = Blue 

def whiteThreshold(image, vMin=78):
    hMin = sMin = 0
    vMin = vMin
    hMax = 179
    sMax = vMax = 255

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(image,image, mask= mask)
    return output

def binaryThreshold(img, threshold):
    ret, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    return img

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def drawSquareOnImage(img, x1: int, y1: int, x2: int, y2: int):
    green = (0, 255, 0)
    thickness = 1
    img = cv2.rectangle(img, (x1, y1), (x2, y2), green, thickness)
    return img

def displayImage(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)

def drawRows():
    x1 = 3182-1920
    y1 = 423
    x2 = 4780-1920
    y2 = 525
    img = cv2.imread('soul_03.jpeg')
    itemsOnScreen = 9
    rowWidth = 103
    offset = 5
    #TODO might need to fix the offset eventually
    for i in range(itemsOnScreen):
        y1 += offset
        y2 -= offset
        img = drawSquareOnImage(img, x1, y1, x2, y2)
        y1 += rowWidth - offset 
        y2 += rowWidth + offset
    displayImage(img)

def crop(img, x1, y1, x2, y2):
    return img[y1:y2, x1:x2]

def getRow(img, x1, y1, x2, y2):
    return img[y1:y2, x1:x2]

def sliceRow(img, x1, x2):
    return img[:, x1:x2]

def identifyText(img):
    # data = pytesseract.image_to_data(img)
    data = pytesseract.image_to_string(img)
    return data

def preprocessItem(img, vMin):
    img = whiteThreshold(img, vMin)
    img = grayscale(img)
    # img = cv2.GaussianBlur(img, (7, 7), 0)
    # img = binaryThreshol d(img, 32)
    return img

def getTextFromData(data):
    text = ''
    for i in range(len(data['level'])):
        if float(data['conf'][i]) > 90.0:        #TODO -1 in conf means multi-word....
            text += (data["text"][i] + ' ')
    text = str.strip(text)
    return text

def identifyTextItem(original_image):
    attempts = 0
    max_attempts = 5
    vMin_max = 80
    vMin = vMin_max
    while (attempts < max_attempts):
        img = original_image.copy()
        img = preprocessItem(img, vMin)
        data = pytesseract.image_to_data(img, lang='eng',
                config='--psm 7',output_type=Output.DICT)
                # psm 7 - Treat the image as a single text line.

        text = getTextFromData(data)
        # print( repr(data))
        attempts += 1
        
        if text == "":
            vMin -= 10 
            continue
        else:
            return text
    return None

        # n_boxes = len(d['level'])
        # for i in range(n_boxes):
        #     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

def identifyTextPrice(original_image):
    attempts = 0
    max_attempts = 5
    vMin_max = 80
    vMin = vMin_max
    pattern = re.compile(r"^\$?(([1-9]\d{0,2}(,\d{3})*)|0)?\.\d{1,2}$")
    while (attempts < max_attempts):
        img = original_image.copy()
        img = preprocessItem(img, vMin)
        data = pytesseract.image_to_data(img, lang='eng',
                config='--psm 7 -c tessedit_char_whitelist=0123456789.,',output_type=Output.DICT)
                # psm 7 - Treat the image as a single text line.

        text = getTextFromData(data)
        # print( repr(data))
        attempts += 1
        
        if (text == "") or ( not pattern.fullmatch(text)):
            vMin -= 10 
            continue
        else:
            return text
    return None

def identifyTextAvail(original_image):
    attempts = 0
    max_attempts = 5
    vMin_max = 80
    vMin = vMin_max
    pattern = re.compile(r"^[0-9]+$")
    while (attempts < max_attempts):
        img = original_image.copy()
        img = preprocessItem(img, vMin)
        data = pytesseract.image_to_data(img, lang='eng',
                config='--psm 7 -c tessedit_char_whitelist=0123456789',output_type=Output.DICT)
                # psm 7 - Treat the image as a single text line.

        text = getTextFromData(data)
        attempts += 1
        
        if (text == "") or ( not pattern.match(text)):
            vMin -= 10 
            continue
        else:
            return text
    return None

def identifyTextTime(original_image):
    attempts = 0
    max_attempts = 5
    vMin_max = 80
    vMin = vMin_max
    # pattern = re.compile(r"^[0-9]+$") #TODO define a regex match 
    while (attempts < max_attempts):
        img = original_image.copy()
        img = preprocessItem(img, vMin)
        data = pytesseract.image_to_data(img, lang='eng',
                config='--psm 7 -c tessedit_char_whitelist=0123456789DdHhMmSs',output_type=Output.DICT)
                # psm 7 - Treat the image as a single text line.

        text = getTextFromData(data)
        attempts += 1
        # or ( not pattern.match(text))
        if (text == "") :
            vMin -= 10 
            continue
        else:
            return text
    return None

def identifyTextLocation(original_image):
    attempts = 0
    max_attempts = 5
    vMin_max = 80
    vMin = vMin_max
    pattern = ["Brightwood", "Cleave's Point", "Cutlass Keys",
    "Eastburn", "Ebonscale Reach", "Everfall", "First Light",
    "Last Stand", "Monarch's Bluffs", "Mountainhome",
    "Mountainrise", "Mourningdale", "Reekwater", "Restless Shore"
    "Valor Hold", "Weaver's Fen", "Windsward"]
    while (attempts < max_attempts):
        img = original_image.copy()
        img = preprocessItem(img, vMin)
        data = pytesseract.image_to_data(img, lang='eng',
                config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\\'",
                output_type=Output.DICT)
                # psm 7 - Treat the image as a single text line.

        text = getTextFromData(data)
        attempts += 1
        # or ( not pattern.match(text))
        if (text == "") or (text not in pattern):
            vMin -= 10 
            continue
        else:
            return text
    return None

def processRow(img):

    row = []

    x_offset = 3182
    left_bound = 3278 - x_offset
    right_bound = 3647 - x_offset
    item = sliceRow(img, left_bound, right_bound)
    row.append(identifyTextItem(item))

    left_bound = 3650 - x_offset
    right_bound = 3838 - x_offset
    price = sliceRow(img, left_bound, right_bound)
    row.append(identifyTextPrice(price))

    left_bound = 4370 - x_offset
    right_bound = 4452 - x_offset
    quantity_available = sliceRow(img, left_bound, right_bound)
    row.append(identifyTextAvail(quantity_available))

    left_bound = 4545 - x_offset
    right_bound = 4629 - x_offset
    time_left = sliceRow(img, left_bound, right_bound)
    row.append(identifyTextTime(time_left))

    left_bound = 4632 - x_offset
    right_bound = 4780 - x_offset
    location = sliceRow(img, left_bound, right_bound)
    row.append(identifyTextLocation(location))

    return row

def processRowWithoutItem(img, item_name):

    row = []
    row.append(item_name)

    x_offset = 3182

    left_bound = 3650 - x_offset
    right_bound = 3838 - x_offset
    price = sliceRow(img, left_bound, right_bound)
    row.append(identifyTextPrice(price))

    left_bound = 4370 - x_offset
    right_bound = 4452 - x_offset
    quantity_available = sliceRow(img, left_bound, right_bound)
    row.append(identifyTextAvail(quantity_available))

    left_bound = 4545 - x_offset
    right_bound = 4629 - x_offset
    time_left = sliceRow(img, left_bound, right_bound)
    row.append(identifyTextTime(time_left))

    left_bound = 4632 - x_offset
    right_bound = 4780 - x_offset
    location = sliceRow(img, left_bound, right_bound)
    row.append(identifyTextLocation(location))

    return row

def defineRegularExpression():
    price_pattern = re.compile("^\$?(([1-9]\d{0,2}(,\d{3})*)|0)?\.\d{1,2}$")
    test = "0.01"
    match = price_pattern.fullmatch(test)
    if match:
        print(match)

def getMeThatData(img):
    x1 = 3182-1920
    y1 = 423
    x2 = 4780-1920
    y2 = 525
    # img = cv2.imread('soul_03.jpeg')
    rowsData = []

    itemsOnScreen = 9
    rowWidth = 103
    offset = 5
    for i in range(itemsOnScreen):
        y1 += offset
        y2 -= offset
        row = getRow(img, x1, y1, x2, y2)
        rowData = processRow(row)
        rowsData.append(rowData)    
        y1 += rowWidth - offset 
        y2 += rowWidth + offset

    df = pd.DataFrame(rowsData, columns=["Name", "Price", "Amount Available", "Time Available", "Location"])
    print(df)

def singleItemData(img):
    x1 = 3182-1920
    y1 = 423
    x2 = 4780-1920
    y2 = 525
    # img = cv2.imread('soul_03.jpeg')
    rowsData = []

    itemsOnScreen = 9
    rowWidth = 103
    offset = 5
    
    y1 += offset
    y2 -= offset
    row = getRow(img, x1, y1, x2, y2)
    rowData = processRow(row)
    item_name = rowData[0]
    rowsData.append(rowData)    
    y1 += rowWidth - offset 
    y2 += rowWidth + offset
    for i in range(itemsOnScreen-1):
        y1 += offset
        y2 -= offset
        row = getRow(img, x1, y1, x2, y2)
        rowData = processRowWithoutItem(row, item_name)
        rowsData.append(rowData)    
        y1 += rowWidth - offset 
        y2 += rowWidth + offset

    df = pd.DataFrame(rowsData, columns=["Name", "Price", "Amount Available", "Time Available", "Location"])
    print(df)



def selectAllSettlements():
    click_duration = 0.2
    pyautogui.moveTo(2608, 193, duration=click_duration, tween=pyautogui.easeInOutQuad)
    pyautogui.click(interval=click_duration)
    pyautogui.moveTo(2892, 233, duration=click_duration, tween=pyautogui.easeInOutQuad)
    pyautogui.click(interval=click_duration)
    pyautogui.moveTo(2656, 253, duration=click_duration, tween=pyautogui.easeInOutQuad)
    pyautogui.click(interval=click_duration)
    pyautogui.moveTo(2608, 193, duration=click_duration, tween=pyautogui.easeInOutQuad)
    pyautogui.click(interval=click_duration)
    

def getBuyOrderScreen(target_item):
    getTradingPost()
    getToItemScreen(target_item)
    pyautogui.moveTo(908, 899, duration=0.2, tween=pyautogui.easeInOutQuad)
    pyautogui.click()

def getSellOrderScreen(target_item):
    getTradingPost()
    getToItemScreen(target_item)
    pyautogui.moveTo(927, 984, duration=0.2, tween=pyautogui.easeInOutQuad)
    pyautogui.click()

def getToItemScreen(target_item):
    # pyautogui.click(759,297,duration=0.3)
    pyautogui.moveTo(759, 297, duration=0.2, tween=pyautogui.easeInOutQuad)
    pyautogui.click()
    pyautogui.click()
    # time.sleep(0.53)
    pyautogui.write(target_item,interval=0.1)
    pyautogui.moveTo(721, 478, duration=0.2, tween=pyautogui.easeInOutQuad)
    pyautogui.click(interval=0.3)
    time.sleep(1.5) # need to wait for the screen to load
    #TODO adjust this wait time

def getMinimumPrice(img):
    x1 = 3182-1920
    y1 = 423
    x2 = 4780-1920
    y2 = 525
    # img = cv2.imread('soul_03.jpeg')
    offset = 5
    y1 += offset
    y2 -= offset
    row = getRow(img, x1, y1, x2, y2)
    rowData = processRow(row)
    return rowData

def getItemData(list_of_items, items_on_screen=9):
    datetime_start = datetime.now(timezone.utc)

    data_list = []
    for item_name in list_of_items:
        getTradingPost()
        getToItemScreen(item_name)
        datetime_of_image = datetime.now(timezone.utc)
        image = windowCapture()

        x1 = 1262
        y1 = 423
        x2 = 2860
        y2 = 525   

        row_width = 103
        offset = 5
        for i in range(items_on_screen):
            y1 += offset
            y2 -= offset
            row_image = getRow(image, x1, y1, x2, y2)
            data_row = processRowMinimal(row_image)
            data_row.insert(0,item_name)
            data_row.insert(0,datetime_of_image)
            data_row.insert(0,datetime_start)
            data_list.append(data_row)    
            y1 += row_width - offset 
            y2 += row_width + offset

    db = pd.DataFrame(data_list, columns=["Run Started", "Data Recorded", "Name", "Price", "Amount Available"])
    return db


def processRowMinimal(img):

    row = []

    x_offset = 3182
    left_bound = 468
    right_bound = 656
    price = sliceRow(img, left_bound, right_bound)
    row.append(identifyTextPrice(price))

    left_bound = 1188
    right_bound = 1270
    quantity_available = sliceRow(img, left_bound, right_bound)
    row.append(identifyTextAvail(quantity_available))

    return row
        

def isTradingPostOpen(): 
    '''
    The in-game header bar is blue when the Trading Post is open.
    Returns True if the difference of the root-mean-square of the 
    average color is less than 1. This works because of how blue
    the header bar is 
    '''
    x1=1530
    y1=14
    x2=1994
    y2=106
    # cv2.imwrite('trading_post_header.jpeg',img)

    img = windowCapture()
    img = crop(img,x1,y1,x2,y2)

    img_header = cv2.imread('trading_post_header.jpeg')

    average = img.mean(axis=0).mean(axis=0)
    average_header = img_header.mean(axis=0).mean(axis=0)

    delta = 0
    for c1, c2 in zip(average, average_header):
        delta += (c2-c1)*(c2-c1)
    delta = delta**0.5
    print(delta)
    if delta <= 3:
        return True
    else:
        return False        

def openTradingPost():
    # pyautogui.click(50,50)
    pyautogui.press('f')

def getTradingPost():
    while True: 
        if isTradingPostOpen():
            return
        else:
            openTradingPost()
            time.sleep(2)
            
def loadDatabase(path_to_database="./database.pkl"):
    #TODO transfer to JSON files 
    return pd.read_pickle(path_to_database)

def saveDatabase(database_dataframe, database_file_name):
    database_dataframe.to_pickle("./" + database_file_name + ".pkl")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # screenWidth, screenHeight = pyautogui.size() # Get the size of the primary monitor.

    item_list = items.item_arcana
    pyautogui.click(50,50)
    getTradingPost()

    item_list = items.item_arcana

    db = getItemData(item_list,1)
    saveDatabase(db,"item_prices_2")
        

    # print(pyautogui.position())




