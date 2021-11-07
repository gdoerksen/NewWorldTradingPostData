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
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

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

def cropImage(img, rectangle):
    r = rectangle
    x = r.x
    y = r.y
    h = r.h
    w = r.w
    cropped = img[y:y + h, x:x + w]
    return cropped
    

def tesseractTest(img):
    # img = cv2.imread('soul_03.jpeg')

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresh, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # gray = cv2.bitwise_not(img_bin)

    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    print(d.keys())
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if float(d['conf'][i]) > 60:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)

def whiteThreshold(image):
    hMin = sMin = 0
    vMin = 78
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

def initDataframe():
    pass


def initGather():
    now = datetime.now()
    timeStart = now
    timeCurrent = []
    name = []
    price = []
    tier = []
    rarity = []
    available = []
    owned = []
    timeLeftOnMarket = []
    location = []

def loopThroughImage():
    # test by drawing boxes to see what is getting captured
    itemsOnScreen = 9

    x1 = 3182-1920
    y1 = 379
    x2 = 3647-1920
    y2 = 420
    
    x1 = 3182-1920
    y1 = 423
    x2 = 3647-1920
    y2 = 526

    box1 = squareCrop(x1, y1, x2, y2)


class squareCrop:
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.w = x2 - x1
        self.h = y2 - y1


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


def getRow(img, x1, y1, x2, y2):
    return img[y1:y2, x1:x2]

def sliceRow(img, x1, x2):
    return img[:, x1:x2]

def preprocess(img):
    img = whiteThreshold(img)
    img = grayscale(img)
    # img = cv2.GaussianBlur(img, (7, 7), 0)
    # img = binaryThreshol d(img, 32)
    return img

def identifyText(img):
    # data = pytesseract.image_to_data(img)
    data = pytesseract.image_to_string(img)
    return data
    
def processRow(img):
    x_offset = 3182
    x1_item = 3278 - x_offset
    x2_item = 3647 - x_offset
    x1_price = 3650 - x_offset
    x2_price = 3838 - x_offset
    x1_avail = 4370 - x_offset
    x2_avail = 4452 - x_offset
    x1_time = 4545 - x_offset
    x2_time = 4629 - x_offset
    x1_location = 4632 - x_offset
    x2_location = 4780 - x_offset

    item = sliceRow(img, x1_item, x2_item)
    price = sliceRow(img, x1_price, x2_price)
    available = sliceRow(img, x1_avail, x2_avail)
    timeLeft = sliceRow(img, x1_time, x2_time)
    location = sliceRow(img, x1_location, x2_location)

    processThese = [item, price, available, timeLeft, location]
    row = []
    for img in processThese:
        img = preprocess(img)
        data = identifyText(img)
        row.append(data)
    
    return row


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # windowCaptureRealtime()
    # listWindowNames()
    # windowCaptureSave('Yeet')
    # tesseractTest()

    # img = cv2.imread('soul_03.jpeg')
    # tesseractTest(whiteThreshold(img))

    # drawRows()

    x1 = 3182-1920
    y1 = 423
    x2 = 4780-1920
    y2 = 525
    img = cv2.imread('soul_03.jpeg')
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

    # price = sliceRow(row, 100, 250)
    # whiteprice = preprocess(price)
    # identifyText(whiteprice)
    # displayImage(whiteprice)
    
    
    
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    # thresh = cv2.bitwise_not(thresh1)

    # x = 4002-1920
    # y = 379
    # x2 = 4089-1920
    # y2 = 420
    # w = x2-x
    # h = y2-y
    # cropped = thresh[y:y + h, x:x + w]
    
    # file = open("recognized.txt", "a")
    # text = pytesseract.image_to_string(cropped)
    # file.write(text)
    # file.write("\n")
    # file.close

    # cv2.imshow('img', cropped)
    # cv2.waitKey(0)



'''
Price "XXXX.XX" 
Tier = ["I", "II", "III", "IV", "V"] 
G.S.: Number or -
Gem: 
Perk: "Symbols so don't do this one"
RarityList = ["Common", "Uncommon", "Rare", "Epic", "Legendary" ]
Avail.: Integer  
Owned: Integer
Time: ["2d", "1m", "1h"]
Location = ["Windsward", "Monarch's Bluff", ]
'''

# Could use regex probably

# could tune the thresholding value over time
# Honestly just do a for loop perhaps? 

# Binary search for thresholding value? 

'''
Ojbective
* Get the global prices of soul motes 

Steps
* Trading post can time out so make sure it's always still opened
    * Save last state so we can go back

* Click "Showing orders at" and scroll up 
* Click All Settlements
* Check that All Settlements are selected
* Close the Settlements selection window

* Search for soul motes in the SEARCH ITEMS bar
* Check whether item was succesfully found
* Select soul mote
* Make sure items are sorted by Price downwards
* Start grabbing lines
* Scroll to bottom if applicable (currently this will miss two middle rows)
* Go to next page (if applicable)
* repeat for all pages 

* repeat for all wanted items

* save data into database
* generate report

 Place into database

'''