def closest(list, number):
    pass

def convertToSec(progTime):
    [h, m, s] = map(float, progTime.split(":"))
    return h * 3600 + m * 60 + s