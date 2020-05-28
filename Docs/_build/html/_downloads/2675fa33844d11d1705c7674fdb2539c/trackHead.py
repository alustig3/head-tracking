import cv2
import numpy as np
import time as t
import math
print("OpenCV Version : %s " % cv2.__version__)
import csv
import os

def getVideoList():
    vidList = []
    for filename in os.listdir(os.getcwd()): #create list of mpg video files in this directory.
        if filename.endswith(".mpg"):
            vidList.append(filename)
    print(vidList)
    return vidList,vidList[0][:-9]

def createOutputFilenames(rootName):
    csvOUT = rootName + "_track.csv"
    videoOUT = rootName + "_track.mp4"
    return csvOUT,videoOUT

def createVideoOutput(sampleClip,outputFilename,fps):
    tempCap = cv2.VideoCapture(sampleClip)
    width = tempCap.get(3) #cv2.CAP_PROP_FRAME_WIDTH
    height = tempCap.get(4) #cv2.CAP_PROP_FRAME_HEIGHT
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoObj = cv2.VideoWriter(outputFilename,fourcc, fps, (3*int(width),int(height))) #Create output video
    return videoObj

def createOutputCSV(csvFilename,rootName):
    outFile = open(csvFilename, 'w',newline='')
    writer = csv.writer(outFile, delimiter=',')
    stampFile = open(rootName+"timestamps.csv" ,'rt')
    reader = csv.reader(stampFile)
    return outFile,writer,reader

def printTimeStats(_fps,wasInterrupted=False):
    print("\n")
    elapsed = t.time()-startTime
    m,s = divmod(elapsed,60)
    h,m = divmod(m,60)
    if wasInterrupted:
        print ("Interupted on FrameNum", frameNum)
    print ("Elapsed time: {}h {}m {}s".format(int(h),int(m),round(s,2)))
    print ("Speed: {} frames processed/sec. It took {}x the playback speed to complete the processing ".format(round(frameNum/elapsed,2), round(_fps/(frameNum/elapsed),2)))

def cleanFrame(startFrame):
    startFrame[:3,:10] = 0       #overwrites timestamp pixels in the top left corner
    startFrame[185:192,398:405] = 0 #overwrites right nosepoke IR LED that shows up on camera

def createOutputFrame(frame):
    outFrame = frame.copy()
    outFrame[:] =25
    return outFrame

class colorMask:
    def __init__(self,lowHue,highHue,lowSat,highSat,lowBright,highBright,hueIsBetween=True):
        self.isBetween = hueIsBetween
        lowHue,highHue = round(lowHue/255*179),round(highHue/255*179) #input will be based on imageJ scale of 0-255. openCV uses scale 0-179 (see:http://docs.opencv.org/trunk/df/d9d/tutorial_py_colorspaces.html)
        if self.isBetween:
            self.lowThresh = np.array([lowHue,lowSat,lowBright])
            self.highThresh = np.array([highHue,highSat,highBright])
        else: #the hues are on the outside the given threshold values. This will be true if you are tring to create a mask in the red spectrum
            self.leftLowThresh = np.array([0,lowSat,lowBright])
            self.leftHighThresh = np.array([lowHue,highSat,highBright])
            self.rightLowThresh = np.array([highHue,lowSat,lowBright])
            self.rightHighThresh = np.array([179,highSat,highBright])

    def applyMask(self,frame):
        hsvFrame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV) #convert frame to HSV color space
        if self.isBetween:
            temp = cv2.inRange(hsvFrame, self.lowThresh, self.highThresh)
        else:
            lowerMask = cv2.inRange(hsvFrame, self.leftLowThresh, self.leftHighThresh)
            higherMask = cv2.inRange(hsvFrame, self.rightLowThresh, self.rightHighThresh)
            temp = lowerMask + higherMask
        
        temp = cv2.erode(temp, np.ones((2,2)), iterations=1)
        self.colorPixels = cv2.dilate(temp, np.ones((2,2)), iterations =3)>0 ## convert to boolean array after dilation

class clusterPair:
    x1,x2,y1,y2 = 0,0,0,0
    def __init__(self,mask=None):
        if mask is not None :
            A,B = np.where(mask)
            X = np.array([A,B]).T
            Z = np.float32(X)
            # define criteria and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            self.compactness,label,self.centers= cv2.kmeans(Z,2,None,criteria,2,cv2.KMEANS_RANDOM_CENTERS) #compactness is the sum of squared distance from each point to their corresponding centers
            # check that there are actually 2 separate clusters
            delta_y,delta_x = self.centers[1] - self.centers[0]
            distance_between_centers = np.sqrt(delta_x**2 + delta_y**2)
            if distance_between_centers<8:
                # take average
                self.centers[0] = (self.centers[0] + self.centers[1])/2 
                self.centers[1] = (-1,-1)
            self.y1, self.x1 = self.centers[0] #cluster 1 center
            self.y2, self.x2 = self.centers[1] #cluster 2 center

        else:
            self.centers = np.array([[0,0],[0,0]])
            self.compactness = 0
    def swap(self):
        self.centers = self.centers[::-1]
        self.x1,self.x2,self.y1,self.y2 = self.x2,self.x1,self.y2,self.y1 #swap cluster coordinates
    def show(self):
        print(self.x1,self.y1,self.x2,self.y2) #print cluster coordinates

def compare_clusters(startPair,endPair):
    #startPair contains 2 clusters A(x1,y1) and B(x2,y2), endPair contains 2 clusters a(x1,y1) and b(x2,y2).
    #as the clusters are moving around, we want to label them
    #there are two possibilities:
        # Sequence 1 (AaBb): A becomes a and B becomes b
        # _OR_
        # Sequence 2 (AbBa): A becomes b and B becomes a
    AaBb = endPair.centers-startPair.centers #delta x's and delta y's for Sequence 1
    AbBa = endPair.centers[::-1]-startPair.centers #delta x's and delta y's for Sequence 2
    AaBb_dist = np.sqrt((AaBb**2).sum(axis=1)) #[distance between cluster A and a], [distance between cluster B and b]]
    AbBa_dist = np.sqrt((AbBa**2).sum(axis=1)) #[distance between cluster A and b], [distance between cluster B and a]]

    # start pair (S) and end pair (E), has both clusters (y) or is missing a cluster (n). There are 4 states we could be in:
        # SnEn
        # SnEy
        # SyEn
        # SyEy
    if endPair.x2 == -1: #(SnEn or SyEn) only one new cluster was found 
        Aa_dist,Ba_dist = AaBb_dist[0], AbBa_dist[1]
        if Ba_dist==0 or Aa_dist/Ba_dist>1.2: # figure out whether it was cluster A or cluster B that has become occluded. Compare the distance for both possibilities and choose the one that travels the shortest distance
            endPair.swap()
    elif startPair.x1 == -1 or startPair.x2 ==-1: # (SnEy) in the last frame, one of the clusters was occluded, but now it has reappeared 
        if startPair.x1 == -1: # if it's the first cluster that's reappearing 
            Bb_dist,Ba_dist = AaBb_dist[1], AbBa_dist[1]
            if Bb_dist>Ba_dist:
                endPair.swap()
        else:
            Aa_dist,Ab_dist = AaBb_dist[0], AbBa_dist[0]
            if Aa_dist>Ab_dist:
                endPair.swap()
    elif AbBa_dist.sum()==0 or AaBb_dist.sum()/AbBa_dist.sum()>1.2 : # (SyEy) whichever sequence results in the smallest combined distance traveled by the clusters, is the correct sequence
        endPair.swap()

####################################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~SCRIPT START~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
####################################################################

mpgList,baseFilename = getVideoList()
output_csv_filename, output_video_filename = createOutputFilenames(baseFilename)
csvFile,csvWriter,stampReader = createOutputCSV(output_csv_filename,baseFilename)
fps = 60
VidOutObject = createVideoOutput(mpgList[0],output_video_filename,fps)

# global variables
font = cv2.FONT_HERSHEY_SIMPLEX
frameNum = 0
past = -1
mills = 0
radius =  3 #cluster center marker size
numVideo = 0
#cluster colors BGR
orange = [0,153,255]
pink = [153,0,204]
green = [102,255,51]
blue = [255,102,0]
#theshold mask colors
lightRed = [100,100,255]
lightBlue = [165,255,100]
#Create mask objects
blueMask = colorMask(140,180,30,255,105,255)
redMask = colorMask(10,200,0,255,60,255,hueIsBetween=False)

print("Processing footage")
print("Press `ctrl + c` to stop")
startTime = t.time()
try:
    redPair,bluePair = clusterPair(),clusterPair() #initialize cluster pairs
    oldredPair,oldbluePair = redPair,bluePair      #initialize old pairs
    for clip in mpgList:
        print (int(t.time()-startTime),"s:", clip)
        cap = cv2.VideoCapture(clip)
        while(cap.isOpened()):
            ret, rawFrame = cap.read()
            if ret==True:# and frameNum<10:
                #create blank arrays of the same size as the footage
                pyramidRatFrame = rawFrame.copy()
                threshFrame = rawFrame.copy()
                threshFrame[:] = 25
                clusterFrame = threshFrame.copy()
                cleanFrame(pyramidRatFrame)
                blueMask.applyMask(pyramidRatFrame)
                redMask.applyMask(pyramidRatFrame)
                if redMask.colorPixels.sum()>2: # there should be at least 2 pixels, otherwise there is no point in applying a kmeans for 2 clusters
                    oldredPair = redPair
                    redPair = clusterPair(redMask.colorPixels) #get centers for both red clusters
                    compare_clusters(oldredPair,redPair)
                else:
                    redPair.x1,redPair.y1,redPair.x2,redPair.y2 = -1,-1,-1,-1
                if blueMask.colorPixels.sum()>2:
                    oldbluePair = bluePair
                    bluePair = clusterPair(blueMask.colorPixels)
                    compare_clusters(oldbluePair,bluePair)
                else:
                    bluePair.x1,bluePair.y1,bluePair.x2,bluePair.y2 = -1,-1,-1,-1

                cv2.circle(clusterFrame,(int(redPair.x1),int(redPair.y1)), radius, orange, -1)    #draw circle at center of 1st cluster
                cv2.circle(clusterFrame,(int(redPair.x2),int(redPair.y2)), radius, pink, -1)      #draw circle at center of 2nd cluster
                cv2.circle(clusterFrame,(int(bluePair.x1),int(bluePair.y1)), radius, green, -1)
                cv2.circle(clusterFrame,(int(bluePair.x2),int(bluePair.y2)), radius, blue, -1)

                stampRow = next(stampReader)
                now = int(stampRow[1])
                if past<0:
                    elapsed = now
                else:
                    elapsed = now-past
                if (now-past)<0: #milliseconds have rolled over
                    elapsed = elapsed + 1000
                mills = mills + elapsed
                past = now

                threshFrame[blueMask.colorPixels] = lightBlue
                threshFrame[redMask.colorPixels] = lightRed
                cv2.putText(clusterFrame,'frame {}, {}'.format(frameNum,mills) ,(10,30), font,.5,(255,255,255))
                # cv2.putText(threshFrame,'red {}, blue {}'.format(int(redPair.compactness),int(bluePair.compactness)) ,(10,30), font, .75,(255,255,255))

                combined = np.hstack((rawFrame,threshFrame,clusterFrame))

                if False: #change to True for live rendering
                    cv2.imshow('ratTube',combined)
                    cv2.waitKey(1)

                VidOutObject.write(combined)
                csvWriter.writerow([frameNum,stampRow[0],stampRow[1],stampRow[2],stampRow[3],mills,
                                    redPair.x1,redPair.y1,redPair.x2,redPair.y2,
                                    bluePair.x1,bluePair.y1,bluePair.x2,bluePair.y2])
                frameNum+=1
            else:
                break
        cap.release()
    VidOutObject.release()
    cv2.destroyAllWindows()
    csvFile.close()
    printTimeStats(fps)
except KeyboardInterrupt:
    cap.release()
    VidOutObject.release()
    cv2.destroyAllWindows()
    csvFile.close()
    printTimeStats(fps,wasInterrupted=True)