import cv2
import numpy as np
import time

np.random.seed(20)
class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        #################################

        self.net = cv2.dnn.DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, "r") as f:
            self.classesList = f.read().splitlines()
            self.classesList.insert(0, "__Background__")

            self.colorList = np.random.uniform(0, 255, size=(len(self.classesList), 3))


            #print(self.classesList)

    def onvideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if not cap.isOpened():
            print("Error: Opening file......")
            return
        startTime = 0

        while True:
            success, image = cap.read()  
            if not success:
                break
        
            currentTime = time.time()
            fps = 1/(currentTime - startTime)
            startTime = currentTime
            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold=0.5)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)

            if len(bboxIdx) != 0:
                for i in range(len(bboxIdx)):
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]

                    displayText = "{}:{:.2f}".format(classLabel, classConfidence)
                    x, y, w, h = bbox

                    cv2.rectangle(image, (x, y), (x + w, y + h), color=classColor, thickness=1)
                    cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, classColor, 2)

                    ##############################
                    linewidth = min(int(w * 0.25), int(h * 0.25) ) # Adjust this value as needed


                    cv2.line(image, (x, y), (x + linewidth, y), color=classColor, thickness=5)
                    cv2.line(image, (x, y), (x , y+ linewidth), color=classColor, thickness=5)

                    cv2.line(image, (x + w, y), (x + w - linewidth, y), color=classColor, thickness=5)
                    cv2.line(image, (x + w, y), (x + w, y+ linewidth), color=classColor, thickness=5)
                    
                    ###########################
                    cv2.line(image, (x, y + h), (x + linewidth, y + h), color=classColor, thickness=5)
                    cv2.line(image, (x, y + h), (x , y+ h - linewidth), color=classColor, thickness=5)

                    cv2.line(image, (x + w, y + h), (x + w - linewidth, y + h), color=classColor, thickness=5)
                    cv2.line(image, (x + w, y + h), (x + w, y+ h - linewidth), color=classColor, thickness=5)

            cv2.putText(image, "FPS: " + str(int(fps)), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Result", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
 
        cap.release()
        cv2.destroyAllWindows()
