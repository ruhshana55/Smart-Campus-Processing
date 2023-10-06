"""
Usefull Links:
https://github.com/ultralytics/yolov5 documentiation for YOLOv5

https://www.youtube.com/watch?v=3wdqO_vYMpA&t=1273s Video going through the code that was used a referance for this
Note that for this to properly work you need to comment out line of code in panfy relating to dislike_count
"""
import torch
import numpy as np
import cv2
from time import time
from PIL import Image 
import PIL
import GetImageClusterV1Mouse

class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a image using OpenCV.
    """
    
    def __init__(self, image):
        """
        Initializes the class with image and output file.
        :param image: Has to be as image,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.Image = image
        self.model = self.load_model()
        self.classes = self.model.names
        #print(self.classes)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:",self.device)


    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        #note to change complexity of model change 'yolov5s'; look at https://github.com/ultralytics/yolov5 for different model complexities
        model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
        return model


    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
     
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        #labels, cord both output a list of information on found objects
        #print(cord)#outputs in [x1, y1, x2, y2, confidence]; x1,y1 and x2,y2 are the corner locations of the object bounding box; confidence is how confident the model is that its lable is correct
        #print(labels)#outputs an int between [0, 79] corisponding to an object lable
        
        return labels, cord


    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]


    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)#creates rectange around object
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def crop_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        frameList = []
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.5:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                #bgr = (0, 255, 0)
                #cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)#creates rectange around object
                position = [x1, y1, x2, y2]
                crop_frame = frame[y1:y2, x1:x2]
                #cv2.imshow("cropped", crop_frame)
                #cv2.waitKey(1000)
                frameList.append({'lable':self.class_to_label(labels[i]), 'image':crop_frame, 'position':position})
                #cv2.imwrite(r"C:\Users\lukez\PycharmProjects\CMSC499\TestFolder\Test" + str(i) +r".jpg", crop_frame)

        return frameList


    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
    
        #start_time = time()
        frame = self.Image
        results = self.score_frame(frame)
        frame = self.crop_boxes(results, frame)
        #end_time = time()
        #fps = 1/np.round(end_time - start_time, 3)
        #print(f"Frames Per Second : {fps}")
        #print('Frame: ', frame)
        return frame



def main():
    # Create a new object and execute.
    #InputIMG = cv2.imread(r"C:\Users\lukez\PycharmProjects\CMSC499\Room.jpg")
    ImageName = 'Room' 
    extension = '.jpg'
    #path1 = r"C:\Users\lukez\PycharmProjects\CMSC499"
    #path = path1 + '\\' + ImageName + extension
    path = r"C:\\Users\\lukez\\PycharmProjects\\CMSC499Project\\YOLOv8_test\\Room.jpg"
    image = path
    InputIMG = cv2.imread(image)
    detection = ObjectDetection(InputIMG)
    OutputIMG = detection()
    #print(OutputIMG)
    InpaintPositions = []

    for x in OutputIMG:
        print(x['lable'])
        
        if(x['lable'] == 'chair'):
            data = GetImageClusterV1Mouse.getImageInfo(x['image'], x['lable'])
            print(data)
            cost = float(data['Price'][1:len(data['Price'])])
            print(cost)
            if(cost > 80.00):
                print("append to InpaintPositions")
                InpaintPositions.append(x['position'])
        
    OutputIMG = InputIMG
    x_shape, y_shape = OutputIMG.shape[1], OutputIMG.shape[0]
    mask_img = np.zeros((y_shape, x_shape, 3), dtype = "uint8")
    bgr = (255, 255, 255)
    for position1 in InpaintPositions:
        cv2.rectangle(mask_img, (position1[0], position1[1]), (position1[2], position1[3]), bgr, -1)#creates black square at position 

    mask2 = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)#gray scale mask_img
    # Inpaint.
    OutputIMG1 = cv2.inpaint(OutputIMG, mask2, 3, cv2.INPAINT_NS)
    # Write the output.
    cv2.imwrite(ImageName + '_OutPut.png', OutputIMG1)



if __name__ == "__main__":

    import cProfile
    cProfile.run('main()', 'output.dat')

    import pstats
    from pstats import SortKey

    with open("output_time.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("time").print_stats()

    with open("output_calls.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("calls").print_stats()
