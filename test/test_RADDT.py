import torch 
from torchvision import datasets, transforms
import pands as pd 
import os

import sys
sys.path.append('./src/')

from warmStart import CARTClfWarmStart
from treeFunc import readTreePath, objv_cost_CalMetrics_binaryClass
from imgDatasetFunc import PathDatasetReader
from RADDT import  multiStartTreeOptbyGRAD_withC



if __name__ == "__main__":
    

    ################## main Code ##################
    # torch.autograd.set_detect_anomaly(True)

    torch.manual_seed(970425)
    
    ## Args 
    treeDepth = int(sys.argv[1])     # 3 
    epochNum = int(sys.argv[2])      # 3000

    deviceArg = str(sys.argv[3])                 
    device = torch.device(deviceArg)
    startNum = int(sys.argv[4])                     # e.g. 1, 2, 3, 4, 5...
    numScale = int(sys.argv[5])                     # e.g. 1, 2, 3, 4, 5...

    nClass = 2 


    ## Prepare the image dataset
    dataBasePath = "../../data/"
    trueTrainDataCSVDir = "xxxxxxxx"
    soft_data_dir = "xxxxxxx"
    hardTestDataCSVDir = "xxxxxxxx"

    transformer = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])



    # take vid_i as the testing set, the rest as training set
    for test_vid in range(1, 11):
        print(f"\nProcessing video {test_vid}...")

        trainTrueNormalImgPath_pd = pd.read_csv(os.path.join(trueTrainDataCSVDir, f"trainNormalImgPath_{test_vid}.csv"), header=0)
        trainTrueNormalImgPath = trainTrueNormalImgPath_pd['imagePath'].tolist()
        trainTrueNormalImgFullPath = [dataBasePath + path for path in trainTrueNormalImgPath]
        trainTrueNormalLabels = trainTrueNormalImgPath_pd['label'].tolist()
        
        trainTrueSmokeImgPath_pd = pd.read_csv(os.path.join(trueTrainDataCSVDir, f"trainSmokeImgPath_{test_vid}.csv"), header=0)
        trainTrueSmokeImgPath = trainTrueSmokeImgPath_pd['imagePath'].tolist()
        trainTrueSmokeImgFullPath = [dataBasePath + path for path in trainTrueSmokeImgPath]
        trainTrueSmokeLabels = trainTrueSmokeImgPath_pd['label'].tolist()
        
        trainTrueImgFullPath = trainTrueNormalImgFullPath + trainTrueSmokeImgFullPath
        trainTrueLabels = trainTrueNormalLabels + trainTrueSmokeLabels


        trainSoftImgPath_pd = pd.read_csv(os.path.join(soft_data_dir, f"softCSVForUnlabeledData_100_{test_vid}.csv"), header=0)
        trainSoftImgPath = trainSoftImgPath_pd['imagePath'].tolist()
        trainSoftImgFullPath = [dataBasePath + path for path in trainSoftImgPath]
        trainSoftLabels = trainSoftImgPath_pd['label'].tolist()


        finalTrainImgPath = trainTrueImgFullPath + trainSoftImgFullPath
        finalTrainLabel = trainTrueLabels + trainSoftLabels


        testTrueNormalImgPath_pd = pd.read_csv(os.path.join(hardTestDataCSVDir, f"testNormalImgPath_{test_vid}.csv"), header=0)
        testTrueNormalImgPath = testTrueNormalImgPath_pd['imagePath'].tolist()
        testTrueNormalImgFullPath = [dataBasePath + path for path in testTrueNormalImgPath]
        testTrueNormalLabels = testTrueNormalImgPath_pd['label'].tolist()
        testTrueSmokeImgPath_pd = pd.read_csv(os.path.join(hardTestDataCSVDir, f"testSmokeImgPath_{test_vid}.csv"), header=0)
        testTrueSmokeImgPath = testTrueSmokeImgPath_pd['imagePath'].tolist()
        testTrueSmokeImgFullPath = [dataBasePath + path for path in testTrueSmokeImgPath]
        testTrueSmokeLabels = testTrueSmokeImgPath_pd['label'].tolist()
        testTrueImgFullPath = testTrueNormalImgFullPath + testTrueSmokeImgFullPath
        testTrueLabels = testTrueNormalLabels + testTrueSmokeLabels


        ## Create datasets
        trainDataset = PathDatasetReader(finalTrainImgPath, finalTrainLabel, transform=transformer)
        testDataset = PathDatasetReader(testTrueImgFullPath, testTrueLabels, transform=transformer)
        nClass = trainDataset.print_class_distribution()
        print(f"Train dataset size: {len(trainDataset)}")
        print(f"Test dataset size: {len(testDataset)}")

        # stack tensors 
        XTrain = torch.stack([img for img, _, _ in trainDataset]).view(len(trainDataset), -1).float().to(device) # training
        YTrain = torch.tensor([label for _, label, _ in trainDataset]).long().to(device)  # training labels
        XTest = torch.stack([img for img, _, _ in testDataset]).view(len(testDataset), -1).float().to(device)  # testing
        YTest = torch.tensor([label for _, label, _ in testDataset]).long().to(device)  # testing labels
        print(f"XTrain shape: {XTrain.shape}, YTrain shape: {YTrain.shape}")
        print(f"XTest shape: {XTest.shape}, YTest shape: {YTest.shape}")



        ##  Train DT model 
        # read the treePath from the HDF5 file
        indices_flags_dict = readTreePath(treeDepth, device)

        # cart warm start
        aInit, bInit, cInit = CARTClfWarmStart(XTrain, YTrain, treeDepth, nClass, device)
        cartWarmStart_dict = {"a": aInit, "b": bInit, "c": cInit}
        warmStart = [cartWarmStart_dict]
        acc_DDTCur, treeDDTCur = multiStartTreeOptbyGRAD_withC(XTrain, YTrain, treeDepth, nClass, indices_flags_dict, epochNum, device, warmStart, startNum, numScale)

        print("\n\nTesting results:")
        treeDDT_accuracy, treeDDT_precision, treeDDT_recall, treeDDT_f1score, treeDDT_FPRate, treeDDT_FNRate = objv_cost_CalMetrics_binaryClass(XTest, YTest, treeDepth, treeDDTCur)

        ## final results
        print("\nFinal Results...")
        print("acc_test:{}; precision_test:{}; recall_test:{}; f1score_test:{}; FPRate_test:{}; FNRate_test:{}".format(treeDDT_accuracy, treeDDT_precision, treeDDT_recall, treeDDT_f1score, treeDDT_FPRate, treeDDT_FNRate))