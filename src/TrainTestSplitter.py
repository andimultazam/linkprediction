from pyspark import SparkConf, SparkContext
import os
from shutil import copyfile
import Utils

TRAIN_PCT=0.7
TEST_PCT=0.3
FEATNAMES="featnames"
TAILS = ["circles", "edges", "egofeat", "feat"]

#Create SparkContext
conf = SparkConf()\
    .setAppName("Train/Test Data Splitter Program")
sc = SparkContext(conf=conf)

#Input parameters
#src = sys.argv[1]
src = '/home/kienguye/NUS/BigData/FinalProject/twitter_t1'

#Train-Test folders
trainFolder = src + "/train"
testFolder = src + "/test"
Utils.removeExistingDir(trainFolder)
Utils.removeExistingDir(testFolder)


#Splitting data
filenames = os.listdir(src)
for filename in filenames:
    srcFilename = src+"/"+filename
    print("Splitting train-test data for file {}".format(srcFilename))
    if(filename.endswith(FEATNAMES)):
        copyfile(srcFilename, trainFolder+"/"+filename)
        copyfile(srcFilename, testFolder+"/"+filename)
    else:
        lines = sc.textFile(srcFilename).filter(lambda l: l != None).cache()
        trainLines = lines.sample(False, TRAIN_PCT).cache()
        trainLines.saveAsTextFile(trainFolder+"/"+filename)
        testLines = lines.subtract(trainLines).saveAsTextFile(testFolder+"/"+filename)
    print("Finished splitting train-test data for file {}".format(srcFilename))
