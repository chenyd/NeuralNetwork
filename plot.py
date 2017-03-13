import re
import sys
import math
import random
import matplotlib.pyplot as plt

class Data:#include attribute, attrnumber, attrvaluenumber, instance, instnumber
	def __init__(self, attribute, attrnumber, attrvaluenumber, instance, instnumber):
		self.attribute = attribute
		self.attrnumber = attrnumber
		self.attrvaluenumber = attrvaluenumber
		self.instance = instance
		self.instnumber = instnumber

class Network:#include inputnum, hiddenunitsnum, inputandhidden, hiddenandoutput
	def __init__(self, inputnum, hiddenunitsnum, inputandhidden, hiddenandoutput):
		self.inputnum = inputnum
		self.hiddenunitsnum = hiddenunitsnum

		inputandhidden = []

		for i in range(0, hiddenunitsnum):
			weight = []
			for j in range(0, inputnum+1): #the last one functions as the bias
				weight.append(random.uniform(-0.05, 0.05))
				#weight.append(0)
			#weight.append(0)
			inputandhidden.append(weight)

		self.inputandhidden = inputandhidden

		hiddenandoutput = []

		for i in range(0, hiddenunitsnum+1): #the last one functions as the bias
			hiddenandoutput.append(random.uniform(-0.05,0.05))
			#hiddenandoutput.append(0)
		#hiddenandoutput.append(random.uniform(0,0.1))
		self.hiddenandoutput = hiddenandoutput

	def show(self):
		for i in range(0, self.hiddenunitsnum):
			print self.inputandhidden[i][-1]
		print ""
		print self.hiddenandoutput
		
def readArff(filename):#return arffData (class Data)
	arffData = Data([], 0, [], [], 0)
	arff = open(filename)
	i = 0
	p = re.compile('@attribute \'')
	q = re.compile('\' { ')
	r = re.compile(' ')
	s = re.compile('}')
	t = re.compile(',')
	for line in arff:
		if line[0] == "%":
			continue
		if line[0: 5] == "@data":
			break
		if line[0: 10] == "@attribute":
			#adjust the form of attribute
			attribute = p.sub('',line)
			attribute = q.sub(',',attribute)
			attribute = s.sub('',attribute)
			attribute = r.sub('',attribute)
			attriline = t.split(attribute[0:len(attribute)-1])
			arffData.attribute.append(attriline)
			#Save as [attribute name, attribute values] 
			arffData.attrvaluenumber.append(len(attriline)-1)
			i = i + 1
	arffData.attrnumber = i

	count = 0

	inwhichclass = 1

	for line in arff:
		if line[0] == "@" or line[0] == "%" or line[0] == " ":
			continue
		if line[-1] == '\n':
				training = t.split(line[0:len(line)-1])
		else:
				training = t.split(line[0:len(line)])
		numericline = []
		for i in range(0,len(training)-1):
			float(training[i])
			numericline.append(float(training[i]))
		numericline.append(training[len(training)-1])
		numericline.append(count)
		arffData.instance.append(numericline)
		count = count + 1
	arffData.instnumber = count

	arff.close()	

	return arffData

def sigmoid(x):
	return 1.0/(1.0 + math.exp(-x))

def forward(network, inputs):
	output = []
	hidden = []
	result2 = 0.0

	for hiddenlayer in range(0, network.hiddenunitsnum):

		result1 = 0.0

		for inputlayer in range(0, network.inputnum):
			result1 += network.inputandhidden[hiddenlayer][inputlayer]*inputs[inputlayer]
		result1 += network.inputandhidden[hiddenlayer][network.inputnum]	
		result = sigmoid(result1)

		hidden.append(result)

		result2 += result*network.hiddenandoutput[hiddenlayer]

	result2 += network.hiddenandoutput[-1]
	
	result2 = sigmoid(result2)

	output.append(hidden)
	output.append(result2)
	return output

def backward(network, data, traindata, learningrate,numepochs):
	random.shuffle(traindata)
	for ite in range(0, numepochs): 
		for instance in traindata:
			inputs = instance[0:len(instance)-2]
			if instance[-2] == data.attribute[-1][1]:
				label = 0
			else:
				label = 1
			output = forward(network, inputs)
			#for output unit
			deltaoutput = output[1]*(1 - output[1])*(label - output[1])	

			#for each hidden unit
			delta = []
			for i in range(0,network.hiddenunitsnum):
				delta.append(output[0][i]*(1-output[0][i])*network.hiddenandoutput[i]*deltaoutput)	

			output[0].append(1)	

			for i in range(0,network.hiddenunitsnum+1):
				network.hiddenandoutput[i] += learningrate*deltaoutput*output[0][i]

			inputs.append(1)	

			for i in range(0,network.hiddenunitsnum):	

				for j in range(0,network.inputnum+1):
					network.inputandhidden[i][j] += learningrate*delta[i]*inputs[j]
			
	output=[]
	loss = 0
	for instance in data.instance:
		a = forward(network, instance[0:len(instance)-2])[1]
		output.append(a)
			
	
	return output

def preparefolds(traindata, numfolds):
	#prepare the training set
	trainset = []
	class1 = []
	class2 = []	

	##Stratification
	for instance in traindata.instance:
		if instance[-2] == traindata.attribute[-1][1]:
			class1.append(instance)
		else:
			class2.append(instance)	

	#prepare the fold
	trainset = []
	trainsummary = []	

	for i in range(0,numfolds):
		trainset.append([])
		trainsummary.append([])	

	for i in range(0,numfolds):
		trainsummary[i].append(0)
		trainsummary[i].append(0)	

	###select instances
	while class1 != []:
		for i in range(0,numfolds):
			if class1 == []:
				break
			trainset[i].append(class1.pop())
			trainsummary[i][0]+=1	

	while class2 != []:
		for i in range(0,numfolds):
			if class2 == []:
				break
			trainset[i].append(class2.pop())
			trainsummary[i][1]+=1

	'''
	##show the proportion of each fold
	for i in range(0,len(trainset)):
		print trainsummary[i][0],trainsummary[i][1],(trainsummary[i][0]+0.0)/(trainsummary[i][0]+trainsummary[i][1])
	'''

	return trainset

def CVtraining(traindata, numfolds, learningrate, numepochs):
	#prepare folds
	trainset = preparefolds(traindata, numfolds)
	
	result = []
	count = 0
	for i in range(0, numfolds):
		#intialize the network
		network = Network(traindata.attrnumber-1,traindata.attrnumber-1,[],[])
		index = range(0,numfolds)
		index.pop(i)
		datafortraining = []
		for j in index:
			for instance in trainset[j]:
				datafortraining.append(instance)
		backward(network,traindata,datafortraining,learningrate, numepochs)
		#test
		for instance in trainset[i]:
			resultline = []
			resultline.append(i)
			a = forward(network,instance[0:len(instance)-2])[1]
			if a<0.5:
				label = traindata.attribute[-1][1]
				#confidence = 1 - a
				confidence = a
			else:
				label = traindata.attribute[-1][2]
				confidence = a
			resultline.append(label)
			resultline.append(instance[-2])
			resultline.append(confidence)
			resultline.append(instance[-1])
			result.append(resultline)
			if label == instance[-2]:
				count += 1

	output = []
	
	for i in range(0,len(traindata.instance)):
		output.append(0)
	'''
	for line in result:
		output[line[-1]] = line[0:4]

	for line in output:
		print line[0]+1,line[1],line[2],line[3]
	'''
	print count,(count+0.0)/traindata.instnumber
	return (count+0.0)/traindata.instnumber

def ROC(traindata, numfolds, learningrate, numepochs):
	#prepare folds
	trainset = preparefolds(traindata, numfolds)
	
	result = []
	count = 0
	PLOT = []
	for i in range(0, numfolds):
		#intialize the network
		network = Network(traindata.attrnumber-1,traindata.attrnumber-1,[],[])
		index = range(0,numfolds)
		index.pop(i)
		datafortraining = []
		for j in index:
			for instance in trainset[j]:
				datafortraining.append(instance)
		output = backward(network,traindata,datafortraining,learningrate, numepochs)
		output.sort()
		output.reverse()
		X=[]
		Y=[]
		Z=[]
		for line in output:
			indicator = 0
			TP = 0.0
			FP = 0.0
			FN = 0.0
			TN = 0.0
			for instance in traindata.instance:
				a = forward(network,instance[0:len(instance)-2])[1]
				if a<line:
					label = traindata.attribute[-1][1]
					if label == instance[-2]:
						TN +=1
					else:
						FN +=1
				else:
					label = traindata.attribute[-1][2]
					if label == instance[-2]:
						TP +=1
					else:
						FP +=1
			FPR = (FP+0.0)/(TN+FP)
			TPR = (TP+0.0)/(TP+FN)
			X.append(FPR)
			Y.append(TPR)
			if indicator==0:
				Z.append(TPR)
				PLOT.append([FPR,TPR])
			else:
				Z.append(TPR-Z[-1])
				PLOT.append([FPR,TPR-Z[-1]])
			indicator +=1
	
	PLOT.sort(key=lambda x:x[0])
	yvalue = 0
	XX=[]
	YY=[]
	for point in PLOT:
		XX.append(point[0])
		yvalue+=(point[1]+0.0)/numfolds
		YY.append(yvalue)
	return [XX,YY]


trainfile = "sonar.arff"

#load dataset
traindata = readArff(trainfile)

'''
#plot for epochs 25,50,75,100, lr=0,1, folds=10
epochs = [25,50,75,100]
accuracy1 = []
for ite in epochs:
	accuracy1.append(CVtraining(traindata, 10, 0.1, ite))

plt.plot(epochs,accuracy1,'bo-')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.savefig("AccuracyVsEpochs.eps")

#plot for folds 5, 10, 15, 20, 25 lr=0.1,epochs=50
folds = [5,10,15,20,25]
accuracy2 = []
for num in folds:
	accuracy2.append(CVtraining(traindata, num, 0.1, 50))

plt.plot(folds,accuracy2,'bo-')
plt.xlabel('Folds')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Folds')
plt.savefig("AccuracyVsFolds.eps")
'''

#ROC for lr=0.1, epochs=50,folds=10
graph = ROC(traindata, 10, 0.1, 5)
plt.plot(graph[0],graph[1],'bo-')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')
plt.savefig("ROC.eps")




