import re
import sys
import math
import random

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
				weight.append(random.uniform(0, 0.1))
				#weight.append(0)
			#weight.append(0)
			inputandhidden.append(weight)

		self.inputandhidden = inputandhidden

		hiddenandoutput = []

		for i in range(0, hiddenunitsnum+1): #the last one functions as the bias
			hiddenandoutput.append(random.uniform(0,0.1))
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
	#print x,1/(1 + math.exp(-x)) 
	return 1.0/(1.0 + math.exp(-x))

def forward(network, inputs):
	#inputs.append(1.0)#add bias
	output = []
	hidden = []
	result2 = 0.0

	for hiddenlayer in range(0, network.hiddenunitsnum):

		result1 = 0.0

		for inputlayer in range(0, network.inputnum):
			result1 += network.inputandhidden[hiddenlayer][inputlayer]*inputs[inputlayer]
		result1 += network.inputandhidden[hiddenlayer][network.inputnum]
		#print result1	
		result = sigmoid(result1)

		hidden.append(result)

		result2 += result*network.hiddenandoutput[hiddenlayer]

	result2 += network.hiddenandoutput[-1]
	
	result2 = sigmoid(result2)

	output.append(hidden)
	output.append(result2)
	#inputs.pop()
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
			#loss1 = (output[1]-label)*(output[1]-label)
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
					network.inputandhidden[i][j] += learningrate*delta[i]*output[0][i]
			#output = forward(network, inputs)
			#loss2 = (output[1]-label)*(output[1]-label)
			
			'''
			if loss1<loss2:
				print 'warning',loss1,loss2
			
			if loss1>0.25 and loss2<0.25:
				print 'attention',loss1, loss2
			'''
			
		
		count =0
		if ite%50 == 0:
			loss = 0
			for instance in data.instance:
				if instance[-2] == data.attribute[-1][1]:
					label = 0
				else:
					label = 1
				#print instance
				a = forward(network, instance[0:len(instance)-2])[1]
				if  a<0.5:
					label0 = data.attribute[-1][1]
				else:
					label0 = data.attribute[-1][2]
				if label0==instance[-2]:
					count+=1
				loss+=(label-a)*(label-a)
				#print label0,instance[-2],a
			print ite,count,(count+0.0)/data.instnumber,loss
		#return network
		
	#return count

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

	for line in result:
		output[line[-1]] = line[0:4]

	for line in output:
		print line[0]+1,line[1],line[2],line[3]
	print count,(count+0.0)/traindata.instnumber





# get parameter
#trainfile = sys.argv[1]
#numfolds = int(sys.argv[2])
#learningrate = float(sys.argv[3])
#numepochs = int(sys.argv[4])

trainfile = "sonar.arff"
numfolds = 10
learningrate = 0.1
numepochs = 1000

#load dataset
traindata = readArff(trainfile)

#intialize the network
network = Network(traindata.attrnumber-1,traindata.attrnumber-1,[],[])
#network.show()
#training with cross validation
#CVtraining(traindata, numfolds, learningrate, numepochs)


backward(network,traindata,traindata.instance,1,500)
backward(network,traindata,traindata.instance,0.5,500)
backward(network,traindata,traindata.instance,0.4,500)
backward(network,traindata,traindata.instance,0.3,500)
backward(network,traindata,traindata.instance,0.2,500)
backward(network,traindata,traindata.instance,0.1,500)
backward(network,traindata,traindata.instance,0.05,1000)
	

'''
for i in range(0,numepochs):
	a = backward(network,traindata,traindata.instance,learningrate,i)
	
	if a < 80:
		learningrate = 0.05
'''

'''
count =0
for instance in traindata.instance:
	#print instance
	a = forward(network, instance[0:len(instance)-2])[1]
	#print a
	if  a<0.5:
		label = traindata.attribute[-1][1]
	else:
		label = traindata.attribute[-1][2]
	if label!=instance[-1]:
		count+=1
	print label,instance[-2],a
'''




