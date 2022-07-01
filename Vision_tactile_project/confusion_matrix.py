from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk


labels = ['0','1']

true = open("E:/github/fu_Experimental_results/C3D_vision_C1D_tactile-y_targets.txt","r")
y_trueStr = true.read()  
y_trueList = list(y_trueStr) 
y_trueArrayTemp = np.array(y_trueList) 
if y_trueArrayTemp[-1]=='\n':
   y_trueArray = y_trueArrayTemp[0:-1]  
else:
   y_trueArray =y_trueArrayTemp

y_true = []
for i in range(0, len(y_trueArray), 25):  
    for j in range(1,24,3):
        if(i+j>len(y_trueArray)):
            break
        else:
            y_true.append(int(y_trueArray[i+j])) 

true.close()  
y_true = np.array(y_true)


pred = open("E:/github/fu_Experimental_results/C3D_vision_C1D_tactile-y_pred.txt","r")
y_predStr =pred.read()     
y_predList = list(y_predStr) 
y_predArrayTemp = np.array(y_predList) 
if y_predArrayTemp[-1]=='\n':
   y_predArray = y_predArrayTemp[0:-1] 
else:
   y_predArray =y_predArrayTemp


y_pred = []
for n in range(0, len(y_predArray), 25):  
    for m in range(1,24,3):
        if( n+m > len(y_trueArray)):
            break
        else:
            y_pred.append(int(y_predArray[n+m])) 
pred.close()   
y_pred = np.array(y_pred)

print("Precision", sk.metrics.precision_score(y_true, y_pred,average='macro'))
print( "Recall", sk.metrics.recall_score(y_true, y_pred,average='macro'))
print( "f1_score", sk.metrics.f1_score(y_true, y_pred,average='macro'))


tick_marks = np.array(range(len(labels))) + 0.5
# ):
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.BuPu):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90,fontsize=25)
    plt.yticks(xlocations, labels,fontsize=25)
    plt.ylabel('True label',fontsize=25)
    plt.xlabel('Predicted label',fontsize=25)


cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print (cm_normalized)
plt.figure(figsize=(15,12), dpi=300)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.6f" % (c,), color='red', fontsize=30, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
#plt.savefig('Plots/confusion_matrix_log_1_421.png', format='png')
plt.show()