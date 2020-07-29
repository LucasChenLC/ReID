import tensorflow as tf
import numpy as np
import random

def evaluate_model_mAP(model,query,query_labels,test_data,test_labels,id_num, rank=100,query_num=1000):
    mAP = 0
    for i in range(query_num):
        predictions = []
        for _ in range(100):
            image = test_data[random.randint(0,test_data.shape[0])]
            data = np.append(query[i], image, axis=2)
            data = np.expand_dims(data, axis=0)
            predict = model.predict(data)
            predictions.append(predict[0])
        predictions = np.array(predictions)
        precision_recall = {}
        tf_id = query_labels[i]
        tf_num = 0
        for j in range(rank):
            match = np.argmax(predictions)
            del predictions[match]
            if test_labels[match] == tf_id:
                tf_num = tf_num+1
            precision = tf_num / (j+1)
            recall = tf_num / id_num[tf_id]
            if recall in precision:
                if precision_recall[recall] < precision:
                    precision_recall[recall] = precision
            else:
                precision_recall[recall] = precision

        sorted(precision_recall)
        first = True
        for recall, precision in precision_recall:
            if first:
                first = False
                AP = precision * recall
                last_precision = precision
                last_recall = recall
            else:
                AP = AP + ((last_precision + precision) * (recall - last_recall))/2
        mAP = mAP + AP

    mAP = mAP/query_num
    print(mAP)
