from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from operator import itemgetter
import os
from collections import Counter

import shutil
import sys
import numpy

def report_average(reports):
    mean_dict = dict()
    for label in reports[0].keys():
        dictionary = dict()

        if label in 'accuracy':
            mean_dict[label] = sum(d[label] for d in reports) / len(reports)
            continue

        for key in reports[0][label].keys():
            dictionary[key] = sum(d[label][key] for d in reports) / len(reports)
        mean_dict[label] = dictionary

    return mean_dict

crossValRootPath = "C:\\Users\\chsalgad\\source\\repos\\Project_NLP_Salgado\\Project_NLP_Salgado\\Config\\Dataset\\songs\\CrossVal\\"
crossValCount = sum(os.path.isdir(os.path.join(crossValRootPath, i)) for i in os.listdir(crossValRootPath))

reports = dict()
for x in range(1, crossValCount + 1):
    crossValIterationPath = os.path.join(crossValRootPath, str(x))

    with open(os.path.join(crossValIterationPath, "groundTruth")) as f:
        y_true = [line.rstrip().split() for line in f]

    y_true = sorted(y_true, key=itemgetter(0))
    y_true = [i[1:] for i in y_true]

    for file in os.listdir(crossValIterationPath):
        if file.startswith("predictions"):
            if file not in reports:
                reports[file] = []

            with open(os.path.join(crossValIterationPath, file)) as f:
                y_pred = [line.rstrip().split() for line in f]

                y_pred = sorted(y_pred, key=itemgetter(0))
                y_pred = [i[1:] for i in y_pred]

                m = MultiLabelBinarizer().fit(y_true)
                # Print the confusion matrix
                # print(metrics.confusion_matrix(y_true, y_pred))

                # Print the precision and recall, among other metrics
                report = metrics.classification_report(m.transform(y_true), m.transform(y_pred), digits=4, target_names=m.classes_, output_dict=True)
                reports[file].append(report)

    # print(report)

for key in reports:
    average_report = report_average(reports[key])
    print(key)
    print(*average_report.items(), sep='\n')
    print("\n")


def report_average(reports):
    mean_dict = dict()
    for label in reports[0].keys():
        dictionary = dict()

        if label in 'accuracy':
            mean_dict[label] = sum(d[label] for d in reports) / len(reports)
            continue

        for key in reports[0][label].keys():
            dictionary[key] = sum(d[label][key] for d in reports) / len(reports)
        mean_dict[label] = dictionary

    return mean_dict

# for x in range(1, 6):
#     with open("C:\\Users\\chsalgad\\source\\repos\\Project_NLP_Salgado\\Project_NLP_Salgado\\Config\\Dataset\\songs\\CrossVal\\{}\\test{}.txt".format(x, x)) as f:
#         lines = ["test/{}".format(' '.join(numpy.array(line.rstrip().split('|'))[[0, 5]])) for line in f]
#
#     # with open("C:\\Users\\chsalgad\\source\\repos\\Project_NLP_Salgado\\Project_NLP_Salgado\\Config\\Dataset\\songs\\CrossVal\\{}\\training{}.txt".format(x, x)) as f:
#     #     lines += ["training/{}".format(' '.join(numpy.array(line.rstrip().split('|'))[[0, 5]])) for line in f]
#
#     with open("C:\\Users\\chsalgad\\source\\repos\\Project_NLP_Salgado\\Project_NLP_Salgado\\Config\\Dataset\\songs\\CrossVal\\{}\\groundTruth".format(x), "w") as file:
#         file.writelines("%s\n" % l for l in lines)

# for x in range(1, 6):
#     with open("C:\\Users\\chsalgad\\source\\repos\\Project_NLP_Salgado\\Project_NLP_Salgado\\Config\\Dataset\\songs\\CrossVal\\{}\\cats.txt".format(x, x)) as f:
#         lines = [line.rstrip().split()[0].split('/') for line in f]
#
#     if not os.path.exists(os.path.join("C:\\Users\\chsalgad\\source\\repos\\Project_NLP_Salgado\\Project_NLP_Salgado\\Config\\Dataset\\songs\\CrossVal\\{}\\".format(x), 'training')):
#         os.makedirs(os.path.join("C:\\Users\\chsalgad\\source\\repos\\Project_NLP_Salgado\\Project_NLP_Salgado\\Config\\Dataset\\songs\\CrossVal\\{}\\".format(x), 'training'))
#
#     if not os.path.exists(os.path.join("C:\\Users\\chsalgad\\source\\repos\\Project_NLP_Salgado\\Project_NLP_Salgado\\Config\\Dataset\\songs\\CrossVal\\{}\\".format(x), 'test')):
#         os.makedirs(os.path.join("C:\\Users\\chsalgad\\source\\repos\\Project_NLP_Salgado\\Project_NLP_Salgado\\Config\\Dataset\\songs\\CrossVal\\{}\\".format(x), 'test'))
#
#     for line in lines:
#         shutil.copyfile("C:\\Users\\chsalgad\\source\\repos\\Project_NLP_Salgado\\Project_NLP_Salgado\\Config\\Dataset\\songs\\Lyrics\\{}.txt".format(line[1]),
#                             "C:\\Users\\chsalgad\\source\\repos\\Project_NLP_Salgado\\Project_NLP_Salgado\\Config\\Dataset\\songs\\CrossVal\\{}\\{}\\{}".format(x, line[0], line[1]))
