import numpy as np
import nltk
import csv
import operator
import sklearn
from sklearn.naive_bayes import MultinomialNB
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# import feature
from extract_features import lexicalFeatures, syntacticFeatures, bagOfWordsFeatures, extract_mean_syllables_per_word_features, extract_unique_word_features

# import main training function
from train import train

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def predictAuthors(training_fvs, labels, test_fvs):
    """
    Predict author using Naive bayes Classifier

    :param training_fvs: list of training feature vectors
    :param labels: list of actual labels mapped onto training_fvs
    :param test_fvs: list of test feature vectors
    :return: list of predicted labels mapped onto tes_fvs
    """
    clf = MultinomialNB()
    clf.fit(training_fvs, labels)
    return clf.predict(test_fvs)

def probability(training_data):
    total_count_map = {}
    for id, text, author in training_data:
        if author not in total_count_map.keys():
            total_count_map[author] = 0
        total_count_map[author] += 1

    for k, v in total_count_map.items():
        total_count_map[k] = v * 100 / float(len(training_data))

    count_list = reversed(sorted(total_count_map.items(), key=operator.itemgetter(1)))
    result_list = []
    print('\n', 'random probability')
    for ele in count_list:
        print(ele)
        result_list.append(ele)

    return result_list

# Calculate metrics
def calculate_metrics(confusion_matrix):
    tp = {k: confusion_matrix[k][k] for k in confusion_matrix}
    fp = {k: sum(confusion_matrix[j][k] for j in confusion_matrix if j != k) for k in confusion_matrix}
    fn = {k: sum(confusion_matrix[k][j] for j in confusion_matrix if j != k) for k in confusion_matrix}
    
    accuracy = sum(tp.values()) / sum(sum(v.values()) for v in confusion_matrix.values())
    
    precision = {k: tp[k] / (tp[k] + fp[k]) if tp[k] + fp[k] > 0 else 0 for k in tp}
    recall = {k: tp[k] / (tp[k] + fn[k]) if tp[k] + fn[k] > 0 else 0 for k in tp}
    f1 = {k: 2 * precision[k] * recall[k] / (precision[k] + recall[k]) if precision[k] + recall[k] > 0 else 0 for k in tp}
    
    return accuracy, precision, recall, f1


if __name__ == '__main__':
    print('training the machine on data...')
    training_data, all_text_dict = train('training_data/train_data.csv', True)
    test_data = train('training_data/test_validation.csv', False)

    feature_sets = list(lexicalFeatures(training_data, test_data))
    feature_sets.append(bagOfWordsFeatures(all_text_dict, test_data))
    feature_sets.append(syntacticFeatures(all_text_dict, test_data))
    #feature_sets.append(extract_unique_word_features(all_text_dict, test_data))
    # remove for speed related reasons
    #feature_sets.append(extract_mean_syllables_per_word_features(all_text_dict, test_data))
    print("predicting:\n")
    classifications = [predictAuthors(fvs, labels, test) for fvs, labels, test in feature_sets]
    truth_values = []
    for el in test_data:
        truth_values.append(el[0][:3])
    #print(truth_values)
    # part4: evaluate the probability of random choice
    count_list = probability(training_data)

    print('\n', 'result table')
    final_answer = {}
    for results in classifications:
        print(' '.join(results))
        for test_count, result in enumerate(results, 0):
            if test_count not in final_answer:
                final_answer[test_count] = []
            final_answer[test_count].append(result)

    print('\n', 'final result')

    test_id_list = [''] * len(test_data)
    for e, (id, text, author) in enumerate(test_data):
        test_id_list[e] = id

    for k, v in final_answer.items():
        count_map = {'AG': 0, 'JB': 0, 'RK': 0, 'SS': 0, 'MA': 0}
        for name in v:
            count_map[name] += 1
        max_val = []
        max_count = 0
        for name, num in count_map.items():
            if max_count < num:
                max_val = [name]
                max_count = num
            elif max_count == num:
                max_val.append(name)

        max_val = max_val[0]

        final_answer[k] = max_val

    """for i in range(len(test_id_list)):
        print('{}\t{}'.format(test_id_list[i], final_answer[i]))"""

    print(final_answer)

    # Sample final_answer and test_id_list
    #final_answer = ['AG', 'jb', 'ss', 'MA', 'rk']
    #test_id_list = ['id1', 'id2', 'id3', 'id4', 'id6']
    testdict = {0: 'AG', 1: 'AG', 2: 'AG', 3: 'AG', 4: 'AG', 5: 'AG', 6: 'AG', 7: 'AG', 8: 'AG', 9: 'AG',
           10: 'JB', 11: 'JB', 12: 'JB', 13: 'JB', 14: 'JB', 15: 'JB', 16: 'JB', 17: 'JB', 18: 'JB', 19: 'JB',
           20: 'RK', 21: 'RK', 22: 'RK', 23: 'RK', 24: 'RK', 25: 'RK', 26: 'RK', 27: 'RK', 28: 'RK', 29: 'RK',
           30: 'SS', 31: 'SS', 32: 'SS', 33: 'SS', 34: 'SS', 35: 'SS', 36: 'SS', 37: 'SS', 38: 'SS', 39: 'SS',
           40: 'MA', 41: 'MA', 42: 'MA', 43: 'MA', 44: 'MA', 45: 'MA', 46: 'MA', 47: 'MA', 48: 'MA', 49: 'MA', 50: 'MA'}

    # Mapping of ids to corresponding final_answer values
    id_map = {'id1': 'AG', 'id2': 'jb', 'id3': 'ss', 'id4': 'MA', 'id6': 'rk'}

   # Convert ground truth IDs to the corresponding labels
    ground_truth_labels = [id_map[id_] for id_ in truth_values]

    # Extract the predicted labels
    predicted_labels = list(final_answer.values())

    # Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(ground_truth_labels, predicted_labels)
    precision = precision_score(ground_truth_labels, predicted_labels, average='weighted')
    recall = recall_score(ground_truth_labels, predicted_labels, average='weighted')
    f1 = f1_score(ground_truth_labels, predicted_labels, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Results
results = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
}

# Write the results to a CSV file
with open("validation_results_table.csv", "w", newline="") as csvfile:
    fieldnames = ["Metric", "Value"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for key, value in results.items():
        writer.writerow({"Metric": key, "Value": value})

# Write the results to a CSV file
with open("randomprob.csv", "w", newline="") as csvfile:
    fieldnames = ["Metric", "Value"]
    writer = csv.writer(csvfile)

    for el in count_list:
        writer.writerow([el])


print("Results saved to 'validation_results_table.csv'")
