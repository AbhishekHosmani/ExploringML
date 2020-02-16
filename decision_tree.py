# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
# Anjum Chida (anjum.chida@utdallas.edu)
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import os
import graphviz
import math
from scipy import stats
import collections
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    part_dict = {}
    keys = np.unique(x)
    for key in keys:
        part_dict[key] = np.where(x == key)
    return part_dict
    # INSERT YOUR CODE HERE

    # CODE ENDS HERE
    raise Exception('Function not yet implemented!')


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # INSERT YOUR CODE HERE
    counter = collections.Counter(y)
    total = 0
    entropy = 0
    for i in counter:
        total += counter[i]
    for i in counter:    
        entropy += counter[i]/total * np.log2(counter[i]/total)
    return -entropy
    # CUSTOM CODE ENDS HERE
    raise Exception('Function not yet implemented!')


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    # INSERT YOUR CODE HERE
    dict = {}
    S = entropy(y)
    subsets = partition(x)
    split_value = 0

    y_len = np.size(y, 0)
    for set in subsets:
        indices = subsets[set][0]
        if len(indices) != y_len:
            split_value = (len(indices) / y_len * entropy(y[indices])) + ((y_len - len(indices)) / y_len * entropy(np.delete(y, indices)))      
            MI = S - split_value
            dict[set] = MI
        else:
            dict[set] = 0
    return dict
    raise Exception('Function not yet implemented!')


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    # My Code starts here.
    attribute_value_pairs = []
    temp_dict = {}
    col_num = x.shape[1]
    for column in range(col_num):
        col_name = column
        temp_dict[col_name] = np.unique(x[:,column])
    for key, value in temp_dict.items():
        for val in value:
            temp = (key, val)
            attribute_value_pairs.append(temp)

    #Base conditions
    if entropy(y) == 0:
        return stats.mode(y)[0][0]
    elif len(attribute_value_pairs[0]) == 0:
        return stats.mode(y)[0][0]
    elif max_depth == depth:
        return stats.mode(y)[0][0]
    else:
        info_gains = {}
        for col, val in attribute_value_pairs:
            gain = mutual_information(x[:,col],y)
            gain_key = (col, val)
            if val in gain:
                info_gains[gain_key] = gain[val]
            else:
                info_gains[gain_key] = 0

        split_criteria = max(info_gains, key=info_gains.get)
        attribute_value_pairs.remove(split_criteria)
        # print("Chosen Node is : ", gain_rec)
        # print("updated attribute list : ", attribute_value_pairs)
        # positives = (gain_rec[0], gain_rec[1], True)
        # negatives = (gain_rec[0], gain_rec[1], False)
        # return dict({positives : id3(x[np.where(x[:,gain_rec[0]]==gain_rec[1])],y[np.where(x[:,gain_rec[0]]==gain_rec[1])],attribute_value_pairs,depth+1,max_depth),
        # negatives: id3(x[np.where(x[:,gain_rec[0]]!=gain_rec[1])],y[np.where(x[:,gain_rec[0]]!=gain_rec[1])],attribute_value_pairs,depth+1,max_depth)})

        
        tree = {}
        col = int(split_criteria[0])
        value = int(split_criteria[1])  
        indices = partition(x[:, col])  
        tree[split_criteria + (True, )] = id3(x[indices[value]], y[indices[value]], attribute_value_pairs, depth+1, max_depth)
        tree[split_criteria + (False, )] = id3(np.delete(x, indices[value], 0), np.delete(y, indices[value]), attribute_value_pairs, depth+1, max_depth)
    return tree

    raise Exception('Function not yet implemented!')


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    for parent, child in tree.items():
        if x[parent[0]] == parent[1] and parent[2] == True:
            if isinstance(child, dict):
                return predict_example(x, child)
            else:
                return child
        elif x[parent[0]] != parent[1] and parent[2] == False:
            if isinstance(child, dict):
                return predict_example(x, child)
            else:
                return child            
    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    if len(y_true) != len(y_pred):
        print("Not equal comparision")
    return (1/len(y_true))*(sum(y_true != y_pred))
    return error
    # INSERT YOUR CODE HERE
    raise Exception('Function not yet implemented!')


def analysis(dataset, Xtrn, ytrn, Xtst, ytst, error_dict):
    M = np.genfromtxt('./data/monks_data/'+dataset+'.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./data/monks_data/'+dataset+'.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    for i in range(1,11):
        decision_tree = id3(Xtrn, ytrn, max_depth=i)
        # Pretty print it to console
        pretty_print(decision_tree)
        # Visualize the tree and save it as a PNG image
        dot_str = to_graphviz(decision_tree)
        render_dot_file(dot_str, './my_learned_tree')
        # Compute the test error
        y_pred_test = [predict_example(x, decision_tree) for x in Xtst]
        y_pred_train = [predict_example(x, decision_tree) for x in Xtrn]
        tst_err = compute_error(ytst, y_pred_test)
        trn_err = compute_error(ytrn, y_pred_train)
        print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
        error_dict[(dataset, i)] = [(tst_err*100),(trn_err*100)]
    print(error_dict)

def error_plots(error_dict, dataset):
    train_errors = []
    test_errors = []

    for key, val in error_dict.items():
        if key[0] == dataset:
            train_errors.append(val[1])
            test_errors.append(val[0])

    plt.xlabel("Depth of the Decision Tree")
    plt.ylabel("Error of the Decision Tree")
    plt.title("Train vs Test Error Plots for " + dataset)
    plt.plot(list(range(1,11)),test_errors, label="Test Values")
    plt.plot(list(range(1,11)),train_errors, label="Train Values")
    plt.legend(loc="upper right")
    plt.show()

def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./data/monks_data/monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./data/monks_data/monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)

    # Pretty print it to console
    pretty_print(decision_tree)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree)
    render_dot_file(dot_str, './my_learned_tree')

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    print(y_pred)
    tst_err = compute_error(ytst, y_pred)
    error_dict = {}
    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
    analysis('monks-1', Xtrn, ytrn, Xtst, ytst, error_dict)
    analysis('monks-2', Xtrn, ytrn, Xtst, ytst, error_dict)
    analysis('monks-3', Xtrn, ytrn, Xtst, ytst, error_dict)

    error_plots(error_dict, 'monks-1')
    error_plots(error_dict, 'monks-2')
    error_plots(error_dict, 'monks-3')

    #Confusion matrix using sklean confusion matrix module
    M = np.genfromtxt('./data/monks_data/monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./data/monks_data/monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    
    # Learn a decision tree of depth 1, 3 and 5
    for i in [1, 3, 5]:
        decision_tree = id3(Xtrn, ytrn, max_depth=i)
        # if i == 1:
        #     print("Tree of Monks-1 Dataset is:::")
        #     pretty_print(decision_tree)
        #     dot_str = to_graphviz(decision_tree)
        #     render_dot_file(dot_str, './my_learned_tree')
        y_pred = [predict_example(x, decision_tree) for x in Xtst]
        matrix = confusion_matrix(ytst, y_pred, labels=[0, 1])
        print("confusion Matrix of monks-1 for depth {} is :{}".format(i, matrix))

    # Decision tree using sklearn of depth 1, 3 and 5
    for i in [1, 3, 5]:
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=i, random_state=0)
        clf.fit(Xtrn,ytrn)
        matrix = confusion_matrix(ytst, y_pred, labels=[0, 1])
        print("confusion Matrix of monks-1 for depth {} is :{}".format(i, matrix))