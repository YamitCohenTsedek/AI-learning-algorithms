from math import log2


def read_dataset(file_name):
    """
    Read the examples from the specified file.

    :param file_name: the name of the file to read from.
    :returns:
        - attributes - a list of the attributes of the examples.
        - dataset - a list of all the examples including their classifications.
    """
    attributes = []
    dataset = []
    try:
        with open(file_name, 'r') as file:
            line = file.readline()
            attributes.extend(line.split())
            while line:
                line = file.readline()
                # split_line - a list of the attribute-values of the example, where the last element is the classification.
                split_line = line.split("\t")
                split_line[-1] = split_line[-1][0:-1]  # Cut the last "\n" in the classification.
                dataset.append(split_line)  # Add the example to the list.
            # If there is a blank line at the end of the file - delete it.
            if dataset[-1] == ['']:
                dataset = dataset[:-1]
    finally:
        file.close()
    # Return a list of the attributes of the examples and a list of all the examples including their classifications.
    return attributes, dataset


def hamming_distance(example, neighbor):
    """
    Compute the Hamming distance between an example and its neighbor. Hamming distance between two strings is
    the number of positions at which the corresponding symbols are different.

    :param example: an example from the dataset.
    :param neighbor: a neighbor of the given example.
    :returns: the Hamming distance between example and neighbor.
    """
    example = example[:-1]  # Cut the classifications - only the attributes are relevant for the computation.
    neighbor = neighbor[:-1]
    num_of_diffs = 0  # A counter for the differences.
    for char_1, char_2 in zip(example, neighbor):
        if char_1 != char_2:
            num_of_diffs += 1
    return num_of_diffs


def return_first_element(obj):
    """
    :param obj: an object that contains elements.
    :returns: the first element of the object.
    """
    return obj[0]


def return_second_element(obj):
    """
    :param obj: an object that contains elements.
    :returns: the second element of the object.
    """
    return obj[1]


def knn(train_set, test_set):
    """
    KNN learning algorithm - the examples are classified by the majority vote of their k nearest neighbors.

    :param train_set: the train set examples.
    :param test_set: the test set examples.
    :returns: predictions - the predictions of the test set examples.
    """
    k = 5  # k is the number of the nearest neighbors.
    predictions = []
    for example in test_set:
        distances = []
        # Compute the distances from the example to all its neighbors.
        for neighbor in train_set:
            distance = hamming_distance(example, neighbor)
            distances.append((distance, neighbor))  # Save the distance to the neighbor and the neighbor object.
        distances.sort(key=return_first_element)  # Sort the distances from the smallest to the biggest.
        k_nearest_neighbors = distances[:k]  # Choose the k nearest neighbors.
        # Cut the distances since they are no longer needed.
        for i in range(0, len(k_nearest_neighbors)):
            k_nearest_neighbors[i] = k_nearest_neighbors[i][-1]
        neighbors_votes = dict()
        # Find the votes of the k nearest neighbors.
        for neighbor in k_nearest_neighbors:
            vote = neighbor[-1]
            if vote in neighbors_votes:
                neighbors_votes[vote] += 1
            else:
                neighbors_votes[vote] = 1
        # Classify according to the majority vote of the k nearest neighbors.
        sorted_votes = sorted(neighbors_votes.items(), reverse=True, key=return_second_element)
        predictions.append(sorted_votes[0][0])
    return predictions


def attributes_possible_values(data_set):
    """
    Create a list of sets - the index of each set in the list corresponds to the attribute index. The elements
    of the set represent all the possible values of the attribute.

    :param data_set: the data set of the examples.
    :returns: list of sets of all the possible values of the attributes.
    """
    attributes_values = []
    for i in range(len(data_set[0]) - 1):
        attributes_values.append(set())
    for example in data_set:
        for i in range(len(example) - 1):
            attributes_values[i].add(example[i])
    return attributes_values


def count_same_values_in_class(separated_classes, class_val, attribute_index, attribute_value):
    """
    Count how many examples with the specified classification have the same value to the specified attribute.

    :param separated_classes: a dictionary which its keys are classifications and the values are the examples.
    :param attribute_index: the index of the attribute.
    :param attribute_value: the required value of the attribute.
    :returns: the number of examples with the specified classification and the same value to the specified attribute.
    """
    counter = 0
    for instance in separated_classes[class_val]:
        if instance[attribute_index] == attribute_value:
            counter += 1
    return counter


def same_values_in_all_classes(attribute_index, attribute_value, examples):
    """
    Count for each classification separately how many examples have the same value to the specified attribute.

    :param attribute_index: the index of the attribute.
    :param attribute_value: the required value of the attribute.
    :param examples: a list of all the examples.
    :returns:
        - class_votes - a dictionary which its keys are classifications and the values are the number of examples
                        that have these classifications.
        - sub_examples - a list of all the examples that have the same value to the specified attribute.
    """
    class_votes = {}
    sub_examples = []
    for example in examples:
        if example[attribute_index] == attribute_value:
            sub_examples.append(example)
            # The classification is the last element of the list that represents the example.
            classification = example[-1]
            if classification not in class_votes.keys():
                class_votes[classification] = 1
            else:
                class_votes[classification] += 1
    return class_votes, sub_examples


def separate_to_classes(examples):
    """
    Separate the examples by their classifications.

    :param examples: a list of all the examples.
    :returns: separated_classes - a dictionary which its keys are classes and its values are lists
              of all the examples of that class.
    """
    separated_classes = dict()
    for i in range(len(examples)):
        instance = examples[i]
        # The classification is the last element of the list that represents the example.
        curr_class = instance[-1]
        if curr_class not in separated_classes:
            separated_classes[curr_class] = list()
        separated_classes[curr_class].append(instance)
    return separated_classes


def naive_bayes(train_set, test_set):
    """
    Applying Bayes' theorem with strong (naive) independence assumptions between the attributes.

    :param train_set: the train set examples.
    :param test_set: the test set examples.
    :returns: test_predictions - a list of the predictions of the test set.
    """
    separated_classes = separate_to_classes(train_set)  # Separate the examples by their classifications.
    attributes_values = attributes_possible_values(train_set)  # The attributes and all their possible values.
    # The prior probabilities of the classifications is the proportion between the number of examples with a specific
    # classification to the number of all the examples.
    prior_classes_probabilities = dict()
    for curr_class in separated_classes.keys():
        prior_classes_probabilities[curr_class] = len(separated_classes[curr_class])/float(len(train_set))
    test_predictions = []  # The predictions of the test examples.
    for instance in test_set:
        class_prediction = ""
        max_probability = 0
        for curr_class in separated_classes.keys():
            # Multiply the probabilities of aj | ci when aj is the attribute and ci is the classification.
            cond_probs_mult = 1
            for i in range(len(instance) - 1):
                k = len(attributes_values[i])  # Use k for smoothing.
                cond_probs_mult *= (count_same_values_in_class(separated_classes, curr_class, i, instance[i]) + 1) \
                    / (len(separated_classes[curr_class]) + k)
            probability = prior_classes_probabilities[curr_class] * cond_probs_mult
            # Find the argmax - the classification that maximizes the probability.
            if probability > max_probability:
                max_probability = probability
                class_prediction = curr_class
        test_predictions.append(class_prediction)
    return test_predictions


class Tree(object):
    """ Class Tree represents a decision tree for ID3 algorithm. """

    def __init__(self, root):
        """
        :param root: the root of the tree.
        """
        self.root = root
        self.children = []

    def add_child(self, v_i, sub_tree=None):
        """
        Add a branch to the root.

        :param v_i: the value of the branch.
        :param sub_tree: the subtree of the branch.
        :returns: None.
        """
        self.children.append((v_i, sub_tree))

    def get_root(self):
        """
        :returns: the root of the tree.
        """
        return self.root

    def get_children(self):
        """
        :returns: all the children of the root.
        """
        return self.children

    def is_leaf(self):
        """
        :returns: True if the current tree is a leaf (has no children), False - otherwise.
        """
        return self.children == []

    def tree_representation(self, attributes_names, depth):
        """
        :param attributes_names: the names of the attributes of the tree.
        :param depth: the depth of the tree.
        :returns: the string that represents the tree as required.
        """
        tree_str = ""
        # If the current tree is not a leaf - get the current attribute.
        if not self.is_leaf():
            current_attribute = attributes_names[self.root]
        children = self.get_children()  # Get the children of the current root.
        # Iterate over the children when they are lexicographically sorted.
        for child in sorted(children, key=return_first_element):
            # Create the depth string.
            if depth == 0:
                depth_str = ""
            else:
                depth_str = depth * "\t" + "|"
            # If the subtree of the current child is a leaf.
            if child[1].is_leaf():
                tree_str += '{}{}={}:{}\n'.format(depth_str, current_attribute, child[0], child[1].get_root())
            # If we reach to a subtree.
            else:
                tree_str += '{}{}={}\n'.format(depth_str, current_attribute, child[0])
                # A recursive call to the tree representation of the child.
                tree_str += (child[1]).tree_representation(attributes_names, depth + 1)
        return tree_str

    def predict(self, test_example):
        """
        Predict the classification of a test example by the decision tree.

        :param test_example: the test example that we want predict its classification.
        :returns: the classification of the test example.
        """
        # The classifications are in the leaves.
        if self.is_leaf():
            return self.root
        attribute = self.get_root()
        children = sorted(self.get_children(), key=return_first_element)
        value = test_example[attribute]
        for child_value, subtree in children:
            if child_value == value:
                break
        # A recursive call to the prediction of the child.
        return subtree.predict(test_example)


def mode(examples):
    """
    :param examples: a list of all the examples.
    :returns:
        - The classification of the majority.
        - A boolean type - True if this classification has the biggest number of votes, False if the number of
                           votes of this classification is equal to the number of votes of the other classification.
    """
    # class_votes - a dictionary which its keys are classifications, and the values are the number of the examples
    #               that have these classifications.
    class_votes = dict()
    for example in examples:
        curr_class = example[-1]
        if curr_class not in class_votes.keys():
            class_votes[curr_class] = 1
        else:
            class_votes[curr_class] += 1
    # Sort the classifications by the number of votes - from the biggest number to the smallest number.
    sorted_votes = sorted(class_votes.items(), reverse=True, key=return_second_element)
    if sorted_votes[0][1] > sorted_votes[1][1]:
        return sorted_votes[0][0], True
    else:
        return sorted_votes[0][0], False


def entropy(separated_classes, num_of_examples):
    """
    Compute the entroy value of the examples.

    :param separated_classes:  a dictionary which its keys are classes and its values are lists
                               of all the examples of that class.
    :param num_of_examples: the number of examples.
    :returns: the entroy value of the examples.
    """
    if num_of_examples == 0:
        return 0
    entropy_val = 0
    prior_classes_probabilities = dict()
    for curr_class in separated_classes.keys():
        # The prior probabilities of the classifications is the proportion between the number of examples with
        # a specific classification to the number of all the examples.
        prior_classes_probabilities[curr_class] = separated_classes[curr_class] / float(num_of_examples)
    for curr_class in prior_classes_probabilities.keys():
        prob = prior_classes_probabilities[curr_class]
        # Compute the entropy.
        if prob == 1:
            return 0
        elif prob != 0:
            entropy_val += (-prob * log2(prob))
    return entropy_val


def choose_attribute(attributes_values, examples):
    """
    Choose the best current attribute.

    :param attributes_values - a dictionary which its keys are the attributes and its values are the possible values.
    :returns: best_attribute - the best current attribute.
    """
    separated_classes = separate_to_classes(examples)
    for curr_class in separated_classes.keys():
        separated_classes[curr_class] = len(separated_classes[curr_class])
    initial_entropy = entropy(separated_classes, len(examples))  # The initial entropy of all the examples.
    max_information_gain = -1
    best_attribute = -1
    for attribute in attributes_values.keys():
        possible_values = attributes_values[attribute]
        information_gain = initial_entropy
        for value in possible_values:
            # class_votes - a dictionary which its keys are classifications and the values are the number of the examples
            #               that have these classifications.
            # sub_examples - a list of all the examples that have the same value to the specified attribute.
            class_votes, sub_examples = same_values_in_all_classes(attribute, value, examples)
            # Compute the entropy.
            entropy_val = entropy(class_votes, len(sub_examples))
            # Compute the information gain.
            information_gain -= (len(sub_examples) / float(len(examples))) * entropy_val
        # Find the max gain.
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            best_attribute = attribute
    return best_attribute


def all_examples_same_classification(examples):
    """
    :param: examples - a list of all the examples.
    :returns: the classification if all the examples have the same classification, otherwise - an empty string.
    """
    flag = 1
    classification = examples[0][-1]
    for example in examples:
        if example[-1] != classification:
            flag = 0
            break
    if flag:
        return classification
    else:
        return ""


def DTL(examples, attributes_values, default):
    """
    DTL algorithm - Top-Down Induction of Decision Trees ID3.

    :param examples - a list of all the examples.
    :param attributes_values - a dictionary which its values are the attributes and its values are the possible values.
    :returns: tree - a decision tree.
    """
    # If there are no more examples - return a tree which its root is the default.
    if len(examples) == 0:
        return Tree(default)
    else:
        # If all the examples have the same classification - return it.
        same_classification = all_examples_same_classification(examples)
        if same_classification != "":
            return Tree(same_classification)
        # If the are no more attributes - return the choice of the majority.
        if len(attributes_values) == 0:
            mode_val, is_biggest = mode(examples)
            if is_biggest:
                return Tree(mode_val)
            # If there is equality - return the father's default.
            else:
                return Tree(default)
        best = choose_attribute(attributes_values, examples)  # Choose the current best attribute.
        tree = Tree(best)  # Create a tree which its root is the best attribute.
        # Iterate over the possible values of the best attribute.
        for v_i in attributes_values[best]:
            # Count for each classification separately how many examples have the same value to the best attribute.
            class_votes, sub_examples = same_values_in_all_classes(best, v_i, examples)
            sub_attributes = attributes_values.copy()  # Create a copy of attributes_values.
            del sub_attributes[best]  # Delete the current best attribute.
            child_default, is_biggest = mode(examples)  # The mode of the child.
            # If the number of the votes of the classifications is equal - choose the default of the father.
            if not is_biggest:
                child_default = default
            sub_tree = DTL(sub_examples, sub_attributes, child_default)  # Create the sub tree of the child.
            tree.add_child(v_i, sub_tree)  # Add the branch of the child to the current tree.
        return tree


def ID3(train_set):
    """
    ID3 algorithm creates a decision tree.

    :param train_set - the train set examples.
    :returns: decision_tree - a decision tree.
    """

    # attributes_values is a list of sets. The index of each set in the list corresponds to the attribute index.
    # The elements of the set represent all the possible values of the attribute.
    attributes_values = attributes_possible_values(train_set)
    attributes = dict()
    # Convert attributes_values, which is a list, to a dictionary for convenience.
    for i in range(len(attributes_values)):
        attributes[i] = attributes_values[i]
    # Create the decision tree by Top-Down Induction of Decision Trees ID3.
    decision_tree = DTL(train_set, attributes, mode(train_set))
    return decision_tree


def k_fold_cross_validation(dataset):
    """
    # Split the data into k folds - k-1 folds for the train set and the last fold for the test set.

    :param: dataset - a list of the examples.
    :returns: folds - the division of the data into k folds.
    """
    k = 5  # k is the number of the folds.
    fold_length = int((len(dataset))/k)
    start_index = 0
    ending_index = fold_length
    folds = []
    # Create the folds.
    for i in range(k - 1):
        folds.append(dataset[start_index:ending_index])
        start_index = ending_index
        ending_index += fold_length
    folds.append(dataset[start_index:])
    return folds


def accuracy_k_fold_cross_validation(learning_algorithm, dataset):
    """
    Compute the accuracy of the given learning algorithm with k fold cross validation method.

    :param learning_algorithm: the learning algorithm to compute the accuracy for.
    :param: dataset - a list of the examples.
    :returns: averaged_accuracy - the averaged accuracy of the of k folds.
    """
    folds = k_fold_cross_validation(dataset)  # Split the data into k folds.
    averaged_accuracy = 0
    # Create the train set and the test set from the folds.
    for i in range(0, len(folds)):
        test_set = folds[i]
        train_set = []
        for j in range(len(folds)):
            if j != i:
                train_set += folds[j]
        # If the learning algorithm is ID3, create the decision tree and find the predictions of all the examples.
        if learning_algorithm == ID3:
            decision_tree = ID3(train_set)
            test_predictions = []
            for example in test_set:
                test_predictions.append(decision_tree.predict(example))
        else:
            test_predictions = learning_algorithm(train_set, test_set)
        correct = 0
        # Compute the current accuracy.
        for l in range(len(test_predictions)):
            if test_set[l][-1] == test_predictions[l]:
                correct += 1
        # The averaged accuracy is the average of the accuracy results of all the rounds.
        averaged_accuracy += (correct / float(len(test_predictions)))
    averaged_accuracy /= float(len(folds))
    return averaged_accuracy


def write_accuracy_in_file():
    """
    Write the accuracy results in file in the required format.
    """
    dataset_file_name = 'dataset.txt'
    dataset = read_dataset(dataset_file_name)[1]  # Read the data, i.e. - all the examples.
    # Find the accuracy results of the algorithms.
    knn_accuracy = accuracy_k_fold_cross_validation(knn, dataset)
    naive_bayes_accuracy = accuracy_k_fold_cross_validation(naive_bayes, dataset)
    ID3_accuracy = accuracy_k_fold_cross_validation(ID3, dataset)
    # Write the accuracy results in file in the required format and then close the file.
    try:
        with open("accuracy.txt", 'w') as accuracy_file:
            accuracy_file.write("{:.2f}".format(ID3_accuracy) + "\t" + "{:.2f}".format(knn_accuracy) + "\t" +
                                "{:.2f}".format(naive_bayes_accuracy))
    finally:
        accuracy_file.close()


def write_decision_tree_in_file():
    """
    Write the decision tree in a file in the required format.
    """
    dataset_file_name = 'dataset.txt'
    attributes, dataset = read_dataset(dataset_file_name)
    tree = ID3(dataset)
    # Get the string that represents the tree as required.
    tree_representation = tree.tree_representation(attributes, 0)
    # Write the decision tree in a file in the required format and then close the file.
    try:
        with open("tree.txt", 'w') as tree_file:
            tree_file.write(tree_representation)
    finally:
        tree_file.close()


# main function will run the real train and test files.
def main():
    train_file = "train.txt"
    test_file = "test.txt"
    # Read the examples of the train and the examples of the test.
    attributes, train_set = read_dataset(train_file)
    attributes, test_set = read_dataset(test_file)
    # Find the predictions of the test set.
    knn_predictions = knn(train_set, test_set)
    naive_bayes_predictions = naive_bayes(train_set, test_set)
    decision_tree = ID3(train_set)
    decision_tree_predictions = []
    for example in test_set:
        decision_tree_predictions.append(decision_tree.predict(example))
    # Compute the accuracy of each algorithm.
    ID3_accuracy = 0
    knn_accuracy = 0
    naive_bayes_accuracy = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == decision_tree_predictions[i]:
            ID3_accuracy += 1
        if test_set[i][-1] == knn_predictions[i]:
            knn_accuracy += 1
        if test_set[i][-1] == naive_bayes_predictions[i]:
            naive_bayes_accuracy += 1
    ID3_accuracy /= len(test_set)
    knn_accuracy /= len(test_set)
    naive_bayes_accuracy /= len(test_set)
    # Write the results in a file in the required format and then close the file.
    try:
        with open("output.txt", 'w') as output_file:
            output_file.write(decision_tree.tree_representation(attributes, 0) + "\n")

            output_file.write("{:.2f}".format(ID3_accuracy) + "\t" + "{:.2f}".format(knn_accuracy) + "\t" +
                              "{:.2f}".format(naive_bayes_accuracy))
    finally:
        output_file.close()
    write_decision_tree_in_file()
    write_accuracy_in_file()


if __name__ == "__main__":
    main()
