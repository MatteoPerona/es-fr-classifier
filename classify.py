import numpy as np 
import pandas as pd 

import itertools
import string
import re

class DecisionTree():
    """
    
    Decision Tree Classifier
    
    Attributes:
        root: Root Node of the tree.
        max_depth: Max depth allowed for the tree
        size_allowed : Min_size split, smallest size allowed for split 
        n_features: Number of features to use during building the tree.(Random Forest)
        n_split:  Number of split for each feature. (Random Forest)
    
    """

    def __init__(self, max_depth = 1000, size_allowed = 1, n_features = None, n_split = None):
        """
            Initializations for class attributes.
        """
        self.root = None             
        self.max_depth = max_depth         
        self.size_allowed = size_allowed      
        self.n_features = n_features        
        self.n_split = n_split           
    
    
    class Node():
        """
            Node Class for the building the tree.

            Attribute: 
                threshold: The threshold like if x1 < threshold, for spliting.
                feature: The index of feature on this current node.
                left: Pointer to the node on the left.
                right: Pointer to the node on the right.
                pure: Bool, describe if this node is pure.
                predict: Class, indicate what the most common Y on this node.

        """
        def __init__(self, threshold = None, feature = None):
            """
                Initializations for class attributes.
            """
            self.threshold = threshold   
            self.feature = feature    
            self.left = None
            self.right = None
            self.pure = None
            self.depth = None
            self.predict = None
    
    
    def entropy(self, lst):
        """
            Function Calculate the entropy given lst.
            
            Attributes: 
                entro: variable store entropy for each step.
                classes: all possible classes. (without repeating terms)
                counts: counts of each possible classes.
                total_counts: number of instances in this lst.
                
            lst is vector of labels.
        """
        
        entro = 0  
        classes = set(lst)  
        counts = [list(lst).count(c) for c in classes] 
        total_counts = len(lst)    
        for count in counts:       
            if count != 0:    
                p = count/total_counts    
                entro -= p * np.log2(p)
        return entro

    def information_gain(self, lst, values, threshold):
        """
            Function Calculate the information gain, by using entropy function.
            
            lst is vector of labels.
            values is vector of values for individule feature.
            threshold is the split threshold we want to use for calculating the entropy. 
        """
        
        left_prop = sum([1 for val in values if val < threshold]) / len(values) 
        right_prop = 1 - left_prop    

        left_labels = [
            label for i, label in enumerate(lst) if values[i] < threshold
        ]
        right_labels = [
            label for i, label in enumerate(lst) if values[i] >= threshold
        ]

        assert sorted(left_labels + right_labels) == sorted(lst)

        left_entropy = self.entropy(left_labels)    
        right_entropy = self.entropy(right_labels)   
        
        information_gain = (
            self.entropy(lst) 
            - 
            (left_prop * left_entropy + right_prop * right_entropy)
        )

        return information_gain  
    
    def find_rules(self, data):
        
        """
            Helper function to find the split rules.
            
            data is a matrix or 2-D numpy array, represnting training instances. 
            Each training instance is a feature vector. 
        """
        rules = []        
        for col in range(data.shape[1]):          
            unique_values = np.unique(data[:, col])       
            mid_points = [
                (unique_values[i] + unique_values[i+1]) / 2 for i in range(len(unique_values) - 1)
            ]        
            rules.append(mid_points)             
        return rules
    
    def next_split(self, data, label):
        """
            Helper function to find the split with most information gain, by using find_rules, and information gain.
            
            data is a matrix or 2-D numpy array, represnting training instances. 
            Each training instance is a feature vector. 
            
            label contains the corresponding labels. There might be multiple (i.e., > 2) classes.
        """
        
        rules = self.find_rules(data)             
        max_info = -1          
        num_col = None          
        threshold = None       
        entropy_y = None   

        """
            Check Number of features to use, None means all featurs. (Decision Tree always use all feature)
            If n_features is a int, use n_features of features by random choice. 
            If n_features == 'sqrt', use sqrt(Total Number of Features ) by random choice.
        """
        if num_col is None:
            index_col = list(range(data.shape[1]))
        else:
            if num_col == 'sqrt': 
                num_index = int(np.sqrt(data.shape[1]))
            else:
                num_index = num_col
            np.random.seed()  
            index_col = np.random.choice(
                data.shape[1], num_index, replace=False
            )
        
        """
            Do the similar selection we did for features, n_split take in None or int or 'sqrt'.
            For all selected feature and corresponding rules, we check it's information gain.       
        """
        for i in index_col:

            index_rules = []
            num_rules = 0
            
            if self.n_split is None:
                index_rules = rules[i]
            elif len(rules[i]) > 0:
                if self.n_split == 'sqrt':
                    num_rules = int(np.sqrt(len(rules[i])))
                else:
                    num_rules = self.n_split
                np.random.seed()
                index_rules = np.random.choice(
                    rules[i], num_rules, replace=False
                )
            
            for rule in index_rules:
                info = self.information_gain(label, data[:, i], rule)     
                if info > max_info:  
                    max_info = info
                    num_col = i
                    threshold = rule
        
        # print(f'threshold: {threshold}, label: {label[num_col]}')
        return threshold, num_col
        
    def build_tree(self, X, y, depth):
            """
                Helper function for building the tree.
                
                First build the root node.
            """
            first_threshold, first_feature = self.next_split(X, y)
            current = self.Node(first_threshold, first_feature)  
            current.depth = depth
            
            if (self.max_depth is not None and depth > self.max_depth) \
                or (first_feature is None) \
                or (first_threshold is None):
                current.predict = max(set(list(y)), key=list(y).count)
                current.pure = True
                return current
            
            if len(np.unique(y)) == 1:
                current.predict = y[0]
                current.pure = True
                return current
            
            left_index = X[:, first_feature] <= first_threshold
            right_index = X[:, first_feature] > first_threshold
            
            if len(left_index)==0 or len(right_index)==0:
                current.predict = y[first_feature]
                current.pure = True 
                return current
            
            
            left_X, left_y = X[left_index,:], y[left_index]

            current.left = self.build_tree(left_X, left_y, depth + 1)
                
            right_X, right_y = X[right_index,:], y[right_index]
            current.right = self.build_tree(right_X, right_y, depth + 1)
            
            return current
    

        
    def fit(self, X, y):
        
        """
            The fit function fits the Decision Tree model based on the training data. 
            
            X_train is a matrix or 2-D numpy array, represnting training instances. 
            Each training instance is a feature vector. 

            y_train contains the corresponding labels. There might be multiple (i.e., > 2) classes.
        """
        self.root = self.build_tree(np.array(X), np.array(y), 1)
        
        return self
            
    def ind_predict(self, inp):
        """
            Predict the most likely class label of one test instance based on its feature vector x.
        """
        cur = self.root  
        while not cur.pure:  
            
            feature = cur.feature  
            threshold = cur.threshold 
            
            if inp[feature] <= threshold:  
                cur = cur.left
            else:
                cur = cur.right
        return cur.predict
    
    def predict(self, inp):
        """
            X is a matrix or 2-D numpy array, represnting testing instances. 
            Each testing instance is a feature vector. 
            
            Return the predictions of all instances in a list.
        """
        
        result = []
        for i in range(inp.shape[0]):
            result.append(self.ind_predict(inp[i]))
        return result
    


class RandomForest():
    
    """
    
    RandomForest Classifier
    
    Attributes:
        n_trees: Number of trees. 
        trees: List store each individule tree
        n_features: Number of features to use during building each individule tree.
        n_split: Number of split for each feature.
        max_depth: Max depth allowed for the tree
        size_allowed : Min_size split, smallest size allowed for split 
    
    """
    
    def __init__(self,n_trees = 10, n_features = 'sqrt', n_split = 'sqrt', max_depth = None, size_allowed = 1):
        
        """
            Initilize all Attributes.
        """
        self.n_trees = n_trees
        self.trees = []
        self.n_features = n_features
        self.n_split = n_split
        self.max_depth = max_depth
        self.size_allowed = size_allowed
        
    def bootstrap_sample(self, X, y):
        """
        Generates a bootstrap sample from the dataset (X, y).

        Parameters:
            X (numpy.ndarray): 2D feature matrix where each row is a sample.
            y (numpy.ndarray): 1D target array where each entry is a label.

        Returns:
            X_sample (numpy.ndarray): The bootstrapped 2D feature matrix.
            y_sample (numpy.ndarray): The bootstrapped 1D target array.
        """
        # Number of samples in X
        n_samples = X.shape[0]

        # Generate random indices with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        # Sample from X and y using the generated indices
        X_sample = X[indices]
        y_sample = np.array(y)[indices]

        return X_sample, y_sample
        
    def fit(self, X, y):
        """
        The fit function fits the Random Forest model based on the training data. 
        
        X_train is a matrix or 2-D numpy array, representing training instances. 
        Each training instance is a feature vector. 
        
        y_train contains the corresponding labels. There might be multiple (i.e., > 2) classes.
        """
        # self.for_running = y[4]
        
        for i in range(self.n_trees):
            np.random.seed()
            temp_clf = DecisionTree(
                n_features=self.n_features, 
                n_split=self.n_split, 
                max_depth=self.max_depth, 
                size_allowed=self.size_allowed
            )
            X_sample, y_sample = self.bootstrap_sample(X, y)
            temp_clf.fit(X_sample, y_sample)
            self.trees.append(temp_clf)
        return self
            
    def ind_predict(self, inp):
        
        """
            Predict the most likely class label of one test instance based on its feature vector x.
        """
        result = []
        for tree in self.trees:
            pred = tree.ind_predict(inp)
            result.append(pred)

        
        labels, counts = np.unique(result, return_counts=True)
        max_label_count = counts[np.argmax(counts)]
        max_label_inds = np.where(counts == max_label_count)
        if len(max_label_inds) > 1:
            selected_index = np.random.choice(max_label_inds)
            return [labels[selected_index]]
        else:
            return labels[np.argmax(counts)]
    
    def predict_all(self, inp):
        
        """
            X is a matrix or 2-D numpy array, represnting testing instances. 
            Each testing instance is a feature vector. 
            
            Return the predictions of all instances in a list.
        """
        result = []
        for i in range(inp.shape[0]):
            result.append(self.ind_predict(inp[i]))
        return result


# Generate all possible combinations of 2 letters from the alphabet
combinations = itertools.product(string.ascii_lowercase+'<>', repeat=2)
pairs_ref = {}
for i, combination in enumerate(combinations, start=0):  
    combination_str = ''.join(combination)
    pairs_ref[combination_str] = i


combinations = itertools.product(string.ascii_lowercase, repeat=3)
tripples_ref = {}
for i, combination in enumerate(combinations, start=0): 
    combination_str = ''.join(combination)
    tripples_ref[combination_str] = i


def count_double_letters(word):
    # Regular expression pattern to find double letters
    pattern = r"(.)\1"
    matches = re.findall(pattern, word)
    return len(matches)


def vowel_proportion(word):
    # Regular expression pattern to find vowels (a, e, i, o, u)
    # Adding 'y' as a vowel as well, though it's sometimes considered a vowel
    pattern = r"[aeiouyAEIOUY]"
    
    # Find all vowels in the word
    vowels = re.findall(pattern, word)
    vowel_count = len(vowels)
    total_letters = len(word)
    
    proportion = vowel_count / max(total_letters, 1)
    
    return proportion


def pair_count_vec(word):
    vec = np.zeros(len(pairs_ref)+3)
    # vec[-5] = tripples_ref[word[:3]]
    vec[-2] = tripples_ref[word[-3:]]
    # vec[-2] = vowel_proportion(word)
    vec[-1] = count_double_letters(word)

    word = f'<{word}>'
    for i in range(1, len(word)):
        index = pairs_ref[word[i-1:i+1]]
        vec[index] += 1
    return vec


def classify(train_words, train_labels, test_words):
    train_vecs = np.vstack(pd.Series(train_words).apply(pair_count_vec))
    test_vecs = np.vstack(pd.Series(test_words).apply(pair_count_vec))

    clf = RandomForest(n_trees=20, n_split='sqrt')
    clf.fit(train_vecs, np.array(train_labels))

    return clf.predict_all(test_vecs)
