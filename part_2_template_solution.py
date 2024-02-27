# Add your imports here.
# Note: only sklearn, numpy, utils and new_utils are allowed.

import numpy as np
from numpy.typing import NDArray
from typing import Any

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

import utils as u

import new_utils as nu
from sklearn.metrics import accuracy_score

# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.


class Section2:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None
        """
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train

    # ---------------------------------------------------------

    """
    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
        all 10 classes by also printing out the number of elements in each class y and 
        print out the number of classes for both training and testing datasets. 
    """

    def partA(
        self,
    ) -> tuple[
        dict[str, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        # Prepare data for training and testing. Scale the features
        Xtrain, ytrain, Xtest, ytest = u.prepare_data()
        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest)
        answer = {}
        
        
        # Enter your code and fill the `answer`` dictionary

        # `answer` is a dictionary with the following keys:
        # - nb_classes_train: number of classes in the training set
        # - nb_classes_test: number of classes in the testing set
        # - class_count_train: number of elements in each class in the training set
        # - class_count_test: number of elements in each class in the testing set
        # - length_Xtrain: number of elements in the training set
        # - length_Xtest: number of elements in the testing set
        # - length_ytrain: number of labels in the training set
        # - length_ytest: number of labels in the testing set
        # - max_Xtrain: maximum value in the training set
        # - max_Xtest: maximum value in the testing set

        # return values:
        # Xtrain, ytrain, Xtest, ytest: the data used to fill the `answer`` dictionary

        #Xtrain = Xtest = np.zeros([1, 1], dtype="float")
        #ytrain = ytest = np.zeros([1], dtype="int")

        # Compute and store various statistics
        answer['nb_classes_train'] = len(np.unique(ytrain))
        answer['nb_classes_test'] = len(np.unique(ytest))
        answer['class_count_train'] = np.bincount(ytrain)
        answer['class_count_test'] = np.bincount(ytest)
        answer['length_Xtrain'] = len(Xtrain)
        answer['length_Xtest'] = len(Xtest)
        answer['length_ytrain'] = len(ytrain)
        answer['length_ytest'] = len(ytest)
        answer['max_Xtrain'] = Xtrain.max()
        answer['max_Xtest'] = Xtrain.max()
        
        return answer, Xtrain, ytrain, Xtest, ytest

    """
    B.  Repeat part 1.C, 1.D, and 1.F, for the multiclass problem. 
        Use the Logistic Regression for part F with 300 iterations. 
        Explain how multi-class logistic regression works (inherent, 
        one-vs-one, one-vs-the-rest, etc.).
        Repeat the experiment for ntrain=1000, 5000, 10000, ntest = 200, 1000, 2000.
        Comment on the results. Is the accuracy higher for the training or testing set?
        What is the scores as a function of ntrain.

        Given X, y from mnist, use:
        Xtrain = X[0:ntrain, :]
        ytrain = y[0:ntrain]
        Xtest = X[ntrain:ntrain+test]
        ytest = y[ntrain:ntrain+test]
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
        ntrain_list: list[int] = [],
        ntest_list: list[int] = [],
    ) -> dict[int, dict[str, Any]]:
        """ """
        # Enter your code and fill the `answer`` dictionary
        answer = {}
        # Loop over different combinations of training and testing sizes
        for i in range(0, len(ntrain_list)):
            # Extract the current training and testing sizes
            train_rows = ntrain_list[i]
            test_rows = ntest_list[i]

            # Subset the data based on the specified sizes
            Xtrain = X[0:train_rows,:]
            ytrain = y[0:train_rows]
            Xtest = Xtest[0:test_rows]
            ytest = ytest[0:test_rows]
            
            X = Xtrain
            y = ytrain
      
    
            answer1= {}
            # Part 1C: Decision Tree with K-Fold Cross-Validation
            clf = DecisionTreeClassifier(random_state=self.seed)
            cv = KFold(n_splits=5,shuffle = True,random_state=self.seed)
            dec_tree = u.train_simple_classifier_with_cv(Xtrain=X,ytrain=y,clf=clf,cv=cv) 
    
            answer_sub ={}
            res_key ={}
            res_key['mean_fit_time'] = dec_tree['fit_time'].mean()
            res_key['std_fit_time'] = dec_tree['fit_time'].std()
            res_key['mean_accuracy'] = dec_tree['test_score'].mean()
            res_key['std_accuracy'] = dec_tree['test_score'].std()
        
            answer_sub["scores_C"] = res_key
            answer_sub["clf"] = clf  
            answer_sub["cv"] = cv  
            
    
            # Part 1D: Decision Tree with Shuffle-Split Cross-Validation    
            answer_sub1 ={}
            clf = DecisionTreeClassifier(random_state=self.seed)
            cv_ss = ShuffleSplit(n_splits=5,random_state=self.seed)
    
            dec_tree_ss = u.train_simple_classifier_with_cv(Xtrain=X,ytrain=y,clf=clf,cv=cv_ss)
            res_key_ss ={}
            res_key_ss['mean_fit_time'] = dec_tree_ss['fit_time'].mean()
            res_key_ss['std_fit_time'] = dec_tree_ss['fit_time'].std()
            res_key_ss['mean_accuracy'] = dec_tree_ss['test_score'].mean()
            res_key_ss['std_accuracy'] = dec_tree_ss['test_score'].std()
    
            answer_sub1["scores_D"] = res_key_ss
            answer_sub1["clf"] = clf
            answer_sub1["cv"] = cv_ss
    
            # Part 1F: Logistic Regression with Shuffle-Split Cross-Validation
            answer_sub2 ={}
            clf = LogisticRegression(max_iter=300,random_state=self.seed)
            cv_ss = ShuffleSplit(n_splits=5,random_state=self.seed)
            scores = cross_validate(clf, X, y, cv=cv_ss, return_train_score=True)
            clf.fit(X, y)
            scores_train_F = clf.score(X, y)
            scores_test_F = clf.score(Xtest, ytest) 
            train_pred =clf.predict(X)
            test_pred = clf.predict(Xtest)
            conf_mat_train = confusion_matrix(y,train_pred)
            conf_mat_test = confusion_matrix(ytest,test_pred)
            mean_cv_accuracy_F = scores["test_score"].mean()
            answer_sub2 = {
                "scores_train_F": scores_train_F,
                "scores_test_F": scores_test_F,
                "mean_cv_accuracy_F": mean_cv_accuracy_F,
                "clf": clf,
                "cv": cv_ss,
                "conf_mat_train": conf_mat_train,
                "conf_mat_test": conf_mat_test
            }
           
            # Store results for the current training size
            answer[ntrain_list[i]] = {
                "partC": answer_sub ,
                "partD": answer_sub1,
                "partF": answer_sub2,
                "ntrain": train_rows,
                "ntest": test_rows,
                "class_count_train": list(np.bincount(ytrain)) ,
                "class_count_test": list(np.bincount(ytest))
            }
        

        """
        `answer` is a dictionary with the following keys:
           - 1000, 5000, 10000: each key is the number of training samples

           answer[k] is itself a dictionary with the following keys
            - "partC": dictionary returned by partC section 1
            - "partD": dictionary returned by partD section 1
            - "partF": dictionary returned by partF section 1
            - "ntrain": number of training samples
            - "ntest": number of test samples
            - "class_count_train": number of elements in each class in
                               the training set (a list, not a numpy array)
            - "class_count_test": number of elements in each class in
                               the training set (a list, not a numpy array)
        """

        return answer
