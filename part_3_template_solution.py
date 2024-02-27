import numpy as np
from numpy.typing import NDArray
from typing import Any

"""
   In the first two set of tasks, we will narrowly focus on accuracy - 
   what fraction of our predictions were correct. However, there are several 
   popular evaluation metrics. You will learn how (and when) to use these evaluation metrics.
"""


# ======================================================================
class Section3:
    def __init__(
        self,
        normalize: bool = True,
        frac_train=0.2,
        seed=42,
    ):
        self.seed = seed
        self.normalize = normalize

    def analyze_class_distribution(self, y: NDArray[np.int32]) -> dict[str, Any]:
        """
        Analyzes and prints the class distribution in the dataset.

        Parameters:
        - y (array-like): Labels dataset.

        Returns:
        - dict: A dictionary containing the count of elements in each class and the total number of classes.
        """
        # Your code here to analyze class distribution
        # Hint: Consider using collections.Counter or numpy.unique for counting

        uniq, counts = np.unique(y, return_counts=True)
       
        #Classes, counts
        print(f"Unique Classes: {uniq}")
        print(f"Counts per class: {counts}")
        print(f"Total count of classes: {np.sum(counts)}")
        
        #Converting to dictionary format
        class_counts = dict(zip(uniq, counts))

        return {
            "class_counts": class_counts,
            "num_classes": len(uniq)
        }

    # --------------------------------------------------------------------------
    """
    A. Using the same classifier and hyperparameters as the one used at the end of part 2.B. 
       Get the accuracies of the training/test set scores using the top_k_accuracy score for k=1,2,3,4,5. 
       Make a plot of k vs. score for both training and testing data and comment on the rate of accuracy change. 
       Do you think this metric is useful for this dataset?
    """

    def partA(
        self,
        Xtrain: NDArray[np.floating],
        ytrain: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """ """
        # Enter code and return the `answer`` dictionary

        answer = {}
        # Train the classifier
        clf = self.train_classifier(Xtrain, ytrain)

        # Initialize lists to store k-values and corresponding scores for training and testing data
        plot_k_vs_score_train = []
        plot_k_vs_score_test = []

        # Define k-values to be evaluated
        k_values = [1, 2, 3, 4, 5]

        # Loop over each k-value
        for k in k_values:
            score_train = self.top_k_accuracy(clf, Xtrain, ytrain, k)
            score_test = self.top_k_accuracy(clf, Xtest, ytest, k)
            
            plot_k_vs_score_train.append((k, score_train))
            plot_k_vs_score_test.append((k, score_test))

            # Store the scores for this k-value in a dictionary
            answer[k] = {
                "score_train": score_train,
                "score_test": score_test
            }
        """
        # `answer` is a dictionary with the following keys:
        - integers for each topk (1,2,3,4,5)
        - "clf" : the classifier
        - "plot_k_vs_score_train" : the plot of k vs. score for the training data, 
                                    a list of tuples (k, score) for k=1,2,3,4,5
        - "plot_k_vs_score_test" : the plot of k vs. score for the testing data
                                    a list of tuples (k, score) for k=1,2,3,4,5

        # Comment on the rate of accuracy change for testing data
        - "text_rate_accuracy_change" : the rate of accuracy change for the testing data

        # Comment on the rate of accuracy change
        - "text_is_topk_useful_and_why" : provide a description as a string

        answer[k] (k=1,2,3,4,5) is a dictionary with the following keys: 
        - "score_train" : the topk accuracy score for the training set
        - "score_test" : the topk accuracy score for the testing set
        """
        # Comment on the rate of accuracy change for testing data
        text_rate_accuracy_change : "The rate at which accuracy changes for the testing data decreases with higher values of k."
        
        # Comment on the rate of accuracy change
        text_is_topk_useful_and_why : "Top-k accuracy is advantageous for analyzing this dataset because it provides valuable insights into the model's performance by considering multiple potential predictions. This metric is particularly relevant for applications where precise predictions are not essential, but having a spectrum of probable predictions holds significance."
            
        answer["clf"] = clf
        answer["plot_k_vs_score_train"] = plot_k_vs_score_train
        answer["plot_k_vs_score_test"] = plot_k_vs_score_test
        answer["text_rate_accuracy_change"] = rate_of_accuracy_change
        answer["text_is_topk_useful_and_why"] = topk_useful_and_why    

        return answer, Xtrain, ytrain, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    B. Repeat part 1.B but return an imbalanced dataset consisting of 90% of all 9s removed.  Also convert the 7s to 0s and 9s to 1s.
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}

        # Prepare the data for training, testing and filtering out samples labeled as 7 or 9 from the training and testing sets
        X, y, Xtest, ytest = u.prepare_data()
        
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        
        Xtrain = Xtrain/255.0
        Xtest = Xtest/255.0
        
        indices_of_9 = np.where(ytrain == 9)[0]
        indices_to_remove = np.random.choice(indices_of_9, size=int(0.9 * len(indices_of_9)), replace=False)
        Xtrain = np.delete(Xtrain, indices_to_remove, axis=0)
        ytrain = np.delete(ytrain, indices_to_remove)
        ytrain = np.where(ytrain == 7, 0, ytrain)
        ytrain = np.where(ytrain == 9, 1, ytrain)
        
        ytest = np.where(ytest == 7, 0, ytest)
        ytest = np.where(ytest == 9, 1, ytest)

        # Answer is a dictionary with the same keys as part 1.B

        # Populate the answer dictionary with relevant information
        answer["length_Xtrain"] = len(Xtrain)
        answer["length_Xtest"] = len(Xtest)
        answer["length_ytrain"] = len(ytrain)
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = np.max(Xtrain)
        answer["max_Xtest"] = np.max(Xtest)

        return answer, X, y, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    C. Repeat part 1.C for this dataset but use a support vector machine (SVC in sklearn). 
        Make sure to use a stratified cross-validation strategy. In addition to regular accuracy 
        also print out the mean/std of the F1 score, precision, and recall. As usual, use 5 splits. 
        Is precision or recall higher? Explain. Finally, train the classifier on all the training data 
        and plot the confusion matrix.
        Hint: use the make_scorer function with the average='macro' argument for a multiclass dataset. 
    """

    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""

        # Enter your code and fill the `answer` dictionary
        answer = {}
       
        # Set up cross-validation strategy and initialize the classifier
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        clf = SVC(random_state=42)
        
        # Define scoring metrics
        scorers = {
            'precision': make_scorer(precision_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro'),
            'f1': make_scorer(f1_score, average='macro'),
            'accuracy': make_scorer(accuracy_score)
        }
        
        # Perform cross-validation and train classifer
        cv_results = cross_validate(clf, X, y, cv=cv_strategy, scoring=scorers)
        clf.fit(X, y)
        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "is_precision_higher_than_recall" : a boolean
        - "explain_is_precision_higher_than_recall" : a string
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        
        answer["scores"] is dictionary with the following keys, generated from the cross-validator:
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1
        """
        # Construct the answer dictionary with relevant information
        answer['cv'] = cv_strategy
        answer['clf'] = clf
        answer['scores'] = {
            'mean_accuracy': np.mean(cv_results['test_accuracy']),
            'std_accuracy': np.std(cv_results['test_accuracy']),
            'mean_precision': np.mean(cv_results['test_precision']),
            'std_precision': np.std(cv_results['test_precision']),
            'mean_recall': np.mean(cv_results['test_recall']),
            'std_recall': np.std(cv_results['test_recall']),
            'mean_f1': np.mean(cv_results['test_f1']),
            'std_f1': np.std(cv_results['test_f1']),
        }
        # Determine if precision is higher than recall
        answer['is_precision_higher_than_recall'] = answer['scores']['mean_precision'] > answer['scores']['mean_recall']
        answer['explain_is_precision_higher_than_recall'] = ("Precision is higher than recall." if answer['is_precision_higher_than_recall'] 
    else "Recall is higher than precision.")
        answer['confusion_matrix_train'] = confusion_matrix(y, clf.predict(X))
        answer['confusion_matrix_test'] = confusion_matrix(ytest, clf.predict(Xtest))
        
        # Plot confusion matrix for the training set
        plot_confusion_matrix(clf, X, y)
        plt.title("Confusion Matrix - Training Set")
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()
        
        # Plot confusion matrix for the test set
        plot_confusion_matrix(clf, Xtest, ytest)
        plt.title("Confusion Matrix - Test Set")
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()
        
        return answer

        return answer

    # --------------------------------------------------------------------------
    """
    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the class_weights parameter).  Print out the class weights, and comment on the performance difference. Use the `compute_class_weight` argument of the estimator to compute the class weights. 
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}
        ## Compute class weights based on the distribution of classes in y
        unique_classes_D = np.unique(y)
        class_weights_D = compute_class_weight(class_weight='balanced', classes=unique_classes_D, y=y)
        class_weights_dict = dict(zip(unique_classes_D, class_weights_D))

        #Set up cross-validation strategy with stratified K-fold and Initialize the classifier with computed class weights
        cv_strategy_D = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        clf_D = SVC(class_weight=class_weights_dict, random_state=42)
        
        # Define scoring metrics for evaluation
        scorers_D = {
            'precision': make_scorer(precision_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro'),
            'f1': make_scorer(f1_score, average='macro'),
            'accuracy': make_scorer(accuracy_score)
        }
        
        # Perform cross-validation and Train the classifier
        cv_results_D = cross_validate(clf_D, X, y, cv=cv_strategy_D, scoring=scorers_D)
        clf_D.fit(X, y)
        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "class_weights" : the class weights
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        - "explain_purpose_of_class_weights" : explanatory string
        - "explain_performance_difference" : explanatory string

        answer["scores"] has the following keys: 
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1

        Recall: The scores are based on the results of the cross-validation step
        """
        #Store computed class weights, cross-validation strategy classifier and trained with class weights
        answer['class_weights'] = class_weights_dict
        answer['cv'] = cv_strategy_D
        answer['clf'] = clf_D
       
        # Calculate and store evaluation scores
        answer['scores'] = {
            'mean_accuracy': np.mean(cv_results_D['test_accuracy']),
            'std_accuracy': np.std(cv_results_D['test_accuracy']),
            'mean_precision': np.mean(cv_results_D['test_precision']),
            'std_precision': np.std(cv_results_D['test_precision']),
            'mean_recall': np.mean(cv_results_D['test_recall']),
            'std_recall': np.std(cv_results_D['test_recall']),
            'mean_f1': np.mean(cv_results_D['test_f1']),
            'std_f1': np.std(cv_results_D['test_f1']),
        }
        # Compute and store confusion matrices for both training and testing sets
        answer['confusion_matrix_train'] = confusion_matrix(y, clf_D.predict(X))
        answer['confusion_matrix_test'] = confusion_matrix(ytest, clf_D.predict(Xtest))
       
        answer['explain_purpose_of_class_weights'] = "Class weights are employed to tackle class imbalance by assigning greater importance to less common classes. This ensures that the classifier focuses more on minority classes during training."
        answer['explain_performance_difference'] = "A rationale stemming from observed variations in performance between utilizing default and weighted loss functions. Generally, incorporating class weights enhances recall for underrepresented classes while potentially impacting precision."

        return answer
