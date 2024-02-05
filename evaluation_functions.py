from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import matplotlib.pyplot as plt
import pandas as pd


def evaluate_model(y_true, y_pred):

    # pr auc curve
    no_skill = len(y_true[y_true == 1]) / len(y_true)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f"Baseline Model (area = {pr_auc:.2f})")
    plt.plot(
        [0, 1],
        [no_skill, no_skill],
        linestyle="--",
        label=f"No Skill (area = {no_skill:.2f})",
    )
    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()

    # roc auc curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label=f"no-skill (area = 0.50)")
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()


def evaluate_model_with_train(train_y_true, test_y_true, train_y_pred, test_y_pred):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    no_skill = len(train_y_true[train_y_true == 1]) / len(train_y_true)
    precision, recall, _ = precision_recall_curve(train_y_true, train_y_pred)
    pr_auc = auc(recall, precision)

    fig.suptitle("Train and Test Evaluation")
    ax[0].plot(recall, precision, label=f"Baseline Model (area = {pr_auc:.2f})")
    ax[0].plot(
        [0, 1],
        [no_skill, no_skill],
        linestyle="--",
        label=f"No Skill (area = {no_skill:.2f})",
    )
    ax[0].legend()
    ax[0].set_xlabel("Recall")
    ax[0].set_ylabel("Precision")
    ax[0].set_title("Training Precision-Recall Curve")

    fpr, tpr, _ = roc_curve(train_y_true, train_y_pred)
    roc_auc = auc(fpr, tpr)
    ax[1].plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    ax[1].plot([0, 1], [0, 1], "k--", label=f"no-skill (area = 0.50)")
    ax[1].legend()
    ax[1].set_xlabel("False Positive Rate")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].set_title("Training ROC Curve")
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    no_skill = len(test_y_true[test_y_true == 1]) / len(test_y_true)
    precision, recall, _ = precision_recall_curve(test_y_true, test_y_pred)
    pr_auc = auc(recall, precision)
    ax[0].plot(recall, precision, label=f"Baseline Model (area = {pr_auc:.2f})")
    ax[0].plot(
        [0, 1],
        [no_skill, no_skill],
        linestyle="--",
        label=f"No Skill (area = {no_skill:.2f})",
    )
    ax[0].legend()
    ax[0].set_xlabel("Recall")
    ax[0].set_ylabel("Precision")
    ax[0].set_title("Test Precision-Recall Curve")

    fpr, tpr, _ = roc_curve(test_y_true, test_y_pred)
    roc_auc = auc(fpr, tpr)
    ax[1].plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    ax[1].plot([0, 1], [0, 1], "k--", label=f"no-skill (area = 0.50)")
    ax[1].legend()
    ax[1].set_xlabel("False Positive Rate")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].set_title("Test ROC Curve")
    plt.show()


def time_based_evaluation(scored_df, date_col, target_col):
    score_dict = {}
    for date in sorted(scored_df[date_col].unique()):
        auc = roc_auc_score(
            scored_df[scored_df[date_col] == date][target_col],
            scored_df[scored_df[date_col] == date]["preds"],
        )
        positive_instances = len(
            scored_df[scored_df[date_col] == date][
                scored_df[scored_df[date_col] == date][target_col] == 1
            ]
        )
        negative_instances = len(
            scored_df[scored_df[date_col] == date][
                scored_df[scored_df[date_col] == date][target_col] == 0
            ]
        )
        positive_ratio = positive_instances / (positive_instances + negative_instances)
        score_dict[date] = {
            "auc": auc,
            "positive_instances": positive_instances,
            "negative_instances": negative_instances,
            "positive_ratio": positive_ratio,
        }

    return pd.DataFrame(score_dict).T
