import numpy as np
import matplotlib.pyplot as plt
import os
from ID_management import ID_manager
from paths_and_constants import *
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import svm


def plot_svc_decision_boundary(clf, X, y, ax):

    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=3, cmap=plt.cm.Paired, alpha=0.7)
    handles, _ = scatter.legend_elements()
    ax.legend(handles, ["Closed Set", "Open Set"], loc="upper right", title="Turtle IDs")

    # Create grid to plot boundary
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # Plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    ax.grid(True)


def analyze_feature_file(filename, sunset=0, convert_to_logits=False):

    def softmax(x):
        # x shape: (num_examples, num_classes)

        # 1. Subtract max for numerical stability (along the class axis)
        # This ensures the largest value in each row becomes 0, so exp(0) = 1
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))

        # 2. Divide by the sum of exponentials
        return e_x / e_x.sum(axis=1, keepdims=True)


    data = np.load(filename)
    features = data['all_features'].squeeze()
    labels = data['all_labels'].squeeze()

    logits = softmax(features)
    x = logits if convert_to_logits else features

    id_mng = ID_manager()
    id_mng.load_labels(labels)
    in_IDs, out_IDs, mask_close, mask_open, re_in_IDs, re_out_IDs = id_mng.get_subset_for_analysis(subset=sunset)


    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    dtct = np.argmax(x.squeeze(), axis=1)
    cm = confusion_matrix(re_in_IDs, dtct)
    sns.heatmap(cm[:40, :40], ax=ax[0], annot=True, fmt='d', xticklabels=in_IDs[:40], yticklabels=in_IDs[:40])
    acc = np.diag(cm[1:][:, 1:]).sum() / cm[1:][:, 1:].sum()
    ax[0].set_title('accuracy (on train examples) = {:5.2f}'.format(acc))
    #
    # tests
    winners_close = np.sort(x.squeeze()[mask_close], axis=1)[:, ::-1][:, :2]
    winners_open = np.sort(x.squeeze()[mask_open], axis=1)[:, ::-1][:, :2]
    # for (m, r) in zip(winners_close[:, 0], winners_close[:, 1]):
    #     ax[1].scatter(m, r, s=10, c='b', edgecolors='none')
    # for (m, r) in zip(winners_open[:, 0], winners_open[:, 1]):
    #     ax[1].scatter(m, r, s=10, c='r', edgecolors='none')
    # ax[1].grid(True)
    #
    # make decision closed set / open set
    # Prepare features (X) and labels (y)
    X = np.vstack((winners_close, winners_open))
    y = np.hstack((np.zeros(len(winners_close)), np.ones(len(winners_open))))
    # Create and fit a Linear SVM
    #clf = svm.SVC(kernel='linear', C=1.0)
    clf = svm.SVC(kernel='poly', degree=3, C=1.0, class_weight={0: 1.0, 1: 2.0})
    clf.fit(X, y)
    plot_svc_decision_boundary(clf, X, y, ax[1])

    return fig, clf


if __name__ == '__main__':

    filname = os.path.join(ROOT_FEATURES, 'SeaTurtleID2022_resnet.npz')
    fig_feat, clf_feat = analyze_feature_file(filname, sunset=1)
    fig_logits, clf_logits = analyze_feature_file(filname, sunset=1, convert_to_logits=True)

    plt.show()





