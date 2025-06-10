#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
import sys
import os

# Get absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils import load_keel_dataset
from utils.noising import add_noise
import PRF
import PRF4DF
from deepforest import CascadeForestClassifier

# === Seed setting function ===
def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# === Data loading & visualization ===
def load_and_visualize_dataset(file_path):
    X_train, X_test, y_train, y_test, label_map = load_keel_dataset(
        train_path=file_path,
        already_split=False
    )

    print("Head of training data: \n", X_train[:10])
    print("Head of labels: \n", y_train[:10])

    feature_names_train = [f'Feature_{i+1}' for i in range(X_train.shape[1])]

    df1 = pd.DataFrame(X_train, columns=feature_names_train)
    df1['Label'] = y_train
    df1['Dataset'] = 'X_train'

    feature_names_test = [f'Feature_{i+1}' for i in range(X_test.shape[1])]

    df2 = pd.DataFrame(X_test, columns=feature_names_test)
    df2['Label'] = y_test
    df2['Dataset'] = 'X_test'

    df_all = pd.concat([df1, df2], ignore_index=True)

    plt.figure(figsize=(20, 15))
    for i, feature in enumerate(feature_names_train):
        plt.subplot(5, 4, i + 1)
        sns.kdeplot(data=df_all, x=feature, hue='Dataset', common_norm=False, fill=True, alpha=0.4, bw_adjust=0.9)
        plt.title(f"Feature: {feature}")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.countplot(x='Label', data=df1, order=sorted(df1['Label'].unique()))
    plt.title("Label Distribution (Train Set)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

    return X_train, X_test, y_train, y_test, label_map


# === Add noise function ===
def add_gaussian_noise(X_train, noise_scale=0.6):
    X_train_noisy, _, dX, _ = add_noise(X_train, noise_type='gaussian', gaussian_scale=noise_scale)
    print("Head of noise (dX): \n", dX[:5])

    feature_names = [f'Feature_{i+1}' for i in range(X_train.shape[1])]
    df1 = pd.DataFrame(X_train, columns=feature_names)
    df1['Label'] = y_train
    df1['Dataset'] = 'X'

    df2 = pd.DataFrame(dX, columns=feature_names)
    df2['Dataset'] = 'dX'

    df_all = pd.concat([df1, df2], ignore_index=True)

    plt.figure(figsize=(20, 15))
    for i, feature in enumerate(feature_names):
        plt.subplot(5, 4, i + 1)
        sns.kdeplot(data=df_all, x=feature, hue='Dataset', common_norm=False, fill=True, alpha=0.4, bw_adjust=0.5)
        plt.title(f"Feature: {feature}")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.countplot(x='Label', data=df1, order=sorted(df1['Label'].unique()))
    plt.title("Label Distribution (Train set)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

    return X_train_noisy, dX


# === PDRF Model ===
def run_pdrf(X_train_noisy, dX, y_train, X_test, y_test, label_map, seeds, n_cascade_estimators=4):
    n_classes = len(label_map)
    n_features = X_train_noisy.shape[1]
    accuracies_PDRF = []

    for seed in seeds:
        set_all_seeds(seed)

        model = CascadeForestClassifier(n_bins=n_classes, random_state=seed)

        prf_estimators = []
        for i in range(n_cascade_estimators):
            estimator = PRF4DF.SklearnCompatiblePRF(
                n_classes_=n_classes,
                n_features_=n_features,
                use_probabilistic_labels=False,
                use_feature_uncertainties=True,
                n_estimators=10,
                max_depth=10,
                random_state=i,
                n_jobs=1
            )
            prf_estimators.append(estimator)

        model.set_estimator(prf_estimators)

        X_train_combined = np.hstack((X_train_noisy, dX))
        model.fit(X=X_train_combined, y=y_train)

        X_test_combined = np.hstack((X_test, np.zeros_like(X_test)))
        acc = model.score(X_test_combined, y_test)
        accuracies_PDRF.append(acc)

    mean_acc = np.mean(accuracies_PDRF)
    std_acc = np.std(accuracies_PDRF)
    print(f"PDRF Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    return accuracies_PDRF


# === Probabilistic Random Forest ===
def run_prf(X_train_noisy, dX, y_train, X_test, y_test, seeds):
    accuracies_PRF = []
    for seed in seeds:
        set_all_seeds(seed)
        prf_cls = PRF.prf(n_estimators=10, bootstrap=True)
        prf_cls.fit(X=X_train_noisy, y=y_train, dX=dX)
        score = prf_cls.score(X_test, y_test)
        accuracies_PRF.append(score)

    mean_acc = np.mean(accuracies_PRF)
    std_acc = np.std(accuracies_PRF)
    print(f"PRF Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    return accuracies_PRF


# === Random Forest ===
def run_rf(X_train_noisy, y_train, X_test, y_test, seeds, n_estimators=50):
    accuracies_RF = []
    for seed in seeds:
        set_all_seeds(seed)
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
        rf.fit(X_train_noisy, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies_RF.append(acc)

    mean_acc = np.mean(accuracies_RF)
    std_acc = np.std(accuracies_RF)
    print(f"Random Forest Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    return accuracies_RF


# === Deep Forest ===
def run_deep_forest(X_train_noisy, y_train, X_test, y_test, seeds, n_estimators=2):
    accuracies_DF = []
    for seed in seeds:
        set_all_seeds(seed)
        clf = CascadeForestClassifier(n_estimators=n_estimators, random_state=seed)
        clf.fit(X_train_noisy, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies_DF.append(acc)

    mean_acc = np.mean(accuracies_DF)
    std_acc = np.std(accuracies_DF)
    print(f"Deep Forest Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    return accuracies_DF


# === Neural Network ===
def run_neural_network(X_train_noisy, y_train, X_test, y_test, seeds,
                       epochs=20, batch_size=16, hidden_units=64, dropout_rate=0.5, optimizer='adam'):

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only
    accuracies_NN = []

    unique_classes = np.unique(y_train)
    num_classes = len(unique_classes)
    is_binary = num_classes == 2

    if not is_binary:
        y_train_cat = to_categorical(np.searchsorted(unique_classes, y_train))
        y_test_cat = to_categorical(np.searchsorted(unique_classes, y_test))
    else:
        y_train_cat = y_train
        y_test_cat = y_test

    def create_model(hidden_units=hidden_units, dropout_rate=dropout_rate, optimizer=optimizer):
        model = Sequential()
        model.add(Dense(hidden_units, input_shape=(X_train_noisy.shape[1],), activation='relu'))
        model.add(Dropout(dropout_rate))
        if is_binary:
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.add(Dense(num_classes, activation='softmax'))
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    for seed in seeds:
        set_all_seeds(seed)

        model = KerasClassifier(model=create_model, epochs=epochs, batch_size=batch_size, verbose=0)
        model.fit(X_train_noisy, y_train_cat)

        if is_binary:
            y_pred = model.predict(X_test)
            y_true = y_test
        else:
            y_pred = np.argmax(model.predict_proba(X_test), axis=1)
            y_true = np.searchsorted(unique_classes, y_test)

        acc = accuracy_score(y_true, y_pred)
        accuracies_NN.append(acc)

    mean_acc = np.mean(accuracies_NN)
    std_acc = np.std(accuracies_NN)
    print(f"Neural Network Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    return accuracies_NN


# === Kernel SVM ===
def run_kernel_svm(X_train_noisy, y_train, X_test, y_test, seeds):
    accuracies_KSVM = []

    # Use first seed for reproducibility
    set_all_seeds(seeds[0])

    model = SVC()
    param_grid = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train_noisy, y_train)

    best_svm = grid.best_estimator_
    y_pred = best_svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Repeat accuracy to match other models' lengths
    accuracies_KSVM = [acc] * len(seeds)

    mean_acc = np.mean(accuracies_KSVM)
    std_acc = np.std(accuracies_KSVM)
    print(f"Kernel SVM Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    return accuracies_KSVM


# === Plot results ===
def plot_results(accuracies_dict, file_path, output_dir="../results/1_low_noise/plots/"):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    title_name = base_name.replace('_', ' ').title()
    output_filename = os.path.join(output_dir, f"{base_name}_low_noise.png")

    comparison_df = pd.DataFrame({
        "Model": list(accuracies_dict.keys()),
        "Mean Accuracy": [np.mean(v) for v in accuracies_dict.values()],
        "Std": [np.std(v) for v in accuracies_dict.values()]
    })

    comparison_df = comparison_df.sort_values("Mean Accuracy", ascending=False)

    def get_color(model):
        return "#55bfc7" if model == "PDRF" else "lightgray"

    colors = comparison_df["Model"].apply(get_color)

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.barplot(
        data=comparison_df,
        x="Model",
        y="Mean Accuracy",
        palette=colors,
        errorbar=None,
        ax=ax
    )

    x_positions = range(len(comparison_df))
    means = comparison_df["Mean Accuracy"].values
    stds = comparison_df["Std"].values

    ax.errorbar(
        x=x_positions,
        y=means,
        yerr=stds,
        fmt='none',
        ecolor='black',
        elinewidth=1.5,
        capsize=5
    )

    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1.05)
    sns.despine()

    for x, (mean, std) in enumerate(zip(means, stds)):
        ax.text(
            x, mean + std + 0.02,
            f'{mean:.2f}±{std:.2f}',
            ha='center', va='bottom',
            fontsize=10, color='black'
        )

    ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')

    fig.text(
        0.1, 0.93,
        f"{title_name} dataset – Low Noise",
        ha='left',
        fontsize=14,
        fontweight='bold'
    )
    fig.text(
        0.1, 0.88,
        "Average model accuracy and standard deviation across runs, with low levels of feature noise",
        ha='left',
        fontsize=11,
        color="#333333"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.85])
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()


# === Save results to CSV ===
def save_results(accuracies_dict, seeds, file_path, output_dir="../results/1_low_noise/tables/"):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    results_df = pd.DataFrame({
        "Model": list(accuracies_dict.keys()),
        "Mean Accuracy": [np.mean(v) for v in accuracies_dict.values()],
        "Std": [np.std(v) for v in accuracies_dict.values()],
        "Seeds": [seeds] * len(accuracies_dict)
    })

    csv_path = os.path.join(output_dir, f"{base_name}_low_noise.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")


# === Main pipeline runner ===
def run_pipeline(file_path, seeds=[17, 42, 123], noise_scale=0.6):
    # Set seeds
    set_all_seeds(42)

    # Load data & visualize
    X_train, X_test, y_train, y_test, label_map = load_and_visualize_dataset(file_path)

    # Add noise
    X_train_noisy, dX = add_gaussian_noise(X_train, noise_scale=noise_scale)

    # Run models
    acc_PDRF = run_pdrf(X_train_noisy, dX, y_train, X_test, y_test, label_map, seeds)
    acc_PRF = run_prf(X_train_noisy, dX, y_train, X_test, y_test, seeds)
    acc_RF = run_rf(X_train_noisy, y_train, X_test, y_test, seeds)
    acc_DF = run_deep_forest(X_train_noisy, y_train, X_test, y_test, seeds)
    acc_NN = run_neural_network(X_train_noisy, y_train, X_test, y_test, seeds)
    acc_KSVM = run_kernel_svm(X_train_noisy, y_train, X_test, y_test, seeds)

    accuracies_dict = {
        "PDRF": acc_PDRF,
        "PRF": acc_PRF,
        "RF": acc_RF,
        "Deep Forest": acc_DF,
        "Neural Network": acc_NN,
        "Kernel SVM": acc_KSVM,
    }

    # Save results
    save_results(accuracies_dict, seeds, file_path)

    # Plot results
    plot_results(accuracies_dict, file_path)


# === Run script ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run classification pipeline with noise and different models.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--noise_scale", type=float, default=0.2, help="Gaussian noise scale for training data.")
    args = parser.parse_args()

    run_pipeline(args.dataset_path, noise_scale=args.noise_scale)
