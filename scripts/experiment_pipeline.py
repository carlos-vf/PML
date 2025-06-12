import numpy as np
import os
from utils import load_keel_dataset
from utils.noising import add_noise
from utils.noising import add_label_noise
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from deepforest import CascadeForestClassifier
import PRF
import PRF4DF
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import probabilistic_deep_forest as pdf
from sklearn.metrics import accuracy_score

def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def create_nn_model(input_dim, num_classes, is_binary, hidden_units, dropout_rate, optimizer):
    model = Sequential()
    model.add(Dense(hidden_units, input_shape=(input_dim,), activation='relu'))
    model.add(Dropout(dropout_rate))
    if is_binary:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def run_pipeline(config):
    # Load dataset
    dataset_path = config['dataset_path']
    seeds = config.get('seeds', [27, 272, 2727, 1, 30])  # default seeds
    noise_scale = config.get('noise_scale', 0.0)  # default noise 0.0
    label_noise_scale = config.get('label_noise_scale', 0.0)  # default noise 0.0
    label_noise_range = config.get('label_noise_range', (0.0, 0.5))  # default range
    noise_type = config.get('noise_type', "gaussian")  # default noise 0.0

    print(f"--- DEBUG: Current noise_scale from config: {noise_scale} ---")

    models_config = config.get('models', {})

    # Load dataset and split
    X_train, X_test, y_train, y_test, label_map = load_keel_dataset(
        train_path=dataset_path,
        already_split=False
    )
    # Ensure labels are adjusted to start from 0
    y_train = y_train - 1  
    y_test = y_test - 1  
    # Add noise to train set
    X_train_noisy, _, dX, _ = add_noise(X_train, noise_type=noise_type, noise_scale=noise_scale)
    y_train_noisy, _, py, _ = add_label_noise(y_train, mode="random_prob", noise_level=label_noise_scale, random_seed=30, prob_range=label_noise_range)

    # print label noise scale for debugging
    print(f"--- DEBUG: label_noise_range: {label_noise_range} ---")
    # print head of py for debugging
    print(f"--- DEBUG: py head: {py[:10]} ---")
    #print head of y_train for debugging
    print(f"--- DEBUG: y_train head: {y_train[:10]} ---")
    # print head of y_train_noisy for debugging
    print(f"--- DEBUG: y_train_noisy head: {y_train_noisy[:10]} ---")


    n_classes = len(label_map)
    n_features = X_train.shape[1]

    accuracies = {}

    # --- PDF ---
    if 'pdf' in models_config:
        n_cascade_estimators = models_config['pdf'].get('n_cascade_estimators', 4)
        n_trees_pdf = models_config['pdf'].get('n_trees_pdf', 10)
        max_depth_pdf = models_config['pdf'].get('max_depth_pdf', 10)
        accuracies_pdf = []
        for seed in seeds:
            set_all_seeds(seed)
            model = pdf.CascadeForestClassifier(random_state=seed)
            prf_estimators = []
            for i in range(n_cascade_estimators):
                estimator = PRF4DF.SklearnCompatiblePRF(
                    n_classes_=n_classes,
                    n_features_=n_features,
                    n_estimators=n_trees_pdf,
                    max_depth=max_depth_pdf,
                    n_jobs=1
                )
                prf_estimators.append(estimator)
            model.set_estimator(prf_estimators)
            model.fit(X=X_train_noisy, y=y_train_noisy, dX=dX, py=py)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_pred, y_test)
            accuracies_pdf.append(acc)
        accuracies['PDF'] = (np.mean(accuracies_pdf), np.std(accuracies_pdf))

    # --- Random Forest ---
    if 'rf' in models_config:
        n_estimators = models_config['rf'].get('n_estimators', 50)
        accuracies_RF = []
        for seed in seeds:
            set_all_seeds(seed)
            rf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
            rf.fit(X_train_noisy, y_train_noisy)
            y_pred = rf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies_RF.append(acc)
        accuracies['RF'] = (np.mean(accuracies_RF), np.std(accuracies_RF))

    # --- Deep Forest ---
    if 'df' in models_config:
        n_estimators = models_config['df'].get('n_estimators', 2)
        n_trees_df = models_config['df'].get('n_trees_df', 10)

        accuracies_DF = []
        for seed in seeds:
            set_all_seeds(seed)
            clf = CascadeForestClassifier(n_estimators=n_estimators, random_state=seed, n_trees=n_trees_df)
            clf.fit(X_train_noisy, y_train_noisy)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies_DF.append(acc)
        accuracies['DF'] = (np.mean(accuracies_DF), np.std(accuracies_DF))

    # --- Neural Network ---
    if 'nn' in models_config:
        nn_params = models_config['neural_network']
        epochs = nn_params.get('epochs', 20)
        batch_size = nn_params.get('batch_size', 16)
        hidden_units = nn_params.get('hidden_units', 64)
        dropout_rate = nn_params.get('dropout_rate', 0.5)
        optimizer = nn_params.get('optimizer', 'adam')

        accuracies_NN = []

        unique_classes = np.unique(y_train_noisy)
        num_classes = len(unique_classes)
        is_binary = num_classes == 2

        if not is_binary:
            y_train_cat = to_categorical(np.searchsorted(unique_classes, y_train_noisy))
            y_test_cat = to_categorical(np.searchsorted(unique_classes, y_test))
        else:
            y_train_cat = y_train_noisy
            y_test_cat = y_test

        for seed in seeds:
            set_all_seeds(seed)

            model = KerasClassifier(
                model=lambda: create_nn_model(
                    input_dim=n_features,
                    num_classes=num_classes,
                    is_binary=is_binary,
                    hidden_units=hidden_units,
                    dropout_rate=dropout_rate,
                    optimizer=optimizer,
                ),
                epochs=epochs,
                batch_size=batch_size,
            )

            model.fit(X_train_noisy, y_train_cat)

            if is_binary:
                y_pred = model.predict(X_test)
                y_true = y_test
            else:
                y_pred = np.argmax(model.predict_proba(X_test), axis=1)
                y_true = np.searchsorted(unique_classes, y_test)

            acc = accuracy_score(y_true, y_pred)
            accuracies_NN.append(acc)

        accuracies['NN'] = (np.mean(accuracies_NN), np.std(accuracies_NN))

    # --- Kernel SVM ---

    if 'ksvm' in models_config:
        set_all_seeds(seeds[0])  # reproducibility for grid search

        model = SVC()
        param_grid = {
            'kernel': ['linear', 'rbf'],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train_noisy, y_train_noisy)
        best_svm = grid.best_estimator_

        y_pred = best_svm.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Repeat to keep consistent length
        accuracies_KSVM = [acc] * len(seeds)
        accuracies['KSVM'] = (np.mean(accuracies_KSVM), np.std(accuracies_KSVM))


    # --- PRF Model ---
    if 'prf' in models_config:
        prf_params = models_config['prf']
        n_estimators = prf_params.get('n_estimators', 10)
        bootstrap = prf_params.get('bootstrap', True)
        max_depth = prf_params.get('max_depth', None)
        max_features = prf_params.get('max_features', 'sqrt')

        accuracies_PRF = [] 

        for seed in seeds:
            set_all_seeds(seed)
            prf_cls = PRF.prf(n_estimators=n_estimators, bootstrap=bootstrap, max_depth=max_depth, max_features=max_features)
            prf_cls.fit(X=X_train_noisy, dX=dX, py=py)
            score = prf_cls.score(X_test, y=y_test)
            accuracies_PRF.append(score)

        accuracies['PRF'] = (np.mean(accuracies_PRF), np.std(accuracies_PRF))

    # Base output directories and paths
    if noise_scale < 0.4:
        noise_level_prefix = "1" # Changed to just "1"
    elif 0.4 <= noise_scale <= 0.9:
        noise_level_prefix = "2" # Changed to just "2"
    else:
        noise_level_prefix = "3" # Changed to just "3"
    
    # Ensure noise_type is defined and accessible here. 
    # Assuming 'noise_type' is passed into this function or is a global/closure variable.
    # For example, if it's passed as an argument: def run_pipeline(config, noise_type): ...
    
    # Construct a more descriptive noise string for filenames
    # This will append the numeric level and the noise_type string (e.g., "1_gaussian" or "2_beta")
    noise_filename_suffix = f"{noise_level_prefix}_{noise_type}" 

    output_base = f"results/{noise_filename_suffix}" # Use the new suffix here
    os.makedirs(output_base, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(dataset_path))[0]
    title_name = base_name.replace('_', ' ').title()

    output_dir = os.path.join(output_base, "plots")
    os.makedirs(output_dir, exist_ok=True)
    # Changed output_filename to include noise_type
    output_filename = os.path.join(output_dir, f"{base_name}_{noise_filename_suffix}_({label_noise_range[0]},{label_noise_range[1]}).png") 

    csv_dir = os.path.join(output_base, "tables")
    os.makedirs(csv_dir, exist_ok=True)
    # Changed csv_path to include noise_type
    csv_path = os.path.join(csv_dir, f"{base_name}_{noise_filename_suffix}_({label_noise_range[0]},{label_noise_range[1]}).csv") 

    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        "Model": list(accuracies.keys()),
        "Mean Accuracy": [v[0] for v in accuracies.values()],
        "Std": [v[1] for v in accuracies.values()]
    })

    # Sort by mean accuracy descending
    comparison_df = comparison_df.sort_values("Mean Accuracy", ascending=False).reset_index(drop=True)

    # Color highlight for PDF
    colors = comparison_df["Model"].apply(lambda x: "#55bfc7" if x == "PDF" else "lightgray")

    # Create the plot
    fig, ax = plt.subplots(figsize=(9, 6))

    # Bar plot
    sns.barplot(
        data=comparison_df,
        x="Model",
        y="Mean Accuracy",
        palette=colors,
        errorbar=None,
        ax=ax
    )

    # Manual error bars
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

    # Axis settings
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1.05)
    sns.despine()

    # Annotate error bars
    for x, (mean, std) in enumerate(zip(means, stds)):
        ax.text(
            x, mean + std + 0.02,
            f'{mean:.2f}±{std:.2f}',
            ha='center', va='bottom',
            fontsize=10, color='black'
        )

    # Formatting
    ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')

    # Title and subtitle
    fig.text(
        0.1, 0.93,
        f"{title_name} dataset – Noise Levels: {noise_scale} ({noise_type}), {label_noise_range}",
        ha='left',
        fontsize=14,
        fontweight='bold'
    )
    fig.text(
        0.1, 0.88,
        "Average model accuracy and standard deviation across 5 runs",
        ha='left',
        fontsize=11, color="#333333"
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.85])

    # Save figure
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')

    # Save CSV
    comparison_df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")

    # Print summary
    print(f"Results for dataset: {os.path.basename(dataset_path)}")
    for _, row in comparison_df.iterrows():
        print(f"{row['Model']} Accuracy: {row['Mean Accuracy']:.4f} ± {row['Std']:.4f}")

    return accuracies