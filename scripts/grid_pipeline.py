import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import struct
from array import array

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier

from deepforest import CascadeForestClassifier
import probabilistic_deep_forest as pdf
import PRF
import PRF4DF

from utils import load_keel_dataset
from utils.noising import add_noise
from utils.noising import add_label_noise



class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)   
    

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
    
    if config["name"] == "mnist":

        base_dir = os.path.dirname(os.path.abspath(__file__))
        training_images_filepath = os.path.join(base_dir, "..", "data", "MNIST", "train-images.idx3-ubyte")
        training_labels_filepath = os.path.join(base_dir, "..", "data", "MNIST", "train-labels.idx1-ubyte")
        test_images_filepath = os.path.join(base_dir, "..", "data", "MNIST", "t10k-images.idx3-ubyte")
        test_labels_filepath = os.path.join(base_dir, "..", "data", "MNIST", "t10k-labels.idx1-ubyte")
        mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
        (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist_dataloader.load_data()

        # Convert to numpy arrays for easier manipulation
        X_train_raw = np.array(X_train_raw[:400])
        y_train = np.array(y_train_raw[:400])
        X_test_raw = np.array(X_test_raw[:100])
        y_test = np.array(y_test_raw[:100])

        # Reshape the 28x28 images into 1D vectors of 784 features
        n_train_samples = X_train_raw.shape[0]
        n_test_samples = X_test_raw.shape[0]
        n_features = X_train_raw.shape[1] * X_train_raw.shape[2] # 28 * 28 = 784
        X_train = X_train_raw.reshape(n_train_samples, n_features)
        X_test = X_test_raw.reshape(n_test_samples, n_features)

        # Normalize pixel values to be between 0 and 1
        X_train = X_train / 255.0
        X_test = X_test / 255.0


    else:
        # Load dataset and split
        X_train, X_test, y_train, y_test, label_map = load_keel_dataset(
            train_path=config["dataset_path"],
            already_split=False
        )
        # Ensure labels are adjusted to start from 0
        y_train = y_train - 1  
        y_test = y_test - 1  


    dataset_path = config['dataset_path']
    seeds = config.get('seeds', [27, 272, 2727, 1, 30])  # default seeds

    noise_scale = config.get('noise_scale', 0.0)  # default noise 0.0
    label_noise_scale = config.get('label_noise_scale', 0.0)  # default noise 0.0
    label_noise_range = config.get('label_noise_range', (0.0, 0.5))  # default range
    noise_type = config.get('noise_type', "gaussian")  # default noise 0.0

    models_config = config.get('models', {})

    # Add noise to train set
    X_train_noisy, _, dX, _ = add_noise(X_train, noise_type=noise_type, noise_scale=noise_scale)
    y_train_noisy, _, py, _ = add_label_noise(y_train, mode="random_prob", noise_level=label_noise_scale, random_seed=30, prob_range=label_noise_range)
    
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    # Helper to wrap scalar into list
    def ensure_list(val):
        if isinstance(val, list) or isinstance(val, tuple):
            return list(val)
        else:
            return [val]

    # Prepare noise-based output dirs
    if noise_scale < 0.4:
        noise_level_prefix = "1"
    elif 0.4 <= noise_scale <= 0.9:
        noise_level_prefix = "2"
    else:
        noise_level_prefix = "3"
    noise_filename_suffix = f"{noise_level_prefix}_{noise_type}"
    output_base = f"results/{noise_filename_suffix}"
    os.makedirs(output_base, exist_ok=True)
    grid_base = os.path.join(output_base, "grid")
    os.makedirs(grid_base, exist_ok=True)

    base_name = config.get('name', os.path.splitext(os.path.basename(dataset_path))[0])
    title_name = base_name.replace('_', ' ').title()
    
    # To collect best per model
    best_configs = []

    # --- Random Forest grid ---
    rf_cfg = models_config.get('rf', None)
    if rf_cfg is not None:
        rf_n_list = ensure_list(rf_cfg.get('n_estimators', 50))
        # Prepare output dirs
        model_name = "rf"
        model_base = os.path.join(grid_base, model_name)
        plots_dir = os.path.join(model_base, "plots")
        tables_dir = os.path.join(model_base, "tables")
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)

        records = []
        for n_estimators in rf_n_list:
            accuracies_RF = []
            for seed in seeds:
                set_all_seeds(seed)
                rf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=1)
                rf.fit(X_train_noisy, y_train_noisy)
                y_pred = rf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                accuracies_RF.append(acc)
            mean_acc = np.mean(accuracies_RF)
            std_acc = np.std(accuracies_RF)
            records.append({
                'n_estimators': n_estimators,
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc
            })
            print(f"RF n_estimators={n_estimators}: {mean_acc:.4f} ± {std_acc:.4f}")

        # Save table
        df_rf = pd.DataFrame(records)
        csv_path = os.path.join(tables_dir,
                                f"{base_name}_{noise_filename_suffix}_rf_grid_({label_noise_range[0]},{label_noise_range[1]}).csv")
        df_rf.to_csv(csv_path, index=False)
        print(f"Saved RF grid CSV to {csv_path}")

        # Plot: bar of mean_accuracy vs n_estimators
        fig, ax = plt.subplots(figsize=(7,5))
        sns.barplot(data=df_rf, x='n_estimators', y='mean_accuracy', ax=ax, errorbar=None)
        # errorbars manually
        x_positions = range(len(df_rf))
        ax.errorbar(x_positions, df_rf['mean_accuracy'].values,
                    yerr=df_rf['std_accuracy'].values, fmt='none',
                    ecolor='black', elinewidth=1.5, capsize=5)
        ax.set_xlabel("n_estimators")
        ax.set_ylabel("Mean Accuracy")
        ax.set_ylim(0,1.05)
        sns.despine()
        for x, (m,s) in enumerate(zip(df_rf['mean_accuracy'], df_rf['std_accuracy'])):
            ax.text(x, m + s + 0.02, f'{m:.2f}±{s:.2f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticklabels([str(v) for v in df_rf['n_estimators']], fontweight='bold')
        fig.suptitle(f"RF grid on {title_name}, noise {noise_scale} ({noise_type})", y=1.02)
        plot_path = os.path.join(plots_dir,
                                 f"{base_name}_{noise_filename_suffix}_rf_grid_({label_noise_range[0]},{label_noise_range[1]}).png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved RF grid plot to {plot_path}")

        # Determine best
        best_row = df_rf.loc[df_rf['mean_accuracy'].idxmax()]
        best_configs.append({
            'Model': 'RF',
            'hyperparams': f"n_estimators={int(best_row['n_estimators'])}",
            'mean_accuracy': best_row['mean_accuracy'],
            'std_accuracy': best_row['std_accuracy']
        })

    # --- PRF grid ---
    prf_cfg = models_config.get('prf', None)
    if prf_cfg is not None:
        # share n_estimators with RF if rf_cfg exists
        if rf_cfg is not None:
            prf_n_list = ensure_list(rf_cfg.get('n_estimators', 50))
        else:
            prf_n_list = ensure_list(prf_cfg.get('n_estimators', 10))
        prf_bootstrap_list = ensure_list(prf_cfg.get('bootstrap', True))
        prf_max_depth_list = ensure_list(prf_cfg.get('max_depth', None))
        prf_max_features_list = ensure_list(prf_cfg.get('max_features', 'sqrt'))

        model_name = "prf"
        model_base = os.path.join(grid_base, model_name)
        plots_dir = os.path.join(model_base, "plots")
        tables_dir = os.path.join(model_base, "tables")
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)

        records = []
        for n_estimators in prf_n_list:
            for bootstrap in prf_bootstrap_list:
                for max_depth in prf_max_depth_list:
                    for max_features in prf_max_features_list:
                        # Validate max_depth is int or None
                        if max_depth is not None and not isinstance(max_depth, int):
                            raise ValueError(f"prf max_depth must be int or None, got {max_depth!r}")
                        key = f"n={n_estimators}_bs={bootstrap}_md={max_depth}_mf={max_features}"
                        accuracies_PRF = []
                        for seed in seeds:
                            set_all_seeds(seed)
                            prf_cls = PRF.prf(
                                n_estimators=n_estimators,
                                bootstrap=bootstrap,
                                max_depth=max_depth,
                                max_features=max_features
                            )
                            prf_cls.fit(X=X_train_noisy, dX=dX, py=py)
                            score = prf_cls.score(X_test, y=y_test)
                            accuracies_PRF.append(score)
                        mean_acc = np.mean(accuracies_PRF)
                        std_acc = np.std(accuracies_PRF)
                        records.append({
                            'n_estimators': n_estimators,
                            'bootstrap': bootstrap,
                            'max_depth': max_depth,
                            'max_features': max_features,
                            'mean_accuracy': mean_acc,
                            'std_accuracy': std_acc
                        })
                        print(f"PRF {key}: {mean_acc:.4f} ± {std_acc:.4f}")

        df_prf = pd.DataFrame(records)
        # Save CSV
        csv_path = os.path.join(tables_dir,
                                f"{base_name}_{noise_filename_suffix}_prf_grid_({label_noise_range[0]},{label_noise_range[1]}).csv")
        df_prf.to_csv(csv_path, index=False)
        print(f"Saved PRF grid CSV to {csv_path}")

        # Plot: we can plot mean_accuracy for each combination; labels may be long, so skip barplot or use a simpler summary.
        # Here: pick top K (e.g. top 5) and plot, or plot all with rotated labels.
        fig, ax = plt.subplots(figsize=(10,6))
        df_prf['combo_label'] = df_prf.apply(
            lambda row: f"n{row['n_estimators']}_bs{row['bootstrap']}_md{row['max_depth']}_mf{row['max_features']}", axis=1
        )
        sns.barplot(data=df_prf, x='combo_label', y='mean_accuracy', ax=ax, errorbar=None)
        x_positions = range(len(df_prf))
        ax.errorbar(x_positions, df_prf['mean_accuracy'].values,
                    yerr=df_prf['std_accuracy'].values, fmt='none',
                    ecolor='black', elinewidth=1.5, capsize=5)
        ax.set_xlabel("PRF config")
        ax.set_ylabel("Mean Accuracy")
        ax.set_ylim(0,1.05)
        plt.xticks(rotation=45, ha='right')
        sns.despine()
        for x, (m,s) in enumerate(zip(df_prf['mean_accuracy'], df_prf['std_accuracy'])):
            ax.text(x, m + s + 0.02, f'{m:.2f}±{s:.2f}', ha='center', va='bottom', fontsize=8)
        fig.suptitle(f"PRF grid on {title_name}, noise {noise_scale} ({noise_type})", y=1.02)
        plt.tight_layout()
        plot_path = os.path.join(plots_dir,
                                 f"{base_name}_{noise_filename_suffix}_prf_grid_({label_noise_range[0]},{label_noise_range[1]}).png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved PRF grid plot to {plot_path}")

        # Determine best
        best_idx = df_prf['mean_accuracy'].idxmax()
        best_row = df_prf.loc[best_idx]
        best_configs.append({
            'Model': 'PRF',
            'hyperparams': f"n_estimators={int(best_row['n_estimators'])}, bootstrap={best_row['bootstrap']}, "
                           f"max_depth={best_row['max_depth']}, max_features={best_row['max_features']}",
            'mean_accuracy': best_row['mean_accuracy'],
            'std_accuracy': best_row['std_accuracy']
        })

    # --- DF grid ---
    df_cfg = models_config.get('df', None)
    if df_cfg is not None:
        df_n_estimators_list = ensure_list(df_cfg.get('n_estimators', 2))
        df_n_trees_list = ensure_list(df_cfg.get('n_trees_df', 10))

        model_name = "deep_forest"
        model_base = os.path.join(grid_base, model_name)
        plots_dir = os.path.join(model_base, "plots")
        tables_dir = os.path.join(model_base, "tables")
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)

        records = []
        for n_estimators_df in df_n_estimators_list:
            for n_trees_df in df_n_trees_list:
                accuracies_DF = []
                for seed in seeds:
                    set_all_seeds(seed)
                    clf = CascadeForestClassifier(n_estimators=n_estimators_df, random_state=seed, n_trees=n_trees_df, n_jobs=1)
                    clf.fit(X_train_noisy, y_train_noisy)
                    y_pred = clf.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    accuracies_DF.append(acc)
                mean_acc = np.mean(accuracies_DF)
                std_acc = np.std(accuracies_DF)
                records.append({
                    'n_estimators': n_estimators_df,
                    'n_trees_df': n_trees_df,
                    'mean_accuracy': mean_acc,
                    'std_accuracy': std_acc
                })
                print(f"DF n_estimators={n_estimators_df}, n_trees={n_trees_df}: {mean_acc:.4f} ± {std_acc:.4f}")

        df_df = pd.DataFrame(records)
        csv_path = os.path.join(tables_dir,
                                f"{base_name}_{noise_filename_suffix}_deep_forest_grid_({label_noise_range[0]},{label_noise_range[1]}).csv")
        df_df.to_csv(csv_path, index=False)
        print(f"Saved Deep Forest grid CSV to {csv_path}")

        # Plot
        fig, ax = plt.subplots(figsize=(8,5))
        df_df['combo_label'] = df_df.apply(
            lambda row: f"e{row['n_estimators']}_t{row['n_trees_df']}", axis=1
        )
        sns.barplot(data=df_df, x='combo_label', y='mean_accuracy', ax=ax, errorbar=None)
        x_positions = range(len(df_df))
        ax.errorbar(x_positions, df_df['mean_accuracy'].values,
                    yerr=df_df['std_accuracy'].values, fmt='none',
                    ecolor='black', elinewidth=1.5, capsize=5)
        ax.set_xlabel("DF config")
        ax.set_ylabel("Mean Accuracy")
        ax.set_ylim(0,1.05)
        plt.xticks(rotation=45, ha='right')
        sns.despine()
        for x, (m,s) in enumerate(zip(df_df['mean_accuracy'], df_df['std_accuracy'])):
            ax.text(x, m + s + 0.02, f'{m:.2f}±{s:.2f}', ha='center', va='bottom', fontsize=8)
        fig.suptitle(f"DF grid on {title_name}, noise {noise_scale} ({noise_type})", y=1.02)
        plt.tight_layout()
        plot_path = os.path.join(plots_dir,
                                 f"{base_name}_{noise_filename_suffix}_deep_forest_grid_({label_noise_range[0]},{label_noise_range[1]}).png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved Deep Forest grid plot to {plot_path}")

        # Determine best
        best_row = df_df.loc[df_df['mean_accuracy'].idxmax()]
        best_configs.append({
            'Model': 'DF',
            'hyperparams': f"n_estimators={int(best_row['n_estimators'])}, n_trees={int(best_row['n_trees_df'])}",
            'mean_accuracy': best_row['mean_accuracy'],
            'std_accuracy': best_row['std_accuracy']
        })

    # --- PDF grid ---
    pdf_cfg = models_config.get('pdf', None)
    if pdf_cfg is not None:
        pdf_n_cascade_list = ensure_list(pdf_cfg.get('n_cascade_estimators', 4))
        pdf_n_trees_list = ensure_list(pdf_cfg.get('n_trees_pdf', 10))
        pdf_max_depth_list = ensure_list(pdf_cfg.get('max_depth_pdf', 10))

        model_name = "pdf"
        model_base = os.path.join(grid_base, model_name)
        plots_dir = os.path.join(model_base, "plots")
        tables_dir = os.path.join(model_base, "tables")
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)

        records = []
        for n_cascade in pdf_n_cascade_list:
            for n_trees_pdf in pdf_n_trees_list:
                for max_depth_pdf in pdf_max_depth_list:
                    # Validate
                    if max_depth_pdf is not None and not isinstance(max_depth_pdf, int):
                        raise ValueError(f"pdf max_depth must be int, got {max_depth_pdf!r}")
                    key = f"c{n_cascade}_t{n_trees_pdf}_md{max_depth_pdf}"
                    accuracies_pdf = []
                    for seed in seeds:
                        set_all_seeds(seed)
                        model = pdf.CascadeForestClassifier(random_state=seed)
                        prf_estimators = []
                        for i in range(n_cascade):
                            estimator = PRF4DF.SklearnCompatiblePRF(
                                n_classes_=n_classes,
                                n_features_=n_features,
                                n_estimators=n_trees_pdf,
                                max_depth=max_depth_pdf,
                                n_jobs=1
                            )
                            prf_estimators.append(estimator)
                        model.set_estimator(prf_estimators)
                        model.fit(X=X_train_noisy, dX=dX, py=py)
                        y_pred = model.predict(X_test)
                        acc = accuracy_score(y_pred, y_test)
                        accuracies_pdf.append(acc)
                    mean_acc = np.mean(accuracies_pdf)
                    std_acc = np.std(accuracies_pdf)
                    records.append({
                        'n_cascade_estimators': n_cascade,
                        'n_trees_pdf': n_trees_pdf,
                        'max_depth_pdf': max_depth_pdf,
                        'mean_accuracy': mean_acc,
                        'std_accuracy': std_acc
                    })
                    print(f"pdf {key}: {mean_acc:.4f} ± {std_acc:.4f}")

        df_pdf = pd.DataFrame(records)
        csv_path = os.path.join(tables_dir,
                                f"{base_name}_{noise_filename_suffix}_pdf_grid_({label_noise_range[0]},{label_noise_range[1]}).csv")
        df_pdf.to_csv(csv_path, index=False)
        print(f"Saved PDF grid CSV to {csv_path}")

        # Plot
        fig, ax = plt.subplots(figsize=(10,6))
        df_pdf['combo_label'] = df_pdf.apply(
            lambda row: f"c{row['n_cascade_estimators']}_t{row['n_trees_pdf']}_md{row['max_depth_pdf']}", axis=1
        )
        sns.barplot(data=df_pdf, x='combo_label', y='mean_accuracy', ax=ax, errorbar=None)
        x_positions = range(len(df_pdf))
        ax.errorbar(x_positions, df_pdf['mean_accuracy'].values,
                    yerr=df_pdf['std_accuracy'].values, fmt='none',
                    ecolor='black', elinewidth=1.5, capsize=5)
        ax.set_xlabel("PDF config")
        ax.set_ylabel("Mean Accuracy")
        ax.set_ylim(0,1.05)
        plt.xticks(rotation=45, ha='right')
        sns.despine()
        for x, (m,s) in enumerate(zip(df_pdf['mean_accuracy'], df_pdf['std_accuracy'])):
            ax.text(x, m + s + 0.02, f'{m:.2f}±{s:.2f}', ha='center', va='bottom', fontsize=8)
        fig.suptitle(f"pdf grid on {title_name}, noise {noise_scale} ({noise_type})", y=1.02)
        plt.tight_layout()
        plot_path = os.path.join(plots_dir,
                                 f"{base_name}_{noise_filename_suffix}_pdf_grid_({label_noise_range[0]},{label_noise_range[1]}).png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved PDF grid plot to {plot_path}")

        # Determine best
        best_row = df_pdf.loc[df_pdf['mean_accuracy'].idxmax()]
        best_configs.append({
            'Model': 'PDF',
            'hyperparams': f"n_cascade_estimators={int(best_row['n_cascade_estimators'])}, "
                           f"n_trees_pdf={int(best_row['n_trees_pdf'])}, max_depth_pdf={best_row['max_depth_pdf']}",
            'mean_accuracy': best_row['mean_accuracy'],
            'std_accuracy': best_row['std_accuracy']
        })

    # --- Neural Network (unchanged) ---
    if 'neural_network' in models_config:
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

        mean_acc = np.mean(accuracies_NN)
        std_acc = np.std(accuracies_NN)
        print(f"NN Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        best_configs.append({
            'Model': 'NN',
            'hyperparams': f"epochs={epochs}, batch_size={batch_size}, hidden_units={hidden_units}, dropout_rate={dropout_rate}, optimizer={optimizer}",
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc
        })

    # --- Kernel SVM (unchanged) ---
    if 'ksvm' in models_config:
        set_all_seeds(seeds[0])
        model = SVC()
        param_grid = {
            'kernel': ['linear', 'rbf'],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=1)
        grid.fit(X_train_noisy, y_train_noisy)
        best_svm = grid.best_estimator_

        y_pred = best_svm.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mean_acc = acc
        std_acc = 0.0
        print(f"KSVM best config {grid.best_params_}: {mean_acc:.4f}")
        best_configs.append({
            'Model': 'KSVM',
            'hyperparams': str(grid.best_params_),
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc
        })

    # --- After all models: save combined best_configs ---
    if best_configs:
        df_best = pd.DataFrame(best_configs)
        best_csv = os.path.join(output_base, f"{base_name}_{noise_filename_suffix}_best_configs.csv")
        df_best.to_csv(best_csv, index=False)
        print(f"Saved best configs summary to {best_csv}")

        # Combined bar plot of best mean_accuracy per model
        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(data=df_best, x='Model', y='mean_accuracy', ax=ax, errorbar=None,
                    palette=["#55bfc7" if m=='PDF' else "lightgray" for m in df_best['Model']])
        x_positions = range(len(df_best))
        ax.errorbar(x_positions, df_best['mean_accuracy'].values,
                    yerr=df_best['std_accuracy'].values, fmt='none',
                    ecolor='black', elinewidth=1.5, capsize=5)
        ax.set_ylabel("Best Mean Accuracy")
        ax.set_ylim(0,1.05)
        sns.despine()
        for x, (m,s) in enumerate(zip(df_best['mean_accuracy'], df_best['std_accuracy'])):
            ax.text(x, m + s + 0.02, f'{m:.2f}±{s:.2f}', ha='center', va='bottom', fontsize=9)
        ax.set_title(f"Result with best configuration on {title_name}, noise {noise_scale} ({noise_type})")
        plt.tight_layout()
        best_plot = os.path.join(output_base, f"{base_name}_{noise_filename_suffix}_best_overall.png")
        plt.savefig(best_plot, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved combined best plot to {best_plot}")

    return None
