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
import itertools
import copy

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

    # Load dataset and split once
    X_train, X_test, y_train, y_test, label_map = load_keel_dataset(
        train_path=dataset_path,
        already_split=False
    )
    # Ensure labels are adjusted to start from 0
    y_train = y_train - 1
    y_test = y_test - 1
    # Add noise to train set once
    X_train_noisy, _, dX, _ = add_noise(X_train, noise_type=noise_type, noise_scale=noise_scale)
    y_train_noisy, _, py, _ = add_label_noise(
        y_train, mode="random_prob", noise_level=label_noise_scale,
        random_seed=30, prob_range=label_noise_range
    )

    # print label noise scale for debugging
    print(f"--- DEBUG: label_noise_range: {label_noise_range} ---")
    print(f"--- DEBUG: py head: {py[:10]} ---")
    print(f"--- DEBUG: y_train head: {y_train[:10]} ---")
    print(f"--- DEBUG: y_train_noisy head: {y_train_noisy[:10]} ---")

    n_classes = len(label_map)
    n_features = X_train.shape[1]

    # Determine grid parameter lists for rf, prf, deep_forest (df), pdrf
    # RF & PRF must share same n_estimators list
    # Each entry in models_config can be a dict with scalar values or lists; wrap scalars into single-element lists
    def ensure_list(val):
        if isinstance(val, list) or isinstance(val, tuple):
            return list(val)
        else:
            return [val]

    # RF
    rf_cfg = models_config.get('rf', None)
    if rf_cfg is not None:
        rf_n_list = ensure_list(rf_cfg.get('n_estimators', 50))
    else:
        rf_n_list = []
    # PRF: share n_estimators
    prf_cfg = models_config.get('prf', None)
    if prf_cfg is not None:
        # use same n_estimators list
        if rf_cfg is not None:
            prf_n_list = rf_n_list
        else:
            prf_n_list = ensure_list(prf_cfg.get('n_estimators', 10))
        prf_bootstrap_list = ensure_list(prf_cfg.get('bootstrap', True))
        prf_max_depth_list = ensure_list(prf_cfg.get('max_depth', None))
        prf_max_features_list = ensure_list(prf_cfg.get('max_features', 'sqrt'))
    else:
        prf_n_list = []
        prf_bootstrap_list = []
        prf_max_depth_list = []
        prf_max_features_list = []

    # Deep Forest
    df_cfg = models_config.get('deep_forest', None)
    if df_cfg is not None:
        df_n_estimators_list = ensure_list(df_cfg.get('n_estimators', 2))
        df_n_trees_list = ensure_list(df_cfg.get('n_trees_drf', 10))
    else:
        df_n_estimators_list = []
        df_n_trees_list = []

    # PDRF
    pdrf_cfg = models_config.get('pdrf', None)
    if pdrf_cfg is not None:
        pdrf_n_cascade_list = ensure_list(pdrf_cfg.get('n_cascade_estimators', 4))
        pdrf_n_trees_list = ensure_list(pdrf_cfg.get('n_trees_pdrf', 10))
        pdrf_max_depth_list = ensure_list(pdrf_cfg.get('max_depth_pdrf', 10))
    else:
        pdrf_n_cascade_list = []
        pdrf_n_trees_list = []
        pdrf_max_depth_list = []

    # Build grid of combinations
    # Only vary among those models present
    grid_param_combinations = []
    # If none of these present, run single default config
    if any([rf_cfg, prf_cfg, df_cfg, pdrf_cfg]):
        # Prepare lists, using single-element lists if config not present to allow iteration
        if not rf_cfg:
            rf_n_list_iter = [None]
        else:
            rf_n_list_iter = rf_n_list
        if not prf_cfg:
            prf_n_list_iter = [None]
            prf_bootstrap_list = [None]
            prf_max_depth_list = [None]
            prf_max_features_list = [None]
        else:
            prf_n_list_iter = prf_n_list
        if not df_cfg:
            df_n_estimators_list_iter = [None]
            df_n_trees_list_iter = [None]
        else:
            df_n_estimators_list_iter = df_n_estimators_list
            df_n_trees_list_iter = df_n_trees_list
        if not pdrf_cfg:
            pdrf_n_cascade_list_iter = [None]
            pdrf_n_trees_list_iter = [None]
            pdrf_max_depth_list_iter = [None]
        else:
            pdrf_n_cascade_list_iter = pdrf_n_cascade_list
            pdrf_n_trees_list_iter = pdrf_n_trees_list
            pdrf_max_depth_list_iter = pdrf_max_depth_list

        # For RF & PRF sharing tree counts: when both present, iterate over rf_n_list; if only one present, handled above
        for rf_n in rf_n_list_iter:
            for prf_n in prf_n_list_iter:
                # enforce sharing: if both present and rf_cfg exists, prf_n == rf_n
                if rf_cfg and prf_cfg and prf_n is not None and rf_n is not None and prf_n != rf_n:
                    continue
                for prf_bootstrap in prf_bootstrap_list:
                    for prf_max_depth in prf_max_depth_list:
                        for prf_max_features in prf_max_features_list:
                            for df_n_estimators in df_n_estimators_list_iter:
                                for df_n_trees in df_n_trees_list_iter:
                                    for pdrf_n_cascade in pdrf_n_cascade_list_iter:
                                        for pdrf_n_trees in pdrf_n_trees_list_iter:
                                            for pdrf_max_depth in pdrf_max_depth_list_iter:
                                                combo = {
                                                    'rf_n': rf_n,
                                                    'prf_n': prf_n,
                                                    'prf_bootstrap': prf_bootstrap,
                                                    'prf_max_depth': prf_max_depth,
                                                    'prf_max_features': prf_max_features,
                                                    'df_n_estimators': df_n_estimators,
                                                    'df_n_trees': df_n_trees,
                                                    'pdrf_n_cascade': pdrf_n_cascade,
                                                    'pdrf_n_trees': pdrf_n_trees,
                                                    'pdrf_max_depth': pdrf_max_depth
                                                }
                                                grid_param_combinations.append(combo)
    else:
        grid_param_combinations = [dict()]  # single run

    # Prepare output directories
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

    base_name = os.path.splitext(os.path.basename(dataset_path))[0]
    title_name = base_name.replace('_', ' ').title()

    # Iterate grid combinations
    for combo in grid_param_combinations:
        # Label parts for filenames
        parts = []
        # RF
        if rf_cfg and combo.get('rf_n') is not None:
            parts.append(f"rf{combo['rf_n']}")
        # PRF
        if prf_cfg and combo.get('prf_n') is not None:
            parts.append(f"prf{combo['prf_n']}")
            if combo.get('prf_bootstrap') is not None:
                parts.append(f"bs{str(combo['prf_bootstrap'])}")
            if combo.get('prf_max_depth') is not None:
                parts.append(f"md{combo['prf_max_depth']}")
            if combo.get('prf_max_features') is not None:
                parts.append(f"mf{combo['prf_max_features']}")
        # Deep Forest
        if df_cfg and combo.get('df_n_estimators') is not None:
            parts.append(f"dfest{combo['df_n_estimators']}")
        if df_cfg and combo.get('df_n_trees') is not None:
            parts.append(f"dftrees{combo['df_n_trees']}")
        # PDRF
        if pdrf_cfg and combo.get('pdrf_n_cascade') is not None:
            parts.append(f"pdrfc{combo['pdrf_n_cascade']}")
        if pdrf_cfg and combo.get('pdrf_n_trees') is not None:
            parts.append(f"pdrft{combo['pdrf_n_trees']}")
        if pdrf_cfg and combo.get('pdrf_max_depth') is not None:
            parts.append(f"pdrfmd{combo['pdrf_max_depth']}")

        combo_suffix = "_".join(parts) if parts else "default"
        # Create per-grid-folder
        combo_plot_dir = os.path.join(grid_base, "plots")
        combo_table_dir = os.path.join(grid_base, "tables")
        os.makedirs(combo_plot_dir, exist_ok=True)
        os.makedirs(combo_table_dir, exist_ok=True)
        output_plot_path = os.path.join(combo_plot_dir,
                                        f"{base_name}_{noise_filename_suffix}_{combo_suffix}_({label_noise_range[0]},{label_noise_range[1]}).png")
        output_csv_path = os.path.join(combo_table_dir,
                                       f"{base_name}_{noise_filename_suffix}_{combo_suffix}_({label_noise_range[0]},{label_noise_range[1]}).csv")

        # For this combination, compute accuracies
        accuracies = {}

        # --- PDRF ---
        if pdrf_cfg:
            # get params
            n_cascade_estimators = combo['pdrf_n_cascade'] if combo['pdrf_n_cascade'] is not None else pdrf_cfg.get('n_cascade_estimators', 4)
            n_trees_pdrf = combo['pdrf_n_trees'] if combo['pdrf_n_trees'] is not None else pdrf_cfg.get('n_trees_pdrf', 10)
            max_depth_pdrf = combo['pdrf_max_depth'] if combo['pdrf_max_depth'] is not None else pdrf_cfg.get('max_depth_pdrf', 10)
            accuracies_PDRF = []
            for seed in seeds:
                set_all_seeds(seed)
                model = pdf.CascadeForestClassifier(random_state=seed)
                prf_estimators = []
                for i in range(n_cascade_estimators):
                    estimator = PRF4DF.SklearnCompatiblePRF(
                        n_classes_=n_classes,
                        n_features_=n_features,
                        n_estimators=n_trees_pdrf,
                        max_depth=max_depth_pdrf,
                        n_jobs=1
                    )
                    prf_estimators.append(estimator)
                model.set_estimator(prf_estimators)
                model.fit(X=X_train_noisy, y=y_train_noisy, dX=dX, py=py)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_pred, y_test)
                accuracies_PDRF.append(acc)
            accuracies['PDRF'] = (np.mean(accuracies_PDRF), np.std(accuracies_PDRF))

        # --- Random Forest ---
        if rf_cfg:
            n_estimators = combo['rf_n'] if combo['rf_n'] is not None else rf_cfg.get('n_estimators', 50)
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
        if df_cfg:
            n_estimators_df = combo['df_n_estimators'] if combo['df_n_estimators'] is not None else df_cfg.get('n_estimators', 2)
            n_trees_drf = combo['df_n_trees'] if combo['df_n_trees'] is not None else df_cfg.get('n_trees_drf', 10)
            accuracies_DF = []
            for seed in seeds:
                set_all_seeds(seed)
                clf = CascadeForestClassifier(n_estimators=n_estimators_df, random_state=seed, n_trees=n_trees_drf)
                clf.fit(X_train_noisy, y_train_noisy)
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                accuracies_DF.append(acc)
            accuracies['DF'] = (np.mean(accuracies_DF), np.std(accuracies_DF))

        # --- Neural Network ---
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
        if prf_cfg:
            # prf shares n_estimators: combo['prf_n']
            n_estimators_prf = combo['prf_n'] if combo['prf_n'] is not None else prf_cfg.get('n_estimators', 10)
            bootstrap = combo['prf_bootstrap'] if combo['prf_bootstrap'] is not None else prf_cfg.get('bootstrap', True)
            max_depth = combo['prf_max_depth'] if combo['prf_max_depth'] is not None else prf_cfg.get('max_depth', None)
            max_features = combo['prf_max_features'] if combo['prf_max_features'] is not None else prf_cfg.get('max_features', 'sqrt')

            accuracies_PRF = []
            for seed in seeds:
                set_all_seeds(seed)
                prf_cls = PRF.prf(
                    n_estimators=n_estimators_prf,
                    bootstrap=bootstrap,
                    max_depth=max_depth,
                    max_features=max_features
                )
                prf_cls.fit(X=X_train_noisy, dX=dX, py=py)
                score = prf_cls.score(X_test, y=y_test)
                accuracies_PRF.append(score)
            accuracies['PRF'] = (np.mean(accuracies_PRF), np.std(accuracies_PRF))

        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            "Model": list(accuracies.keys()),
            "Mean Accuracy": [v[0] for v in accuracies.values()],
            "Std": [v[1] for v in accuracies.values()]
        })

        # Sort by mean accuracy descending
        comparison_df = comparison_df.sort_values("Mean Accuracy", ascending=False).reset_index(drop=True)

        # Color highlight for PDRF
        colors = comparison_df["Model"].apply(lambda x: "#55bfc7" if x == "PDRF" else "lightgray")

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
            "Average model accuracy and standard deviation across runs",
            ha='left',
            fontsize=11, color="#333333"
        )

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.85])

        # Save figure and CSV for this grid combo
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        comparison_df.to_csv(output_csv_path, index=False)
        plt.close(fig)

        print(f"Saved to {output_csv_path}")
        # Print summary
        print(f"Results for dataset: {os.path.basename(dataset_path)}, combo: {combo_suffix}")
        for _, row in comparison_df.iterrows():
            print(f"{row['Model']} Accuracy: {row['Mean Accuracy']:.4f} ± {row['Std']:.4f}")

    # Return nothing or optionally return summaries
    return None
