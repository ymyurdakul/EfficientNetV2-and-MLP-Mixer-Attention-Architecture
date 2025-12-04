import os
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.utils.class_weight import compute_class_weight

import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Reproducibility: global seeds (for multi-seed experiments)
# ---------------------------------------------------------------------
def set_global_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------
from keras_cv_attention_models import cspnext

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

def build_model(num_classes: int):
    """
    This is a placeholder example using CSPNeXtTiny.
    In the manuscript, replace this with your EfficientNetV2
    + MLP-Mixer-Attention architecture and keep the rest identical.
    """
    model = cspnext.CSPNeXtTiny(
        input_shape=INPUT_SHAPE,
        pretrained="imagenet",
        num_classes=num_classes
    )

    # Optional: you can recompile here with your desired optimizer and weight decay
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ---------------------------------------------------------------------
# Paths and training configuration
# ---------------------------------------------------------------------
DATA_ROOT = "folds_image"       # should contain fold1, fold2, ..., fold5
BASE_SAVE_DIR = "SoftmaxResults_PatientWise5Fold"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 1
FOLDS = [1, 2, 3, 4, 5]
SEEDS = [42]  # if you want multi-seed: e.g., [0, 42, 1234]

# Optional: external / shifted test set for generalization experiments
EXTERNAL_TEST_DIR = None  # e.g., "external_data/figshare_shift" or BraTS/Kaggle


# ---------------------------------------------------------------------
# Data generators with explicit augmentations and preprocessing
# ---------------------------------------------------------------------
def get_data_generators(fold_num: int, seed: int):
    data_path = os.path.join(DATA_ROOT, f"fold{fold_num}")
    path_train = os.path.join(data_path, "train")
    path_val = os.path.join(data_path, "test")   # TODO: gerçek val klasörü varsa burayı "val" yapın
    path_test = os.path.join(data_path, "test")

    # Training: strong but realistic augmentations
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    train_generator = train_datagen.flow_from_directory(
        directory=path_train,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        shuffle=True,
        seed=seed
    )

    # Validation and test: only rescaling, no augmentations
    test_val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    valid_generator = test_val_datagen.flow_from_directory(
        directory=path_val,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        shuffle=False
    )

    test_generator = test_val_datagen.flow_from_directory(
        directory=path_test,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        shuffle=False
    )

    return train_generator, valid_generator, test_generator


# ---------------------------------------------------------------------
# (Optional) Patient-level leakage check helper
# You must adapt extract_patient_id() to your filename convention if you want to use it.
# ---------------------------------------------------------------------
def extract_patient_id_from_filename(filename: str):
    """
    Example implementation; ADAPT THIS to your own naming convention.
    E.g., if filenames are 'patient123_slice_004.mat', parse 'patient123'.
    """
    # Example: split by '_' and take the first token
    # return filename.split("_")[0]
    return filename  # placeholder – replace with your own logic


def check_patient_leakage(train_generator, test_generator):
    train_files = train_generator.filenames
    test_files = test_generator.filenames

    train_patients = {extract_patient_id_from_filename(os.path.basename(f)) for f in train_files}
    test_patients = {extract_patient_id_from_filename(os.path.basename(f)) for f in test_files}

    intersection = train_patients.intersection(test_patients)
    if len(intersection) > 0:
        print("WARNING: Patient-wise leakage detected between train and test!")
        print("Overlapping patient IDs:", intersection)
    else:
        print("Patient-wise leakage check: PASSED (no overlapping patient IDs).")


# ---------------------------------------------------------------------
# Calibration helper (Brier score) – can be reported foldwise if desired
# ---------------------------------------------------------------------
def brier_score_multiclass(y_true_ohe: np.ndarray, y_pred_prob: np.ndarray):
    """
    Compute mean Brier score for multiclass probabilities.
    """
    return np.mean(np.sum((y_pred_prob - y_true_ohe) ** 2, axis=1))


# ---------------------------------------------------------------------
# ROC Curve helper: plot per-class multiclass ROC curves
# ---------------------------------------------------------------------
def plot_multiclass_roc(y_true_ohe, y_pred_prob, num_classes, fold_save_dir, seed, fold_num):
    """
    y_true_ohe : (N, C) one-hot encoded true labels
    y_pred_prob: (N, C) predicted probabilities (softmax outputs)
    """
    fpr = {}
    tpr = {}
    roc_auc_vals = {}

    # Her sınıf için ROC eğrisi (eğer test setinde o sınıf varsa)
    for c in range(num_classes):
        # Eğer bu sınıfa ait hiç pozitif örnek yoksa ROC tanımlı değil → o sınıfı atla
        if y_true_ohe[:, c].sum() == 0:
            print(f"Fold {fold_num}, class {c}: no positive samples, skipping ROC.")
            continue
        fpr[c], tpr[c], _ = roc_curve(y_true_ohe[:, c], y_pred_prob[:, c])
        roc_auc_vals[c] = auc(fpr[c], tpr[c])

    if len(roc_auc_vals) == 0:
        print(f"Fold {fold_num}: No classes with positive samples; ROC curve not plotted.")
        return

    # Micro-average ROC (tüm sınıfları tek problemmiş gibi)
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_ohe.ravel(), y_pred_prob.ravel())
    roc_auc_vals["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot
    plt.figure(figsize=(8, 6))
    # Micro curve
    plt.plot(fpr["micro"], tpr["micro"],
             label=f"micro-average ROC (AUC = {roc_auc_vals['micro']:.3f})",
             linestyle=":", linewidth=3)

    # Per-class curves
    for c, auc_val in roc_auc_vals.items():
        if c == "micro":
            continue
        plt.plot(fpr[c], tpr[c],
                 label=f"Class {c} (AUC = {auc_val:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves (Seed {seed}, Fold {fold_num})")
    plt.legend(loc="lower right")
    plt.grid(True)

    roc_path = os.path.join(fold_save_dir, f"roc_curves_seed{seed}_fold{fold_num}.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC curves to: {roc_path}")


# ---------------------------------------------------------------------
# Main Training Loop over seeds and folds
# ---------------------------------------------------------------------
all_results_rows = []  # to aggregate across folds and seeds for summary CSV

for seed in SEEDS:
    print(f"\n\n========== SEED {seed} ==========\n")
    set_global_seed(seed)

    # metrics per fold for this seed
    fold_metrics = {
        "accuracy": [],
        "precision_weighted": [],
        "recall_weighted": [],
        "f1_weighted": [],
        "kappa": [],
        "auc_macro": [],
        "auc_ovr_weighted": [],
        "brier": [],
        "train_time_sec": [],
        "test_time_sec": [],
    }

    for fold_num in FOLDS:
        print(f"\n----- Training on fold {fold_num} (seed={seed}) -----\n")

        train_gen, val_gen, test_gen = get_data_generators(fold_num, seed)

        num_classes = train_gen.num_classes
        class_indices = train_gen.class_indices
        print("Class indices:", class_indices)

        # Optional leakage check (only meaningful if extract_patient_id_from_filename is implemented)
        # check_patient_leakage(train_gen, test_gen)

        # Compute class weights to handle imbalance
        train_labels = train_gen.classes
        class_weights_array = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weights = {
            i: w for i, w in enumerate(class_weights_array)
        }
        print("Class weights:", class_weights)

        # Build and compile model for this fold
        model = build_model(num_classes=num_classes)

        # Callbacks: EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
        fold_save_dir = os.path.join(
            BASE_SAVE_DIR,
            f"seed_{seed}",
            f"fold_{fold_num}"
        )
        os.makedirs(fold_save_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            fold_save_dir,
            f"best_model_seed{seed}_fold{fold_num}.h5"
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            )
        ]

        # Parameters count (for Params column in tables)
        num_params = model.count_params()
        print(f"Number of trainable parameters: {num_params}")

        # Training with timing
        start_train = time.time()
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        train_duration = time.time() - start_train

        # Testing with timing
        start_test = time.time()
        y_pred_prob = model.predict(test_gen, verbose=1)
        test_duration = time.time() - start_test

        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = test_gen.classes
        y_true_ohe = keras.utils.to_categorical(y_true, num_classes=num_classes)

        # Metrics
        kappa = cohen_kappa_score(y_true, y_pred)

        # OVR and macro ROC-AUC (hata durumunda NaN yazalım ki script çökmesin)
        try:
            auc_macro = roc_auc_score(
                y_true_ohe, y_pred_prob,
                multi_class="ovr", average="macro"
            )
            auc_ovr_weighted = roc_auc_score(
                y_true_ohe, y_pred_prob,
                multi_class="ovr", average="weighted"
            )
        except ValueError as e:
            print(f"ROC-AUC could not be computed for fold {fold_num}: {e}")
            auc_macro = np.nan
            auc_ovr_weighted = np.nan

        brier = brier_score_multiclass(y_true_ohe, y_pred_prob)

        report_dict = classification_report(
            y_true, y_pred, digits=6, output_dict=True
        )
        f1_weighted = report_dict["weighted avg"]["f1-score"]
        precision_weighted = report_dict["weighted avg"]["precision"]
        recall_weighted = report_dict["weighted avg"]["recall"]
        accuracy = report_dict["accuracy"]

        cm = confusion_matrix(y_true, y_pred)

        # Store fold-level metrics
        fold_metrics["accuracy"].append(accuracy)
        fold_metrics["precision_weighted"].append(precision_weighted)
        fold_metrics["recall_weighted"].append(recall_weighted)
        fold_metrics["f1_weighted"].append(f1_weighted)
        fold_metrics["kappa"].append(kappa)
        fold_metrics["auc_macro"].append(auc_macro)
        fold_metrics["auc_ovr_weighted"].append(auc_ovr_weighted)
        fold_metrics["brier"].append(brier)
        fold_metrics["train_time_sec"].append(train_duration)
        fold_metrics["test_time_sec"].append(test_duration)

        # -----------------------------------------------------------------
        # Save everything needed for paper: metrics, curves, confusion matrix
        # -----------------------------------------------------------------
        metrics_txt_path = os.path.join(fold_save_dir, "classification_metrics.txt")
        with open(metrics_txt_path, "w") as f:
            f.write(f"Seed: {seed}\n")
            f.write(f"Fold: {fold_num}\n")
            f.write(f"Num parameters: {num_params}\n\n")

            f.write(f"Cohen's Kappa: {kappa:.6f}\n")
            f.write(f"Macro AUC (OVR): {auc_macro:.6f}\n")
            f.write(f"Weighted AUC (OVR): {auc_ovr_weighted:.6f}\n")
            f.write(f"Brier Score (multiclass): {brier:.6f}\n")
            f.write(f"Training Time (s): {train_duration:.3f}\n")
            f.write(f"Testing Time (s): {test_duration:.3f}\n\n")

            f.write(f"Accuracy: {accuracy:.6f}\n")
            f.write(f"Precision (Weighted Avg): {precision_weighted:.6f}\n")
            f.write(f"Recall/Sensitivity (Weighted Avg): {recall_weighted:.6f}\n")
            f.write(f"F1-Score (Weighted Avg): {f1_weighted:.6f}\n\n")

            f.write("Per-class classification report:\n")
            f.write(classification_report(y_true, y_pred, digits=6))

            f.write("\nConfusion Matrix:\n")
            f.write(str(cm))

        # ROC EĞRİLERİNİ KAYDET
        plot_multiclass_roc(
            y_true_ohe=y_true_ohe,
            y_pred_prob=y_pred_prob,
            num_classes=num_classes,
            fold_save_dir=fold_save_dir,
            seed=seed,
            fold_num=fold_num
        )

        # Also save predictions and probabilities for later statistical tests / ROC / calibration
        preds_save_path = os.path.join(fold_save_dir, "predictions_probabilities.npz")
        np.savez(
            preds_save_path,
            y_true=y_true,
            y_pred=y_pred,
            y_pred_prob=y_pred_prob,
            class_indices=class_indices
        )

        # History plots (accuracy and loss)
        history_path = os.path.join(fold_save_dir, "history.txt")
        with open(history_path, "w") as f:
            for key, value_list in history.history.items():
                f.write(f"{key}: {', '.join(map(str, value_list))}\n")

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="Train")
        plt.plot(history.history["val_accuracy"], label="Validation")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")

        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="Train")
        plt.plot(history.history["val_loss"], label="Validation")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(os.path.join(fold_save_dir, "accuracy_loss_curves.png"))
        plt.close()

        # Confusion matrix heatmap (publication-grade)
        df_cm = pd.DataFrame(
            cm,
            index=[f"True_{c}" for c in range(num_classes)],
            columns=[f"Pred_{c}" for c in range(num_classes)]
        )
        plt.figure(figsize=(6, 5))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix (Seed {seed}, Fold {fold_num})")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(os.path.join(fold_save_dir, "confusion_matrix.png"))
        plt.close()

        # Add row to global results log
        all_results_rows.append({
            "seed": seed,
            "fold": fold_num,
            "num_params": num_params,
            "accuracy": accuracy,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "kappa": kappa,
            "auc_macro": auc_macro,
            "auc_ovr_weighted": auc_ovr_weighted,
            "brier": brier,
            "train_time_sec": train_duration,
            "test_time_sec": test_duration,
        })

    # -----------------------------------------------------------------
    # After all folds for this seed: compute mean, std, and 95% CI
    # ---------------------------------------------------------------------
    seed_summary_path = os.path.join(BASE_SAVE_DIR, f"summary_seed_{seed}.txt")
    with open(seed_summary_path, "w") as f:
        f.write(f"Cross-validated results for seed={seed} (5-fold)\n\n")
        for metric_name, values in fold_metrics.items():
            values = np.array(values)
            mean_val = np.nanmean(values)
            std_val = np.nanstd(values, ddof=1)
            ci95 = 1.96 * std_val / np.sqrt(np.sum(~np.isnan(values)))
            f.write(
                f"{metric_name}: "
                f"mean={mean_val:.6f}, std={std_val:.6f}, 95% CI = [{mean_val - ci95:.6f}, {mean_val + ci95:.6f}]\n"
            )

# ---------------------------------------------------------------------
# Save all fold/seed results in one CSV file for later significance tests (McNemar, bootstraps, etc.)
# ---------------------------------------------------------------------
results_df = pd.DataFrame(all_results_rows)
results_csv_path = os.path.join(BASE_SAVE_DIR, "all_seed_fold_results.csv")
results_df.to_csv(results_csv_path, index=False)
print("Saved global results table to:", results_csv_path)
