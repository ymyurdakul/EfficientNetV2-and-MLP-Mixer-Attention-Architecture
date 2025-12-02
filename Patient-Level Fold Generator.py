
import os
import shutil
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DATA_DIR = r"BrainTumorReview/data/allinone"     
OUTPUT_DIR = r"BrainTumorReview/data/folds_z"     
PID_EXCEL = "pid_list.xlsx"                     
# ================================================================

df = pd.read_excel(PID_EXCEL)  

def decode_pid(pid_str):
    # "49,48,48,51,54,48" -> "100360"
    codes = [int(x) for x in str(pid_str).split(",")]
    return "".join(chr(c) for c in codes)

df["pid_str"] = df["pid"].apply(decode_pid)

label_counts_per_pid = df.groupby("pid_str")["label"].nunique()
if (label_counts_per_pid > 1).any():
    pids_multi_label = label_counts_per_pid[label_counts_per_pid > 1].index.tolist()
    raise ValueError(f"Birden fazla label'a sahip PID(ler) bulundu: {pids_multi_label}")

patient_df = (
    df.groupby("pid_str")
      .agg(
          label=("label", lambda x: x.iloc[0]),  # tek label
          n_slices=("filename", "count"),
          filenames=("filename", lambda x: ",".join(str(int(v)) for v in sorted(x)))
      )
      .reset_index()
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
patient_df["fold"] = -1

for fold_idx, (_, test_idx) in enumerate(
    skf.split(patient_df["pid_str"], patient_df["label"]), start=1
):
    patient_df.loc[test_idx, "fold"] = fold_idx

df = df.merge(patient_df[["pid_str", "fold"]], on="pid_str", how="left")

df.to_excel("slice_folds.xlsx", index=False)
print("Slice-level fold bilgisi slice_folds.xlsx olarak kaydedildi.")

# ------------------------------------------------------------------
# ------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

labels = sorted(df["label"].unique())  # örn: [1, 2, 3]

for k in range(1, 6):  # fold1..fold5
    fold_root = os.path.join(OUTPUT_DIR, f"fold{k}")
    for phase in ["train", "test"]:
        for lbl in labels:
            class_dir = os.path.join(fold_root, phase, str(int(lbl)))
            os.makedirs(class_dir, exist_ok=True)

# ------------------------------------------------------------------
# ------------------------------------------------------------------
for idx, row in df.iterrows():
    filename = int(row["filename"])
    fold_id = int(row["fold"])
    label = int(row["label"])

    src_path = os.path.join(DATA_DIR, f"{filename}.mat")
    if not os.path.isfile(src_path):
        print(f"Uyarı: .mat dosyası bulunamadı, atlanıyor: {src_path}")
        continue

    for k in range(1, 6):
        fold_root = os.path.join(OUTPUT_DIR, f"fold{k}")
        if fold_id == k:
            phase = "test"
        else:
            phase = "train"

        dst_dir = os.path.join(fold_root, phase, str(label))
        os.makedirs(dst_dir, exist_ok=True)

        dst_path = os.path.join(dst_dir, f"{filename}.mat")

        shutil.copy2(src_path, dst_path)

print("Tüm fold/train-test/label kopyalama işlemi tamamlandı.")
print(f"Fold yapısı: {OUTPUT_DIR} altında oluşturuldu.")
