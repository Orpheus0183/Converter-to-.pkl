"""
Zip/7z to PKL Converter
========================
Streamlit app untuk mengonversi archive (.zip / .7z) yang isinya gambar
menjadi satu file .pkl berisi array gambar + label, siap dipakai untuk
training model machine learning.

Struktur folder yang didukung di dalam archive:
    dataset.zip
    ├── kucing/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── anjing/
        ├── img1.jpg
        └── img2.jpg

Nama subfolder otomatis jadi label. Kalau gambar diletakkan langsung
tanpa subfolder, dataset akan dibuat tanpa label (unlabeled).
"""

import os
import io
import pickle
import zipfile
import tempfile

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

try:
    import py7zr
    HAS_7Z = True
except ImportError:
    HAS_7Z = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

st.set_page_config(page_title="Zip/7z → PKL Dataset Converter", page_icon="🗂️", layout="wide")

st.title("🗂️ Konversi Zip/7z Gambar → Dataset .pkl")
st.caption(
    "Upload archive berisi gambar (boleh dikelompokkan per folder kelas), "
    "atur ukuran & mode warna, lalu unduh hasilnya sebagai file .pkl "
    "(array NumPy + label) yang siap dipakai untuk training ML."
)

# ---------------------------------------------------------------------------
# Sidebar - pengaturan
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Pengaturan")

col_w, col_h = st.sidebar.columns(2)
img_w = col_w.number_input("Lebar (px)", min_value=8, max_value=1024, value=64, step=8)
img_h = col_h.number_input("Tinggi (px)", min_value=8, max_value=1024, value=64, step=8)

color_mode = st.sidebar.selectbox("Mode warna", ["RGB", "Grayscale"])

do_split = st.sidebar.checkbox("Bagi jadi train / test split", value=False)
test_size = 0.2
if do_split:
    test_size = st.sidebar.slider("Proporsi data test", 0.05, 0.5, 0.2, 0.05)

st.sidebar.divider()
st.sidebar.caption(
    "💡 Gambar dengan ukuran besar + jumlah banyak akan menghasilkan file .pkl "
    "yang besar dan butuh memori lebih saat diproses. Sesuaikan ukuran resize "
    "kalau dataset-nya besar."
)

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
allowed_types = ["zip", "7z"] if HAS_7Z else ["zip"]
if not HAS_7Z:
    st.warning("Library `py7zr` belum terpasang di environment ini, jadi upload .7z dinonaktifkan. "
               "Pastikan `py7zr` ada di requirements.txt saat deploy.")

uploaded_file = st.file_uploader(
    "Upload file archive (.zip atau .7z)", type=allowed_types
)


def extract_archive(archive_path: str, extract_dir: str, filename: str) -> None:
    lower = filename.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    elif lower.endswith(".7z"):
        if not HAS_7Z:
            raise RuntimeError("py7zr tidak tersedia untuk membaca file .7z")
        with py7zr.SevenZipFile(archive_path, mode="r") as zf:
            zf.extractall(path=extract_dir)
    else:
        raise ValueError("Format file tidak didukung. Gunakan .zip atau .7z")


def collect_samples(extract_dir: str):
    """Cari semua file gambar, label diambil dari nama folder induk langsung."""
    samples = []
    for root, _dirs, files in os.walk(extract_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTS:
                filepath = os.path.join(root, fname)
                rel_dir = os.path.relpath(root, extract_dir)
                label = os.path.basename(root) if rel_dir != "." else None
                relpath = os.path.relpath(filepath, extract_dir)
                samples.append((filepath, label, relpath))
    return samples


def build_dataset(samples, size, grayscale):
    has_labels = any(s[1] is not None for s in samples)
    label_names = sorted({s[1] for s in samples if s[1] is not None}) if has_labels else []
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    mode = "L" if grayscale else "RGB"
    data, labels, filenames, errors = [], [], [], []

    progress = st.progress(0.0, text="Memproses gambar...")
    total = len(samples)
    for i, (filepath, label, relpath) in enumerate(samples):
        try:
            img = Image.open(filepath).convert(mode).resize(size)
            data.append(np.array(img))
            labels.append(label_to_idx[label] if has_labels else -1)
            filenames.append(relpath)
        except (UnidentifiedImageError, OSError, ValueError) as e:
            errors.append((relpath, str(e)))
        if total:
            progress.progress((i + 1) / total, text=f"Memproses gambar... ({i + 1}/{total})")
    progress.empty()

    data_arr = np.stack(data) if data else np.empty((0,))
    labels_arr = np.array(labels)
    return {
        "data": data_arr,
        "labels": labels_arr,
        "label_names": label_names,
        "filenames": filenames,
        "has_labels": has_labels,
    }, errors


if uploaded_file is not None:
    process_clicked = st.button("🚀 Proses Archive", type="primary")

    if process_clicked:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = os.path.join(tmpdir, uploaded_file.name)
            with open(archive_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            extract_dir = os.path.join(tmpdir, "extracted")
            os.makedirs(extract_dir, exist_ok=True)

            try:
                with st.spinner("Mengekstrak archive..."):
                    extract_archive(archive_path, extract_dir, uploaded_file.name)
            except Exception as e:
                st.error(f"Gagal mengekstrak file: {e}")
                st.stop()

            samples = collect_samples(extract_dir)
            if not samples:
                st.error("Tidak ditemukan file gambar di dalam archive ini.")
                st.stop()

            result, errors = build_dataset(samples, size=(int(img_w), int(img_h)), grayscale=(color_mode == "Grayscale"))

            if result["data"].size == 0:
                st.error("Semua file gambar gagal diproses. Cek kembali isi archive.")
                st.stop()

            data, labels = result["data"], result["labels"]
            label_names, has_labels = result["label_names"], result["has_labels"]

            # ---------------------------------------------------------------
            # Ringkasan
            # ---------------------------------------------------------------
            st.success("Selesai diproses!")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total gambar", len(data))
            m2.metric("Jumlah kelas", len(label_names) if has_labels else "Tanpa label")
            m3.metric("Shape array", str(data.shape))

            if errors:
                with st.expander(f"⚠️ {len(errors)} file gagal diproses (diabaikan)"):
                    for relpath, msg in errors:
                        st.write(f"- `{relpath}`: {msg}")

            if has_labels:
                st.subheader("Distribusi kelas")
                counts = {name: int(np.sum(labels == i)) for i, name in enumerate(label_names)}
                st.bar_chart(counts)

            st.subheader("Preview")
            preview_n = min(8, len(data))
            cols = st.columns(preview_n)
            for i in range(preview_n):
                caption = label_names[labels[i]] if has_labels else result["filenames"][i]
                cols[i].image(data[i], caption=caption, use_container_width=True)

            # ---------------------------------------------------------------
            # Build dict final + pickle
            # ---------------------------------------------------------------
            if do_split:
                rng = np.random.RandomState(42)
                idx = rng.permutation(len(data))
                split_at = int(len(data) * (1 - test_size))
                train_idx, test_idx = idx[:split_at], idx[split_at:]
                dataset = {
                    "X_train": data[train_idx],
                    "y_train": labels[train_idx],
                    "X_test": data[test_idx],
                    "y_test": labels[test_idx],
                    "label_names": label_names,
                }
            else:
                dataset = {
                    "data": data,
                    "labels": labels,
                    "label_names": label_names,
                    "filenames": result["filenames"],
                }

            pkl_bytes = pickle.dumps(dataset, protocol=4)
            out_name = os.path.splitext(uploaded_file.name)[0] + "_dataset.pkl"

            st.download_button(
                "⬇️ Download dataset.pkl",
                data=pkl_bytes,
                file_name=out_name,
                mime="application/octet-stream",
                type="primary",
            )

            with st.expander("📄 Cara load file .pkl ini di Python"):
                st.code(
                    "import pickle\n\n"
                    f"with open('{out_name}', 'rb') as f:\n"
                    "    dataset = pickle.load(f)\n\n"
                    + (
                        "X_train, y_train = dataset['X_train'], dataset['y_train']\n"
                        "X_test, y_test = dataset['X_test'], dataset['y_test']\n"
                        "label_names = dataset['label_names']"
                        if do_split
                        else
                        "data, labels = dataset['data'], dataset['labels']\n"
                        "label_names = dataset['label_names']  # kosong jika tanpa label\n"
                        "filenames = dataset['filenames']"
                    ),
                    language="python",
                )
else:
    st.info("Upload file .zip atau .7z untuk mulai.")
