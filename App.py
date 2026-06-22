import os
import pickle
import shutil
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

allowed_types = ["zip", "7z"] if HAS_7Z else ["zip"]
if not HAS_7Z:
    st.sidebar.warning(
        "Library `py7zr` belum terpasang, upload .7z dinonaktifkan. "
        "Pastikan `py7zr` ada di requirements.txt saat deploy."
    )


# ---------------------------------------------------------------------------
# Fungsi-fungsi inti (dipakai oleh kedua mode upload)
# ---------------------------------------------------------------------------
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


def estimate_memory_mb(n_images: int, size, grayscale: bool) -> float:
    """Estimasi kasar RAM (MB) yang dibutuhkan untuk menampung array hasil resize."""
    width, height = size
    channels = 1 if grayscale else 3
    return (n_images * height * width * channels) / (1024 ** 2)


def build_dataset(samples, size, grayscale):
    """Versi hemat memori: array hasil di-preallocate sekali di awal (bukan
    dikumpulkan ke list lalu di-stack di akhir, yang sempat menduplikasi
    memori saat proses stacking). Gambar dibuka satu-satu lewat context
    manager supaya buffer-nya langsung dilepas tiap iterasi, dan PIL diberi
    hint `draft()` supaya JPEG besar di-decode langsung dalam ukuran kecil
    (lebih hemat RAM & lebih cepat) daripada decode full-res baru di-resize.
    """
    has_labels = any(s[1] is not None for s in samples)
    label_names = sorted({s[1] for s in samples if s[1] is not None}) if has_labels else []
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    mode = "L" if grayscale else "RGB"
    width, height = size
    n_total = len(samples)
    shape = (n_total, height, width) if grayscale else (n_total, height, width, 3)

    data = np.zeros(shape, dtype=np.uint8)
    labels = np.full(n_total, -1, dtype=np.int32)
    filenames, errors = [], []
    valid = 0

    progress = st.progress(0.0, text="Memproses gambar...")
    for i, (filepath, label, relpath) in enumerate(samples):
        try:
            with Image.open(filepath) as img:
                try:
                    img.draft(mode, size)  # hint decode kecil utk JPEG, no-op utk format lain
                except Exception:
                    pass
                converted = img.convert(mode).resize(size)
                data[valid] = np.asarray(converted, dtype=np.uint8)
            labels[valid] = label_to_idx[label] if has_labels else -1
            filenames.append(relpath)
            valid += 1
        except (UnidentifiedImageError, OSError, ValueError) as e:
            errors.append((relpath, str(e)))
        if n_total:
            progress.progress((i + 1) / n_total, text=f"Memproses gambar... ({i + 1}/{n_total})")
    progress.empty()

    data = data[:valid]  # buang slot kosong kalau ada file yang gagal diproses
    labels = labels[:valid]
    return {
        "data": data,
        "labels": labels,
        "label_names": label_names,
        "filenames": filenames,
        "has_labels": has_labels,
    }, errors


def process_archive(archive_path: str, filename_hint: str) -> bool:
    """Pipeline lengkap: ekstrak -> build dataset -> tampilkan hasil -> pickle -> download.
    Dipakai dari mode upload langsung maupun mode upload bertahap.
    Return True kalau berhasil sampai tahap download button ditampilkan."""
    with tempfile.TemporaryDirectory() as tmpdir:
        extract_dir = os.path.join(tmpdir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)

        try:
            with st.spinner("Mengekstrak archive..."):
                extract_archive(archive_path, extract_dir, filename_hint)
        except Exception as e:
            st.error(f"Gagal mengekstrak file: {e}")
            return False

        samples = collect_samples(extract_dir)
        if not samples:
            st.error("Tidak ditemukan file gambar di dalam archive ini.")
            return False

        est_mb = estimate_memory_mb(len(samples), (int(img_w), int(img_h)), color_mode == "Grayscale")
        if est_mb > 400:
            st.warning(
                f"Estimasi data hasil resize ~{est_mb:.0f} MB di memori untuk {len(samples)} gambar. "
                "Streamlit Cloud (terutama tier gratis) RAM-nya terbatas (umumnya sekitar 1GB), jadi "
                "ini berisiko bikin app crash/restart. Kalau mau lebih aman, kecilkan ukuran resize di "
                "sidebar dulu sebelum lanjut."
            )
            if not st.checkbox("Saya paham risikonya, tetap lanjutkan proses", key="risk_ack"):
                return False

        result, errors = build_dataset(
            samples, size=(int(img_w), int(img_h)), grayscale=(color_mode == "Grayscale")
        )

        if result["data"].size == 0:
            st.error("Semua file gambar gagal diproses. Cek kembali isi archive.")
            return False

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
        if preview_n > 0:
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
            del data, labels  # lepas array gabungan, sudah tidak dibutuhkan
        else:
            dataset = {
                "data": data,
                "labels": labels,
                "label_names": label_names,
                "filenames": result["filenames"],
            }

        pkl_bytes = pickle.dumps(dataset, protocol=4)
        out_name = os.path.splitext(filename_hint)[0] + "_dataset.pkl"

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
        return True


# ---------------------------------------------------------------------------
# Pilihan mode upload
# ---------------------------------------------------------------------------
st.divider()
mode = st.radio(
    "Cara upload",
    ["Upload langsung (file < 500MB)", "Upload bertahap / per-potongan (untuk file besar)"],
    horizontal=True,
)

if mode.startswith("Upload langsung"):
    uploaded_file = st.file_uploader("Upload file archive (.zip atau .7z)", type=allowed_types)

    if uploaded_file is not None:
        # Reset niat proses kalau file yang di-upload berubah (file baru/berbeda)
        sig = (uploaded_file.name, uploaded_file.size)
        if st.session_state.get("last_uploaded_sig") != sig:
            st.session_state.last_uploaded_sig = sig
            st.session_state.want_process_direct = False

        if st.button("🚀 Proses Archive", type="primary"):
            st.session_state.want_process_direct = True

        if st.session_state.get("want_process_direct"):
            with tempfile.TemporaryDirectory() as tdir:
                archive_path = os.path.join(tdir, uploaded_file.name)
                with open(archive_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                success = process_archive(archive_path, uploaded_file.name)
                if success:
                    st.session_state.want_process_direct = False
    else:
        st.info("Upload file .zip atau .7z untuk mulai.")

else:
    st.markdown(
        "**Cara pakai mode ini (untuk file besar, misal >500MB):**\n"
        "1. Pecah file archive aslinya jadi beberapa potongan kecil di komputermu "
        "(disarankan ±200MB per potongan):\n"
        "   - **Windows (7-Zip)**: klik kanan file archive → `7-Zip` → `Split file...` → "
        "isi ukuran misal `200M`\n"
        "   - **Linux / Mac (terminal)**: `split -b 200m archive.zip archive.zip.part`\n"
        "2. Upload potongan-potongannya di bawah ini **satu per satu, sesuai urutan asli**.\n"
        "3. Setelah semua potongan ter-upload, klik **Selesai & Proses**."
    )

    if "work_dir" not in st.session_state:
        st.session_state.work_dir = tempfile.mkdtemp(prefix="chunked_")
        st.session_state.chunk_count = 0
        st.session_state.total_bytes = 0
        st.session_state.reassembled_path = os.path.join(st.session_state.work_dir, "reassembled.bin")
        open(st.session_state.reassembled_path, "wb").close()

    archive_format = st.selectbox("Format archive aslinya", ["zip", "7z"] if HAS_7Z else ["zip"])

    st.info(
        f"📦 {st.session_state.chunk_count} potongan diterima — "
        f"total terkumpul {st.session_state.total_bytes / (1024 ** 2):.1f} MB"
    )

    chunk_file = st.file_uploader(
        f"Upload potongan ke-{st.session_state.chunk_count + 1}",
        key=f"chunk_uploader_{st.session_state.chunk_count}",
    )

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        add_clicked = st.button("➕ Tambahkan potongan", disabled=(chunk_file is None))
    with col_b:
        finish_clicked = st.button(
            "✅ Selesai & Proses", disabled=(st.session_state.chunk_count == 0), type="primary"
        )
    with col_c:
        reset_clicked = st.button("🔄 Reset")

    if add_clicked and chunk_file is not None:
        with open(st.session_state.reassembled_path, "ab") as f:
            f.write(chunk_file.getbuffer())
        st.session_state.chunk_count += 1
        st.session_state.total_bytes += chunk_file.size
        st.rerun()

    if reset_clicked:
        shutil.rmtree(st.session_state.work_dir, ignore_errors=True)
        for key in ["work_dir", "chunk_count", "total_bytes", "reassembled_path", "want_process_chunked"]:
            st.session_state.pop(key, None)
        st.rerun()

    if finish_clicked:
        st.session_state.want_process_chunked = True

    if st.session_state.get("want_process_chunked"):
        filename_hint = f"dataset.{archive_format}"
        success = process_archive(st.session_state.reassembled_path, filename_hint)
        if success:
            st.session_state.want_process_chunked = False
            # Bersihkan file reassembly di disk, sudah tidak dibutuhkan
            # (hasil pkl sudah dipegang sebagai bytes oleh download_button).
            shutil.rmtree(st.session_state.work_dir, ignore_errors=True)
            for key in ["work_dir", "chunk_count", "total_bytes", "reassembled_path", "want_process_chunked"]:
                st.session_state.pop(key, None)
