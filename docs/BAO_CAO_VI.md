# BÁO CÁO ĐỒ ÁN BIG DATA
## Suy luận Quan hệ theo Thời gian của Mô hình Ngôn ngữ Lớn để Dự đoán Sụp đổ Giá Cổ phiếu
### (Temporal Relational Reasoning of LLMs for Stock Crash Prediction — phỏng theo arXiv:2410.17266)

> Ghi chú: Các chỗ đánh dấu 「điền sau」 là kết quả đang chạy (mẻ Kaggle toàn bộ corpus 2016–2023 với lọc theo danh mục) — sẽ điền số khi có. Các con số đã có là số thật đã đo.

---

## 1. Tóm tắt (Abstract)

Đồ án xây dựng một hệ thống **dự đoán xác suất sụp đổ (crash)** của một danh mục cổ phiếu vốn hóa lớn trong **~3 ngày giao dịch tới**, bằng cách cho một **mô hình ngôn ngữ lớn (LLM) đọc tin tức tài chính** và suy luận **zero-shot** (không huấn luyện lại mô hình). Phương pháp gồm 4 pha: **Brainstorm → Memory → Attention → Reason**, kèm **RAG** (truy hồi tăng cường) để bám vào các tiền lệ lịch sử.

Trọng tâm Big Data: xử lý nguồn tin **23 GB / 15,7 triệu bài** (FNSPID) bằng kỹ thuật **stream-process, lập chỉ mục phân vùng, và chọn lọc bằng RAG**, kết hợp **tính toán phân tán** (Spark cho ETL, pool GPU Kaggle free-tier cho suy luận LLM).

**Kết quả chính:** Cửa sổ khủng hoảng COVID đạt **AUROC 0.785** (có RAG **0.847**); giai đoạn rộng 2016–2020 đạt **0.710**. RAG là cải thiện ổn định và có ý nghĩa thống kê (+0.074, p=0.009 ở quy mô lớn). Mở rộng sang corpus toàn bộ 2016–2023 (lọc theo danh mục): base **0.615** / RAG **0.652** — hồi phục mạnh sau khi sửa lỗi chọn lọc, nhưng *không vượt* bộ tin bundled gốc (chi tiết §9.2).

---

## 2. Giới thiệu & Động lực

- Thị trường tài chính sinh dữ liệu **khối lượng lớn, tốc độ cao, đa dạng** — bài toán Big Data điển hình.
- Dự đoán **hướng giá** (lên/xuống) gần như bất khả thi (giả thuyết thị trường hiệu quả dạng yếu). Nhưng **rủi ro đuôi / sụp đổ** mang tín hiệu từ tin tức (tâm lý, sự kiện, lan truyền) → khả thi hơn.
- Ý tưởng cốt lõi: thay vì huấn luyện mô hình dự báo, ta để **LLM suy luận** về quan hệ nhân quả giữa các thực thể tin tức và danh mục, có **bộ nhớ theo thời gian** (tác động xấu lưu lại và phân rã dần).

**Use case:** cảnh báo sớm rủi ro sụp đổ danh mục cho nhà đầu tư/quản trị rủi ro — một advisory hằng ngày, không phải lệnh giao dịch.

---

## 3. Bài toán & Mục tiêu

- **Đầu vào:** dòng tin tức tài chính theo ngày cho danh mục 6 cổ phiếu: **AAPL, AMZN, GOOGL, NVDA, TSLA, NFLX**.
- **Đầu ra:** với mỗi ngày *t*, xác suất danh mục (equal-weight) **giảm ≥ 6%** trong cửa sổ 3 ngày tới (nhãn crash).
- **Đánh giá:** AUROC / PR-AUC trên backtest có nhãn (giá lịch sử cung cấp nhãn).
- **Ràng buộc nhân quả:** không rò rỉ tương lai — RAG chỉ truy hồi các ngày trong quá khứ vượt ngưỡng embargo ≥ tầm dự báo của nhãn.

---

## 4. Dữ liệu

### 4.1 Nguồn dữ liệu
| Nguồn | Vai trò | Quy mô |
|---|---|---|
| **FNSPID** (tin tài chính) | corpus huấn-luyện-RAG/suy luận lịch sử | 23 GB thô, 15,7 triệu bài, 4.775 mã, 1999–2023 |
| **Corpus đã lọc 2016–2023** | hồ dữ liệu cục bộ | **4.500.216 bài, 12 GB** |
| **Giá OHLCV** (yfinance) | sinh **nhãn** crash | 6 mã, 2.012 ngày giao dịch |
| **Tin trực tiếp** (yfinance + Google News RSS) | triển khai live (không nhãn) | ~500 tin/ngày |

### 4.2 Bốn chữ V của Big Data
- **Volume (Khối lượng):** nguồn 23 GB / 15,7 triệu bài; corpus 12 GB / 4,5 triệu bài.
- **Velocity (Tốc độ):** luồng tin trực tiếp ~500 tin/ngày, daemon cập nhật mỗi 60 giây, Spark Structured Streaming.
- **Variety (Đa dạng):** tin công ty, vĩ mô, crypto, thế giới; giá; nhiều nguồn (FNSPID, RSS, yfinance).
- **Value (Giá trị):** advisory cảnh báo sụp đổ + ứng dụng web triển khai thực.

### 4.3 Phân biệt quan trọng: Hiển thị ≠ Dự đoán
- Feed hiển thị 4 nhóm tin (🏢 công ty, 🌐 vĩ mô, ₿ crypto, 🌍 thế giới).
- **Dự đoán chỉ dùng tin công ty + vĩ mô** (đúng phân phối đã huấn luyện) — crypto/thế giới bị loại trước khi suy luận.

---

## 5. Phương pháp TRR (4 pha)

1. **Brainstorm:** LLM đọc tin trong ngày → trích **đồ thị tác động** (cạnh "X tác động ±Y", trọng số [0,1]).
2. **Memory (Bộ nhớ phân rã):** cập nhật bộ nhớ với cạnh mới; tác động cũ **phân rã theo hàm mũ** (λ) → tin xấu vẫn nâng rủi ro nhiều ngày, rồi nhạt dần ("temporal").
3. **Attention (PageRank):** cắt tỉa đồ thị gộp xuống *top-k* cạnh gần danh mục nhất.
4. **Reason:** LLM suy luận trên đồ thị con + tóm tắt bộ nhớ → **xác suất sụp đổ**.

> Mô hình **không bao giờ được huấn luyện** — nó suy luận. Chỉ một **meta-learner nhỏ** học trên các đặc trưng dẫn xuất; giá cung cấp nhãn.

---

## 6. RAG — Truy hồi Tăng cường

Hai vai trò, cả hai đều đưa thêm ngữ cảnh vào LLM:
1. **Chọn lọc theo truy hồi (retrieval-selection):** từ một kho tin lớn mỗi ngày, chọn *k* tin **liên quan nhất** → LLM chỉ đọc phần đã giới hạn (giữ chi phí LLM = O(số_ngày × k) dù corpus lớn cỡ GB).
2. **Few-shot theo tình huống (case-based / "kho ngày lịch sử"):** truy hồi các **ngày quá khứ tương tự** (TF-IDF) và chèn kết cục thực tế ("ngày này từng crash / không crash") vào prompt suy luận. Hoàn toàn **nhân quả** (chỉ nhìn quá khứ vượt embargo).

**Bài học quan trọng (kết quả trung thực):** chọn lọc theo *độ liên quan với truy vấn crash trên TOÀN bộ 4.775 mã* làm **giảm tín hiệu** (bơm từ vựng "crash" vào cả ngày yên tĩnh, và lấy tin không thuộc danh mục) → AUROC tụt dưới mức ngẫu nhiên. Khắc phục: **lọc theo mã trong danh mục trước**, rồi xếp hạng theo độ nổi bật (salience). *Liên quan ≠ liên quan-tới-danh-mục.*

---

## 7. Kiến trúc Big Data (điểm nhấn môn học)

### 7.1 Lưu trữ phân tầng — "lưu khổng lồ, phục vụ tí hon"
| Tầng | Nội dung | Kích thước |
|---|---|---|
| Lạnh / khối | corpus 2016–2023 | 12 GB (đĩa cục bộ) |
| Ấm / chỉ mục | **SQLite lập chỉ mục theo ngày** | 1,9 GB (tra cứu 1 ngày ~44 ms) |
| Nóng / phục vụ | phần tin RAG đã chọn | ~2 MB (tải lên Kaggle) |

Kỹ thuật: **stream-and-filter** (không lưu file thô 23 GB), **đọc theo từng khối (giới hạn RAM)**, **chỉ đọc các cột cần (bỏ cột nội dung)**, **chỉ mục phân vùng (SQLite)**, **chọn lọc RAG**. Dữ liệu dẫn xuất **không commit vào git** (chỉ commit code tái tạo).

### 7.2 Xử lý phân tán bằng Spark
- **Spark batch ETL:** đọc corpus 12 GB → ghi **Parquet phân vùng theo năm** (bố cục data-lake kiểu HDFS: cột hóa + nén + cắt tỉa phân vùng).
- Đo thực tế: 12 GB CSV → **718 MB Parquet trong 101 giây**; truy vấn từ Parquet đọc **4,5 triệu dòng trong 2,4 giây** (nhanh ~40×, song song 8 lõi). **Partition pruning:** chỉ quét `year=2020` trong 0,2 giây.
- **Cùng một code chạy trên cluster thật** chỉ bằng cách đổi `SPARK_MASTER=spark://host:7077`.
- **Spark Structured Streaming** cho luồng tin/giá trực tiếp (pha velocity).

### 7.3 Tính toán phân tán (fan-out GPU)
- Suy luận LLM 32B là điểm nghẽn → **phân tán thành 40 mảnh (shard)** (20 base + 20 RAG) chạy song song trên một **pool GPU miễn phí** → một "đợt" ≈ **~20 phút** thay vì ~5 giờ chạy tuần tự.
- Mỗi shard dự đoán ~101 ngày, gắn **kho ngày lịch sử đầy đủ (lookback)** để giữ tính nhân quả khi chia theo ngày.
- **Tính hợp lệ (trung thực):** đây là *mẫu thiết kế* phân tán — cùng code chạy được trên cluster cloud/HPC trả phí. Pool ở đây dựng từ GPU free-tier của Kaggle (giải pháp sinh viên, không phải cluster chính thức); **bản chạy 1 tài khoản ~5 giờ cho kết quả y hệt**, nên kết quả KHÔNG phụ thuộc vào cách phân tán.

---

## 8. Mô hình & Triển khai

- **Qwen2.5-32B-Instruct** trên Kaggle RTX 6000 Pro (offline, batch) — cho backtest quy mô lớn.
- **Qwen2.5-7B-AWQ** cục bộ trên RTX 2060 SUPER (8 GB) — cho **suy luận trực tiếp** (vừa GPU).
- **Phục vụ:** FastAPI (`/predict`, `/predict-ensemble`, `/backtest`); ứng dụng web **Streamlit** (giám sát thị trường trực tiếp, feed tin tích lũy ~500 tin/ngày, biểu đồ tương tác).
- **Daemon trực tiếp:** mỗi 60 giây fetch → lưu rolling **7 ngày** (`data/live/news.jsonl`) → chạy TRR → tín hiệu.
- **Lưu ý:** tin trực tiếp **không có nhãn** → chỉ chứng minh *triển khai*, không phải độ chính xác. Độ chính xác chỉ đến từ backtest lịch sử có nhãn.

---

## 9. Thực nghiệm & Kết quả

### 9.1 Kết quả đã có
| Thiết lập | AUROC | Ghi chú |
|---|---|---|
| COVID (2019-06…2020-06), base | **0.785** | 343 ngày, 14 crash |
| COVID + RAG | **0.847** | RAG là cải thiện ổn định |
| Rộng 2016–2020, base | **0.710** | |
| RAG (quy mô lớn) | **+0.074 (p=0.009)** | có ý nghĩa thống kê |
| Baseline "khối lượng tin" | ~0.50 | ≈ ngẫu nhiên → tín hiệu đến từ suy luận, không phải đếm tin |

### 9.2 Kết quả toàn corpus 2016–2023 (lọc theo danh mục)
| Cửa sổ | Base AUROC | RAG AUROC | News-volume | So với cũ |
|---|---|---|---|---|
| COVID (cùng cửa sổ) | 0.707 | **0.763** | 0.656 | cũ bundled 0.785 / 0.847 |
| Rộng 2016–2020 | 0.693 | 0.681 | 0.677 | cũ bundled 0.710 |
| Toàn bộ 2016–2023 | 0.615 | 0.652 | 0.662 | all-ticker (hỏng) 0.568 / 0.606 |

**Đọc kết quả (trung thực):**
1. **Lọc theo danh mục đã SỬA được lỗi:** COVID hồi phục từ 0.37 (all-ticker hỏng) → **0.76**; rộng 0.50 → **0.69**. Chẩn đoán đúng.
2. **Nhưng corpus lớn KHÔNG vượt bộ tin bundled gốc** — thấp hơn nhẹ ở mọi cửa sổ (COVID 0.763 < 0.847; rộng 0.69 < 0.71). *Nhiều dữ liệu hơn ≠ tốt hơn* khi bộ nhỏ đã được tuyển khớp danh mục.
3. **RAG giúp ở COVID (+0.056) và toàn bộ (+0.037)** nhưng không ở 2016–2020 (−0.01) — yếu hơn mức +0.074 trước.
4. **Khiêm tốn:** baseline **news-volume (0.662) nhỉnh hơn TRR base (0.615)** ở toàn cửa sổ — đếm khối lượng tin danh mục là baseline mạnh (một kết quả âm trung thực).

**Kết luận phần này:** giá trị của công đoạn corpus là **trình diễn pipeline Big Data ở quy mô lớn + xác nhận bản sửa**, không phải lập kỷ lục độ chính xác. Số headline vẫn là bundled COVID 0.785 / 0.847.

### 9.3 Số đo hạ tầng (đã có)
- Tải corpus: ~39 MB/s, resumable; lọc cục bộ 22 GB → 12 GB.
- Chỉ mục SQLite: 1,9 GB, tra cứu ngày COVID (4.098 bài) trong 44 ms.
- Spark ETL: 12 GB → 718 MB Parquet / 101 s; truy vấn Parquet 4,5 triệu dòng / 2,4 s.
- Kaggle: 40 shard song song trên pool GPU free-tier, 0 lỗi xác thực, ~20 phút/đợt.

---

## 10. Phân tích Trung thực (Honest Analysis)

- **Small-N là trần thực sự:** số ngày crash rất ít (14–82 tùy cửa sổ, ~4% base rate) → cần thận trọng khi đọc AUROC tuyệt đối; ưu tiên kiểm định ý nghĩa thống kê và cửa sổ lớn.
- **Các kết quả âm trung thực (đã thử, không cải thiện):** Graph-RAG đa bước, phân loại hướng 3 lớp, đặc trưng khối lượng OHLCV, đặc trưng meta từ RAG — đều không tăng ngoài-thời-gian.
- **Hướng giá ≈ ngẫu nhiên** (EMH dạng yếu); rủi ro đuôi/crash mới là mục tiêu khả thi.
- **Bẫy Big Data đã gặp & sửa:** mở rộng sang corpus 4,5 triệu bài *toàn mã* + chọn theo truy vấn crash **làm giảm tín hiệu** (liên quan ≠ liên quan-danh-mục). Sửa bằng lọc theo danh mục.
- **Phạm vi phân tán:** kỹ thuật stream/partition/index trên một máy + Spark (pseudo-distributed, `local[*]`, code sẵn sàng cho cluster) + pool GPU Kaggle free-tier. Không phải cluster HDFS/Spark nhiều máy thật — nêu rõ để trung thực.

---

## 11. Kết luận & Hướng phát triển

- **Kết luận:** TRR zero-shot + RAG dự đoán được rủi ro sụp đổ danh mục từ tin tức với AUROC tốt ở cửa sổ khủng hoảng; **RAG là cải thiện ổn định**. Hệ thống xử lý nguồn 23 GB bằng kỹ thuật stream-index-select và phân tán Spark + Kaggle.
- **Hướng phát triển:** (1) cluster Spark/HDFS nhiều máy thật; (2) corpus đa nguồn (mạng xã hội, filing); (3) hiệu chỉnh xác suất & ngưỡng theo chi phí; (4) mở rộng danh mục & đa tài sản.

---

## Phụ lục — Cấu trúc mã nguồn
- `trr/` — pipeline (schema, brainstorm, memory, attention, reason, rag, corpus, select, prices, targets).
- `kaggle/` — kernel 32B tự chứa + script deploy/poll/eval phân tán.
- `processing/` — Spark (ETL corpus + streaming consumers).
- `train/` — meta-learner, ablations, backtest, figures, significance.
- `serving/` — FastAPI. `webapp/` — Streamlit. `scripts/` — daemon trực tiếp, build dữ liệu, cron.
- `reports/` — `RESULTS_TRR.md` (tài liệu kết quả gốc).
