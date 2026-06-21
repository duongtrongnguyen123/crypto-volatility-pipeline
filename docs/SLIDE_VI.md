# SLIDE THUYẾT TRÌNH (Tiếng Việt)
## Dùng cho AI tạo slide — mỗi slide có: Tiêu đề · Nội dung (gạch đầu dòng) · Gợi ý hình ảnh · Ghi chú thuyết trình

> HƯỚNG DẪN CHO AI TẠO SLIDE: Giữ nguyên thứ tự. Mỗi slide tối đa 5–6 gạch đầu dòng, chữ to, ưu tiên biểu đồ/bảng/icon. Tông màu: xanh lá (#16a34a) làm nhấn, nền sáng. Các chỗ 「điền sau」 để trống/placeholder vì kết quả đang chạy.

---

### SLIDE 1 — Trang bìa
- **Tiêu đề:** Dự đoán Sụp đổ Giá Cổ phiếu bằng Suy luận Quan hệ Thời gian của LLM
- **Phụ đề:** Đồ án Big Data — phỏng theo arXiv:2410.17266
- **Thông tin:** Nhóm: [TÊN NHÓM] · Thành viên: [TÊN] · GVHD: [TÊN GV] · [HỌC KỲ/NĂM]
- *Hình ảnh:* biểu tượng biểu đồ nến đỏ lao dốc + icon não/AI; nền tối giản.
- *Ghi chú:* Mở đầu: "Chúng em xây hệ thống để LLM ĐỌC TIN và CẢNH BÁO sụp đổ danh mục — một bài toán Big Data thực tế."

---

### SLIDE 2 — Vấn đề & Động lực
- Thị trường = dữ liệu **khối lượng lớn, tốc độ cao, đa dạng** → bài toán Big Data.
- Dự đoán **hướng giá** ≈ bất khả thi (thị trường hiệu quả dạng yếu).
- Nhưng **rủi ro sụp đổ (crash)** mang tín hiệu từ TIN TỨC (tâm lý, sự kiện, lan truyền).
- **Use case:** cảnh báo sớm rủi ro cho nhà đầu tư/quản trị rủi ro (advisory, không phải lệnh mua bán).
- *Hình ảnh:* 2 mũi tên: "Hướng giá ❌ ngẫu nhiên" vs "Rủi ro đuôi ✅ khả thi".
- *Ghi chú:* Nhấn: ta không đoán giá lên/xuống, ta đoán XÁC SUẤT SỤP ĐỔ.

---

### SLIDE 3 — Bài toán (định nghĩa chính xác)
- **Đầu vào:** dòng tin tài chính theo ngày cho 6 cổ phiếu: AAPL, AMZN, GOOGL, NVDA, TSLA, NFLX.
- **Đầu ra:** P(danh mục giảm ≥ 6% trong 3 ngày tới) cho mỗi ngày.
- **Đánh giá:** AUROC / PR-AUC trên backtest có nhãn (giá lịch sử = nhãn).
- **Nhân quả:** chỉ dùng dữ liệu quá khứ (embargo ≥ chân trời nhãn) — không rò rỉ tương lai.
- *Hình ảnh:* sơ đồ: [Tin ngày t] → [Mô hình] → [Xác suất crash 3 ngày tới] → so với [Giá thực tế].

---

### SLIDE 4 — Ý tưởng cốt lõi: LLM SUY LUẬN, không huấn luyện
- Mô hình **zero-shot**: không huấn luyện lại — nó **đọc và lập luận**.
- Phỏng theo bài báo TRR (Temporal Relational Reasoning).
- Chỉ một **meta-learner nhỏ** học trên đặc trưng dẫn xuất; **giá** cung cấp nhãn.
- *Hình ảnh:* icon "no training" (gạch chéo bánh răng) + icon "reasoning" (bóng đèn).
- *Ghi chú:* Khác biệt lớn nhất so với ML truyền thống: KHÔNG train mô hình dự báo.

---

### SLIDE 5 — Phương pháp TRR: 4 Pha
- **1. Brainstorm:** tin → đồ thị tác động ("X tác động ±Y").
- **2. Memory:** trí nhớ phân rã theo thời gian (tin xấu nhạt dần theo hàm mũ).
- **3. Attention:** PageRank cắt tỉa xuống top-k cạnh gần danh mục.
- **4. Reason:** LLM suy luận → xác suất sụp đổ.
- *Hình ảnh:* sơ đồ 4 khối nối tiếp Brainstorm→Memory→Attention→Reason, có mũi tên vòng "bộ nhớ mang sang ngày sau".
- *Ghi chú:* "Temporal" = bộ nhớ mang tác động xấu sang các ngày sau rồi phân rã.

---

### SLIDE 6 — RAG: Truy hồi Tăng cường
- **Vai trò 1 — Chọn lọc:** từ bể tin lớn mỗi ngày, lấy k tin liên quan nhất → LLM chỉ đọc phần giới hạn.
- **Vai trò 2 — Few-shot theo tình huống:** truy hồi các NGÀY QUÁ KHỨ tương tự + kết cục thật ("từng crash/không") chèn vào prompt.
- Nhờ vậy corpus to cỡ GB nhưng **chi phí LLM vẫn O(số_ngày × k)**.
- Hoàn toàn nhân quả (chỉ nhìn quá khứ).
- *Hình ảnh:* sơ đồ "Bể tin 4,5 triệu → RAG chọn 20/ngày → LLM"; bên cạnh "Ngày hôm nay ↔ các ngày crash tương tự trong quá khứ".

---

### SLIDE 7 — Dữ liệu
- **FNSPID:** 23 GB thô, **15,7 triệu bài**, 4.775 mã, 1999–2023.
- **Corpus đã lọc 2016–2023:** **4.500.216 bài / 12 GB**.
- **Giá OHLCV:** 6 mã, 2.012 ngày → sinh nhãn crash.
- **Tin trực tiếp:** ~500 tin/ngày (yfinance + Google News RSS).
- *Hình ảnh:* bảng nguồn dữ liệu + biểu đồ cột số bài theo năm (2016–2023).
- *Ghi chú:* Nhấn quy mô: 15,7 triệu bài nguồn, 4,5 triệu bài sau lọc.

---

### SLIDE 8 — Bốn chữ V của Big Data
- **Volume:** 23 GB / 15,7 triệu bài → 12 GB / 4,5 triệu bài.
- **Velocity:** luồng trực tiếp ~500 tin/ngày, cập nhật 60 giây, Spark Streaming.
- **Variety:** tin công ty, vĩ mô, crypto, thế giới; giá; nhiều nguồn.
- **Value:** advisory cảnh báo sụp đổ + web app triển khai thật.
- *Hình ảnh:* 4 ô icon V (Volume/Velocity/Variety/Value) với số liệu.

---

### SLIDE 9 — Kiến trúc Lưu trữ: "Lưu khổng lồ, phục vụ tí hon"
- **Lạnh:** corpus 12 GB (đĩa). **Ấm:** SQLite chỉ mục theo ngày 1,9 GB (tra cứu ~44 ms). **Nóng:** lát RAG ~2 MB (lên Kaggle).
- Kỹ thuật: stream-and-filter (không lưu 23 GB thô), đọc theo khối (RAM-bounded), chiếu cột, chỉ mục phân vùng, chọn lọc RAG.
- Dữ liệu dẫn xuất **không commit git** (chỉ commit code tái tạo).
- *Hình ảnh:* phễu 3 tầng: 12 GB → 1,9 GB → 2 MB.
- *Ghi chú:* Đây là câu trả lời cho "lưu dữ liệu khổng lồ thế nào".

---

### SLIDE 10 — Xử lý Phân tán: Apache Spark
- **Spark ETL:** corpus 12 GB → **Parquet phân vùng theo năm** (bố cục data-lake kiểu HDFS).
- Đo thật: 12 GB CSV → **718 MB Parquet / 101 giây**.
- Truy vấn từ Parquet: **4,5 triệu dòng / 2,4 giây** (~40× nhanh, song song 8 lõi); **partition pruning** quét `year=2020` trong 0,2 giây.
- Cùng code chạy cluster thật: đổi `SPARK_MASTER=spark://...`.
- *Hình ảnh:* sơ đồ Spark: CSV → executors song song → thư mục Parquet `year=2016…2023`.

---

### SLIDE 11 — Tính toán Phân tán: 20 GPU Kaggle
- Điểm nghẽn = suy luận LLM 32B → **fan-out 40 shard** (20 base + 20 RAG).
- **20 tài khoản Kaggle × 2 notebook** = 40 khe GPU → chạy 1 đợt.
- Mỗi shard ~101 ngày + **ngân hàng lookback toàn lịch sử** (giữ nhân quả khi chia shard).
- **~20 phút/đợt** thay vì ~5 giờ chạy đơn; 0 lỗi xác thực.
- *Hình ảnh:* lưới 20 tài khoản × 2 GPU; thanh thời gian "đơn 5h vs phân tán 20 phút".

---

### SLIDE 12 — Mô hình & Triển khai
- **Qwen2.5-32B** (Kaggle RTX 6000 Pro, offline) — backtest quy mô lớn.
- **Qwen2.5-7B-AWQ** (cục bộ RTX 2060 SUPER 8 GB) — suy luận trực tiếp.
- **FastAPI** (`/predict`, `/predict-ensemble`, `/backtest`) + **Web app Streamlit** (giám sát trực tiếp, feed ~500 tin/ngày, biểu đồ tương tác).
- **Daemon trực tiếp** mỗi 60 giây, lưu rolling 7 ngày.
- *Hình ảnh:* ảnh chụp web app (gauge xác suất crash + feed tin) — [CHÈN ẢNH MÀN HÌNH].

---

### SLIDE 13 — Hiển thị ≠ Dự đoán (điểm dễ bị hỏi)
- Feed hiển thị 4 nhóm: 🏢 công ty, 🌐 vĩ mô, ₿ crypto, 🌍 thế giới.
- **Dự đoán CHỈ dùng tin công ty + vĩ mô** (đúng phân phối huấn luyện); crypto/thế giới bị loại trước khi suy luận.
- Tin trực tiếp **không có nhãn** → chứng minh *triển khai*, không phải độ chính xác.
- *Hình ảnh:* 2 cột: "Hiển thị (4 nhóm)" vs "Dự đoán (2 nhóm)".

---

### SLIDE 14 — Kết quả (1): Số đã có
- COVID (2019-06…2020-06): **AUROC 0.785** → **+RAG 0.847**.
- Rộng 2016–2020: **0.710**; RAG **+0.074 (p=0.009)** — có ý nghĩa thống kê.
- Baseline "khối lượng tin" ≈ **0.50** → tín hiệu đến từ **suy luận**, không phải đếm tin.
- *Hình ảnh:* biểu đồ cột AUROC (base vs +RAG) cho COVID & 2016–2020; đường 0.5 = ngẫu nhiên.
- *Ghi chú:* RAG là "chiến thắng ổn định" xuyên các cửa sổ.

---

### SLIDE 15 — Kết quả (2): Toàn bộ Corpus 2016–2023 「điền sau」
- Bảng so sánh trên CÙNG cửa sổ (data mới lọc-danh-mục vs cũ):
  - COVID: base 「điền sau」 / RAG 「điền sau」 (cũ 0.785 / 0.847)
  - Rộng 2016–2020: 「điền sau」 (cũ 0.710)
  - Toàn bộ 2016–2023: base 「điền sau」 / RAG 「điền sau」
- *Hình ảnh:* bảng/biểu đồ so sánh — để trống chờ số.
- *Ghi chú:* Đây là kiểm thử "corpus lớn có giúp không khi chọn lọc ĐÚNG".

---

### SLIDE 16 — Phân tích Trung thực
- **Small-N là trần:** chỉ 14–82 ngày crash (~4%) → đọc AUROC tuyệt đối thận trọng.
- **Kết quả âm trung thực:** Graph-RAG đa bước, hướng 3 lớp, đặc trưng khối lượng OHLCV, meta-RAG — đều không tăng.
- **Bẫy Big Data:** corpus toàn-mã + chọn theo truy vấn crash **làm giảm tín hiệu** (liên quan ≠ liên quan-danh-mục) → đã sửa bằng lọc danh mục.
- **Phạm vi phân tán:** Spark pseudo-distributed + 20 GPU Kaggle (không phải cluster HDFS nhiều máy) — nêu rõ, code sẵn sàng cho cluster.
- *Hình ảnh:* icon "✓ trung thực" + danh sách bài học.

---

### SLIDE 17 — Kết luận & Hướng phát triển
- **Kết luận:** TRR zero-shot + RAG dự đoán rủi ro sụp đổ từ tin tức (AUROC tốt ở cửa sổ khủng hoảng); **RAG cải thiện ổn định**; xử lý 23 GB bằng stream-index-select + Spark + Kaggle.
- **Hướng phát triển:** cluster Spark/HDFS thật; corpus đa nguồn (mạng xã hội, filing); hiệu chỉnh xác suất theo chi phí; mở rộng đa tài sản.
- *Hình ảnh:* sơ đồ tổng kết pipeline end-to-end.

---

### SLIDE 18 — Q&A / Cảm ơn
- **Cảm ơn thầy/cô và các bạn đã lắng nghe!**
- Liên hệ / mã nguồn: https://github.com/duongtrongnguyen123/bigdata-stock-crash-trr
- *Hình ảnh:* logo nhóm + ảnh kiến trúc thu nhỏ.
- *Ghi chú — câu hỏi hay gặp & trả lời nhanh:*
  - "HDFS/Spark cluster đâu?" → có Spark (pseudo-distributed, code sẵn cho cluster) + 20 GPU Kaggle; điểm nghẽn là LLM nên ưu tiên phân tán suy luận.
  - "Sao AUROC không cao tuyệt đối?" → small-N + dự đoán rủi ro đuôi xuyên nhiều năm yên tĩnh là khó; chỉ số trung thực, không cherry-pick.
  - "Tin trực tiếp chính xác bao nhiêu?" → không nhãn, chỉ chứng minh triển khai; độ chính xác từ backtest lịch sử.
