# SLIDE THUYẾT TRÌNH (Tiếng Việt) — kèm đặc tả hình ảnh

## Dùng cho AI tạo slide. Mỗi slide gồm 4 phần:
**① Tiêu đề · ② Nội dung (gạch đầu dòng) · ③ 🎨 Hình ảnh (đặc tả/PROMPT) · ④ 🎤 Ghi chú thuyết trình**

> HƯỚNG DẪN CHUNG CHO AI TẠO SLIDE (đã chốt)
> - Bố cục 16:9, nền sáng (#ffffff / #f8fafc).
> - **Phông chữ: tiêu đề = Montserrat SemiBold (~34–40pt); nội dung = Inter Regular (~20–24pt).**
> - Màu nhấn: **xanh lá #16a34a**; cảnh báo/crash: **đỏ #dc2626**; trung tính: xám #475569; nền chart: #f1f5f9.
> - Mỗi slide tối đa 5–6 gạch đầu dòng; ưu tiên 1 hình lớn bên phải, chữ bên trái.
> - **BIỂU ĐỒ: vẽ chart THẬT bằng đúng số liệu cho sẵn** (không placeholder), ghi nhãn giá trị trên cột/điểm.
> - **SƠ ĐỒ: vẽ SẠCH bằng khối bo góc + đường nối đơn sắc, KHÔNG dùng emoji trong slide render.**
>   Dùng icon line đơn sắc hoặc nhãn chữ thay cho emoji. (Các emoji trong phần 🎨 bên dưới CHỈ là gợi ý loại icon, đừng vẽ emoji vào slide.)
> - Chỗ 「điền sau」 = kết quả đang chạy → **giữ placeholder rõ ràng** (ô nền vàng nhạt + chữ "đang chạy"), đừng bịa số.
> - Đơn vị số theo chuẩn VN (dấu phẩy thập phân) như trong nội dung.
>
> ⏱️ LỘ TRÌNH ~10 PHÚT (12 slide lõi · 6 slide lướt/dự phòng)
> - **Lõi (trình kỹ, ~45–60s/slide):** 1 (bìa, 20s) · 2 (vấn đề) · 3 (bài toán) · 5 (4 pha) · 6 (RAG) · 8 (4 chữ V) · 9 (lưu trữ) · 10 (Spark) · 11 (Kaggle) · 14 (kết quả) · 16 (trung thực) · 17 (kết luận).
> - **Lướt nhanh (~15–20s, hoặc bỏ nếu thiếu giờ):** 4 (ý tưởng — gộp vào slide 5) · 7 (dữ liệu — gộp vào slide 8) · 12 (triển khai) · 13 (hiển thị≠dự đoán) · 15 (số 「điền sau」 — chỉ mở nếu đã có số) · 18 (Q&A).
> - Tổng ước tính: 12 × ~48s + 6 × ~18s ≈ **~10 phút**. Nếu dư giờ, mở thêm slide lướt; nếu thiếu, bỏ 4/7/15.

---

### SLIDE 1 — Trang bìa
**② Nội dung**
- **Tiêu đề lớn:** Dự đoán Sụp đổ Giá Cổ phiếu bằng Suy luận Quan hệ–Thời gian của LLM
- **Phụ đề:** Đồ án Big Data · phỏng theo arXiv:2410.17266
- Nhóm: [TÊN NHÓM] · Thành viên: [TÊN] · GVHD: [TÊN GV] · [HỌC KỲ/NĂM]

**🎨 Hình ảnh — PROMPT vẽ ảnh nền:**
> "Ảnh bìa tối giản, nền trắng-xám gradient nhẹ. Bên phải: biểu đồ nến chứng khoán màu đỏ đang lao dốc mạnh, chồng mờ lên là một mạng nơ-ron/đồ thị node phát sáng màu xanh lá (tượng trưng LLM đọc tin). Phong cách phẳng (flat), hiện đại, chuyên nghiệp, không chữ trong ảnh."

**🎤 Ghi chú:** "Hệ thống để LLM ĐỌC TIN TỨC và CẢNH BÁO sụp đổ danh mục — một bài toán Big Data thực tế." (30 giây)

---

### SLIDE 2 — Vấn đề & Động lực
**② Nội dung**
- Thị trường = dữ liệu **khối lượng lớn, tốc độ cao, đa dạng** → đúng bài toán Big Data.
- Dự đoán **hướng giá** ≈ bất khả thi (thị trường hiệu quả dạng yếu).
- Nhưng **rủi ro sụp đổ (crash)** mang tín hiệu rõ từ TIN TỨC (tâm lý, sự kiện, lan truyền).
- **Use case:** cảnh báo sớm rủi ro — advisory cho nhà đầu tư / quản trị rủi ro (không phải lệnh mua–bán).

**🎨 Hình ảnh — sơ đồ so sánh 2 cột (vẽ bằng khối hộp + icon):**
> Hai thẻ cạnh nhau. Thẻ TRÁI viền đỏ, tiêu đề "Hướng giá (lên/xuống)", icon xúc xắc 🎲, nhãn "≈ Ngẫu nhiên (EMH)". Thẻ PHẢI viền xanh lá, tiêu đề "Rủi ro sụp đổ (tail-risk)", icon khiên/cảnh báo 🛡️, nhãn "✅ Có tín hiệu từ tin tức". Mũi tên lớn từ phải chỉ tới chữ "Mục tiêu của đồ án".

**🎤 Ghi chú:** Nhấn: "Ta KHÔNG đoán giá lên/xuống — ta đoán XÁC SUẤT SỤP ĐỔ."

---

### SLIDE 3 — Bài toán (định nghĩa chính xác)
**② Nội dung**
- **Đầu vào:** dòng tin tài chính theo ngày cho 6 cổ phiếu: AAPL, AMZN, GOOGL, NVDA, TSLA, NFLX.
- **Đầu ra:** P(danh mục equal-weight giảm ≥ 6% trong 3 ngày tới) cho mỗi ngày.
- **Nhãn:** từ giá lịch sử OHLCV (ngày crash nếu low 3 ngày tới ≤ −6%).
- **Đánh giá:** AUROC / PR-AUC. **Nhân quả:** chỉ dùng quá khứ (embargo ≥ chân trời 3 ngày).

**🎨 Hình ảnh — sơ đồ luồng ngang (flowchart, 4 khối + mũi tên):**
> `[📰 Tin ngày t]` → `[🤖 Mô hình TRR]` → `[📊 P(crash 3 ngày tới)]` → `[⚖️ So với giá thực tế → nhãn]`. Khối đầu màu xám, khối mô hình màu xanh lá nổi bật, khối xác suất có thanh gauge nhỏ, khối nhãn màu đỏ. Đường kẻ mảnh, bo góc.

**🎤 Ghi chú:** Giải thích ngưỡng −6% và chân trời 3 ngày là quy ước "crash danh mục".

---

### SLIDE 4 — Ý tưởng cốt lõi: LLM SUY LUẬN, không huấn luyện
**② Nội dung**
- Mô hình **zero-shot**: KHÔNG huấn luyện lại — nó **đọc và lập luận**.
- Phỏng theo khung TRR (Temporal Relational Reasoning) của bài báo gốc.
- Chỉ một **meta-learner nhỏ** học trên đặc trưng dẫn xuất; **giá** cung cấp nhãn.

**🎨 Hình ảnh — đối lập 2 biểu tượng:**
> Bên trái: bánh răng ⚙️ bị gạch chéo đỏ + chữ "KHÔNG train mô hình dự báo". Bên phải: bóng đèn 💡 + đồ thị suy luận, chữ "LLM SUY LUẬN zero-shot". Ở giữa dấu "≠" lớn. Phong cách icon phẳng.

**🎤 Ghi chú:** "Khác biệt lớn nhất với ML truyền thống: không có pha train mô hình dự báo."

---

### SLIDE 5 — Phương pháp TRR: 4 Pha
**② Nội dung**
- **1. Brainstorm:** tin → đồ thị tác động ("X tác động ±Y", có trọng số).
- **2. Memory:** trí nhớ phân rã `R = exp(−t·λ)` — tin xấu nhạt dần theo thời gian.
- **3. Attention:** PageRank cắt tỉa xuống top-k cạnh gần danh mục nhất.
- **4. Reason:** LLM suy luận trên đồ thị con → xác suất sụp đổ.

**🎨 Hình ảnh — sơ đồ pipeline 4 khối nối tiếp + vòng lặp bộ nhớ:**
> 4 hộp bo góc xếp ngang, mỗi hộp 1 màu pastel + icon: Brainstorm (🧠 đồ thị node), Memory (⏳ đường cong phân rã exp), Attention (🔍 đồ thị bị tỉa bớt), Reason (🤖 → gauge %). Mũi tên thẳng nối 4 hộp. Thêm **một mũi tên cong** từ Memory vòng lại đầu vào ngày kế tiếp, nhãn "mang sang ngày sau". Dưới Memory vẽ mini-chart đường cong giảm dần (trục x = ngày, y = độ liên quan).

**🎤 Ghi chú:** Nhấn chữ "Temporal" = bộ nhớ mang tác động xấu sang ngày sau rồi phân rã.

---

### SLIDE 6 — RAG: Truy hồi Tăng cường
**② Nội dung**
- **Vai trò 1 — Chọn lọc:** từ bể tin lớn mỗi ngày, lấy k tin liên quan-danh-mục nhất → LLM chỉ đọc phần giới hạn.
- **Vai trò 2 — Few-shot theo tình huống:** truy hồi các NGÀY QUÁ KHỨ tương tự + kết cục thật ("từng crash/không") chèn vào prompt.
- Nhờ vậy corpus to cỡ GB nhưng **chi phí LLM vẫn O(số_ngày × k)**. Hoàn toàn nhân quả.

**🎨 Hình ảnh — 2 sơ đồ nhỏ cạnh nhau:**
> TRÁI ("Chọn lọc"): hình phễu lớn "Bể tin 4,5 triệu bài/ngày" → lọc → "20 tin/ngày" → icon 🤖 LLM. TexT trên phễu: "RAG select".
> PHẢI ("Few-shot lịch sử"): một dòng thời gian ngang; ô "HÔM NAY" màu xanh; 2–3 ô quá khứ được nối bằng nét đứt tới hôm nay, gắn nhãn đỏ "đã CRASH" / xám "không". Chú thích: "Hôm nay giống ngày crash nào trong quá khứ?"

**🎤 Ghi chú:** Bài học sẽ nói ở slide 16: "liên quan ≠ liên quan-danh-mục".

---

### SLIDE 7 — Dữ liệu
**② Nội dung**
- **FNSPID:** 23 GB thô, **15,7 triệu bài**, 4.775 mã, 1999–2023.
- **Corpus đã lọc 2016–2023:** **4.500.216 bài / 12 GB**.
- **Giá OHLCV:** 6 mã, 2.012 ngày giao dịch → sinh nhãn crash.
- **Tin trực tiếp:** ~500 tin/ngày (yfinance + Google News RSS).

**🎨 Hình ảnh — BIỂU ĐỒ CỘT (bar chart) số bài theo năm:**
> Bar chart dọc, màu xanh lá. Trục X = năm; Trục Y = số bài (triệu/nghìn). Dữ liệu thật (số bài corpus): 2016=817.956; 2017=515.523; 2018=698.590; 2019=700.151; 2020=351.832; 2021=181.623; 2022=280.354; 2023=954.117. Tiêu đề chart: "Số bài tin trong corpus 2016–2023 (tổng 4,5 triệu)". Ghi nhãn giá trị trên mỗi cột.

**🎤 Ghi chú:** Nhấn quy mô: 15,7 triệu bài nguồn → 4,5 triệu bài sau lọc.

---

### SLIDE 8 — Bốn chữ V của Big Data
**② Nội dung**
- **Volume:** 23 GB / 15,7 triệu bài → 12 GB / 4,5 triệu bài.
- **Velocity:** luồng trực tiếp ~500 tin/ngày, cập nhật 60 giây, Spark Streaming.
- **Variety:** tin công ty, vĩ mô, crypto, thế giới; giá; nhiều nguồn.
- **Value:** advisory cảnh báo sụp đổ + web app triển khai thật.

**🎨 Hình ảnh — lưới 2×2 thẻ (4 ô V), mỗi ô 1 icon + số:**
> Ô1 Volume: icon kho/đĩa 💾, "12 GB · 4,5 triệu bài". Ô2 Velocity: icon đồng hồ tốc độ ⏱️, "~500 tin/ngày · 60 s". Ô3 Variety: icon nhiều luồng 🔀, "Công ty · Vĩ mô · Crypto · Thế giới · Giá". Ô4 Value: icon khiên/gauge 🛡️, "Cảnh báo crash + Web app". Viền mỗi ô màu xanh lá, chữ V in đậm góc trên.

**🎤 Ghi chú:** Đây là khung chuẩn để map đồ án vào ngôn ngữ Big Data.

---

### SLIDE 9 — Kiến trúc Lưu trữ: "Lưu khổng lồ, phục vụ tí hon"
**② Nội dung**
- **Lạnh:** corpus 12 GB (đĩa). **Ấm:** SQLite chỉ mục theo ngày 1,9 GB (tra cứu 1 ngày ~44 ms). **Nóng:** lát RAG ~2 MB (tải lên Kaggle).
- Kỹ thuật: stream-and-filter (không lưu 23 GB thô), đọc theo khối (RAM-bounded), chiếu cột, chỉ mục phân vùng, chọn lọc RAG.
- Dữ liệu dẫn xuất **không commit git** (chỉ commit code tái tạo).

**🎨 Hình ảnh — sơ đồ PHỄU 3 tầng (funnel dọc):**
> Phễu 3 tầng từ trên xuống, thu nhỏ dần: Tầng 1 (rộng nhất, xám) "Corpus 12 GB · 4,5 triệu bài"; Tầng 2 (vừa, xanh dương) "SQLite chỉ mục ngày · 1,9 GB · 44 ms/ngày"; Tầng 3 (nhỏ nhất, xanh lá) "Lát RAG ~2 MB → LLM". Bên cạnh mỗi tầng ghi nhãn nhiệt độ: 🧊 Lạnh / 🌡️ Ấm / 🔥 Nóng. Mũi tên giảm dần kích thước.

**🎤 Ghi chú:** Đây là câu trả lời trực tiếp cho "lưu dữ liệu khổng lồ thế nào".

---

### SLIDE 10 — Xử lý Phân tán: Apache Spark
**② Nội dung**
- **Spark ETL:** corpus 12 GB → **Parquet phân vùng theo năm** (bố cục data-lake kiểu HDFS).
- Đo thật: 12 GB CSV → **718 MB Parquet / 101 giây**.
- Truy vấn từ Parquet: **4,5 triệu dòng / 2,4 giây** (~40× nhanh, song song 8 lõi); **partition pruning** quét `year=2020` trong 0,2 giây.
- Cùng code chạy cluster thật: chỉ đổi `SPARK_MASTER=spark://...`.

**🎨 Hình ảnh — (a) sơ đồ Spark + (b) bar chart so sánh tốc độ:**
> (a) Sơ đồ: 1 file "CSV 12 GB" → khối "Spark (8 executors song song)" (vẽ 8 ô worker nhỏ) → nhóm thư mục Parquet nhãn `year=2016 … year=2023` (cột hóa). 
> (b) Bar chart ngang 2 cột so sánh thời gian đọc: "Đọc CSV (1 lõi) = 101 s" (cột đỏ dài) vs "Đọc Parquet (8 lõi) = 2,4 s" (cột xanh ngắn). Tiêu đề: "~40× nhanh hơn khi dữ liệu splittable".

**🎤 Ghi chú:** Trung thực: đây là Spark local[*] (pseudo-distributed) nhưng CÙNG code chạy cluster nhiều máy.

---

### SLIDE 11 — Tính toán Phân tán: 20 GPU Kaggle
**② Nội dung**
- Điểm nghẽn = suy luận LLM 32B → **fan-out 40 shard** (20 base + 20 RAG).
- **20 tài khoản Kaggle × 2 notebook** = 40 khe GPU → chạy trong **1 đợt**.
- Mỗi shard ~101 ngày + **ngân hàng lookback toàn lịch sử** (giữ nhân quả khi chia shard theo ngày).
- **~20 phút/đợt** thay vì ~5 giờ chạy đơn; 0 lỗi xác thực.

**🎨 Hình ảnh — (a) lưới GPU + (b) thanh thời gian:**
> (a) Lưới 20 ô tài khoản, mỗi ô có 2 chip GPU nhỏ (tổng 40), tô màu xanh lá "đang chạy". Nhãn "20 tài khoản × 2 notebook = 40 shard".
> (b) Hai thanh ngang so sánh wall-clock: "Chạy đơn 1 notebook ≈ 5 giờ" (thanh đỏ rất dài) vs "Phân tán 40 shard ≈ 20 phút" (thanh xanh ngắn). Nhấn mạnh tỉ lệ rút gọn ~15×.

**🎤 Ghi chú:** Vì điểm nghẽn là LLM (không phải join dữ liệu) nên ta phân tán SUY LUẬN, không phải phân tán file.

---

### SLIDE 12 — Mô hình & Triển khai
**② Nội dung**
- **Qwen2.5-32B** (Kaggle RTX 6000 Pro, offline) — backtest quy mô lớn.
- **Qwen2.5-7B-AWQ** (cục bộ RTX 2060 SUPER 8 GB) — suy luận trực tiếp.
- **FastAPI** (`/predict`, `/predict-ensemble`, `/backtest`) + **Web app Streamlit** (giám sát trực tiếp, feed ~500 tin/ngày, biểu đồ tương tác).
- **Daemon trực tiếp** mỗi 60 giây, lưu rolling 7 ngày.

**🎨 Hình ảnh — ẢNH CHỤP MÀN HÌNH web app (chèn ảnh thật):**
> [CHÈN ẢNH MÀN HÌNH web app: gauge xác suất crash + feed tin tích lũy]. Nếu chưa có ảnh, vẽ mockup: khung trình duyệt, bên trái 1 gauge bán nguyệt kim chỉ % (màu chuyển xanh→đỏ), bên phải danh sách thẻ tin có icon 🏢/🌐, nhãn "LIVE" nhấp nháy đỏ góc trên.

**🎤 Ghi chú:** "Heavy 32B = offline chất lượng cao; 7B cục bộ = triển khai trực tiếp."

---

### SLIDE 13 — Hiển thị ≠ Dự đoán (điểm dễ bị hỏi)
**② Nội dung**
- Feed HIỂN THỊ 4 nhóm: 🏢 công ty · 🌐 vĩ mô · ₿ crypto · 🌍 thế giới.
- **Dự đoán CHỈ dùng tin công ty + vĩ mô** (đúng phân phối huấn luyện); crypto/thế giới bị **loại trước khi suy luận**.
- Tin trực tiếp **không có nhãn** → chứng minh *triển khai*, không phải độ chính xác.

**🎨 Hình ảnh — sơ đồ "phễu lọc 4→2":**
> Bên trái 4 icon nhóm tin xếp dọc (🏢🌐₿🌍). Mũi tên qua một "bộ lọc" (hình phễu/màng lọc). Bên phải chỉ còn 2 icon (🏢🌐) đi vào khối 🤖 "Dự đoán". 2 icon ₿🌍 rơi ra ngoài, mờ đi, nhãn "chỉ để hiển thị". 

**🎤 Ghi chú:** Câu hay bị hỏi: "Sao web hiện crypto mà bảo không dùng?" → vì hiển thị ≠ dự đoán.

---

### SLIDE 14 — Kết quả (1): Số đã có
**② Nội dung**
- COVID (2019-06…2020-06): **AUROC 0.785** → **+RAG 0.847**.
- Rộng 2016–2020: **0.710**; RAG **+0.074 (p = 0.009)** — có ý nghĩa thống kê.
- Baseline "khối lượng tin" ≈ **0.50** → tín hiệu đến từ **suy luận**, không phải đếm tin.

**🎨 Hình ảnh — BIỂU ĐỒ CỘT NHÓM (grouped bar) AUROC:**
> Grouped bar chart. Trục Y = AUROC (0.4–0.9). Hai nhóm trên trục X: "COVID" và "Rộng 2016–2020". Mỗi nhóm 2 cột: "Base" (xám) và "+RAG" (xanh lá). Giá trị: COVID base 0.785 / +RAG 0.847; Rộng base 0.710 / +RAG (≈0.784, hoặc để 「điền sau」). Vẽ **đường ngang đứt nét tại 0.50** nhãn "Ngẫu nhiên". Ghi nhãn giá trị trên cột.

**🎤 Ghi chú:** RAG là "chiến thắng ổn định" xuyên các cửa sổ; baseline đếm tin ≈ ngẫu nhiên.

---

### SLIDE 15 — Kết quả (2): Mở rộng Toàn bộ Corpus 2016–2023
**② Nội dung**
- So sánh trên **CÙNG cửa sổ** (corpus mới lọc-danh-mục vs số cũ):
  - COVID: base **0.707** / RAG **0.763** (cũ bundled 0.785 / 0.847)
  - Rộng 2016–2020: base **0.693** / RAG **0.681** (cũ bundled 0.710)
  - Toàn bộ 2016–2023: base **0.615** / RAG **0.652** (news-volume 0.662)
- Kết luận: lọc theo danh mục **hồi phục mạnh** (COVID từ 0.37 → 0.76) nhưng corpus lớn **không vượt** bộ bundled gốc → *nhiều dữ liệu ≠ tốt hơn*.

**🎨 Hình ảnh — GROUPED BAR CHART (3 nhóm × 2 cột):**
> Grouped bar, trục Y = AUROC (0.4–0.9). 3 nhóm trục X: "COVID", "2016–2020", "Toàn bộ 2016–2023". Mỗi nhóm 2 cột: Base (xám) và RAG (xanh lá). Giá trị: COVID 0.707/0.763; 2016–2020 0.693/0.681; Toàn bộ 0.615/0.652. Vẽ đường ngang đứt nét tại **0.50** (ngẫu nhiên) và một dấu ★ nhỏ ghi "cũ bundled COVID 0.847" để so sánh. Ghi nhãn giá trị trên mỗi cột.

**🎤 Ghi chú:** Nhấn 2 ý trung thực: (1) đã sửa được lỗi chọn lọc (0.37→0.76); (2) corpus lớn không lập kỷ lục — giá trị là trình diễn quy mô Big Data + xác nhận bản sửa.

---

### SLIDE 16 — Phân tích Trung thực
**② Nội dung**
- **Small-N là trần:** chỉ 14–82 ngày crash (~4%) → đọc AUROC tuyệt đối thận trọng.
- **Kết quả âm trung thực:** Graph-RAG đa bước, hướng 3 lớp, đặc trưng khối lượng OHLCV, meta-RAG — đều không tăng.
- **Bẫy Big Data đã gặp:** corpus toàn-mã + chọn theo truy vấn crash **làm giảm tín hiệu** (liên quan ≠ liên quan-danh-mục) → đã sửa bằng lọc danh mục.
- **Phạm vi phân tán:** Spark pseudo-distributed + 20 GPU Kaggle (không phải cluster HDFS nhiều máy) — nêu rõ, code sẵn cho cluster.

**🎨 Hình ảnh — bảng "✓/✗" trung thực:**
> Bảng 2 cột: "Điều đã hoạt động ✅" (RAG ổn định, stream-index-select, Spark+Kaggle) và "Giới hạn / kết quả âm ❌" (small-N, Graph-RAG, hướng 3 lớp, OHLCV volume, bẫy chọn lọc toàn-mã). Dùng icon ✓ xanh / ✗ đỏ. Tông trung tính, nghiêm túc.

**🎤 Ghi chú:** Sự trung thực là điểm cộng — cho thấy hiểu rõ giới hạn, không cherry-pick.

---

### SLIDE 17 — Kết luận & Hướng phát triển
**② Nội dung**
- **Kết luận:** TRR zero-shot + RAG dự đoán rủi ro sụp đổ từ tin tức (AUROC tốt ở cửa sổ khủng hoảng); **RAG cải thiện ổn định**; xử lý nguồn 23 GB bằng stream-index-select + Spark + 20 GPU Kaggle.
- **Hướng phát triển:** cluster Spark/HDFS nhiều máy thật; corpus đa nguồn (mạng xã hội, filing); hiệu chỉnh xác suất & ngưỡng theo chi phí; mở rộng đa tài sản.

**🎨 Hình ảnh — sơ đồ tổng kết pipeline end-to-end (1 dải ngang):**
> Một dải pipeline ngang gộp tất cả: `FNSPID 23GB → Stream/Index (SQLite) → RAG select → [Brainstorm→Memory→Attention→Reason] → 40 shard Kaggle GPU → AUROC` và nhánh xuống "Web app / FastAPI (live)". Tô màu xanh lá cho LLM, xanh dương cho dữ liệu, cam cho phân tán. Đây là slide "1 hình thấy toàn bộ".

**🎤 Ghi chú:** Chốt 3 đóng góp: (1) áp dụng TRR cho cổ phiếu, (2) RAG ổn định, (3) hạ tầng Big Data thật.

---

### SLIDE 18 — Q&A / Cảm ơn
**② Nội dung**
- **Cảm ơn thầy/cô và các bạn đã lắng nghe!**
- Mã nguồn: https://github.com/duongtrongnguyen123/bigdata-stock-crash-trr

**🎨 Hình ảnh:**
> Nền tối giản, chữ "Q&A" lớn ở giữa, logo nhóm + QR code trỏ tới repo GitHub ở góc. Thu nhỏ sơ đồ pipeline (slide 17) làm watermark mờ phía sau.

**🎤 Ghi chú — câu hỏi hay gặp & trả lời nhanh:**
- "HDFS/Spark cluster đâu?" → có Spark (pseudo-distributed, code sẵn cho cluster) + 20 GPU Kaggle; điểm nghẽn là LLM nên ưu tiên phân tán suy luận.
- "Sao AUROC không cao tuyệt đối?" → small-N + dự đoán rủi ro đuôi xuyên nhiều năm yên tĩnh là khó; chỉ số trung thực, không cherry-pick.
- "Tin trực tiếp chính xác bao nhiêu?" → không nhãn, chỉ chứng minh triển khai; độ chính xác từ backtest lịch sử có nhãn.
- "Vì sao −6% / 3 ngày?" → quy ước crash danh mục equal-weight; đủ hiếm để là rủi ro đuôi, đủ thường để đánh giá thống kê.
