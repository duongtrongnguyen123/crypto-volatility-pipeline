# KỊCH BẢN DEMO WEBSITE (Tiếng Việt) — ~90 giây

> Mở trước: **http://localhost:8501** (hoặc link công khai). Đảm bảo daemon đang chạy (tin + tóm tắt cập nhật sẵn). Nói ngắn, vừa nói vừa click.

---

## 0. Mở đầu (10s)
> "Đây là web demo của hệ thống: một **LLM đọc tin tài chính** và **cảnh báo rủi ro sụp đổ** danh mục. Có 3 tab: Trực tiếp, Nghiên cứu, và Cách hoạt động."

## 1. Tab 🔴 Live & Advisory (35s)
**Chỉ phần "Daily advisory" (trên cùng):**
> "Đây là **dự đoán chính thức**: mô hình 32B chạy mỗi ngày, đưa ra **xác suất sụp đổ trong 3 ngày tới** và mức rủi ro (HIGH/ELEVATED/LOW), kèm các cổ phiếu rủi ro nhất và lý do."

**Kéo xuống "Live market monitor":**
> "Phần này **giám sát + tóm tắt tin trực tiếp** bằng LLM 7B chạy nền — **mô tả** tình hình hiện tại, *không* phải dự đoán. Tin cập nhật từng phút, ưu tiên tin mới nhất."

**Kéo xuống "Live news feed":**
> "Feed tin trực tiếp, 50 tin mới nhất, bấm **'Show 50 more'** để xem thêm. Lọc theo công ty / vĩ mô / crypto / thế giới."

> 💡 Câu chốt (nếu được hỏi): "**Hiển thị ≠ Dự đoán** — feed hiện 4 nhóm tin, nhưng dự đoán chỉ dùng tin công ty + vĩ mô."

## 2. Tab 📊 Research & Backtest (25s)
> "Đây là nơi có **số liệu nghiêm túc**: backtest trên dữ liệu lịch sử **có nhãn**. AUROC dự đoán sụp đổ — cửa sổ COVID đạt **0.785, có RAG 0.847**. Kèm biểu đồ đường vốn, hiệu chỉnh xác suất."

> 💡 Câu chốt: "**Live = chứng minh triển khai** (tin chưa có nhãn); **Research = chứng minh độ chính xác** (dữ liệu có nhãn)."

## 3. Tab ℹ️ How it works (10s)
> "Giải thích 4 pha TRR: Brainstorm → Memory → Attention → Reason, và RAG. Mô hình **zero-shot**, không huấn luyện."

## 4. Kết (10s)
> "Tóm lại: dữ liệu lớn (corpus 4,5 triệu tin) được lọc bằng RAG, suy luận bằng LLM, triển khai trực tiếp + backtest có nhãn. Em xin demo đến đây."

---

## Câu hỏi hay gặp (trả lời nhanh)
- **"Sao có 2 xác suất khác nhau?"** → Daily advisory = 32B chạy mỗi ngày (số chính thức); Live monitor = tóm tắt nhanh, không phải dự đoán.
- **"Sao tin cập nhật từng phút mà dự đoán 3 ngày?"** → Feed sống cập nhật từng phút; *dự đoán 3 ngày* chỉ ở advisory hằng ngày — đúng chân trời.
- **"Tin trực tiếp chính xác bao nhiêu?"** → Không đo được (chưa có nhãn); độ chính xác đến từ backtest lịch sử (tab Research).
- **"Web có chậm không?"** → Không — model chạy nền ở daemon; web chỉ đọc file nên mở/restart tức thì.

---

## Checklist trước khi demo
```bash
# 1. Daemon (model nền) đang chạy?
pgrep -f "scripts.live_daemon" && echo OK
# 2. Web đang chạy?
curl -sI http://localhost:8501 | head -1        # mong đợi 200
# 3. Nếu cần bật lại:
nohup .venv/bin/python -m scripts.live_daemon --backend 7b --poll 60 > data/live/daemon.log 2>&1 &
nohup .venv/bin/streamlit run webapp/app.py > /tmp/streamlit.log 2>&1 &
```
