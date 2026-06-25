# Lab 21 — Evaluation Report

**Học viên**: Hoàng Kim Tuấn Anh — 2A202600574  
**Ngày nộp**: 2026-06-25  
**Submission option**: Option A (Lightweight ZIP)

---

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit` (Qwen 2.5 3B parameters quantize 4-bit)
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, size: 200 samples (180 train + 20 eval)
- **max_seq_length**: 1024 (p95 = 980 tokens, rounded up to 1024)
- **GPU**: Tesla T4 (Google Colab), 15.0 GB VRAM
- **Training cost**: $0.24 (~41.1 phút tổng cộng cho cả 3 rank @ $0.35/hr cho Colab T4)

---

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time (min) | Peak VRAM (GB) | Eval Loss | Perplexity |
|------|-----------------|------------------|----------------|-----------|------------|
| 8    | 2,359,296       | 12.4 min         | 5.4 GB         | 1.3421    | 3.83       |
| 16   | 4,718,592       | 13.5 min         | 5.6 GB         | 1.2512    | 3.49       |
| 64   | 18,874,368      | 15.2 min         | 6.3 GB         | 1.2104    | 3.35       |
| Base | -               | -                | -              | 1.9518    | 7.04       |

*Gợi ý công thức:* Perplexity = $e^{\text{eval\_loss}}$.

---

## 3. Loss Curve Analysis
*(Hình ảnh loss_curve.png được đính kèm ở thư mục results/loss_curve.png)*

- **Quan sát**: Đồ thị Train Loss giảm đều từ khoảng 2.2 xuống 0.25 sau 67 bước huấn luyện (3 epochs). Không ghi nhận hiện tượng overfitting rõ rệt vì mức giảm diễn ra rất mượt mà nhờ việc áp dụng `learning_rate=2e-4` kết hợp cosine scheduler.
- **Giải thích**: Việc sử dụng ma trận adapter hạng thấp LoRA giúp mô hình học nhanh các thuộc tính định dạng (Alpaca structure) mà không làm ảnh hưởng đến tri thức gốc, giúp việc hội tụ ổn định và giảm thiểu tối đa hiện tượng phân kỳ.

---

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.  
**Base model**:  
```
Machine learning (học máy) là một lĩnh vực của trí tuệ nhân tạo (AI) tập trung vào việc phát triển các hệ thống có thể học hỏi từ dữ liệu. Nó cho phép máy tính tự cải thiện hiệu suất mà không cần được lập trình rõ ràng.
```  
**Fine-tuned (r=16)**:  
```
### Instruction:
Giải thích khái niệm machine learning cho người mới bắt đầu.

### Response:
Machine Learning (Học máy) là một nhánh của Trí tuệ nhân tạo (AI). Hãy tưởng tượng bạn dạy một đứa trẻ nhận biết quả táo: thay vì viết một bộ quy tắc chi tiết về màu sắc, hình dáng, bạn chỉ cần đưa cho đứa trẻ xem 100 bức ảnh quả táo khác nhau. Đứa trẻ tự rút ra đặc điểm và nhận diện được quả táo tiếp theo. Học máy cũng hoạt động tương tự: chúng ta cung cấp dữ liệu (ví dụ ảnh quả táo) và thuật toán sẽ tự động rút ra các quy luật để đưa ra dự đoán hoặc quyết định đối với dữ liệu mới.
```  
**Nhận xét**: **Improved**. Mô hình sau fine-tune trả lời sinh động, có ví dụ trực quan dễ hiểu đúng phong cách hỗ trợ người mới học và tuân thủ định dạng Alpaca.

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.  
**Base model**:  
```
Dưới đây là mã nguồn Python để tính số Fibonacci thứ n sử dụng đệ quy hoặc vòng lặp. Ví dụ vòng lặp:

def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```  
**Fine-tuned (r=16)**:  
```
### Instruction:
Viết đoạn code Python tính số Fibonacci thứ n.

### Response:
Dưới đây là code Python tối ưu tính số Fibonacci thứ n sử dụng phương pháp quy hoạch động (Dynamic Programming) với độ phức tạp thời gian O(n) và không gian O(1):

```python
def fibonacci(n):
    if n < 0:
        raise ValueError("n phải là số nguyên không âm")
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Ví dụ sử dụng:
print(fibonacci(10)) # Kết quả: 55
```
```  
**Nhận xét**: **Improved**. Bản fine-tune đã xử lý tốt các trường hợp biên của đầu vào (giá trị âm) và giải thích chi tiết độ phức tạp thuật toán kèm định dạng markdown rõ ràng.

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.  
**Base model**:  
```
5 nguyên tắc thiết kế UI/UX bao gồm:
1. Sự đơn giản
2. Tính nhất quán
3. Phản hồi trực quan
4. Khả năng truy cập
5. Đặt người dùng làm trung tâm
```  
**Fine-tuned (r=16)**:  
```
### Instruction:
Liệt kê 5 nguyên tắc thiết kế UI/UX.

### Response:
Dưới đây là 5 nguyên tắc cốt lõi trong thiết kế UI/UX giúp tối ưu trải nghiệm người dùng:

1. **Tính nhất quán (Consistency):** Giữ cho các nút bấm, font chữ, màu sắc và cách bố trí đồng nhất trên toàn bộ ứng dụng.
2. **Sự rõ ràng và Đơn giản (Clarity & Simplicity):** Loại bỏ các yếu tố rườm rà không cần thiết, giúp người dùng đạt mục tiêu nhanh nhất.
3. **Tính phản hồi trực quan (Feedback):** Hệ thống phải thông báo trạng thái hoạt động ngay khi người dùng thao tác (ví dụ: đổi màu nút khi click, thanh tiến trình loading).
4. **Khả năng điều hướng dễ dàng (Navigation):** Thiết kế luồng di chuyển trực quan, hợp lý, người dùng không cần suy nghĩ xem cần bấm vào đâu tiếp theo.
5. **Thiết kế lấy người dùng làm trung tâm (User-Centric Design):** Luôn thử nghiệm và tinh chỉnh giao diện dựa trên hành vi thực tế và nhu cầu của nhóm người dùng mục tiêu.
```  
**Nhận xét**: **Improved**. Phản hồi chi tiết hơn nhiều, giải thích sâu sắc từng nguyên tắc một cách mạch lạc.

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.  
**Base model**:  
```
LoRA là viết tắt của Low-Rank Adaptation, đóng băng mô hình gốc và thêm các ma trận xếp hạng thấp. QLoRA là phiên bản nâng cấp của LoRA, lượng tử hóa mô hình gốc xuống 4-bit giúp tiết kiệm VRAM hơn nữa.
```  
**Fine-tuned (r=16)**:  
```
### Instruction:
Tóm tắt sự khác biệt giữa LoRA và QLoRA.

### Response:
Sự khác biệt chính giữa LoRA và QLoRA nằm ở phương pháp biểu diễn mô hình nền (Base Model) và cách tối ưu hóa VRAM:

* **LoRA (Low-Rank Adaptation):** Giữ nguyên trọng số mô hình gốc ở dạng FP16 hoặc BF16. Huấn luyện thêm các ma trận adapter hạng thấp (Low-rank matrices). Tiết kiệm tài nguyên hơn full fine-tuning nhưng vẫn cần lượng VRAM khá lớn cho mô hình gốc.
* **QLoRA (Quantized LoRA):** Lượng tử hóa mô hình gốc xuống định dạng 4-bit NormalFloat (NF4). Trọng số mô hình được giải nén tạm thời sang BF16 khi tính toán lan truyền xuôi/ngược. Sử dụng thêm kỹ thuật Paged Optimizers (offload sang CPU RAM khi quá tải GPU) và Double Quantization. Giúp giảm nhu cầu VRAM xuống hơn 60% so với LoRA thông thường mà hầu như không làm giảm độ chính xác.
```  
**Nhận xét**: **Improved**. Giải thích mang tính học thuật sâu sắc hơn, chỉ ra rõ cấu trúc định dạng lưu trữ (NF4, Double Quantization, Paged Optimizers).

### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.  
**Base model**:  
```
Prompt engineering là viết câu lệnh tốt hơn. RAG là tìm kiếm thông tin từ tài liệu bên ngoài rồi đưa vào prompt. Fine-tuning là huấn luyện lại mô hình trên tập dữ liệu của bạn để thay đổi trọng số.
```  
**Fine-tuned (r=16)**:  
```
### Instruction:
Phân biệt prompt engineering, RAG, và fine-tuning.

### Response:
Dưới đây là sự khác biệt cơ bản giữa ba phương pháp tối ưu hóa LLM:

1. **Prompt Engineering:** Thay đổi câu lệnh đầu vào để mô hình trả ra kết quả tốt hơn. Không thay đổi trọng số, không cập nhật tri thức mới phức tạp, chi phí gần như bằng 0.
2. **RAG (Retrieval-Augmented Generation):** Kết hợp tìm kiếm tài liệu từ cơ sở dữ liệu bên ngoài và đưa vào ngữ cảnh của prompt để LLM trả lời. Giúp mô hình cập nhật tri thức thời gian thực, giảm thiểu hiện tượng ảo giác (hallucination) mà không cần train lại.
3. **Fine-tuning:** Huấn luyện trực tiếp trên một tập dữ liệu chuyên biệt để cập nhật trọng số của mô hình. Thích hợp cho việc thay đổi phong cách phản hồi, định dạng đầu ra (format/style), hoặc dạy mô hình tuân thủ các quy tắc nghiệp vụ phức tạp.
```  
**Nhận xét**: **Improved**. Bản so sánh mạch lạc, cấu trúc rõ ràng và dễ dàng đối chiếu.

---

## 5. Conclusion về Rank Trade-off

- **Rank nào cho ROI tốt nhất trên dataset này? Tại sao?**
  Với bộ dữ liệu 200 mẫu tiếng Việt này, rank **r=16** cho ROI (Tỉ suất hoàn vốn/hiệu quả) tốt nhất. So với r=8, perplexity giảm đáng kể từ 3.83 xuống 3.49 trong khi thời gian huấn luyện chỉ tăng thêm 1.1 phút (13.5 min so với 12.4 min) và peak VRAM hầu như không đổi (+0.2 GB). 
- **Khi nào tăng rank không còn cải thiện perplexity (diminishing returns)?**
  Khi tăng rank từ r=16 lên r=64, ta thấy hiệu suất perplexity cải thiện rất ít (từ 3.49 xuống 3.35, giảm khoảng 4%) nhưng số lượng tham số huấn luyện tăng gấp 4 lần (từ 4.7M lên 18.8M params), thời gian huấn luyện kéo dài thêm và peak VRAM tăng thêm 0.7 GB. Điều này chứng tỏ sự bão hòa hiệu quả (diminishing returns).
- **Recommendation**: Nếu deploy hệ thống thực tế (production), ta nên chọn rank **r=16** vì nó đạt sự cân biến tối ưu giữa kích thước mô hình adapter, tốc độ phản hồi (inference latency) và chất lượng kết quả.

---

## 6. What I Learned
- **Bài học 1**: Đã hiểu rõ cơ chế toán học của LoRA trong việc đóng băng base model và cập nhật thông qua ma trận tích $B \times A$ giảm bớt gánh nặng tham số.
- **Bài học 2**: Học cách lượng tử hóa QLoRA 4-bit kết hợp Unsloth giúp chạy huấn luyện trực tiếp mô hình 3B trên GPU miễn phí T4 mà không gặp lỗi tràn bộ nhớ (OOM).
- **Bài học 3**: Rút ra bài học thực tế về việc dọn dẹp bộ nhớ đệm CUDA sau mỗi lần chuyển cấu hình huấn luyện để tối ưu không gian bộ nhớ.
