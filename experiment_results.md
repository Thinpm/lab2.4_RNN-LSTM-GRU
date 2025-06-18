# Phân tích chi tiết quá trình thực thi thử nghiệm

## 1. Tổng quan về run_experiments.py

`run_experiments.py` là một script tự động hóa quá trình thực thi và đánh giá 3 mô hình deep learning khác nhau (RNN, LSTM, và GRU) cho bài toán image captioning. Script này được thiết kế để chạy tất cả các thử nghiệm một cách tuần tự và thu thập kết quả.

## 2. Cấu trúc và chức năng

### 2.1. Các thành phần chính

```python
def run_experiment(script_name):
    # Hàm thực thi một thử nghiệm
    
def main():
    # Hàm chính điều phối các thử nghiệm
```

### 2.2. Quy trình hoạt động

1. **Khởi tạo danh sách thử nghiệm**:
   - LSTM: `image_captioning_improved.py`
   - RNN: `image_captioning_rnn.py`
   - GRU: `image_captioning_gru.py`

2. **Cho mỗi thử nghiệm**:
   - Ghi lại thời điểm bắt đầu
   - Thực thi script tương ứng
   - Theo dõi output và lỗi
   - Tính toán thời gian thực thi
   - Lưu kết quả

3. **Tổng hợp kết quả**:
   - Hiển thị thời gian chạy của từng mô hình
   - So sánh hiệu suất

## 3. Chi tiết quá trình thực thi

### 3.1. Chạy một thử nghiệm đơn lẻ

```python
def run_experiment(script_name):
    print(f"\nĐang chạy thử nghiệm với {script_name}...")
    start_time = time.time()
    process = subprocess.run(['python', script_name], 
                           capture_output=True, 
                           text=True)
    end_time = time.time()
```

- **Input**: Tên file script cần chạy
- **Quá trình**:
  1. Ghi lại thời điểm bắt đầu
  2. Sử dụng `subprocess.run()` để thực thi script
  3. Capture toàn bộ output và error
  4. Ghi lại thời điểm kết thúc
- **Output**: 
  - Thời gian thực thi
  - Kết quả stdout/stderr

### 3.2. Quản lý nhiều thử nghiệm

```python
def main():
    experiments = [
        'image_captioning_improved.py',
        'image_captioning_rnn.py',
        'image_captioning_gru.py'
    ]
    
    results = {}
    for script in experiments:
        runtime = run_experiment(script)
        results[script] = runtime
```

- **Quá trình**:
  1. Định nghĩa danh sách các thử nghiệm
  2. Tạo dictionary lưu kết quả
  3. Chạy tuần tự từng thử nghiệm
  4. Lưu thời gian thực thi

## 4. Ưu điểm của thiết kế

1. **Tự động hóa**:
   - Không cần can thiệp thủ công
   - Giảm thiểu lỗi người dùng
   - Tiết kiệm thời gian

2. **Theo dõi và ghi log**:
   - Capture đầy đủ output
   - Ghi nhận lỗi nếu có
   - Đo lường thời gian chính xác

3. **Dễ mở rộng**:
   - Thêm mô hình mới dễ dàng
   - Có thể thêm metrics đánh giá
   - Linh hoạt trong việc thay đổi cấu hình

## 5. Kết quả thực nghiệm

### 5.1. Thời gian thực thi
- LSTM (image_captioning_improved.py): ~274.35 giây
- RNN (image_captioning_rnn.py): ~238.91 giây
- GRU (image_captioning_gru.py): ~272.65 giây

### 5.2. Độ chính xác
- LSTM: 83.09% (validation accuracy)
- RNN: 82.92% (validation accuracy)
- GRU: 83.13% (validation accuracy)

## 6. Kết luận

Script `run_experiments.py` cung cấp một framework hiệu quả để:
- Tự động hóa quá trình thử nghiệm
- Thu thập và so sánh kết quả
- Đảm bảo tính nhất quán trong quá trình đánh giá
- Tiết kiệm thời gian và công sức trong việc thực hiện nhiều thử nghiệm

Việc sử dụng script này giúp đảm bảo tính khách quan và dễ dàng tái tạo lại các thử nghiệm khi cần thiết.

# Báo cáo tổng hợp kết quả thực nghiệm

## 1. Đánh giá kết quả thực hiện

### 1.1. Thời gian chạy
| Mô hình | Thời gian huấn luyện | Thời gian/epoch | Ghi chú       |
|---------|----------------------|-----------------|---------------|
| RNN     | 238.91 giây          | ~11.95 giây     | Nhanh nhất    |
| LSTM    | 274.35 giây          | ~13.72 giây     | Chậm nhất     |
| GRU     | 272.65 giây          | ~13.63 giây     | Gần bằng LSTM |

### 1.2. Độ chính xác
| Mô hình | Training Accuracy | Validation Accuracy | Loss | Ghi chú           |
|---------|-------------------|---------------------|------|-------------------|
| RNN     | 83.85%            | 82.92%              | 0.89 | Hội tụ nhanh nhất |
| LSTM    | 83.90%            | 83.09%              | 0.86 | Ổn định nhất      |
| GRU     | 83.77%            | 83.13%              | 0.86 | Accuracy cao nhất |

## 2. Đề xuất điều chỉnh tham số

### 2.1. Tham số có thể điều chỉnh
1. **Kiến trúc mô hình**:
   - Tăng số lượng hidden units (512, 1024)
   - Thêm các lớp Dropout (0.4-0.5)
   - Thử nghiệm với nhiều lớp (2-3 layers)
   - Thêm cơ chế Attention

2. **Tham số huấn luyện**:
   - Tăng batch size (128, 256) để cải thiện tốc độ
   - Điều chỉnh learning rate (0.0001-0.001)
   - Thay đổi optimizer (RMSprop, AdamW)
   - Tăng số epochs (30-50)

3. **Xử lý dữ liệu**:
   - Tăng kích thước từ điển (15000-20000 từ)
   - Tăng độ dài câu tối đa (60-80 từ)
   - Áp dụng thêm data augmentation
   - Cải thiện tiền xử lý văn bản

### 2.2. Đề xuất cụ thể cho từng mô hình

#### RNN
- Thêm BatchNormalization sau mỗi lớp
- Tăng dropout rate lên 0.4
- Sử dụng Bidirectional wrapper
- Thêm residual connections

#### LSTM
- Tăng số units lên 512
- Thêm lớp attention
- Giảm learning rate xuống 0.0001
- Sử dụng gradient clipping

#### GRU
- Thêm một lớp GRU thứ hai
- Tăng embedding dimension lên 300
- Sử dụng layer normalization
- Thêm skip connections

## 3. So sánh chi tiết giữa các mô hình

### 3.1. Hiệu suất tổng thể
| Tiêu chí          | RNN        | LSTM     | GRU        |
|-------------------|------------|----------|---------   |
| Độ chính xác      | Tốt        | Rất tốt  | Tốt nhất   |
| Tốc độ huấn luyện | Nhanh nhất | Chậm     | Trung bình |
| Khả năng hội tụ   | Nhanh      | Ổn định  | Ổn định    |
| Bộ nhớ sử dụng    | Thấp nhất  | Cao nhất | Trung bình |

### 3.2. Ưu điểm và nhược điểm

#### RNN
- **Ưu điểm**:
  - Đơn giản, dễ triển khai
  - Tốc độ huấn luyện nhanh
  - Ít tham số nhất
- **Nhược điểm**:
  - Độ chính xác thấp nhất
  - Dễ bị vanishing gradient
  - Khó học dependencies dài

#### LSTM
- **Ưu điểm**:
  - Học tốt dependencies dài
  - Ổn định trong quá trình học
  - Ít bị vanishing gradient
- **Nhược điểm**:
  - Chậm nhất trong huấn luyện
  - Nhiều tham số nhất
  - Tốn nhiều bộ nhớ

#### GRU
- **Ưu điểm**:
  - Độ chính xác cao nhất
  - Cân bằng giữa tốc độ và hiệu suất
  - Ít tham số hơn LSTM
- **Nhược điểm**:
  - Chậm hơn RNN
  - Khó tinh chỉnh hơn RNN
  - Có thể không tốt bằng LSTM với sequences rất dài

## 4. Tổng quan về dự án

### 4.1. Mô tả dự án
- **Mục tiêu**: Xây dựng và so sánh hiệu suất của các mô hình RNN, LSTM, và GRU trong bài toán image captioning bằng tiếng Việt
- **Phạm vi**: Thử nghiệm trên bộ dữ liệu Flickr8k với các caption được dịch sang tiếng Việt
- **Thời gian thực hiện**: ~2 tuần

### 4.2. Dữ liệu
- **Nguồn**: Flickr8k Dataset
- **Kích thước**: 8,000 ảnh, mỗi ảnh 5 caption
- **Phân chia**: 
  - Training: 6,000 ảnh
  - Validation: 1,000 ảnh
  - Test: 1,000 ảnh

### 4.3. Chương trình
- **Ngôn ngữ**: Python
- **Framework**: TensorFlow, Keras
- **Cấu trúc code**:
  - `image_captioning_rnn.py`: Mô hình RNN
  - `image_captioning_improved.py`: Mô hình LSTM
  - `image_captioning_gru.py`: Mô hình GRU
  - `run_experiments.py`: Script chạy thử nghiệm

### 4.4. Kết quả và đánh giá
1. **Kết quả tốt nhất**:
   - Mô hình: GRU
   - Accuracy: 83.13%
   - Loss: 0.86
   - Thời gian huấn luyện: 272.65 giây

2. **Đánh giá tổng thể**:
   - Cả ba mô hình đều cho kết quả tốt (>82% accuracy)
   - GRU cung cấp sự cân bằng tốt nhất giữa hiệu suất và tốc độ
   - Mô hình hiện tại đã cho kết quả khá tốt trên bộ dữ liệu

3. **Hướng phát triển**:
   - Tích hợp cơ chế attention
   - Thử nghiệm với các pretrained models
   - Tối ưu hóa hiệu suất cho tiếng Việt 
