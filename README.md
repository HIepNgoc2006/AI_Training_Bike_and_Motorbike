# AI - Bike & Motorbike Classification

Dự án phân loại hình ảnh **Xe Đạp (Bike)** và **Xe Máy (Motorbike)** sử dụng Deep Learning.

## Cấu trúc luồng làm việc

Toàn bộ luồng làm việc được thiết kế tối giản, tập trung vào 2 quy trình chính:

### 1. Huấn luyện mô hình (Training)
- **File thực thi:** `train.ipynb`
- **Cách dùng:** Mở file và Run All các cell.
- Quá trình này sẽ tự động lo mọi việc từ tải dataset, xử lý dữ liệu cho tới huấn luyện mô hình. Mọi kết quả (biểu đồ, trọng số mô hình) sẽ được xuất ngay tại thư mục hiện tại.

### 2. Dự đoán thực tế (Inference)
- **File thực thi:** `predict.ipynb`
- **Cách dùng:** Mở file này trên Jupyter hoặc Colab. Nó được chia thành các cell để bạn dễ dàng theo dõi từng bước: tải mô hình, xử lý ảnh và hiển thị kết quả một cách trực quan ngay trên màn hình.

## Yêu cầu môi trường

Đảm bảo bạn đã cài đặt các thư viện cơ bản dành cho Computer Vision và PyTorch:
```bash
pip install torch torchvision numpy pandas matplotlib seaborn kagglehub pillow scikit-learn
```

---
*💡 Dự án được thiết kế để dễ dàng tùy biến. Bạn có thể thoải mái nâng cấp kiến trúc mạng, điều chỉnh logic hay tham số bên trong mã nguồn mà không cần quan tâm đến việc cập nhật lại tài liệu này.*