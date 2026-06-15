# AI - Bike & Motorbike Classification (TensorFlow/Keras Version)

Dự án phân loại hình ảnh **Xe Đạp (Bike)** và **Xe Máy (Motorbike)** sử dụng Deep Learning với thư viện TensorFlow/Keras.

## Cấu trúc luồng làm việc

Toàn bộ luồng làm việc được thiết kế tối giản, minh bạch và dễ theo dõi:

### 1. Huấn luyện mô hình (Training)
- **File thực thi:** `train_keras.ipynb`
- **Cách dùng:** Mở file trên Jupyter hoặc Google Colab và Run All các cell.
- Quá trình này sẽ tự động tải dataset từ Kaggle, tổ chức lại thư mục chuẩn của Keras, thiết kế cấu trúc mạng ResNet50 kèm theo các lớp Custom. Đặc biệt, nó sẽ in ra **Bảng Tóm Tắt Mô Hình (Model Summary)** để bạn dễ dàng theo dõi số lượng tham số (Trainable/Non-trainable) trước và sau khi Fine-tuning.

### 2. Dự đoán thực tế (Inference)
- **File thực thi:** `predict_keras.ipynb`
- **Cách dùng:** Mở file này trên Jupyter hoặc Colab. Nó giúp bạn tải lại file trọng số (`best_keras_model.keras`) và dự đoán thử trên một ảnh bất kỳ (ảnh của bạn tải lên hoặc tự động lấy trên mạng).

## Yêu cầu môi trường

Đảm bảo bạn đã cài đặt các thư viện cơ bản dành cho Computer Vision và TensorFlow:
```bash
pip install tensorflow numpy pandas matplotlib kagglehub pillow scikit-learn
```

---
*💡 Dự án được thiết kế để dễ dàng tùy biến. Bạn có thể thoải mái nâng cấp kiến trúc mạng (sửa các lớp Dense, Dropout), điều chỉnh logic hay tham số bên trong mã nguồn một cách trực quan nhờ bảng Model Summary.*