# traffic-sign-recognition
Traffic Sign Recognition project using CNN
## 🧠 Model Download
Do file model lớn (>300 MB) nên không thể upload trực tiếp lên GitHub.  
👉 Tải model tại đây: [Google Drive - best_model_traffic_sign.keras](https://drive.google.com/drive/folders/1VraV-SmLJeEnOwCI81crv2NBAu4gmb0U?usp=sharing)
vô bằng mail trường nhé 
best_model.keras là cái train của model cũ 
best_model_traffic_sign.keras là cái update 
demo_final.py code model final
## 🚀 Load model tự động trong Python
```python
import gdown
from tensorflow.keras.models import load_model

# Tải model từ Google Drive
file_id = "1YOUR_FILE_ID"  # 👈 thay phần ID trong link Drive của bạn
url = f"https://drive.google.com/uc?id={file_id}"
output = "best_model_traffic_sign.keras"

gdown.download(url, output, quiet=False)

# Load model
model = load_model(output)
print("✅ Model loaded successfully!")
