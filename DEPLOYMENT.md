# 🚀 Deployment Guide - Streamlit Sentiment Analysis App

## 📋 Prerequisites

- Python 3.8+
- pip atau conda
- Git

## 🛠️ Local Development

### 1. Clone Repository
```bash
git clone <repository-url>
cd AnalyticsSocialMedia-Sentiment-Streamlit
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Or using conda
conda create -n sentiment-app python=3.9
conda activate sentiment-app
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Application
```bash
streamlit run app.py
```

Aplikasi akan berjalan di `http://localhost:8501`

## 🌐 Cloud Deployment Options

### Option 1: Streamlit Community Cloud (Recommended)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

### Option 2: Heroku

1. **Create Procfile**
   ```bash
   echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile
   ```

2. **Create runtime.txt**
   ```bash
   echo "python-3.9.18" > runtime.txt
   ```

3. **Deploy to Heroku**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 3: Docker

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   EXPOSE 8501

   CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
   ```

2. **Build and Run**
   ```bash
   docker build -t sentiment-app .
   docker run -p 8501:8501 sentiment-app
   ```

### Option 4: Railway

1. **Connect GitHub Repository**
   - Go to [railway.app](https://railway.app)
   - Connect your GitHub account
   - Select your repository

2. **Configure Environment**
   - Railway will auto-detect Python
   - Set start command: `streamlit run app.py --server.port $PORT`

## ⚙️ Configuration

### Environment Variables
```bash
# Optional: Set Hugging Face cache directory
export HF_HOME=/path/to/cache

# Optional: Set model cache
export TRANSFORMERS_CACHE=/path/to/transformers/cache
```

### Streamlit Configuration
File `.streamlit/config.toml` sudah dikonfigurasi untuk production:
- Headless mode enabled
- CORS disabled
- Usage stats disabled
- Custom theme applied

## 📊 Performance Optimization

### Model Caching
Model akan di-cache otomatis setelah pertama kali di-download. Untuk production:

1. **Pre-download model** di build time
2. **Use model caching** untuk menghindari re-download
3. **Consider model quantization** untuk mengurangi memory usage

### Memory Management
- Model RoBERTa membutuhkan ~500MB RAM
- Dataset loading membutuhkan ~100MB RAM
- Total recommended: 1GB+ RAM

## 🔧 Troubleshooting

### Common Issues

1. **Model Download Timeout**
   ```bash
   # Increase timeout
   export HF_HUB_DOWNLOAD_TIMEOUT=300
   ```

2. **Memory Issues**
   ```bash
   # Use CPU-only model
   export CUDA_VISIBLE_DEVICES=""
   ```

3. **Port Already in Use**
   ```bash
   streamlit run app.py --server.port 8502
   ```

### Logs
```bash
# View Streamlit logs
streamlit run app.py --logger.level debug
```

## 📈 Monitoring

### Health Check
Aplikasi menyediakan endpoint health check di `/health` (jika dikonfigurasi)

### Metrics
- Response time untuk prediksi sentimen
- Model loading time
- Memory usage

## 🔒 Security

### Production Checklist
- [ ] Disable debug mode
- [ ] Set secure headers
- [ ] Configure CORS properly
- [ ] Use HTTPS
- [ ] Set up monitoring
- [ ] Regular dependency updates

## 📝 Notes

- Model pertama kali di-download saat startup (bisa memakan waktu 1-2 menit)
- Aplikasi menggunakan model `cardiffnlp/twitter-roberta-base-sentiment`
- Dataset TweetEval di-load secara real-time dari Hugging Face
- Forest fire keyword detection berjalan di client-side

## 🆘 Support

Jika mengalami masalah:
1. Check logs untuk error messages
2. Verify semua dependencies terinstall
3. Pastikan Python version compatible
4. Check network connectivity untuk model download
