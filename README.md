
# Deploy to VPS

```bash
$ apt install -y libgl1 && \
  python3 -mvenv venv && \
  source venv/bin/activate && \
  pip install -r requirements.txt && \
  pip install -r local_requirements.txt
$ gunicorn -w 2 -b 0.0.0.0:7860 --timeout 600 app:app 
```
# Deploy to Beam

Download model required files to Volume:

```bash
python upload.py
```

Deploy to Beam:

```bash
beam deploy app.py:i2v
```
