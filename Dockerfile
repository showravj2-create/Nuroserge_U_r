FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["python", "scripts/check_brats_dataset.py", "--data-root", "/data/BraTS2020"]
