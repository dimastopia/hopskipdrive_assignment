FROM python:3.12
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN mkdir -p ./data ./written_report