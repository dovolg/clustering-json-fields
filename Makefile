IMAGE := clustering-app:latest
ENV_FILE := $(wildcard .env)

run_local:
	streamlit run app.py

build:
	docker build --tag $(IMAGE) .

# --env-file is only passed if .env actually exists -- it's optional (only
# needed for Gemini cluster labeling), so its absence shouldn't break `make run`.
run: build
	docker run --rm -p 8501:8501 $(if $(ENV_FILE),--env-file .env,) $(IMAGE)

clean:
	docker rmi -f $(IMAGE)
