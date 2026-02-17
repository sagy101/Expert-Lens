VENV := backend/.venv
PY   := $(VENV)/bin/python3
PIP  := $(VENV)/bin/pip

.PHONY: install train dev backend frontend screenshot clean

$(VENV)/bin/activate:
	python3 -m venv $(VENV)

install: $(VENV)/bin/activate
	$(PIP) install -r backend/requirements.txt
	cd frontend && npm install

train:
	cd backend && ../$(PY) train.py

backend:
	cd backend && ../$(PY) -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload

frontend:
	cd frontend && npm run dev

dev:
	@echo "Starting backend and frontend..."
	@make backend & make frontend

screenshot:
	$(PY) scripts/screenshot.py

clean:
	rm -rf backend/pretrained backend/data frontend/node_modules frontend/dist
