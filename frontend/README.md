# GlowWise AI - React Frontend Dashboard 🧴✨

This is the premium React + TypeScript frontend dashboard for GlowWise AI. It connects to the FastAPI backend to visualize customer reviews sentiment classifications and unsupervised cluster insights.

---

## 🏃‍♂️ 1. Local Full-Stack Startup Instructions

To run this application as a full-stack system, follow these steps:

### Phase A: Launch FastAPI Backend
1. Open a new terminal in the project root.
2. Navigate to the backend directory and activate the virtual environment:
   ```bash
   cd backend
   .venv\Scripts\Activate.ps1   # On Windows (PowerShell)
   source .venv/bin/activate    # On Linux/macOS
   ```
3. Run the development API server:
   ```bash
   uvicorn app.main:app --reload
   ```
   The backend will start serving endpoints at `http://localhost:8000`.

### Phase B: Launch React Frontend
1. Open a second terminal in the project root.
2. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
3. Install package dependencies (if not already done):
   ```bash
   npm install
   ```
4. Run the Vite development server:
   ```bash
   npm run dev
   ```
   The dashboard interface will open in your browser at `http://localhost:5173`.

---

## 🔌 2. API & Environment Configuration

The React application queries the backend URL defined in the environment:
- **Variable**: `VITE_API_URL`
- **File**: `frontend/.env`
- **Default value**: `http://localhost:8000`

If the backend server is offline or fails to connect, the dashboard page will display warning banners and guide you to start the server, keeping health pill status states updated gracefully.

---

## 👥 3. Customer Cohorts fallbacks

If the unsupervised clustering reports are missing or the backend is offline, the React client automatically falls back to pre-annotated static profiles for the 5 skincare segments:
- **General Beauty Enthusiasts** (27.6% size, 84.4% satisfied)
- **Daily Skincare Users** (40.5% size, 76.9% satisfied)
- **Moisture & Texture Fans** (17.5% size, 87.9% satisfied)
- **Acne & Blemish Care** (9.9% size, 85.9% satisfied)
- **Lip Care Seekers** (4.5% size, 83.9% satisfied)
