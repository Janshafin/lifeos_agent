# LifeOS Agent — Complete Submission Checklist

> **OpenEnv Hackathon 2026 — Final Submission Checklist**
> Every item must be verified before submission deadline.

---

## 🔲 GitHub Repository

- [ ] **Repo is public** — `https://github.com/Janshafin/lifeos_agent`
  - Verify: `curl -s https://api.github.com/repos/Janshafin/lifeos_agent | grep '"private"'` → should say `false`
- [ ] **All files committed and pushed:**
  - [ ] `models.py` — Pydantic data models
  - [ ] `client.py` — OpenEnv HTTP client
  - [ ] `app_ui.py` — Gradio UI
  - [ ] `openenv.yaml` — environment config
  - [ ] `requirements.txt` — dependencies
  - [ ] `Dockerfile` — production container
  - [ ] `README.md` — world-class documentation
  - [ ] `blog_post.md` — HuggingFace blog content
  - [ ] `__init__.py` — package init
  - [ ] `server/__init__.py` — server package init
  - [ ] `server/app.py` — FastAPI server
  - [ ] `server/lifeos_environment.py` — core RL environment
  - [ ] `notebooks/lifeos_training.py` — training notebook
  - Verify: `git status` → clean working tree
- [ ] **reward_curve.png committed** (after Colab training)
- [ ] **components_curve.png committed** (after Colab training)
- [ ] **README has real training numbers** (replace X.XX with Colab output)
- [ ] **No Unsloth references anywhere**
  - Verify: `grep -ri "unsloth" . --include="*.py" --include="*.md" --include="*.yaml" --include="*.txt"` → no results
- [ ] **License file exists** (BSD-3-Clause)
- [ ] **`.gitignore` covers:** `__pycache__/`, `.venv/`, `*.egg-info/`, `.DS_Store`

---

## 🤗 HuggingFace Space

- [ ] **Space created** — `https://huggingface.co/spaces/heyjan/lifeos-agent`
- [ ] **Space SDK set to Gradio**
- [ ] **Files uploaded to Space:**
  - [ ] `app_ui.py` → renamed to `app.py` in Space root
  - [ ] `requirements.txt`
  - Verify: Space builds and runs at the URL above
- [ ] **Space is running and accessible**
  - Verify: Open `https://huggingface.co/spaces/heyjan/lifeos-agent` in browser
- [ ] **All 9 scenarios load in dropdown**
- [ ] **Submit action returns reward scores**
- [ ] **Trained vs Untrained comparison works**
- [ ] **About panel has correct links**

---

## 📓 Google Colab Notebook

- [ ] **Notebook uploaded to Colab** or GitHub (linked from README)
  - Copy `notebooks/lifeos_training.py` content into a Colab notebook
  - Or use: File → Upload notebook → paste cells
- [ ] **Runtime set to T4 GPU**
- [ ] **Cell 1 runs** — all deps install without errors
- [ ] **Cell 2 runs** — W&B initializes, environment loads
- [ ] **Cell 3 runs** — model loads in 8-bit, LoRA applied
  - Should print: `trainable params: X / total params: Y`
- [ ] **Cell 4 runs** — baseline test prints reward table
  - **Record the baseline total reward number**
- [ ] **Cell 5 runs** — 60 training steps complete
  - Should take ~20-30 minutes on T4
  - Watch for OOM — if it happens, reduce `max_new_tokens` to 200
- [ ] **Cell 6 runs** — plots saved as `reward_curve.png` and `components_curve.png`
  - Download both images
- [ ] **Cell 7 runs** — before/after comparison table prints
  - **Record all numbers for README**
- [ ] **Cell 8 runs** — LoRA adapters saved to `lifeos_agent_lora/`
- [ ] **Colab notebook is shared** (anyone with link can view)
  - Share → Anyone with the link → Viewer

---

## 📝 HuggingFace Blog Post

- [ ] **Blog post created** at `https://huggingface.co/blog`
  - Go to: `https://huggingface.co/new-post` (or Settings → Blog)
  - Copy content from `blog_post.md`
- [ ] **Title:** "LifeOS Agent: Training AI to Handle Your Worst Day"
- [ ] **All links work** — Space, Colab, GitHub
- [ ] **Post is published** (not draft)

---

## 📊 Training Artifacts (After Colab)

- [ ] **Download from Colab:**
  - [ ] `reward_curve.png`
  - [ ] `components_curve.png`
  - [ ] `lifeos_agent_lora/` (optional — for model sharing)
- [ ] **Copy images to repo root:**
  ```bash
  cp ~/Downloads/reward_curve.png /Users/janshafin/Desktop/lifeos_agent/
  cp ~/Downloads/components_curve.png /Users/janshafin/Desktop/lifeos_agent/
  ```
- [ ] **Update README.md** with real numbers from Cell 7 output:
  - Replace all `X.XX` in the Training Results table
- [ ] **Commit and push updated files:**
  ```bash
  cd /Users/janshafin/Desktop/lifeos_agent
  git add reward_curve.png components_curve.png README.md
  git commit -m "Add real training results"
  git push
  ```

---

## 🎥 Demo Video (Optional but Recommended)

- [ ] **Record 2-minute screen recording showing:**
  1. Open the HuggingFace Space
  2. Select the hardest scenario (Total Travel Meltdown)
  3. Show the crisis description
  4. Type a bad response → show low reward
  5. Type a good response → show high reward
  6. Click "Compare Trained vs Untrained" → show the difference
  7. Quick flash of the training curves
- [ ] **Upload to YouTube** (unlisted is fine)
- [ ] **Add link to README.md** in the Links table

---

## 🏁 OpenEnv Submission

- [ ] **Verify openenv.yaml is valid:**
  ```bash
  cd /Users/janshafin/Desktop/lifeos_agent
  python -c "import yaml; print(yaml.safe_load(open('openenv.yaml'))['name'])"
  ```
  → Should print: `lifeos-agent`
- [ ] **Verify Docker server runs:**
  ```bash
  docker build -t lifeos-agent .
  docker run -p 8000:8000 lifeos-agent
  # In another terminal:
  curl http://localhost:8000/health
  ```
  → Should return health JSON
- [ ] **Verify reset/step endpoints:**
  ```bash
  curl -X POST http://localhost:8000/reset
  curl -X POST http://localhost:8000/step -H "Content-Type: application/json" \
    -d '{"action_type":"send_message","target_person":"Partner_Jamie","content":"My flight was cancelled. I have booked the 11pm bus arriving at 5:30am tomorrow. I will call you in 5 minutes.","reasoning":"Partner has been waiting 40 minutes and deserves an immediate honest update with a concrete plan","urgency":"immediate"}'
  ```
  → Should return observation with reward breakdown
- [ ] **Submit to OpenEnv hackathon** — follow submission instructions on openenv.dev

---

## ✅ Final Verification Checklist

| Item | URL/Command | Expected |
|---|---|---|
| GitHub repo public | `github.com/Janshafin/lifeos_agent` | Accessible, all files present |
| HF Space running | `huggingface.co/spaces/heyjan/lifeos-agent` | UI loads, scenarios work |
| Colab notebook | Shared link | All 8 cells run on T4 |
| Blog post | `huggingface.co/blog/...` | Published, links work |
| Docker server | `localhost:8000/health` | Returns health JSON |
| openenv.yaml | `cat openenv.yaml` | Valid YAML, all fields |
| Training curves | `reward_curve.png` | Real data, not placeholder |
| README numbers | Check table | Real numbers, not X.XX |
| No Unsloth | `grep -ri unsloth .` | Zero results |

---

**When everything above is checked: you're ready to submit. Good luck! 🏆**
