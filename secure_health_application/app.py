#from flask import Flask
#app = Flask(__name__)

#@app.route('/')
#def home():
#    return "✅ Flask is working perfectly! Welcome to Secure Health Application."

#if __name__ == '__main__':
#    app.run(debug=True)
# from flask import Flask, render_template

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')
# # I'm adding a very simple login page route so the navbar link works
# @app.route('/login')
# def login():
#     # I'm rendering a template called login.html (I'll create it next)
#     return render_template('login.html')

# if __name__ == '__main__':
#     app.run(debug=True)
# # I'm building the Flask app and adding simple, secure login logic
# from flask import Flask, render_template, request, redirect, url_for, flash, session
# from werkzeug.security import generate_password_hash, check_password_hash
# import os

# # I'm creating the Flask application instance
# app = Flask(__name__)

# # I'm setting a secret key so sessions and flash() work securely
# # (Later, for production, this MUST come from an environment variable)
# app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me")

# # I'm defining a tiny in-memory "database" of users for now
# # (Later, we’ll replace this with a real data store)
# USERS = {
#     "demo": generate_password_hash("Password123!")
# }

# @app.route("/")
# def home():
#     # I'm rendering the homepage (extends base.html)
#     return render_template("index.html")

# @app.route("/login", methods=["GET", "POST"])
# def login():
#     # I show the form on GET; on POST I validate credentials
#     if request.method == "POST":
#         username = request.form.get("username", "").strip()
#         password = request.form.get("password", "")

#         # I'm doing minimal input validation first
#         if not username or not password:
#             flash("Please enter both username and password.", "error")
#             return render_template("login.html", username=username)

#         # I'm checking the stored hashed password securely
#         if username in USERS and check_password_hash(USERS[username], password):
#             session["user"] = username
#             flash(f"Welcome back, {username}!", "success")
#             return redirect(url_for("dashboard"))
#         else:
#             flash("Invalid username or password.", "error")
#             return render_template("login.html", username=username), 401

#     # If it's a GET request, I just show the form
#     return render_template("login.html")

# @app.route("/dashboard")
# def dashboard():
#     # I'm protecting this page: if you’re not logged in, I send you to /login
#     if "user" not in session:
#         flash("Please log in to access the dashboard.", "warning")
#         return redirect(url_for("login"))
#     return render_template("dashboard.html", user=session["user"])

# @app.route("/logout")
# def logout():
#     # I'm clearing the session to log the user out
#     session.clear()
#     flash("You have been logged out.", "success")
#     return redirect(url_for("home"))

# if __name__ == "__main__":
#     app.run(debug=True)


# now replacing my app.py a more secure version (an AI assisted code)
# app.py
# building the Flask app with CSRF protection, session hardening,
# basic auth flow, security headers, and a tiny rate-limit for login.


# I'm building the main Flask app for the Secure Health Application.
# This file handles login, analytics dashboard, admin features, and database setup.

# --- Imports (I'm pulling in everything I need up front) ---
from datetime import timedelta, datetime
from collections import defaultdict, deque, Counter
from statistics import mean
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, flash, session, abort
from werkzeug.security import check_password_hash, generate_password_hash

# I'm importing CSRF protection to secure forms from cross-site attacks.
from flask_wtf import CSRFProtect

# I'm importing my login form from forms.py.
from forms import LoginForm

# I'm importing tools to load and run my AI model.
import joblib
import numpy as np

# --- App setup (I'm creating my Flask app) ---
app = Flask(__name__)

# --- Core security config (I'm keeping secrets out of code in real life) ---
# I'm setting a secret key so Flask can encrypt sessions and CSRF tokens.
app.config["SECRET_KEY"] = "change-me-in-real-project"
app.config["WTF_CSRF_ENABLED"] = True  # I'm enabling CSRF protection.

# --- Session security hardening ---
# I'm setting cookies to be secure and reducing session lifetime for safety.
app.config["SESSION_COOKIE_SECURE"] = False  # Should be True when using HTTPS.
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.permanent_session_lifetime = timedelta(minutes=30)

# I'm enabling CSRF protection globally.
csrf = CSRFProtect(app)

# --- Fake user store (temporary fallback for demo) ---
# I'm using hashed passwords here (never plain text!).
USERS = {
    "student": generate_password_hash("password123")
}

# --- Database setup section (SQL for users; Mongo for patients) ---
from flask_sqlalchemy import SQLAlchemy
from pymongo import MongoClient

# I'm setting up SQLite for user authentication.
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# I'm creating a simple SQLAlchemy model to store login users securely.
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

# I'm connecting to MongoDB for patient stroke records (non-SQL data).
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["secure_health_db"]
patients_collection = mongo_db["patients"]

# --- AI model loading (I'm loading my saved stroke risk model) ---
try:
    # I'm loading the trained Logistic Regression model from disk.
    stroke_model = joblib.load("stroke_model.pkl")
    print("✅ Stroke model loaded successfully.")
except Exception as e:
    # If anything goes wrong, I'm falling back safely with no model.
    print("⚠ Could not load stroke_model.pkl:", e)
    stroke_model = None

# --- Tiny login rate limiter to stop brute-force attempts ---
FAIL_WINDOW = 60  # I'm checking login attempts within this time frame (seconds).
FAIL_LIMIT = 5    # I'm allowing up to 5 failed attempts before blocking temporarily.
fail_log = defaultdict(lambda: deque())  # I'm using a deque to track timestamps.

def too_many_failures(ip: str) -> bool:
    # I'm checking if this IP has made too many failed login attempts recently.
    now = datetime.utcnow().timestamp()
    dq = fail_log[ip]
    while dq and now - dq[0] > FAIL_WINDOW:
        dq.popleft()
    return len(dq) >= FAIL_LIMIT

def note_failure(ip: str):
    # I'm logging a failed login attempt.
    now = datetime.utcnow().timestamp()
    dq = fail_log[ip]
    dq.append(now)

# --- Security headers for every response ---
@app.after_request
def add_security_headers(resp):
    # I'm adding HTTP headers to improve browser security.
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # I'm allowing the Chart.js CDN + safe defaults (matches what your charts need).
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'"
    )
    return resp

# --- Helper function to enforce login ---
def require_login():
    if not session.get("user"):
        abort(401)

# --- Tiny role helpers (I'm not changing your DB schema; just inferring by name) ---
def is_admin():
    # I'm treating the username 'admin' as the admin account
    return session.get("user") == "admin"

def is_patient():
    # I'm treating usernames like 'patient-1234' as patient accounts
    u = session.get("user") or ""
    return u.startswith("patient-")

def patient_id_from_username():
    # I'm pulling the numeric id out of 'patient-<id>'
    try:
        return int((session.get("user") or "").split("-", 1)[1])
    except Exception:
        return None
    
def build_stroke_risk_message(doc):
    """
    I'm taking ONE patient record from MongoDB and asking the AI model
    for a stroke-risk prediction, then turning it into a human-friendly message.
    """
    # If the model failed to load, I quietly return nothing.
    if stroke_model is None:
        return None

    try:
        # I'm pulling the same features I used during training.
        age = float(doc.get("age"))
        htn = int(doc.get("hypertension"))
        hd = int(doc.get("heart_disease"))
        glu = float(doc.get("avg_glucose_level"))
        bmi = doc.get("bmi")

        # If BMI is missing, I can't safely predict.
        if bmi in (None, "", "N/A"):
            return None
        bmi = float(bmi)
    except (TypeError, ValueError):
        # If any value is weird or missing, I skip AI for this patient.
        return None

    # I'm building a tiny 2D array for the model: [[age, htn, hd, glu, bmi]]
    X = np.array([[age, htn, hd, glu, bmi]])

    # I'm getting the probability that stroke = 1 from the model.
    proba = stroke_model.predict_proba(X)[0][1]
    confidence = round(proba * 100)

    # I'm mapping the probability into LOW / MEDIUM / HIGH risk.
    if proba < 0.33:
        level = "LOW"
    elif proba < 0.66:
        level = "MEDIUM"
    else:
        level = "HIGH"

    # This is the final message the admin will see in the UI later.
    msg = (
        f"AI Stroke Risk: {level} ({confidence}% confidence) – "
        f"Doctor review required before informing patient."
    )

    return msg

# --- Routes start here ---
@app.route("/")
def home():
    # I'm showing the home page (index.html template).
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    # I'm using my WTForms login form for validation.
    form = LoginForm()
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"

    # I'm blocking login if this IP has too many recent failures.
    if request.method == "POST" and too_many_failures(client_ip):
        flash("Too many failed attempts. Please wait a minute and try again.", "error")
        return render_template("login.html", form=form), 429

    # If form is valid, I'm checking credentials.
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        # secure session handling, I'm checking credentials from the SQLite database first.
        u = User.query.filter_by(username=username).first()
        if u and check_password_hash(u.password_hash, password):
            session.clear()
            session.permanent = True
            session["user"] = username
            flash("You are now signed in.", "success")
            return redirect(url_for("dashboard"))

        # If not in SQLite, I’m checking the fallback USERS dict.
        stored_hash = USERS.get(username)
        if stored_hash and check_password_hash(stored_hash, password):
            session.clear()
            session.permanent = True
            session["user"] = username
            flash("You are now signed in.", "success")
            return redirect(url_for("dashboard"))

        # If both fail, I log and show an error.
        note_failure(client_ip)
        flash("Invalid username or password.", "error")
        return render_template("login.html", form=form), 401

    # If GET request or invalid POST, I show the login form again.
    return render_template("login.html", form=form)

# --- Dashboard + analytics + patient-only branch ---
def _age_band(a):
    # I'm grouping ages into simple buckets for the chart.
    try:
        a = float(a)
    except (TypeError, ValueError):
        return "Unknown"
    if a < 18: return "0–17"
    if a < 36: return "18–35"
    if a < 51: return "36–50"
    if a < 66: return "51–65"
    return "66+"

@app.route("/dashboard")
def dashboard():
    # I'm making sure only logged-in users can access this page.
    if not session.get("user"):
        flash("Please log in to see the dashboard.", "warning")
        return redirect(url_for("login"))

    # --- Patient-only branch (I'm showing ONLY their own row; no charts/tables) ---
    if is_patient():
        pid = patient_id_from_username()
        if pid is None:
            flash("Your account is misconfigured. Please contact support.", "error")
            return redirect(url_for("logout"))

        # I'm fetching exactly one record for this patient id
        doc = patients_collection.find_one({"id": pid}, {"_id": 0})
        if not doc:
            flash("No record found for your account. Please contact support.", "error")
            return redirect(url_for("logout"))

        # I'm cleaning age if it's a whole number (e.g., 80.0 → 80)
        if "age" in doc and isinstance(doc["age"], (float, int)) and float(doc["age"]).is_integer():
            doc["age"] = int(doc["age"])

        # I'm handing off to a simple patient view (no charts)
        return render_template(
            "patient_dashboard.html",
            user=session.get("user"),
            patient=doc,
            support_number="+44 7700 900123"  # forged UK number as requested
        )

    # --- Admin branch (my existing analytics + table + search) ---
    # I'm fetching selected patient fields from MongoDB for display.
    cursor = patients_collection.find({}, {
        "_id": 0, "id": 1, "age": 1, "gender": 1, "stroke": 1,
        "avg_glucose_level": 1, "bmi": 1, "hypertension": 1,
        "heart_disease": 1, "smoking_status": 1, "ever_married": 1,
        "work_type": 1, "residence_type": 1
    })
    docs = list(cursor)

    # I'm computing some basic statistics.
    total = len(docs)
    stroke_yes = sum(1 for d in docs if int(d.get("stroke", 0)) == 1)
    stroke_no = total - stroke_yes
    stroke_rate = round((stroke_yes / total) * 100, 2) if total else 0.0

    by_gender = Counter(d.get("gender") or "Unknown" for d in docs)
    by_age_band = Counter(_age_band(d.get("age")) for d in docs)

    # I'm preparing lists for calculating mean values.
    glucose_stroke = [float(d.get("avg_glucose_level")) for d in docs
                      if d.get("avg_glucose_level") not in (None, "") and int(d.get("stroke", 0)) == 1]
    glucose_no = [float(d.get("avg_glucose_level")) for d in docs
                  if d.get("avg_glucose_level") not in (None, "") and int(d.get("stroke", 0)) == 0]
    bmi_stroke = [float(d.get("bmi")) for d in docs
                  if d.get("bmi") not in (None, "") and int(d.get("stroke", 0)) == 1]
    bmi_no = [float(d.get("bmi")) for d in docs
              if d.get("bmi") not in (None, "") and int(d.get("stroke", 0)) == 0]

    # I'm structuring my summary stats neatly for the dashboard.
    stats = {
        "total": total,
        "stroke_yes": stroke_yes,
        "stroke_no": stroke_no,
        "stroke_rate": stroke_rate,
        "by_gender": by_gender,
        "by_age_band": by_age_band,
        "glucose_mean_stroke": round(mean(glucose_stroke), 1) if glucose_stroke else None,
        "glucose_mean_no": round(mean(glucose_no), 1) if glucose_no else None,
        "bmi_mean_stroke": round(mean(bmi_stroke), 1) if bmi_stroke else None,
        "bmi_mean_no": round(mean(bmi_no), 1) if bmi_no else None,
    }

    # I'm preparing chart labels and values for Chart.js graphs.
    gender_labels = list(stats["by_gender"].keys())
    gender_values = [stats["by_gender"][g] for g in gender_labels]
    age_labels = ["0–17", "18–35", "36–50", "51–65", "66+"]
    age_values = [stats["by_age_band"].get(lbl, 0) for lbl in age_labels]

    # --- SEARCH + PAGINATION (your working block, unchanged except exact matches for gender/smoking) ---
    q = (request.args.get("q") or "").strip()
    page = max(int(request.args.get("page", 1) or 1), 1)
    per_page = 50
    query = {}

    # I'm allowing search on gender, smoking status, text and numbers(this part is an ai assisted code).
    if q:
        or_blocks = []
        for fld in ["gender", "smoking_status", "work_type", "residence_type", "ever_married"]:
            if fld in ["gender", "smoking_status"]:
                or_blocks.append({fld: {"$regex": f"^{q}$", "$options": "i"}})  # whole word match
            else:
                or_blocks.append({fld: {"$regex": q, "$options": "i"}})        # partial ok
        if q.isdigit():
            or_blocks.append({"id": int(q)})
        try:
            num = float(q)
            or_blocks += [
                {"age": num}, {"avg_glucose_level": num}, {"bmi": num},
                {"stroke": int(num)}, {"hypertension": int(num)}, {"heart_disease": int(num)},
            ]
        except ValueError:
            pass
        if or_blocks:
            query = {"$or": or_blocks}

    # I'm applying pagination to MongoDB query.
    patients_cursor = (
        patients_collection.find(query, {"_id": 0})
        .skip((page - 1) * per_page)
        .limit(per_page)
    )
    patients = list(patients_cursor)

    # I'm cleaning float ages like 80.0 → 80 for neat display.
    for p in patients:
        if "age" in p and isinstance(p["age"], (float, int)) and float(p["age"]).is_integer():
            p["age"] = int(p["age"])

    # --- AI stroke risk (admin only, using my Logistic Regression model) ---
    # I'm only doing this extra AI work if:
    # 1) the logged-in user is the admin, and
    # 2) the stroke_model actually loaded correctly at startup.
    if is_admin() and stroke_model is not None:
        for p in patients:
            # I'm asking my helper to build a human-friendly AI message for this patient.
            msg = build_stroke_risk_message(p)
            # If the helper returns something (not None), I attach it to the patient dict.
            p["ai_risk"] = msg
    else:
        # For non-admins, or if the model failed, I make sure there is no AI message at all.
        for p in patients:
            p["ai_risk"] = None

    total_patients = patients_collection.count_documents(query)
    total_pages = max((total_patients + per_page - 1) // per_page, 1)

    # I'm rendering the dashboard template with all prepared data.
    return render_template(
        "dashboard.html",
        user=session.get("user"),
        stats=stats,
        gender_labels=gender_labels, gender_values=gender_values,
        age_labels=age_labels, age_values=age_values,
        patients=patients, page=page, total_pages=total_pages, q=q
    )

@app.route("/api/stats")
def api_stats():
    # I'm returning a small JSON version of basic stats (for potential API use).
    if not session.get("user"):
        return {"error": "unauthorised"}, 401
    cursor = patients_collection.find({}, {"_id": 0, "stroke": 1})
    docs = list(cursor)
    total = len(docs)
    stroke_yes = sum(1 for d in docs if int(d.get("stroke", 0)) == 1)
    return {"total": total, "stroke_yes": stroke_yes, "stroke_no": total - stroke_yes}

@app.route("/logout")
def logout():
    # I'm clearing the user session and flashing a friendly sign-out message.
    session.clear()
    flash("You have been signed out.", "success")
    return redirect(url_for("home"))

# --- ADMIN ROUTES SECTION (AI-assisted code) ---
# I'm adding admin-only routes to insert, edit, delete patient data, and change passwords.

def admin_required(f):
    # I'm restricting access to these routes to admin users only(this part is an ai assisted code).
    @wraps(f)
    def wrapper(*args, **kwargs):
        if session.get("user") != "admin":
            flash("Admins only.", "danger")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return wrapper

@app.route("/settings")
@admin_required
def admin_settings():
    # I'm showing a small admin control panel.
    return render_template("admin_settings.html")

@app.route("/patients/insert", methods=["GET", "POST"])
@admin_required
def insert_record():
    # I'm letting the admin insert new patient records into MongoDB.
    from forms import PatientForm
    form = PatientForm()
    if form.validate_on_submit():
        if patients_collection.count_documents({"id": form.id.data}) > 0:
            flash("A patient with that ID already exists.", "warning")
            return redirect(url_for("insert_record"))
        doc = {
            "id": form.id.data,
            "gender": form.gender.data.strip().title(),
            "age": float(form.age.data),
            "hypertension": int(form.hypertension.data),
            "heart_disease": int(form.heart_disease.data),
            "ever_married": form.ever_married.data.strip().title(),
            "work_type": form.work_type.data.strip(),
            "residence_type": form.residence_type.data.strip().title(),
            "avg_glucose_level": float(form.avg_glucose_level.data),
            "bmi": float(form.bmi.data) if form.bmi.data is not None else None,
            "smoking_status": form.smoking_status.data.strip(),
            "stroke": int(form.stroke.data),
        }
        patients_collection.insert_one(doc)
        flash("Patient inserted.", "success")
        return redirect(url_for("dashboard"))
    return render_template("insert_record.html", form=form)

@app.route("/patients/delete", methods=["GET", "POST"])
@admin_required
def delete_record():
    # I'm letting the admin delete a patient by their ID.
    from forms import DeleteForm
    form = DeleteForm()
    if form.validate_on_submit():
        result = patients_collection.delete_one({"id": form.id.data})
        if result.deleted_count:
            flash("Patient deleted.", "success")
        else:
            flash("No patient with that ID.", "warning")
        return redirect(url_for("dashboard"))
    return render_template("delete_record.html", form=form)

@app.route("/patients/edit", methods=["GET", "POST"])
@admin_required
def edit_record():
    # I'm creating a two-step edit process: Find patient → Edit → Save changes.
    from forms import EditLookupForm, PatientForm
    lookup = EditLookupForm()
    form = PatientForm()

    if lookup.validate_on_submit() and "Find" in request.form.values():
        doc = patients_collection.find_one({"id": lookup.id.data}, {"_id": 0})
        if not doc:
            flash("No patient with that ID.", "warning")
            return redirect(url_for("edit_record"))
        # I'm pre-filling the edit form with the current patient details.
        form.id.data = doc.get("id")
        form.gender.data = doc.get("gender", "")
        form.age.data = doc.get("age", 0)
        form.hypertension.data = int(doc.get("hypertension", 0))
        form.heart_disease.data = int(doc.get("heart_disease", 0))
        form.ever_married.data = doc.get("ever_married", "")
        form.work_type.data = doc.get("work_type", "")
        form.residence_type.data = doc.get("residence_type", "")
        form.avg_glucose_level.data = float(doc.get("avg_glucose_level", 0))
        form.bmi.data = float(doc["bmi"]) if doc.get("bmi") not in (None, "") else None
        form.smoking_status.data = doc.get("smoking_status", "")
        form.stroke.data = int(doc.get("stroke", 0))
        return render_template("edit_record.html", lookup=lookup, form=form, found=True)

    if form.validate_on_submit() and "Save" in request.form.values():
        # I'm saving updated details back to MongoDB.
        updated = {
            "gender": form.gender.data.strip().title(),
            "age": float(form.age.data),
            "hypertension": int(form.hypertension.data),
            "heart_disease": int(form.heart_disease.data),
            "ever_married": form.ever_married.data.strip().title(),
            "work_type": form.work_type.data.strip(),
            "residence_type": form.residence_type.data.strip().title(),
            "avg_glucose_level": float(form.avg_glucose_level.data),
            "bmi": float(form.bmi.data) if form.bmi.data is not None else None,
            "smoking_status": form.smoking_status.data.strip(),
            "stroke": int(form.stroke.data),
        }
        patients_collection.update_one({"id": form.id.data}, {"$set": updated})
        flash("Patient updated.", "success")
        return redirect(url_for("dashboard"))

    return render_template("edit_record.html", lookup=lookup, form=form, found=False)

# i am adding new delete user route
@app.route("/users/delete", methods=["GET", "POST"])
@admin_required
def delete_user():
    from forms import DeleteUserForm
    form = DeleteUserForm()

    if form.validate_on_submit():
        # I'm looking up this user in the SQLite users table.
        u = User.query.filter_by(username=form.username.data).first()

        if not u:
            flash("No user found with that username.", "warning")
            return redirect(url_for("delete_user"))

        # I'm preventing the admin from deleting their own account by mistake.
        if u.username == "admin":
            flash("You can't delete the main admin account.", "danger")
            return redirect(url_for("delete_user"))

        db.session.delete(u)
        db.session.commit()
        flash("User deleted from SQLite users database.", "success")
        return redirect(url_for("dashboard"))

    return render_template("delete_user.html", form=form)

@app.route("/password", methods=["GET", "POST"])
def change_password():
    # I'm allowing logged-in users (admin, student, or patient-<id>) to change their password securely.
    if not session.get("user"):
        flash("Please log in first.", "warning")
        return redirect(url_for("login"))

    from forms import ChangePasswordForm
    form = ChangePasswordForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=session["user"]).first()
        if not user or not check_password_hash(user.password_hash, form.current_password.data):
            flash("Current password is incorrect.", "danger")
            return redirect(url_for("change_password"))
        user.password_hash = generate_password_hash(form.new_password.data)
        db.session.commit()
        flash("Password changed.", "success")
        return redirect(url_for("dashboard"))

    return render_template("change_password.html", form=form)

# --- Dev runner ---
if __name__ == "__main__":
    # I'm running in debug mode during development (never in production).
    app.run(debug=True)