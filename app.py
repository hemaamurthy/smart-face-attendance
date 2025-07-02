from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import os, sqlite3, datetime, base64
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from whitenoise import WhiteNoise

# --- REMOVED HEAVY DEPENDENCIES ---
# Removed: face_recognition, cv2, dlib from imports as they won't run on free tier

# --- EXPLICITLY DEFINE STATIC FOLDER ---
# This tells Flask exactly where to find static files.
# It should be the 'static' folder relative to your app.py.
app = Flask(__name__, static_folder='static') # NEW: Explicit static_folder

app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# --- WhiteNoise Configuration for Static Files ---
# WhiteNoise will now serve files from the explicitly defined static_folder.
app.wsgi_app = WhiteNoise(app.wsgi_app, root=app.static_folder) # Changed root to app.static_folder
print(f"WhiteNoise serving static files from: {app.static_folder}") # DEBUG PRINT

# --- Flask-Login Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model for Flask-Login
class User(UserMixin):
    def __init__(self, id, username, password_hash, role):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.role = role

    def get_id(self_):
        return str(self_.id)
    
    def is_admin(self_):
        return self_.role == 'admin'

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    user_data = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if user_data:
        return User(user_data['id'], user_data['username'], user_data['password_hash'], user_data['role'])
    return None

# --- Database Initialization ---
DATABASE = 'attendance.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE NOT NULL,
            name TEXT NOT NULL,
            roll_number TEXT UNIQUE NOT NULL,
            section TEXT NOT NULL,
            date_of_birth TEXT NOT NULL,
            face_descriptor TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            image_path TEXT,
            latitude REAL,
            longitude REAL,
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    ''')

    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
    if cursor.fetchone()[0] == 0:
        admin_password = generate_password_hash("adminpass")
        conn.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                     ("admin", admin_password, "admin"))
        print("Default admin user created: username='admin', password='adminpass'")
    
    conn.commit()
    conn.close()

if not os.path.exists('face_data/attendance_logs'):
    os.makedirs('face_data/attendance_logs')

with app.app_context():
    init_db()

def insert_attendance_record(student_id, date, time, image_path, latitude, longitude):
    conn = get_db_connection()
    conn.execute("INSERT INTO attendance (student_id, date, time, image_path, latitude, longitude) VALUES (?, ?, ?, ?, ?, ?)",
                 (student_id, date, time, image_path, latitude, longitude))
    conn.commit()
    conn.close()

def compare_face_descriptors(known_descriptor_str, unknown_descriptor_array, tolerance=0.6):
    if not known_descriptor_str:
        return False

    try:
        known_descriptor_array = np.array(eval(known_descriptor_str))
        distance = np.linalg.norm(known_descriptor_array - unknown_descriptor_array)
        return distance < tolerance
    except Exception as e:
        print(f"Error comparing descriptors: {e}")
        return False

@app.route('/')
def home():
    if not current_user.is_authenticated:
        return render_template('landing.html')
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        flash("You are already logged in.", "info")
        if current_user.is_admin():
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('my_attendance'))

    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        conn = get_db_connection()
        user_data = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        
        if user_data and user_data['role'] == 'student':
            student_data = conn.execute('SELECT date_of_birth FROM students WHERE user_id = ?', (user_data['id'],)).fetchone()
            if student_data and check_password_hash(user_data['password_hash'], password):
                user = User(user_data['id'], user_data['username'], user_data['password_hash'], user_data['role'])
                login_user(user)
                flash('Logged in successfully!', 'success')
                conn.close()
                return redirect(url_for('my_attendance')) 
            else:
                flash('Invalid username or password.', 'error')
                conn.close()
                return render_template('login.html')
        
        elif user_data and user_data['role'] == 'admin':
            if check_password_hash(user_data['password_hash'], password):
                user = User(user_data['id'], user_data['username'], user_data['password_hash'], user_data['role'])
                login_user(user)
                flash('Logged in successfully!', 'success')
                conn.close()
                return redirect(url_for('admin_dashboard'))
            else:
                flash('Invalid username or password.', 'error')
                conn.close()
                return render_template('login.html')
        else:
            flash('Invalid username or password.', 'error')
            conn.close()
            return render_template('login.html')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name'].strip()
        roll_number = request.form['roll_number'].strip()
        section = request.form['section'].strip()
        date_of_birth = request.form['date_of_birth'].strip()
        image_data_url = request.form['captured_image']
        face_descriptor_str = request.form.get('face_descriptor')

        if not name or not roll_number or not section or not date_of_birth or not face_descriptor_str:
            flash("All fields (Name, Roll Number, Section, Date of Birth) and face capture are required.", "error")
            return redirect(url_for('register'))

        conn = get_db_connection()
        if conn.execute('SELECT COUNT(*) FROM students WHERE roll_number = ?', (roll_number,)).fetchone()[0] > 0:
            flash(f"Roll Number '{roll_number}' is already registered. Please login or use a unique Roll Number.", "error")
            conn.close()
            return redirect(url_for('register'))
        
        if conn.execute('SELECT COUNT(*) FROM users WHERE username = ?', (roll_number,)).fetchone()[0] > 0:
            flash(f"Username '{roll_number}' already exists. Please login or use a different Roll Number.", "error")
            conn.close()
            return redirect(url_for('register'))
        conn.close()

        try:
            face_descriptor = eval(face_descriptor_str)
            if not isinstance(face_descriptor, list) or len(face_descriptor) == 0:
                flash("❌ Invalid face descriptor received. Please ensure a face is detected.", "error")
                return redirect(url_for('register'))

            conn = get_db_connection()
            student_username = roll_number
            student_password_hash = generate_password_hash(date_of_birth)
            
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                           (student_username, student_password_hash, "student"))
            student_user_id = cursor.lastrowid

            conn.execute("INSERT INTO students (user_id, name, roll_number, section, date_of_birth, face_descriptor) VALUES (?, ?, ?, ?, ?, ?)",
                         (student_user_id, name, roll_number, section, date_of_birth, str(face_descriptor)))
            conn.commit()
            conn.close()

            flash(f"✅ Registration successful! Your username is '{roll_number}' and password is your Date of Birth.", "success")
            return redirect(url_for('login'))

        except Exception as e:
            flash(f"⚠️ Error during registration: {str(e)}. Please try again.", "error")
            print(f"Error in /register: {e}")
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/mark_attendance', methods=['GET', 'POST'])
@login_required
def mark_attendance():
    if current_user.is_admin():
        flash("🚫 Admins cannot mark attendance here. Please use the admin dashboard.", "info")
        return redirect(url_for('admin_dashboard'))

    conn = get_db_connection()
    student_data = conn.execute('SELECT id, name, roll_number, section, face_descriptor FROM students WHERE user_id = ?', (current_user.id,)).fetchone()
    conn.close()

    if not student_data:
        flash("🚫 Your user account is not linked to a student profile. Please contact admin.", "error")
        return redirect(url_for('home'))

    student_id = student_data['id']
    student_name = student_data['name']
    student_roll_number = student_data['roll_number']
    known_face_descriptor_str = student_data['face_descriptor']

    if not known_face_descriptor_str:
        flash("🚫 No face registered for your account. Please register your face first.", "error")
        return redirect(url_for('home'))

    if request.method == 'POST':
        image_data_url = request.form.get('captured_image')
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        captured_face_descriptor_str = request.form.get('face_descriptor')

        try:
            latitude = float(latitude) if latitude else None
            longitude = float(longitude) if longitude else None
        except ValueError:
            flash("Invalid location data.", "error")
            return redirect(url_for('mark_attendance'))

        if not latitude or not longitude:
            flash("❌ Location data not captured. Please allow location access.", "error")
            return redirect(url_for('mark_attendance'))
        
        if not captured_face_descriptor_str:
            flash("❌ No face detected or descriptor captured. Please ensure your face is visible.", "error")
            return redirect(url_for('mark_attendance'))

        now = datetime.datetime.now()
        today_date = now.strftime("%Y-%m-%d")
        conn = get_db_connection()
        attendance_today = conn.execute('SELECT COUNT(*) FROM attendance WHERE student_id = ? AND date = ?',
                                        (student_id, today_date)).fetchone()[0]
        conn.close()

        if attendance_today > 0:
            flash(f"⚠️ Attendance already marked for {student_name} ({student_roll_number}) today ({today_date}).", "warning")
            return redirect(url_for('my_attendance'))

        try:
            captured_descriptor_array = np.array(eval(captured_face_descriptor_str))
            
            is_match = compare_face_descriptors(known_face_descriptor_str, captured_descriptor_array)

            if is_match:
                current_time = now.strftime("%H:%M:%S")
                filename = f"attendance_{student_roll_number}_{today_date}_{now.strftime('%H%M%S')}.jpg"
                
                attendance_log_dir = 'face_data/attendance_logs'
                if not os.path.exists(attendance_log_dir):
                    os.makedirs(attendance_log_dir)

                img_bytes = base64.b64decode(image_data_url.split(",")[1])
                with open(os.path.join(attendance_log_dir, filename), 'wb') as f:
                    f.write(img_bytes)

                insert_attendance_record(student_id, today_date, current_time, os.path.join(attendance_log_dir, filename), latitude, longitude)
                flash(f"✅ Attendance marked for {student_name} ({student_roll_number}) for {today_date} at {current_time}!", "success")
                return redirect(url_for('my_attendance'))
            else:
                flash("❌ Face not recognized or does not match your registered face. Please try again.", "error")
                return redirect(url_for('mark_attendance'))

        except Exception as e:
            flash(f"⚠️ Error processing attendance: {str(e)}. Please try again.", "error")
            print(f"Error in /mark_attendance: {e}")
            return redirect(url_for('mark_attendance'))

    return render_template("mark_attendance.html")

@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    if not current_user.is_admin():
        flash("🚫 You don't have permission to access the admin dashboard.", "error")
        return redirect(url_for('home'))

    conn = get_db_connection()
    
    students = conn.execute('SELECT id, name, roll_number, section, date_of_birth FROM students ORDER BY name').fetchall()

    query = """
        SELECT
            a.id,
            s.name,
            s.roll_number,
            s.section,
            a.date,
            a.time,
            a.latitude,
            a.longitude,
            a.image_path
        FROM
            attendance a
        JOIN
            students s ON a.student_id = s.id
        ORDER BY
            a.date DESC, a.time DESC
    """
    attendance_records = conn.execute(query).fetchall()
    conn.close()
    
    return render_template('admin_dashboard.html', students_and_records={'students': students, 'attendance_records': attendance_records})

@app.route('/delete_student/<int:student_id>', methods=['POST'])
@login_required
def delete_student(student_id):
    if not current_user.is_admin():
        return jsonify({'success': False, 'message': 'Permission denied.'}), 403

    conn = get_db_connection()
    try:
        user_id = conn.execute('SELECT user_id FROM students WHERE id = ?', (student_id,)).fetchone()
        
        if user_id:
            user_id = user_id['user_id']
            conn.execute('DELETE FROM attendance WHERE student_id = ?', (student_id,))
            conn.execute('DELETE FROM students WHERE id = ?', (student_id,))
            conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
            conn.commit()

            flash('Student and all associated records deleted successfully!', 'success')
            return jsonify({'success': True, 'message': 'Student and all associated records deleted successfully!'})
        else:
            flash('Student not found.', 'error')
            return jsonify({'success': False, 'message': 'Student not found.'}), 404
    except Exception as e:
        conn.rollback()
        print(f"Error deleting student: {e}")
        flash(f'Error deleting student: {str(e)}', 'error')
        return jsonify({'success': False, 'message': f'Error deleting student: {str(e)}'}), 500
    finally:
        conn.close()

@app.route('/delete_attendance/<int:record_id>', methods=['POST'])
@login_required
def delete_attendance(record_id):
    if not current_user.is_admin():
        return jsonify({'success': False, 'message': 'Permission denied.'}), 403

    conn = get_db_connection()
    try:
        image_path_data = conn.execute('SELECT image_path FROM attendance WHERE id = ?', (record_id,)).fetchone()
        
        conn.execute('DELETE FROM attendance WHERE id = ?', (record_id,))
        conn.commit()

        if image_path_data and image_path_data['image_path'] and os.path.exists(image_path_data['image_path']):
            os.remove(image_path_data['image_path'])
            print(f"Deleted attendance image: {image_path_data['image_path']}")

        flash('Attendance record deleted successfully!', 'success')
        return jsonify({'success': True, 'message': 'Attendance record deleted successfully!'})
    except Exception as e:
        conn.rollback()
        print(f'Error deleting attendance record: {str(e)}', 'error')
        return jsonify({'success': False, 'message': f'Error deleting attendance record: {str(e)}'}), 500
    finally:
        conn.close()


if __name__ == '__main__':
    app.run(debug=False)
