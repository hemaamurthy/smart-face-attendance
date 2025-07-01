from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import os, sqlite3, datetime, base64
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from whitenoise import WhiteNoise # NEW: Import WhiteNoise

# --- REMOVED HEAVY DEPENDENCIES ---
# Removed: face_recognition, cv2, dlib from imports as they won't run on free tier

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secure random secret key for session management
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Upload limit to 5 MB

# --- NEW: WhiteNoise Configuration for Static Files ---
# This tells WhiteNoise where to find your static files.
# It will serve files from the 'static' directory.
app.wsgi_app = WhiteNoise(app.wsgi_app, root=os.path.join(app.root_path, 'static'))
# Optional: Add specific static files to serve (e.g., models)
# app.wsgi_app.add_files('/path/to/your/static/models', prefix='static/models/') # Not strictly needed if root is 'static'

# --- Flask-Login Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Redirect unauthenticated users to the login page

# User model for Flask-Login
class User(UserMixin):
    def __init__(self, id, username, password_hash, role):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.role = role

    def get_id(self):
        return str(self.id)
    
    def is_admin(self):
        return self.role == 'admin'

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
    conn.row_factory = sqlite3.Row # Allows accessing columns by name
    return conn

def init_db():
    conn = get_db_connection()
    # Create users table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')
    # Create students table (linked to users and storing student-specific data)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE NOT NULL,
            name TEXT NOT NULL,
            roll_number TEXT UNIQUE NOT NULL,
            section TEXT NOT NULL,
            date_of_birth TEXT NOT NULL,
            face_descriptor TEXT, -- NEW: Store face descriptor as a JSON string
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    # Create attendance table
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

    # Add a default admin user if not exists
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
    if cursor.fetchone()[0] == 0:
        admin_password = generate_password_hash("adminpass") # CHANGE THIS PASSWORD FOR DEPLOYMENT!
        conn.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                     ("admin", admin_password, "admin"))
        print("Default admin user created: username='admin', password='adminpass'")
    
    conn.commit()
    conn.close()

# Ensure directories exist (we might not save raw registered faces locally anymore)
if not os.path.exists('face_data/attendance_logs'): # Only need logs, registered_face not needed for server-side storing
    os.makedirs('face_data/attendance_logs')

# Initialize the database on app startup
with app.app_context():
    init_db()

# --- Helper to insert attendance ---
def insert_attendance_record(student_id, date, time, image_path, latitude, longitude):
    conn = get_db_connection()
    conn.execute("INSERT INTO attendance (student_id, date, time, image_path, latitude, longitude) VALUES (?, ?, ?, ?, ?, ?)",
                 (student_id, date, time, image_path, latitude, longitude))
    conn.commit()
    conn.close()

# --- Face Matching Helper (Python side, using numpy) ---
# This will replace face_recognition.compare_faces
def compare_face_descriptors(known_descriptor_str, unknown_descriptor_array, tolerance=0.6):
    """
    Compares a known face descriptor (as a JSON string from DB) with a
    newly provided descriptor (NumPy array from frontend).
    Returns True if they match within tolerance, False otherwise.
    """
    if not known_descriptor_str:
        return False # No known face descriptor to compare against

    try:
        known_descriptor_array = np.array(eval(known_descriptor_str)) # eval() to convert string back to list/array
        distance = np.linalg.norm(known_descriptor_array - unknown_descriptor_array)
        return distance < tolerance
    except Exception as e:
        print(f"Error comparing descriptors: {e}")
        return False

# --- Routes ---
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
        face_descriptor_str = request.form.get('face_descriptor') # NEW: Get face descriptor from frontend

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
            # Parse the face descriptor from string to list/NumPy array
            face_descriptor = eval(face_descriptor_str) # Using eval for simplicity, safer would be json.loads
            if not isinstance(face_descriptor, list) or len(face_descriptor) == 0:
                flash("❌ Invalid face descriptor received. Please ensure a face is detected.", "error")
                return redirect(url_for('register'))

            # --- Registration Logic: Create User and Student ---
            conn = get_db_connection()
            student_username = roll_number
            student_password_hash = generate_password_hash(date_of_birth)
            
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                           (student_username, student_password_hash, "student"))
            student_user_id = cursor.lastrowid

            # Save student details, linked to the new user ID, including DOB and face descriptor
            conn.execute("INSERT INTO students (user_id, name, roll_number, section, date_of_birth, face_descriptor) VALUES (?, ?, ?, ?, ?, ?)",
                         (student_user_id, name, roll_number, section, date_of_birth, str(face_descriptor))) # Store as string
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
    known_face_descriptor_str = student_data['face_descriptor'] # Get the stored descriptor

    if not known_face_descriptor_str:
        flash("🚫 No face registered for your account. Please register your face first.", "error")
        return redirect(url_for('home')) # Or redirect to register page

    if request.method == 'POST':
        image_data_url = request.form.get('captured_image')
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        captured_face_descriptor_str = request.form.get('face_descriptor') # NEW: Get captured descriptor

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

        # Check if attendance is already marked for today for this student
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
            # Convert captured descriptor string to NumPy array
            captured_descriptor_array = np.array(eval(captured_face_descriptor_str))
            
            # Perform face comparison on the server side using numpy
            # (Alternatively, this comparison could be done entirely in JS on client)
            is_match = compare_face_descriptors(known_face_descriptor_str, captured_descriptor_array)

            if is_match:
                current_time = now.strftime("%H:%M:%S")
                filename = f"attendance_{student_roll_number}_{today_date}_{now.strftime('%H%M%S')}.jpg"
                
                # Save the raw image for attendance logs (optional, but good for visual verification)
                attendance_log_dir = 'face_data/attendance_logs'
                if not os.path.exists(attendance_log_dir): # Ensure log directory exists
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


# --- Admin Dashboard ---
@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    if not current_user.is_admin():
        flash("🚫 You don't have permission to access the admin dashboard.", "error")
        return redirect(url_for('home'))

    conn = get_db_connection()
    
    # Fetch all students
    students = conn.execute('SELECT id, name, roll_number, section, date_of_birth FROM students ORDER BY name').fetchall()

    # Fetch all attendance records with student details
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


# --- Delete Student Route ---
@app.route('/delete_student/<int:student_id>', methods=['POST'])
@login_required
def delete_student(student_id):
    if not current_user.is_admin():
        return jsonify({'success': False, 'message': 'Permission denied.'}), 403

    conn = get_db_connection()
    try:
        # Get the user_id associated with the student_id
        user_id = conn.execute('SELECT user_id FROM students WHERE id = ?', (student_id,)).fetchone()
        
        # NOTE: If you decide to save raw registered face images on the server for admin review,
        # you would need to fetch the image path here (e.g., from a new column in students table)
        # and delete it from the file system. For now, we only store the descriptor.
        
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


# --- Delete Attendance Record Route ---
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
