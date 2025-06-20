from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import os, cv2, sqlite3, datetime, base64
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

# Import the utility functions
from utils.face_recognition_utils import load_known_faces, recognize_face, process_image_for_face_recognition
import face_recognition # Explicitly import face_recognition here for direct use if needed

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secure random secret key for session management
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Increased upload limit to 5 MB

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
            date_of_birth TEXT NOT NULL, -- New: Date of Birth column
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
        admin_password = generate_password_hash("nothing123") # CHANGE THIS PASSWORD FOR DEPLOYMENT!
        conn.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                     ("admin", admin_password, "admin"))
        print("Default admin user created: username='admin', password='adminpass'")
    
    conn.commit()
    conn.close()

# Ensure directories exist
if not os.path.exists('face_data/registered_face'):
    os.makedirs('face_data/registered_face')
if not os.path.exists('face_data/attendance_logs'):
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

# --- Routes ---
@app.route('/')
def home():
    # If not logged in, show the new landing page
    if not current_user.is_authenticated:
        return render_template('landing.html') # Render the new landing page
    # If logged in, redirect to their respective dashboard/home as before
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        flash("You are already logged in.", "info")
        if current_user.is_admin():
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('my_attendance')) # Or mark_attendance directly

    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip() # Password is now DOB for students
        conn = get_db_connection()
        user_data = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        
        # If user found, check if it's a student and verify DOB
        if user_data and user_data['role'] == 'student':
            student_data = conn.execute('SELECT date_of_birth FROM students WHERE user_id = ?', (user_data['id'],)).fetchone()
            if student_data and check_password_hash(user_data['password_hash'], password):
                user = User(user_data['id'], user_data['username'], user_data['password_hash'], user_data['role'])
                login_user(user)
                flash('Logged in successfully!', 'success')
                conn.close()
                return redirect(url_for('my_attendance')) 
            else:
                flash('Invalid username or password.', 'error') # Password mismatch for student
                conn.close()
                return render_template('login.html')
        
        # If user found and it's an admin, verify password
        elif user_data and user_data['role'] == 'admin':
            if check_password_hash(user_data['password_hash'], password):
                user = User(user_data['id'], user_data['username'], user_data['password_hash'], user_data['role'])
                login_user(user)
                flash('Logged in successfully!', 'success')
                conn.close()
                return redirect(url_for('admin_dashboard'))
            else:
                flash('Invalid username or password.', 'error') # Password mismatch for admin
                conn.close()
                return render_template('login.html')
        else:
            flash('Invalid username or password.', 'error') # User not found
            conn.close()
            return render_template('login.html')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login')) # Redirect to login after logout

@app.route('/register', methods=['GET', 'POST'])
# NO @login_required here - allowing self-registration for students
def register():
    if request.method == 'POST':
        name = request.form['name'].strip()
        roll_number = request.form['roll_number'].strip()
        section = request.form['section'].strip()
        date_of_birth = request.form['date_of_birth'].strip() # New: Date of Birth
        image_data_url = request.form['captured_image']

        if not name or not roll_number or not section or not date_of_birth:
            flash("All fields (Name, Roll Number, Section, Date of Birth) are required.", "error")
            return redirect(url_for('register'))

        conn = get_db_connection()
        # Check for duplicate roll number (or username)
        if conn.execute('SELECT COUNT(*) FROM students WHERE roll_number = ?', (roll_number,)).fetchone()[0] > 0:
            flash(f"Roll Number '{roll_number}' is already registered. Please login or use a unique Roll Number.", "error")
            conn.close()
            return redirect(url_for('register'))
        
        # Also check if username already exists for the user table (roll_number is username)
        if conn.execute('SELECT COUNT(*) FROM users WHERE username = ?', (roll_number,)).fetchone()[0] > 0:
            flash(f"Username '{roll_number}' already exists. Please login or use a different Roll Number.", "error")
            conn.close()
            return redirect(url_for('register'))
        conn.close()

        if image_data_url:
            try:
                if "," not in image_data_url:
                    flash("❌ Invalid image data format.", "error")
                    print("DEBUG: Invalid image data format (no comma).")
                    return redirect(url_for('register'))
                
                image_data = image_data_url.split(",")[1]
                img_bytes = base64.b64decode(image_data)
                print(f"DEBUG: Received image bytes length: {len(img_bytes)} bytes")

                rgb_img_for_face_rec = process_image_for_face_recognition(img_bytes)

                if rgb_img_for_face_rec is None:
                    flash("❌ Failed to decode or process image. Please try again.", "error")
                    print("DEBUG: process_image_for_face_recognition returned None.")
                    return redirect(url_for('register'))
                
                print(f"DEBUG: Image processed by utility. Shape: {rgb_img_for_face_rec.shape}, Dtype: {rgb_img_for_face_rec.dtype}")

                # Redundant OpenCV conversion for dlib compatibility (as previously debugged)
                try:
                    img_bgr = cv2.cvtColor(rgb_img_for_face_rec, cv2.COLOR_RGB2BGR)
                    final_rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    flash(f"❌ Error during OpenCV color conversion: {str(e)}. Cannot prepare image for face detection.", "error")
                    print(f"DEBUG: Error during OpenCV BGR->RGB conversion: {e}")
                    return redirect(url_for('register'))


                # Check if a face is present in the captured image
                try:
                    face_locations = face_recognition.face_locations(final_rgb_img)
                except NameError:
                    flash("❌ Critical Error: face_recognition library not accessible. Please check server setup.", "error")
                    print("CRITICAL ERROR: 'face_recognition' module not found or accessible in this scope.")
                    return redirect(url_for('register'))
                except Exception as e:
                    flash(f"❌ Error detecting face: {str(e)}. This might indicate an issue with the image data after processing.", "error")
                    print(f"DEBUG: Error calling face_recognition.face_locations: {e}")
                    return redirect(url_for('register'))

                if not face_locations:
                    flash("❌ No face detected in the captured image. Please ensure your face is visible and well-lit.", "error")
                    print("DEBUG: No face detected by face_recognition.")
                    return redirect(url_for('register'))
                else:
                    print(f"DEBUG: Face(s) detected: {len(face_locations)}")

                # --- Registration Logic: Create User and Student ---
                conn = get_db_connection()
                # Create a user account for the student (roll number as username)
                # Password is a hash of DATE OF BIRTH
                student_username = roll_number
                student_password_hash = generate_password_hash(date_of_birth) # Password is DOB
                
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                               (student_username, student_password_hash, "student"))
                student_user_id = cursor.lastrowid # Get the ID of the newly created user

                # Save student details, linked to the new user ID, including DOB
                conn.execute("INSERT INTO students (user_id, name, roll_number, section, date_of_birth) VALUES (?, ?, ?, ?, ?)",
                             (student_user_id, name, roll_number, section, date_of_birth))
                conn.commit()
                conn.close()

                # Save the registered face image (used for face recognition later)
                # Filename format: name_rollnumber.jpg
                save_path = os.path.join('face_data/registered_face', f"{name.replace(' ', '_')}_{roll_number}.jpg")
                img_to_save = cv2.cvtColor(final_rgb_img, cv2.COLOR_RGB2BGR) 
                cv2.imwrite(save_path, img_to_save)
                print(f"DEBUG: Registered face image saved to {save_path}")

                flash(f"✅ Registration successful! Your username is '{roll_number}' and password is your Date of Birth.", "success")
                return redirect(url_for('login')) # Redirect to login after successful registration

            except Exception as e:
                flash(f"⚠️ Error during registration: {str(e)}. Please try again.", "error")
                print(f"Error in /register: {e}")
                return redirect(url_for('register'))
        else:
            flash("❌ No image captured.", "error")
            print("DEBUG: No image data received from frontend.")
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/mark_attendance', methods=['GET', 'POST'])
@login_required # Only logged-in users can mark attendance
def mark_attendance():
    # Only students should mark attendance. Admins should go to admin dashboard.
    if current_user.is_admin():
        flash("🚫 Admins cannot mark attendance here. Please use the admin dashboard.", "info")
        return redirect(url_for('admin_dashboard'))

    conn = get_db_connection()
    # Get student data linked to the current logged-in user
    student_data = conn.execute('SELECT id, name, roll_number, section FROM students WHERE user_id = ?', (current_user.id,)).fetchone()
    conn.close()

    if not student_data:
        flash("🚫 Your user account is not linked to a student profile. Please contact admin.", "error")
        return redirect(url_for('home'))

    student_id = student_data['id']
    student_name = student_data['name']
    student_roll_number = student_data['roll_number']

    if request.method == 'POST':
        image_data_url = request.form.get('captured_image')
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')

        # Basic validation for location data
        try:
            latitude = float(latitude) if latitude else None
            longitude = float(longitude) if longitude else None
        except ValueError:
            flash("Invalid location data.", "error")
            return redirect(url_for('mark_attendance'))

        if not latitude or not longitude:
            flash("❌ Location data not captured. Please allow location access.", "error")
            return redirect(url_for('mark_attendance'))

        if not image_data_url or "," not in image_data_url:
            flash("❌ No image received or invalid format.", "error")
            print("DEBUG (mark_attendance): No image received or invalid format.")
            return redirect(url_for('mark_attendance'))

        try:
            # Check if attendance is already marked for today for this student
            now = datetime.datetime.now()
            today_date = now.strftime("%Y-%m-%d")
            conn = get_db_connection()
            attendance_today = conn.execute('SELECT COUNT(*) FROM attendance WHERE student_id = ? AND date = ?',
                                            (student_id, today_date)).fetchone()[0]
            conn.close()

            if attendance_today > 0:
                flash(f"⚠️ Attendance already marked for {student_name} ({student_roll_number}) today ({today_date}).", "warning")
                print(f"DEBUG (mark_attendance): Attendance already marked for {student_name} today.")
                return redirect(url_for('my_attendance')) # Redirect to their own dashboard
            
            base64_data = image_data_url.split(",")[1]
            img_bytes = base64.b64decode(base64_data)
            print(f"DEBUG (mark_attendance): Received image bytes length: {len(img_bytes)} bytes")
            
            rgb_img_for_face_rec = process_image_for_face_recognition(img_bytes)

            if rgb_img_for_face_rec is None:
                flash("❌ Failed to decode or process image. Please try again.", "error")
                print("DEBUG (mark_attendance): process_image_for_face_recognition returned None.")
                return redirect(url_for('mark_attendance'))
            
            print(f"DEBUG (mark_attendance): Image processed by utility. Shape: {rgb_img_for_face_rec.shape}, Dtype: {rgb_img_for_face_rec.dtype}")

            # Redundant OpenCV conversion for dlib compatibility (as previously debugged)
            try:
                img_bgr = cv2.cvtColor(rgb_img_for_face_rec, cv2.COLOR_RGB2BGR)
                final_rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:
                flash(f"❌ Error during OpenCV color conversion in attendance: {str(e)}. Cannot prepare image.", "error")
                print(f"DEBUG (mark_attendance): Error during OpenCV BGR->RGB conversion: {e}")
                return redirect(url_for('mark_attendance'))

            # Load known faces using the utility function (student_roll_number is used as the id for face_recognition loading)
            known_encodings, known_names, known_ids = load_known_faces('face_data/registered_face')
            
            if not known_encodings:
                flash("⚠️ No registered faces found. Please register users first.", "warning")
                print("DEBUG (mark_attendance): No registered faces found.")
                return redirect(url_for('mark_attendance'))

            # Recognize face using the utility function, passing the final_rgb_img
            recognized_name, recognized_roll_number_from_face = recognize_face(final_rgb_img, known_encodings, known_names, known_ids)

            # --- Authentication Check: Is the recognized face the logged-in user? ---
            if recognized_roll_number_from_face == student_roll_number:
                current_time = now.strftime("%H:%M:%S")
                # Create a unique filename for the attendance record image
                filename = f"attendance_{student_roll_number}_{today_date}_{now.strftime('%H%M%S')}.jpg"
                
                attendance_log_dir = 'face_data/attendance_logs' 
                save_path = os.path.join(attendance_log_dir, filename)
                
                # Convert the RGB NumPy array back to BGR for saving with OpenCV
                img_to_save = cv2.cvtColor(final_rgb_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, img_to_save)

                insert_attendance_record(student_id, today_date, current_time, save_path, latitude, longitude)
                flash(f"✅ Attendance marked for {student_name} ({student_roll_number}) for {today_date} at {current_time}!", "success")
                print(f"DEBUG (mark_attendance): Attendance marked for {student_name} ({student_roll_number}).")
                return redirect(url_for('my_attendance')) # Redirect to their own dashboard
            else:
                flash("❌ Face not recognized or does not match your registered face. Please try again.", "error")
                print(f"DEBUG (mark_attendance): Face not recognized or mismatch. Recognized: {recognized_roll_number_from_face}, Expected: {student_roll_number}")
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
        return redirect(url_for('home')) # Redirect to general home if not admin

    conn = get_db_connection()
    
    # Fetch all students
    students = conn.execute('SELECT id, name, roll_number, section, date_of_birth FROM students ORDER BY name').fetchall()

    # Fetch all attendance records with student details
    query = """
        SELECT
            a.id, -- Added attendance ID for deletion
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
    
    # Pass both lists to the template
    return render_template('admin_dashboard.html', students_and_records={'students': students, 'attendance_records': attendance_records})


# --- My Attendance (Student View) ---
@app.route('/my_attendance')
@login_required
def my_attendance():
    # Admins should not view this page
    if current_user.is_admin():
        flash("🚫 Admins do not have personal attendance records here. Please use the admin dashboard.", "info")
        return redirect(url_for('admin_dashboard'))

    conn = get_db_connection()
    # Get the student_id for the currently logged-in user
    student_data = conn.execute('SELECT id, name FROM students WHERE user_id = ?', (current_user.id,)).fetchone()
    
    if not student_data:
        flash("🚫 Your user account is not linked to a student profile. Please contact admin.", "error")
        conn.close()
        return redirect(url_for('home'))

    student_id = student_data['id']
    student_name = student_data['name']

    # Fetch attendance records only for this specific student
    query = """
        SELECT
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
        WHERE
            s.id = ?
        ORDER BY
            a.date DESC, a.time DESC
    """
    records = conn.execute(query, (student_id,)).fetchall()
    conn.close()
    return render_template('my_attendance.html', records=records, student_name=student_name)

# --- NEW: Delete Student Route ---
@app.route('/delete_student/<int:student_id>', methods=['POST'])
@login_required
def delete_student(student_id):
    if not current_user.is_admin():
        return jsonify({'success': False, 'message': 'Permission denied.'}), 403

    conn = get_db_connection()
    try:
        # Get the user_id associated with the student_id
        user_id = conn.execute('SELECT user_id FROM students WHERE id = ?', (student_id,)).fetchone()
        
        if user_id:
            user_id = user_id['user_id']
            # Delete associated attendance records first (due to foreign key constraint)
            conn.execute('DELETE FROM attendance WHERE student_id = ?', (student_id,))
            # Delete the student record
            conn.execute('DELETE FROM students WHERE id = ?', (student_id,))
            # Delete the user account
            conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
            conn.commit()

            # Attempt to delete the registered face image file
            # Get the student's name and roll_number to construct the filename
            # This requires fetching the student's data *before* deletion, if not already fetched.
            # For simplicity, we'll assume the format: name_rollnumber.jpg for registered faces.
            # This is a bit tricky to reliably delete without having the exact name_roll_number.
            # A more robust solution would store the registered_face_path in the students table.
            # For now, we'll try to guess based on common format.
            # (Re-fetching student data before deletion for image path construction)
            # This is a placeholder, a real solution would need to store image path for student.
            # If you want robust image deletion, you need to store image_path in `students` table.
            # Example: name_roll_number = conn.execute('SELECT name, roll_number FROM students WHERE id = ?', (student_id,)).fetchone()
            # If (name_roll_number):
            #     face_image_path = os.path.join('face_data/registered_face', f"{name_roll_number['name'].replace(' ', '_')}_{name_roll_number['roll_number']}.jpg")
            #     if os.path.exists(face_image_path):
            #         os.remove(face_image_path)
            #         print(f"Deleted face image: {face_image_path}")

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


# --- NEW: Delete Attendance Record Route ---
@app.route('/delete_attendance/<int:record_id>', methods=['POST'])
@login_required
def delete_attendance(record_id):
    if not current_user.is_admin():
        return jsonify({'success': False, 'message': 'Permission denied.'}), 403

    conn = get_db_connection()
    try:
        # Get image path before deleting record
        image_path_data = conn.execute('SELECT image_path FROM attendance WHERE id = ?', (record_id,)).fetchone()
        
        conn.execute('DELETE FROM attendance WHERE id = ?', (record_id,))
        conn.commit()

        # Delete the associated image file if it exists
        if image_path_data and image_path_data['image_path'] and os.path.exists(image_path_data['image_path']):
            os.remove(image_path_data['image_path'])
            print(f"Deleted attendance image: {image_path_data['image_path']}")

        flash('Attendance record deleted successfully!', 'success')
        return jsonify({'success': True, 'message': 'Attendance record deleted successfully!'})
    except Exception as e:
        conn.rollback()
        print(f"Error deleting attendance record: {e}")
        flash(f'Error deleting attendance record: {str(e)}', 'error')
        return jsonify({'success': False, 'message': f'Error deleting attendance record: {str(e)}'}), 500
    finally:
        conn.close()


if __name__ == '__main__':
    app.run(debug=False)

