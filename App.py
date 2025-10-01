from flask import Flask, redirect, render_template, request, jsonify, session,render_template_string
import pickle
from Model import Model
from flask_cors import CORS
from bson.json_util import dumps
from pymongo.mongo_client import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from pymongo.server_api import ServerApi
from flask_pymongo import PyMongo
from flask_session import Session
from urllib.parse import quote_plus
from werkzeug.security import generate_password_hash
from bson import ObjectId  
from bson.errors import InvalidId
from flask_mail import Mail, Message
import random
import string
import joblib
import json
import logging
import re
import os
import bcrypt
from datetime import datetime, timedelta
from flask import url_for

app = Flask(__name__)
app.secret_key = 'NAzhathAara@123'

@app.route('/')
def home():
    session['user'] = 'username'  # This will be signed with the secret key
    return render_template('login.html') 
    
app.permanent_session_lifetime = timedelta(days=1) 



# enable CORS 
CORS(app,supports_credentials=True)  
# Correctly formatted MongoDB connection URL with TLS enabled
uri = "mongodb+srv://researchnazi428:nazi1234567@cluster0.zuzgk1k.mongodb.net/mydatabase?retryWrites=true&w=majority&tls=true&tlsAllowInvalidCertificates=false"

try:
    # Use the correct MongoClient with properly formatted URI
    client = MongoClient(uri, server_api=ServerApi('1') ) # 5 seconds for selecting server)
    
    # Ping the MongoDB server to check the connection
    client.admin.command('ping')
    print("Successfully connected to MongoDB")
except Exception as e:
    print(e)


client = MongoClient(uri, tls=True, tlsAllowInvalidCertificates=False, server_api=ServerApi('1'))
# Create a new client and connect to the server
try:
    client.admin.command('ping')
    print("You successfully connected ")
except Exception as e:
    print ("Try again")

#Specify the database name abd the connection
database = client["remedydata"]
collection = database["remedy"]

# Use joblib to load the model instead of pickle
try:
    model = joblib.load('model.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error during model loading: {e}")

#The remedy recomendation function 
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        remedy_type = request.form['remedy-type']
        conditions = request.form.getlist('condition-types')
        age_group = request.form['age-group']
        
        # Safely access the first element or set to None if the list is empty
        hair_type = request.form.getlist('hair-type')[0] if request.form.getlist('hair-type') else None
        environment_condition = request.form.getlist('environment-condition')[0] if request.form.getlist('environment-condition') else None

        remedy_names, remedy_indexes = model.recommend_remedies(remedy_type, conditions, hair_type, age_group, environment_condition)

        response = [name for name in remedy_indexes]
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/search')
def search():
    keyword = request.args.get('keyword')
    remedies = []
    ingrdients = collection.find({"ingredients": {"$regex": keyword, "$options": "i"}})
    for remedy in ingrdients:
        remedies.append(remedy)
    try:
        return dumps(remedies)
    except Exception as e:
        logging.error(e)
        return jsonify({'remedy not found '}), 500
    

@app.route('/search-remedies')
def search_remedies():
    name = request.args.get('name')
    print("Name: ", name)
    remedy = collection.find_one({"name": {"$regex": str(name), "$options": "i"}})
    print("Remedy: ", remedy)
    if remedy is not None:
        try:
            response = dumps(remedy)
            return response, 200  # Do not manually set CORS headers here
        except Exception as e:
            logging.error(e)
            return jsonify({'message': 'Remedy not found'}), 500
    else:
        return jsonify({'message': 'Remedy not found'}), 404




# MongoDB connection
uri = "mongodb+srv://researchnazi428:nazi1234567@cluster0.zuzgk1k.mongodb.net/mydatabase?retryWrites=true&w=majority&tls=true&tlsAllowInvalidCertificates=false"
client = MongoClient(uri, server_api=ServerApi('1'))

# Access each database and collection
databases = {
    'remedydata': client['remedydata']['remedy'],
    'haircare': client['haircare']['users'],
    'admin_db': client['admin_db']['admin'],
    'ayurveda_advice_db': client['ayurveda_advice_db']['ayurveda_advice'],
    'payment_db': client['payment_db']['payments'],
    'forum': client['forum']['user_forum'],
    'chatbot_db': client['chatbot_db']['user_queries'],
    'chatbot_history_db': client['chatbot_history_db']['chat_history'],
    'doctor_db': client['doctor_db']['doctor_usernames']

}



# Now you can access each collection by name
remedy_collection = databases['remedydata']
users_collection = databases['haircare']
advice_collection = databases['ayurveda_advice_db']
payment_collection = databases['payment_db']
forum_collection = databases['forum']
chat_collection = databases['chatbot_db']
chat_history_collection = databases['chatbot_history_db']
admin_collection = databases ['admin_db']
doctors_collection =databases['doctor_db']
# Helper function to hash passwords
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# Helper function to verify passwords
def verify_password(stored_password, provided_password):
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password)

@app.route('/signup', methods=['POST'])
def signup():
    data = request.form  # Access form data
    username = data['username']
    email = data['email']
    password = data['password']

    # Ensure email and username are not empty
    if not username or not email or not password:
        
        return jsonify({'success': False, 'message': 'All fields are required'})

    # Check if the email or username already exists in the database
    if users_collection.find_one({'email': email}):
        return jsonify({'success': False, 'message': 'Email already exists'})

    if users_collection.find_one({'username': username}):
        return jsonify({'success': False, 'message': 'Username already exists'})

    # Hash the password and store user details in MongoDB
    hashed_password = hash_password(password)
    users_collection.insert_one({
        'username': username,
        'email': email,
        'password': hashed_password
    })
    return jsonify({'success': True, 'message': 'User registered successfully'})



#login route
@app.route('/login', methods=['POST'])
def login():
    data = request.form  # Get form data
    username = data.get('username')
    password = data.get('password')

    # Check if the user is an admin
    admin = admin_collection.find_one({'username': username, 'role': 'admin'})
    if admin and verify_password(admin['password'], password):
        session.clear()
        session['admin_username'] = admin['username']
        session['admin_email'] = admin['email']
        session['is_admin'] = True  # Mark the session as admin
        return jsonify({'success': True, 'message': 'Admin login successful', 'redirect': '/professional_advice'}), 200

    # If not an admin, check if the user exists in the database
    user = users_collection.find_one({'username': username})
    if user and verify_password(user['password'], password):
        session.clear()
        session['username'] = user['username']
        session['email'] = user['email']
        session['is_admin'] = False  # Mark the session as user
        return jsonify({'success': True, 'message': 'User login successful', 'redirect': '/Home'}), 200
    
    # Check if the doctor exists in the database
    doctor = doctors_collection.find_one({'username': username})
    if doctor and verify_password(doctor['password'], password):
        session.clear()
        session['doctor_username'] = doctor['username']
        session['doctor_email'] = doctor['email']
        session['is_doctor'] = True  # Mark the session as doctor
        return jsonify({'success': True, 'message': 'Doctor login successful', 'redirect': '/doctors'}), 200

    # If no match found, return an error message
    return jsonify({'success': False, 'message': 'Invalid username or password'}), 401



# Sample Ayurvedic advice with Ayurvedic doctors' names (add more to MongoDB)
advice_data = [
    {"condition": "dandruff", 
     "advice": "For dandruff, use neem oil or a hair mask made from fenugreek seeds soaked overnight. Apply the mask and rinse after 30 minutes.", 
     "doctor": "Dr. Sharma, Ayurvedic Practitioner"},
    {"condition": "hair fall", 
     "advice": "Apply a mix of bhringraj oil and amla powder to strengthen hair roots. Also, practice regular head massage with warm coconut oil to improve blood circulation.", 
     "doctor": "Dr. Anjali Verma, Ayurvedic Doctor"},
    {"condition": "dry scalp", 
     "advice": "Use a mixture of aloe vera gel and coconut oil for a soothing scalp massage. Ayurvedic herbs like hibiscus and shikakai can be added to your hair care routine.", 
     "doctor": "Dr. Rajesh Gupta, Ayurvedic Specialist"},
    {"condition": "oily scalp", 
     "advice": "To manage oily scalp, apply a paste of multani mitti (Fuller’s earth) mixed with rose water once a week. This helps absorb excess oil and cleanse the scalp.", 
     "doctor": "Dr. Priya Nair, Ayurvedic Expert"}
]



# Insert Ayurvedic advice into MongoDB for testing (if not already present)
if advice_collection.count_documents({}) == 0:
    advice_collection.insert_many(advice_data)

@app.route('/get_advice', methods=['POST'])
def get_advice():
    data = request.json
    condition = data.get('condition')

    # Search for Ayurvedic advice in MongoDB by condition
    advice = advice_collection.find_one({"condition": condition.lower()})

    if advice:
        response = {"advice": advice["advice"], "doctor": advice["doctor"]}
    else:
        response = {"advice": "Sorry, no Ayurvedic advice found for this condition. Please proceed payment or consult an Ayurvedic practitioner.", "doctor": ""}

    return jsonify(response)



@app.route('/payment', methods=['GET', 'POST'])
def payment():
    try:
        logged_in_user = session.get('username') or session.get('admin_username') or session.get('doctor_username')  # Check both keys

        # Check if the user is logged in
        if not logged_in_user:
            return jsonify({'error': 'User not logged in'}), 401

        # Check if the user is an admin
        if session.get('is_admin'):  # Check the admin flag in the session
            return jsonify({'success': 'Admin bypassed payment', 'redirect': '/forum'}), 200
        
        # Check if the user is a doctor
        if session.get('is_doctor'):  # Check the admin flag in the session
            return jsonify({'success': 'Doctor bypassed payment', 'redirect': '/forum'}), 200

        # Check if the user has already made a payment
        existing_payment = payment_collection.find_one({'username': logged_in_user})
        if existing_payment:
            return jsonify({'success': 'Payment has already been made by this user', 'redirect': '/professional_advice'}), 200

        if request.method == 'GET':
            return render_template('payment.html')

        # Handle payment process
        data = request.get_json()
        payment_method = data.get('payment_method')
        card = data.get('card')
        expiry = data.get('expiry')
        amount = 3000  # Fixed amount

        # Validate inputs
        if not payment_method or not card or not expiry:
            return jsonify({'error': 'Please fill out all fields'}), 400
        if len(card) != 16 or not card.isdigit():
            return jsonify({'error': 'Card number must be exactly 16 digits'}), 400

        # Save payment data
        payment_data = {
            'username': logged_in_user,
            'payment_method': payment_method,
            'card': card,
            'expiry': expiry,
            'amount': amount
        }
        payment_collection.insert_one(payment_data)

        return jsonify({'success': 'Payment processed successfully', 'redirect': '/professional_advice'}), 200

    except Exception as e:
        print(f"Error processing payment: {str(e)}")
        return jsonify({'error': 'An error occurred while processing your payment'}), 500



@app.route('/check_payment_status', methods=['GET'])
def check_payment_status():
    try:
        logged_in_user = session.get('username') or session.get('admin_username') or session.get('doctor_username')  # Check both keys
        
        if not logged_in_user:
            return jsonify({'error': 'User not logged in'}), 401

        # Check if the user is an admin
        if session.get('is_admin'):  # Check the admin flag in the session
            return jsonify({'paid': True})  # Admin is considered to have "paid"
        
        # Check if the user is a doctor
        if session.get('is_doctor'):  
            return jsonify({'paid': True}), 200

        # Check payment status in the database
        existing_payment = payment_collection.find_one({'username': logged_in_user})
        
        if existing_payment:
            return jsonify({'paid': True})
        else:
            return jsonify({'paid': False})
    
    except Exception as e:
        print(f"Error checking payment status: {str(e)}")
        return jsonify({'error': 'An error occurred while checking payment status'}), 500





# Flask-Mail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'researchnazi428@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'fbqv lnns rxxs zfux'         # Replace with your password
mail = Mail(app)


@app.route('/forgot_password', methods=['POST'])
def forgot_password():
    data = request.form
    username = data.get('username')

    user = users_collection.find_one({'username': username})
    if not user:
        return jsonify({'success': False, 'message': 'Username does not exist'}), 404

    pin = ''.join(random.choices(string.digits, k=6))
    users_collection.update_one({"username": username}, {"$set": {"reset_pin": pin}})
    
    email = user.get("email")
    if email:
        msg = Message("Password Reset PIN", sender=app.config['MAIL_USERNAME'], recipients=[email])
        msg.body = f"Your password reset PIN is: {pin}"
        mail.send(msg)
        return jsonify({"message": "PIN sent to your email"})
    else:
        return jsonify({"message": "Email not found for this user"}), 400

@app.route('/new_password', methods=['GET', 'POST'])
def new_password():
    if request.method == 'GET':
        return render_template('New_password.html')

    if request.method == 'POST':
        data = request.json
        pin = data['pin']
        new_password = data['newPassword']
        confirm_password = data['confirmPassword']

        if new_password != confirm_password:
            return jsonify({"message": "Passwords do not match"}), 400

        user = users_collection.find_one({"reset_pin": pin})
        if user:
            hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
            users_collection.update_one({"reset_pin": pin}, {"$set": {"password": hashed_password}, "$unset": {"reset_pin": ""}})
            return jsonify({"message": "Password updated successfully","redirect_url": url_for('login')})
        else:
            return jsonify({"message": "Invalid PIN"}), 400
    




# Initialize the model with paths to datasets and model files
model = Model(
    dataset=r'c:\Users\DELL\Desktop\Final project test\dataset.csv', 
    dataset1=r'c:\Users\DELL\Desktop\Final project test\dataset1.csv', 
    model_path=r'c:\Users\DELL\Desktop\Final project test'
)

CORS(app, resources={r"/chatbot_recommendation": {"origins": "http://localhost:5000"}})

# Route to render the chatbot page
@app.route('/')
def index():
    return render_template('professional_advice.html')
@app.route('/chatbot_recommendation', methods=['POST'])
def chatbot_recommendation():
    data = request.json
    print("Received data:", data)  # Print the entire request to check the format
    
    if not data:
        return jsonify({"error": "No data provided"}), 400

    user_query = data.get('query')
    doctor_name = data.get('doctor_name')

    if not user_query and not doctor_name:
        return jsonify({"error": "Either query or doctor_name must be provided"}), 400

    try:
        if user_query:
            # Get recommendations from the general chatbot
            recommendations = model.recommend_chatbot(user_query)

            # Save query and recommendations to MongoDB
            chat_data = {
                "user_query": user_query,
                "recommendations": recommendations
            }
            chat_history_collection.insert_one(chat_data)
            return jsonify({"recommendations": recommendations}), 200
        
        elif doctor_name:
            # Get recommendations for a specific doctor
            recommendations1 = model.recommend_chatbot_doctor(doctor_name)

            # Save doctor query and recommendations to MongoDB
            chat_data1 = {
                "doctor_name": doctor_name,
                "recommendations": recommendations1
            }
            chat_history_collection.insert_one(chat_data1)
            return jsonify({"recommendations": recommendations1}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500

       
@app.route('/forum')
def forum():
    if 'username' not in session and 'admin_username' not in session:
        return redirect('/login')  # Redirect to login if not authenticated

    posts = list(forum_collection.find())
    for post in posts:
        post['_id'] = str(post['_id'])  # Convert ObjectId to string
        post['is_admin'] = session.get('is_admin', False)  # Check if admin
        post['is_owner'] = session.get('username') == post.get('author')  # Check ownership
        for reply in post['replies']:
            reply['is_owner'] = session.get('username') == reply.get('author')

    return render_template('forum.html', posts=posts)

# Add post route
@app.route('/add_post', methods=['POST'])
def add_post():
    if 'username' not in session and 'admin_username' not in session:
        return jsonify({'success': False, 'message': 'User or Admin not logged in'}), 401

    data = request.json
    title = data.get('title')
    content = data.get('content')

    if not title or not content:
        return jsonify({'success': False, 'message': 'Title and content are required'}), 400

    author = session.get('username') or session.get('admin_username')
    new_post = {
        'title': title,
        'content': content,
        'author': author,
        'replies': []
    }
    forum_collection.insert_one(new_post)
    return jsonify({'success': True, 'message': 'Post added successfully'}), 201

# Edit post route
@app.route('/edit_post/<post_id>', methods=['POST'])
def edit_post(post_id):
    try:
        post = forum_collection.find_one({'_id': ObjectId(post_id)})
    except InvalidId:
        return jsonify({'message': 'Invalid post ID'}), 400

    if not post:
        return jsonify({'message': 'Post not found'}), 404

    author = session.get('username') or session.get('admin_username')
    if post['author'] != author and not session.get('is_admin', False):
        return jsonify({'message': 'Unauthorized to edit this post'}), 403

    data = request.json
    new_content = data.get('new_content')

    if not new_content:
        return jsonify({'message': 'New content is required'}), 400

    forum_collection.update_one({'_id': ObjectId(post_id)}, {'$set': {'content': new_content}})
    return jsonify({'message': 'Post updated successfully'}), 200

# Delete post route
@app.route('/delete_post/<post_id>', methods=['POST'])
def delete_post(post_id):
    try:
        post = forum_collection.find_one({'_id': ObjectId(post_id)})
    except InvalidId:
        return jsonify({'message': 'Invalid post ID'}), 400

    if not post:
        return jsonify({'message': 'Post not found'}), 404

    author = session.get('username') or session.get('admin_username')
    if session.get('is_admin', False) or post['author'] == author:
        forum_collection.delete_one({'_id': ObjectId(post_id)})
        return jsonify({'message': 'Post deleted successfully'}), 200

    return jsonify({'message': 'Unauthorized to delete this post'}), 403

# Add reply route
@app.route('/add_reply/<post_id>', methods=['POST'])
def add_reply(post_id):
    if 'username' not in session and 'admin_username' not in session:
        return jsonify({'success': False, 'message': 'User or Admin not logged in'}), 401

    try:
        post = forum_collection.find_one({'_id': ObjectId(post_id)})
    except InvalidId:
        return jsonify({'message': 'Invalid post ID'}), 400

    if not post:
        return jsonify({'message': 'Post not found'}), 404

    data = request.json
    reply_content = data.get('reply_content')

    if not reply_content:
        return jsonify({'message': 'Reply content is required'}), 400

    author = session.get('username') or session.get('admin_username')
    new_reply = {
        'id': ObjectId(),
        'content': reply_content,
        'author': author
    }

    forum_collection.update_one({'_id': ObjectId(post_id)}, {'$push': {'replies': new_reply}})
    return jsonify({'success': True, 'message': 'Reply added successfully'}), 201

# Edit reply route
@app.route('/edit_reply/<post_id>/<reply_id>', methods=['POST'])
def edit_reply(post_id, reply_id):
    try:
        post = forum_collection.find_one({'_id': ObjectId(post_id)})
    except InvalidId:
        return jsonify({'message': 'Invalid post or reply ID'}), 400

    if not post:
        return jsonify({'message': 'Post not found'}), 404

    reply = next((r for r in post['replies'] if str(r['id']) == reply_id), None)

    if not reply:
        return jsonify({'message': 'Reply not found'}), 404

    author = session.get('username') or session.get('admin_username')
    if reply['author'] != author and not session.get('is_admin', False):
        return jsonify({'message': 'Unauthorized to edit this reply'}), 403

    data = request.json
    new_reply_content = data.get('new_reply_content')

    if not new_reply_content:
        return jsonify({'message': 'New reply content is required'}), 400

    forum_collection.update_one(
        {'_id': ObjectId(post_id), 'replies.id': ObjectId(reply_id)},
        {'$set': {'replies.$.content': new_reply_content}}
    )
    return jsonify({'message': 'Reply updated successfully'}), 200

# Delete reply route
@app.route('/delete_reply/<post_id>/<reply_id>', methods=['POST'])
def delete_reply(post_id, reply_id):
    try:
        post = forum_collection.find_one({'_id': ObjectId(post_id)})
    except InvalidId:
        return jsonify({'message': 'Invalid post or reply ID'}), 400

    if not post:
        return jsonify({'message': 'Post not found'}), 404

    reply = next((r for r in post['replies'] if str(r['id']) == reply_id), None)

    if not reply:
        return jsonify({'message': 'Reply not found'}), 404

    author = session.get('username') or session.get('admin_username')
    if reply['author'] != author and not session.get('is_admin', False):
        return jsonify({'message': 'Unauthorized to delete this reply'}), 403

    forum_collection.update_one(
        {'_id': ObjectId(post_id)},
        {'$pull': {'replies': {'id': ObjectId(reply_id)}}}
    )
    return jsonify({'message': 'Reply deleted successfully'}), 200


# Hash password
password = "Admin123!"
hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# Insert admin user
admin_data = {
    "username": "admin",
    "password": hashed_password.decode('utf-8'),  # Store as string
    "email": "admin@example.com",
    "role": "admin"
}

# Check if admin already exists
if not admin_collection.find_one({"username": "admin"}):
    admin_collection.insert_one(admin_data)
    print("Admin user inserted successfully!")
else:
    print("Admin user already exists!")



# Assuming you have the following verify_password function:
def verify_password(stored_password, provided_password):
    # Convert stored password (string) to bytes
    stored_password_bytes = stored_password.encode('utf-8') if isinstance(stored_password, str) else stored_password
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password_bytes)










# Hash password
password1= "Raj123!"
hashed_password1 = bcrypt.hashpw(password1.encode('utf-8'), bcrypt.gensalt())
# Add a new doctor
new_doctor = {
    "username": "drrajpatel",
    "email": "raj.patel@gmail.com",
    "password": hashed_password1.decode('utf-8'),  # Store as string
    "role": "doctor",
    "profile": {
        "name": "Dr. Raj Patel",
        "qualification": "Dermatologist (MD)",
        "profilePic": "path_to_image",
        "contactNumber": "071-425-6354",
        "socialLinks": {
            "linkedin": "https://linkedin.com/in/raj-patel",
            "facebook": "https://facebook.com/raj.patel"
        },
        "posts": []  # Empty initially, to be filled later
    }
}



# Check if admin already exists
if not doctors_collection.find_one({"username": "drrajpatel"}):
    doctors_collection.insert_one(new_doctor)
    print("Doctor  inserted successfully!")
else:
    print("Doctor already exists!")



from flask import session, flash

@app.route('/doctor_wall/<username>', methods=['GET', 'POST'])
def doctor_wall(username):
    # Check if the user is logged in and is a doctor
    if 'username' not in session or session.get('is_doctor') != True:
        return "Unauthorized access", 403

    # Ensure the logged-in doctor matches the username
    if session['doctor_username'] != username:
        return "Unauthorized access", 403

    doctor = doctors_collection.find_one({"username": username})
    if not doctor:
        return "Doctor not found", 404

    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'add_post':
            # Add a new post
            post_title = request.form.get('title')
            post_content = request.form.get('content')
            new_post = {
                "title": post_title,
                "content": post_content,
                "date": datetime.datetime.now().strftime("%Y-%m-%d")
            }
            doctors_collection.update_one(
                {"username": username},
                {"$push": {"profile.posts": new_post}}
            )
            flash("Post added successfully!", "success")
        
        elif action == 'edit_post':
            # Edit an existing post
            post_id = request.form.get('post_id')
            updated_title = request.form.get('title')
            updated_content = request.form.get('content')
            doctors_collection.update_one(
                {"username": username, "profile.posts._id": post_id},
                {"$set": {
                    "profile.posts.$.title": updated_title,
                    "profile.posts.$.content": updated_content
                }}
            )
            flash("Post updated successfully!", "success")
        
        elif action == 'delete_post':
            # Delete a post
            post_id = request.form.get('post_id')
            doctors_collection.update_one(
                {"username": username},
                {"$pull": {"profile.posts": {"_id": post_id}}}
            )
            flash("Post deleted successfully!", "success")
        
        elif action == 'add_video':
            # Add a new video
            video_url = request.form.get('video_url')
            new_video = {"url": video_url, "date": datetime.datetime.now().strftime("%Y-%m-%d")}
            doctors_collection.update_one(
                {"username": username},
                {"$push": {"profile.videos": new_video}}
            )
            flash("Video added successfully!", "success")
        
        elif action == 'delete_video':
            # Delete a video
            video_url = request.form.get('video_url')
            doctors_collection.update_one(
                {"username": username},
                {"$pull": {"profile.videos": {"url": video_url}}}
            )
            flash("Video deleted successfully!", "success")
        
        elif action == 'add_picture':
            # Add a new picture
            picture_url = request.form.get('picture_url')
            new_picture = {"url": picture_url, "date": datetime.datetime.now().strftime("%Y-%m-%d")}
            doctors_collection.update_one(
                {"username": username},
                {"$push": {"profile.pictures": new_picture}}
            )
            flash("Picture added successfully!", "success")
        
        elif action == 'delete_picture':
            # Delete a picture
            picture_url = request.form.get('picture_url')
            doctors_collection.update_one(
                {"username": username},
                {"$pull": {"profile.pictures": {"url": picture_url}}}
            )
            flash("Picture deleted successfully!", "success")
        
        return redirect(f'/doctor_wall/{username}')

    return render_template('doctor_wall.html', doctor=doctor)



@app.route('/About')
def about():
    return render_template('About.html')

@app.route('/doctors')
def doctors():
    return render_template('doctors.html')

# Route to render Ayurveda Advice page
@app.route('/ayurveda_advice')
def ayurveda_advice():
    return render_template('ayurveda_advice.html')

# Route to render Home page
@app.route('/Home')
def home_page():
    return render_template('Home.html')


# Route to render Login page
@app.route('/login')
def login_page():
    return render_template('login.html')

# Route to render Login page
@app.route('/admin-login')
def admin_page():
    return render_template('admin.html')

# Route to render Signup page
@app.route('/signup')
def signup_page():
    return render_template('signup.html')

# Route to render Forgot Password page
@app.route('/forgot_password')
def forgot_password_page():
    return render_template('forgot_password.html')

# Route to render Remedy Page
@app.route('/RemedyPage')
def remedy_page():
    return render_template('RemedyPage.html')

# Route to render Index page
@app.route('/index')
def index_page():
    return render_template('index.html')

# Route to render Forum page
@app.route('/forum')
def forum_page():
    return render_template('forum.html')

# Route to render Thank You page
@app.route('/thankyou')
def thankyou_page():
    return render_template('thankyou.html')

@app.route('/professional_advice')
def professional_advice():
    # Render the professional_advice page if the user has paid
    return render_template('professional_advice.html')

@app.route('/admin-login')
def admin_login1():
    # Render the admin login page .
    return render_template('admin.html')




if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
