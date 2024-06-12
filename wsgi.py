# wsgi.py
from main import app

if _name_ == "_main_":
    app.run()