#!/bin/sh
cd {{DIST_PATH}}
#Todo: import propertie files as dictionnaries in XLD
export DEBUG=True
export DJANGO_SECRET_KEY=eX@SecR3TkeY
export ALLOWED_HOSTS=localhost,127.0.0.1 
export DATABASE_URL= 
python3.8 -m venv envPython
source ./envPython/bin/activate
tar -xvf exa-python-django-starter-kit.tar.gz
cd app-src
python -m pip install -r requirements.txt --no-index --find-links ../dependencies/
python -m gunicorn config.wsgi:application --daemon

