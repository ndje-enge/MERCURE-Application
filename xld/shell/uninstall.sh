#!/bin/sh
cd {{DIST_PATH}}
# Ménage de l'environnement virtuel
source ./envPython/bin/activate
# Récupérer la liste des process de l'application
gunicorns=$(ps -aef | grep 'python3 -m gunicorn config.wsgi:application --daemon' | awk '{ print $2 }')
kill -9 $gunicorns
# Ménage de l'environnement virtuel
python3 -m pip uninstall -y -r requirements.txt
rm -rf app-src dependencies exa-python-django-starter-kit.tar.gz pyproject.toml envPython

