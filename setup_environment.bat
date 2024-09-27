@echo off
:: Create a Python virtual environment in the 'venv' directory
python -m venv venv

:: Activate the virtual environment
call venv\Scripts\activate

:: Upgrade pip to the latest version
python -m pip install --upgrade pip

:: Install the required packages from requirements.txt
pip install -r requirements.txt

:: Confirmation message
echo Environment setup complete!
pause
