# Human Activity Recognition Using Smartphones

This project trains a Random Forest classifier on the UCI HAR Dataset to recognize human activities from smartphone sensor data.

Repository: https://github.com/mohananagaraj2005-bot/human-activity-recognition

Requirements:
- Place the extracted `UCI HAR Dataset` folder in the same directory as this project.
- Python 3.8+

How to run:
1. (Optional) Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows

2. Install dependencies:
   pip install -r requirements.txt

3. Ensure the folder `UCI HAR Dataset/` is next to `human_activity.py`.

4. Run the script:
   python human_activity.py

Files included:
- human_activity.py
- requirements.txt
- .gitignore
- README.md

Notes:
- The raw dataset folder (`UCI HAR Dataset/`) is included in `.gitignore` and should NOT be committed to the repository. Download and extract the dataset locally before running the script.
- If you want me to add a license file, example outputs, or CI workflow, tell me and I'll add them.
