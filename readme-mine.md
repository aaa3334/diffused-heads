
#### Update git


git add .
git commit -m ""
git push origin main



### Starting the VENV to create the tables:



python3.10 -m venv myenv
source myenv/bin/activate
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
pip install -r requirements.txt
python main.py

To exit: 
deactivate