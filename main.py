#flask is witteen in python it is a web framework 
# it is used to implement web server and can serve html files to it
from flask import Flask,render_template,request
import pickle
app = Flask(__name__)

# we open the file for reading
fileObject = open('model.pkl','rb')  
# load the object from the file into var b
clf = pickle.load(fileObject)
fileObject.close()

@app.route('/',methods=["GET","POST"])
def myrender_function():
    if request.method == "POST":
        myDict = request.form
        firstName = myDict.get('firstName')
        lastName = myDict.get('lastName')
        fever = int(myDict.get('fever'))
        bodyPain = int(myDict.get('bodyPain'))
        age = int(myDict.get('age'))
        runnyNose = int(myDict.get('runnyNose'))
        diffBreathe = int(myDict.get('diffBreathe'))
        # code for inference   
        # input_features = [100,1,22,-1,1]
        input_features = [fever,bodyPain,age,runnyNose,diffBreathe]
        inf= clf.predict_proba([input_features])
        # infProb==> array([[0.47945127, 0.52054873]])
        inf = inf[0][1] #actual probability
        print(inf)
        params = {
            'infProb': round(inf*100,2),
            'firstName':firstName,
            'lastName':lastName
            }
        return render_template('show.html',para = params)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)