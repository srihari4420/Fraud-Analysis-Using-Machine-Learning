from flask import Flask , render_template , request
import numpy as np 
import pickle
import sklearn

from sklearn.linear_model import LogisticRegression

model = pickle.load(open('D:\\Python codes for data science\\Ekkeda Nit Bacth\\ML\\Mini Projects\\classification project\\credit_project.pkl','rb'))
standard_scalar = pickle.load(open('D:\\Python codes for data science\\Ekkeda Nit Bacth\\ML\\Mini Projects\\classification project\\stadardscalar.pkl','rb'))



app = Flask(__name__)

@app.route('/')
def fun():
    return render_template('index.html')

@app.route('/predict' , methods = ['GET','POST'])
def fun1():
    a = [i for i in request.form.values()]

    a = [int(j) if j.isdigit() else float(j) for j in a]

    a = np.array([a])

    res = standard_scalar.transform(a)

    sol = model.predict(res)[0]

    if sol == 0:
        return render_template('index.html' , value = 'It is a Bad Transcation')
    else:
        return render_template('index.html' , value = 'It is a Good Transcation')



if __name__ == '__main__':
    app.run(debug = True)