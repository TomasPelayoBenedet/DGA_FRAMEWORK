from flask import Flask

from APIrest.apirest import APIrest

from classifiers.common.dataElementCommon import DataElementCommon

apirest = APIrest()

app = Flask(__name__)

@app.route('/')
def index():
  return 'Server Works!'
  
@app.route('/predict/<domainame>', methods=['GET'])
def predict(domainame):
  
  if len(domainame) > 70:
    return "Domain name too large to this version of APIrest! (Len < 70 characters)"

  dataElement = DataElementCommon(domainame, False)
  
  prob_dga = apirest.predict(dataElement=dataElement)
  
  ret_str = "Domain name: " + domainame + "\n"
  ret_str = ret_str + "Probability of being DGA: " + str(prob_dga)
  return ret_str