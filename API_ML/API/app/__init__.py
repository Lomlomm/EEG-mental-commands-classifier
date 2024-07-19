from flask import Flask
import services.processed_data as ProcessData
import model.svm_model as SVModel

app = Flask(__name__)


@app.route('/home', methods=['GET'])
def home():
    return 'HELLO'

@app.route('/processData', methods=['GET'])
def getProcessData():
    return ProcessData.getProcessData()

@app.route('/getEvaluationData', methods=['GET'])
def getEvaluationData():
    return ProcessData.getEvaluationData()


@app.route('/modelSVM', methods=['GET'])
def modelSVM():
    return SVModel.main()

if __name__ == '__main__':
    app.run(debug=True)
