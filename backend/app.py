from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from model.RBMBakeOff import model
# End of imports


app = Flask(__name__)

# creating an API object
api = Api(app)

class RBM(Resource):
    def get(self):
        # call the function here with required parameters
        rbm = model()
        rbm.train_model()
        recs = rbm.recs('harasees_singh', 5)
        return jsonify({"recommendations": recs})

api.add_resource(RBM, '/rbm')

if __name__ == '__main__':
    app.run()