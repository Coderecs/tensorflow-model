from flask import Flask, jsonify, request
from flask_restful import Resource, Api
# End of imports


app = Flask(__name__)

# creating an API object
api = Api(app)

class RBM(Resource):
    def get(self):
        # call the function here with required parameters
        return jsonify({"data": "Recommendations"})

api.add_resource(RBM, '/rbm')

if __name__ == '__main__':
    app.run()